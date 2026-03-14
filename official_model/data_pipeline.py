# official_model/data_pipeline.py
from __future__ import annotations
from dataclasses import dataclass
from genericpath import exists
from pathlib import Path
from typing import Optional, Tuple, Dict, Sequence

import os, math
import numpy as np
import pandas as pd
from pyparsing import col
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.data.experimental import copy_to_device
from tensorflow.data.experimental import assert_cardinality

# =========================
# Constantes/labels
# =========================
CLASS_NAMES = ["Normal", "Benign", "Malignant"]
LABEL_CANDIDATES = ("label", "classification", "class", "class_id", "target", "labels")
DEFAULT_LABEL_MAP = {"Normal": 0, "Benign": 1, "Malignant": 2}

# Aceita sinônimos/idiomas e normaliza
_LABEL_NORMALIZER = {
    # inglês
    "normal": "Normal", "benign": "Benign", "malignant": "Malignant",
    # pt-br
    "benigno": "Benign", "maligno": "Malignant",
    # atalhos comuns
    "negativo": "Normal", "positivo": "Malignant",
}

def _ensure_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante df['label'] em {0,1,2}, a partir de qualquer coluna candidata:
    - numérica: 0/1/2
    - string: Normal/Benign/Malignant (ou PT-BR), case/whitespace-insensitive
    """
    # 1) acha a coluna candidata
    label_col = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
    if label_col is None:
        raise KeyError(f"Nenhuma coluna de rótulo encontrada. Procurei {LABEL_CANDIDATES}. Colunas do CSV: {list(df.columns)}")

    col = df[label_col]
    # 2) numérica {0,1,2}
    if np.issubdtype(col.dtype, np.number):
        vals = set(pd.unique(col.dropna()).tolist())
        if not vals.issubset({0, 1, 2}):
            raise ValueError(f"Coluna '{label_col}' é numérica mas não está em {{0,1,2}}. Valores únicos: {sorted(vals)}")
        df["label"] = col.astype(np.int32)
        # opcional: traduz para string legível
        df["classification"] = df["label"].map({0: "Normal", 1: "Benign", 2: "Malignant"})
        return df

    # 3) string -> normaliza (strip, case) e mapeia EN/PT
    norm = col.astype(str).str.strip().str.lower()
    norm = norm.replace(_LABEL_NORMALIZER)  # pode voltar "Normal/Benign/..."
    canon = norm.str.lower()                # COMPARA SEMPRE EM lower

    ok_vals = {"normal", "benign", "malignant"}
    ok_mask = canon.isin(ok_vals)
    if not ok_mask.any():
        raise ValueError(
            f"Não reconheci rótulos em '{label_col}'. Exemplos: {col.head(5).tolist()}. "
            f"Aceito: Normal/Benign/Malignant (ou Benigno/Maligno/Normal)."
        )

    df = df[ok_mask].copy()
    df["classification"] = canon[ok_mask].str.title()           # "Normal/Benign/Malignant"
    df["label"] = df["classification"].map(DEFAULT_LABEL_MAP).astype(np.int32)
    return df

def _join_csv_path(p: str, base_dir: Path, img_root: Path) -> str:
    fast = os.environ.get("MAMMO_FAST_RESOLVE","1") == "1"
    s = str(p).strip().replace("\\", "/")
    if not s: return ""
    q = Path(s)
    if q.is_absolute(): return s
    if s.startswith("Preprocessed_Dataset/") or s.startswith("Original_Dataset/"):
        return str((base_dir / s) if fast else (base_dir / s).resolve())
    return str((img_root / s) if fast else (img_root / s).resolve())

# =========================
# Augment config
# =========================
@dataclass(frozen=True)
class AugmentConfig:
    flip_lr: bool = True                 # flip horizontal
    brightness_delta: float = 0.03       # ±delta (0..1)
    contrast_lower: float = 0.92         # multiplicador min
    contrast_upper: float = 1.08         # multiplicador max
    translate_frac: float = 0.05         # até 5% H/W
    zoom_min: float = 0.96               # 0.96..1.04
    zoom_max: float = 1.04
    noise_std: float = 0.005             # N(0, std)
    cutout_prob: float = 0.25            # prob. aplicar
    cutout_max_frac: float = 0.08        # área máx. do retângulo

# =========================
# Resolvedor de caminhos
# =========================
def _resolve_paths(
    data_dir: Optional[str] = None,
    csv_path: Optional[str] = None,
) -> Tuple[Path, Path, Path]:
    """
    Retorna (BASE_DIR, CSV_FILE, IMG_ROOT).
    - BASE_DIR = .../Mammo-Bench
    - CSV_FILE = CSV_Files/mammo-bench_nbm_classification*.csv
    - IMG_ROOT = Preprocessed_Dataset
    """
    if csv_path:
        csv = Path(csv_path).expanduser().resolve()
        assert csv.exists(), f"CSV não encontrado: {csv}"
        base = csv.parent.parent  # .../Mammo-Bench/CSV_Files/<csv>
        return base, csv, base / "Preprocessed_Dataset"

    if data_dir:
        base = Path(data_dir).expanduser().resolve()
    else:
        env = os.environ.get("MAMMO_PATH")
        if env:
            base = Path(env).expanduser().resolve()
        else:
            repo_root = Path(__file__).resolve().parents[1]
            base = (repo_root / "dataset" / "Mammo_Data" / "Mammo-Bench").resolve()

    csv_candidates = list((base / "CSV_Files").glob("mammo-bench_nbm_classification*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"CSV N/B/M não encontrado em {base/'CSV_Files'}")
    return base, csv_candidates[0], base / "Preprocessed_Dataset"

# =========================
# Label helpers
# =========================
def _normalize_label_column(df: pd.DataFrame, label_col=None, label_map=None) -> pd.DataFrame:
    """Garante coluna 'label' int [0..K-1]."""
    col = None
    if label_col and label_col in df.columns:
        col = label_col
    else:
        for c in LABEL_CANDIDATES:
            if c in df.columns:
                col = c; break
    if col is None:
        raise KeyError(f"Nenhuma coluna de label encontrada. Colunas={list(df.columns)}")

    ser = df[col]
    lm = (label_map or DEFAULT_LABEL_MAP).copy()
    lm.update({"0": 0, "1": 1, "2": 2})

    if ser.dtype == object:
        mapped = ser.astype(str).map(lm)
    else:
        mapped = ser

    try:
        mapped = mapped.astype("int64")
    except Exception:
        codes, _ = pd.factorize(ser.astype(str))
        mapped = pd.Series(codes, index=df.index, dtype="int64")

    df = df.rename(columns={col: "label"})
    df["label"] = mapped.astype("int64")
    return df

# =========================
# Leitura e split
# =========================
def read_nbm_dataset(
    verify_paths: bool = False,
    data_dir: Optional[str] = None,
    csv_path: Optional[str] = None,
    val_size: float = 0.15,
    test_size: float = 0.15, # <<< ADICIONADO
    max_per_class: int = 7000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]: # <<< MODIFICADO (retorna test_df)
    base_dir, csv_file, img_root = _resolve_paths(data_dir, csv_path)
    df = pd.read_csv(csv_file, low_memory=False)
    # garante 'label' robusto (0/1/2) e 'classification' legível
    df = _ensure_label_column(df)

    # Log útil para diagnosticar
    try:
        print("[DATA] CSV:", csv_file.name, "| colunas:", list(df.columns)[:12], "...")
        print("[DATA] Dist. por label:", df["label"].value_counts().to_dict())
    except Exception:
        pass

    # escolhe coluna do caminho (pré-processado)
    img_col = "preprocessed_image_path" if "preprocessed_image_path" in df.columns else \
              "processed_image_path" if "processed_image_path" in df.columns else \
              "image_path"
    if img_col not in df.columns:
        raise KeyError("Nenhuma coluna de caminho encontrada (preprocessed_image_path/processed_image_path/image_path).")

    # --- monta caminhos (sem .resolve() quando FAST) ---
    df["img_path"] = df[img_col].apply(lambda p: _join_csv_path(p, base_dir, img_root))

    # --- verificação de existência (robusta e opcional por env) ---
    _fast_no_exists = os.environ.get("FAST_NO_EXISTS", "0") == "1"
    if not _fast_no_exists:
        exists_mask = df["img_path"].map(os.path.exists)
        missing = int((~exists_mask).sum())
        if missing > 0:
            print(f"[WARN] {missing} caminhos ausentes; filtrando.")
        df = df[exists_mask].reset_index(drop=True)
    else:
        print("[FAST] Pulando verificação de existência de arquivos.")

    # --- garante 'label' como coluna (evita sumir no índice) ---
    if ("label" not in df.columns) and ("label" in getattr(df.index, "names", [])):
        df = df.reset_index()
    if "label" not in df.columns:
        if "classification" in df.columns:
            df["label"] = df["classification"].map(DEFAULT_LABEL_MAP).astype(np.int32)
        else:
            raise KeyError("Coluna 'label' ausente após normalização.")

    # --- limite máximo por classe (versão estável p/ qualquer pandas) ---
    if max_per_class and max_per_class > 0:
        chunks = []
        for k, g in df.groupby("label", sort=False):
            n = min(len(g), max_per_class)
            chunks.append(g.sample(n=n, random_state=seed))
        df = pd.concat(chunks, ignore_index=True)

    # --- split estratificado ---
    _min = df.groupby("label").size().min()
    strat = df["label"] if _min >= 2 else None

    # <<< INÍCIO DO BLOCO MODIFICADO >>>
    # 1. Separar Teste do resto (Train+Val)
    # Se test_size=0, test_df ficará vazio e train_val_df será o dataframe completo
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=strat,
        random_state=seed
    )

    # 2. Calcular o val_size proporcional para o que sobrou
    # (Ex: se test_size=0.15 e val_size=0.15, queremos 15% do total,
    # que é 0.15 / (1.0 - 0.15) = 0.17647... dos 85% restantes)
    remaining_frac = 1.0 - test_size
    if remaining_frac > 0.001: # Evitar divisão por zero
        # Ajusta val_size para ser relativo ao dataframe train_val_df
        val_size_relative = val_size / remaining_frac
    else:
        # Se test_size for 1.0 (ou mto perto), não sobra nada para val
        val_size_relative = 0.0

    # 3. Separar Train e Val do 'train_val_df'
    # Se val_size_relative for 0 (ou 1.0), o split ainda funciona
    _min_tv = train_val_df.groupby("label").size().min() if len(train_val_df) > 0 else 0
    strat_tv = train_val_df["label"] if _min_tv >= 2 else None

    if len(train_val_df) > 0:
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_relative,
            stratify=strat_tv,
            random_state=seed
        )
    else:
        # Caso extremo: test_size = 1.0, train_df e val_df ficam vazios
        train_df = train_val_df.copy()
        val_df = train_val_df.copy()

    # Resetar índices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    # <<< FIM DO BLOCO MODIFICADO >>>


    # class weights (baseado apenas no treino)
    classes_sorted = np.array(sorted(train_df["label"].unique()), dtype=np.int64)
    y_train = train_df["label"].to_numpy(dtype=np.int64)
    
    class_weights = {}
    if len(y_train) > 0:
        cw_arr = compute_class_weight(class_weight="balanced", classes=classes_sorted, y=y_train)
        class_weights = {int(c): float(w) for c, w in zip(classes_sorted, cw_arr)}

    meta = dict(
        n_train=len(train_df),
        n_val=len(val_df),
        n_test=len(test_df), # <<< ADICIONADO
        class_weights=class_weights,
        class_map=DEFAULT_LABEL_MAP,
        base_dir=str(base_dir),
        csv=str(csv_file),
    )
    return train_df, val_df, test_df, meta # <<< MODIFICADO (retorna test_df)

# =========================
# tf.data helpers
# =========================
def _decode_resize(path: tf.Tensor, label: tf.Tensor, img_size: int, channels: int):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=channels, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, [img_size, img_size], antialias=True)
    return img, label

# ---- stateless aug pieces ----
def _apply_flip_lr(x, do, seed):
    return tf.image.stateless_random_flip_left_right(x, seed) if do else x

def _apply_brightness_contrast(x, cfg: AugmentConfig, seed):
    if cfg.brightness_delta and cfg.brightness_delta > 0:
        delta = tf.random.stateless_uniform([], seed=seed, minval=-cfg.brightness_delta, maxval=cfg.brightness_delta)
        x = tf.clip_by_value(x + delta, 0.0, 1.0)
    if cfg.contrast_lower and cfg.contrast_upper and cfg.contrast_upper > cfg.contrast_lower:
        x = tf.image.stateless_random_contrast(x, lower=cfg.contrast_lower, upper=cfg.contrast_upper, seed=seed)
    return x

def _apply_translate(x, cfg: AugmentConfig, seed):
    if cfg.translate_frac <= 0:
        return x
    h = tf.shape(x)[0]; w = tf.shape(x)[1]
    pad_y = tf.cast(tf.round(tf.cast(h, tf.float32) * cfg.translate_frac), tf.int32)
    pad_x = tf.cast(tf.round(tf.cast(w, tf.float32) * cfg.translate_frac), tf.int32)
    xpad = tf.pad(x, [[pad_y, pad_y], [pad_x, pad_x], [0, 0]], mode="REFLECT")
    off_y = tf.random.stateless_uniform([], seed=seed, minval=0, maxval=pad_y*2+1, dtype=tf.int32)
    off_x = tf.random.stateless_uniform([], seed=[seed[0], seed[1]+1], minval=0, maxval=pad_x*2+1, dtype=tf.int32)
    return tf.image.crop_to_bounding_box(xpad, off_y, off_x, h, w)

def _apply_zoom(x, cfg, seed):
    if cfg.zoom_min == 1.0 and cfg.zoom_max == 1.0:
        return x
    h = tf.shape(x)[0]; w = tf.shape(x)[1]
    z = tf.random.stateless_uniform([], seed=seed, minval=cfg.zoom_min, maxval=cfg.zoom_max)
    new_h = tf.cast(tf.round(tf.cast(h, tf.float32)*z), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(w, tf.float32)*z), tf.int32)
    x_resized = tf.image.resize(x, (new_h, new_w), method="bilinear")

    def _zoom_in():
        max_off_y = tf.maximum(new_h - h + 1, 1)
        max_off_x = tf.maximum(new_w - w + 1, 1)
        off_y = tf.random.stateless_uniform([], seed=[seed[0], seed[1]+2], minval=0, maxval=max_off_y, dtype=tf.int32)
        off_x = tf.random.stateless_uniform([], seed=[seed[0], seed[1]+3], minval=0, maxval=max_off_x, dtype=tf.int32)
        return tf.image.crop_to_bounding_box(x_resized, off_y, off_x, h, w)

    def _zoom_out():
        pad_y = (h - new_h) // 2
        pad_x = (w - new_w) // 2
        return tf.pad(x_resized, [[pad_y, h-new_h-pad_y], [pad_x, w-new_w-pad_x], [0,0]], mode="REFLECT")

    return tf.cond(z > 1.0, _zoom_in, _zoom_out)


def _apply_noise(x, cfg: AugmentConfig, seed):
    if cfg.noise_std and cfg.noise_std > 0:
        n = tf.random.stateless_normal(tf.shape(x), seed=seed, stddev=cfg.noise_std, dtype=tf.float32)
        x = tf.clip_by_value(x + n, 0.0, 1.0)
    return x

def _apply_cutout(x, cfg: AugmentConfig, seed):
    if cfg.cutout_prob <= 0 or cfg.cutout_max_frac <= 0:
        return x
    p = tf.random.stateless_uniform([], seed=seed)
    def _do():
        h = tf.shape(x)[0]; w = tf.shape(x)[1]
        max_area = tf.cast(tf.round(tf.cast(h*w, tf.float32) * cfg.cutout_max_frac), tf.int32)
        side = tf.maximum(tf.cast(tf.round(tf.sqrt(tf.cast(max_area, tf.float32))), tf.int32), 1)
        off_y = tf.random.stateless_uniform([], seed=[seed[0], seed[1]+5], minval=0, maxval=tf.maximum(h - side + 1, 1), dtype=tf.int32)
        off_x = tf.random.stateless_uniform([], seed=[seed[0], seed[1]+6], minval=0, maxval=tf.maximum(w - side + 1, 1), dtype=tf.int32)
        yy = tf.range(h)[:, None]; xx = tf.range(w)[None, :]
        m_y = tf.logical_and(tf.greater_equal(yy, off_y), tf.less(yy, off_y + side))
        m_x = tf.logical_and(tf.greater_equal(xx, off_x), tf.less(xx, off_x + side))
        mask = tf.logical_not(tf.logical_and(m_y, m_x))
        mask = tf.cast(mask, tf.float32)[..., None]
        fill_val = tf.reduce_mean(x)
        return x * mask + (1.0 - mask) * fill_val
    return tf.cond(p < cfg.cutout_prob, _do, lambda: x)

def _augment_image(x, cfg: AugmentConfig, seed):
    # ordem: flip -> translate -> zoom -> brilho/contraste -> noise -> cutout
    x = _apply_flip_lr(x, cfg.flip_lr, seed)
    x = _apply_translate(x, cfg, [seed[0], seed[1]+10])
    x = _apply_zoom(x, cfg, [seed[0], seed[1]+20])
    x = _apply_brightness_contrast(x, cfg, [seed[0], seed[1]+30])
    x = _apply_noise(x, cfg, [seed[0], seed[1]+40])
    x = _apply_cutout(x, cfg, [seed[0], seed[1]+50])
    return tf.clip_by_value(x, 0.0, 1.0)

def _aug_summary(cfg: AugmentConfig) -> str:
    parts = []
    if cfg.flip_lr: parts.append("flip LR")
    if cfg.translate_frac>0: parts.append(f"translate ±{int(cfg.translate_frac*100)}%")
    if cfg.zoom_min!=1.0 or cfg.zoom_max!=1.0: parts.append(f"zoom [{cfg.zoom_min:.2f},{cfg.zoom_max:.2f}]")
    if cfg.brightness_delta>0: parts.append(f"brightness ±{cfg.brightness_delta:.02f}")
    if cfg.contrast_lower!=1.0 or cfg.contrast_upper!=1.0: parts.append(f"contrast [{cfg.contrast_lower:.2f},{cfg.contrast_upper:.2f}]")
    if cfg.noise_std>0: parts.append(f"noise σ={cfg.noise_std:.3f}")
    if cfg.cutout_prob>0: parts.append(f"cutout p={cfg.cutout_prob:.2f} a≤{int(cfg.cutout_max_frac*100)}%")
    return " + ".join(parts) if parts else "none"

def make_val_ds(paths, labels, img_size: int, channels: int, batch_size: int):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p,y: _decode_resize(p,y,img_size,channels),
                num_parallel_calls=tf.data.AUTOTUNE)
    opt = tf.data.Options(); opt.experimental_deterministic = False
    ds = ds.with_options(opt).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # cardinalidade conhecida = ceil(len(paths)/batch)
    val_steps = (len(paths) + batch_size - 1) // batch_size
    ds = ds.apply(assert_cardinality(val_steps))
    return ds

def make_train_ds_plus_aug(
    paths, labels, img_size: int, channels: int, batch_size: int,
    aug_cfg: AugmentConfig, extra_fraction: float = 0.0, seed: int = 42
):
    # limites para não estourar RAM/CPU/SSD
    PARALLEL = int(os.getenv("TFDATA_PARALLEL", 10))  # 2–8 conforme sua CPU
    PREFETCH = int(os.getenv("TFDATA_PREFETCH", 6))  # 1–2 costuma ser ótimo
    N = len(paths)
    # base
    base = tf.data.Dataset.from_tensor_slices((paths, labels))
    base = base.shuffle(buffer_size=N, seed=seed, reshuffle_each_iteration=True)
    base = base.map(lambda p,y: _decode_resize(p,y,img_size,channels),
                    num_parallel_calls=PARALLEL)
    base = base.batch(batch_size)

    # --- contagem de passos ---
    base_steps = int(np.ceil(N / batch_size))
    num_extra   = int(N * max(0.0, float(extra_fraction)))
    extra_steps = int(np.ceil(num_extra / batch_size)) if num_extra > 0 else 0

    # augment (fila finita; nada de repeat infinito)
    def _map_aug(x, y):
        sd = tf.stack([tf.cast(1000 * tf.timestamp(), tf.int32), tf.cast(y, tf.int32)])
        xa = _augment_image(x, aug_cfg, sd)
        return xa, y

    aug = tf.data.Dataset.from_tensor_slices((paths, labels))
    aug = aug.shuffle(buffer_size=N, seed=seed, reshuffle_each_iteration=True)
    aug = aug.map(lambda p,y: _decode_resize(p,y,img_size,channels),
                  num_parallel_calls=PARALLEL)
    aug = aug.map(_map_aug, num_parallel_calls=PARALLEL)

    # repete só o suficiente para produzir 'extra_steps' lotes
    repeats = max(1, (extra_steps + base_steps - 1) // base_steps) if extra_steps > 0 else 1
    aug = aug.repeat(repeats).batch(batch_size)

    # empurrar lotes para GPU e reduzir cópias (FP16)
    base = base.map(lambda x,y: (tf.cast(x, tf.float16), y))
    base = base.apply(copy_to_device('/GPU:0')).prefetch(1)

    aug  = aug.map(lambda x,y: (tf.cast(x, tf.float16), y))
    aug  = aug.apply(copy_to_device('/GPU:0')).prefetch(1)

    ds = base if extra_steps == 0 else base.concatenate(aug.take(extra_steps))

    ds = ds.apply(assert_cardinality(base_steps + extra_steps))
    return ds.prefetch(PREFETCH), dict(
        N=N,
        base_steps=base_steps,
        aug_steps=extra_steps,
        expected_aug_images=min(num_extra, extra_steps * batch_size),
    )

# =========================
# Função única chamada pelo check/train
# =========================
def build_datasets_nbm(
    img_size=1024,
    channels=1,
    batch_size=8,
    max_per_class=7000,
    val_size=0.15,
    test_size=0.15, # <<< ADICIONADO
    seed=42,
    extra_fraction=2.0,
    aug_cfg: AugmentConfig | None = None,
    data_dir: Optional[str] = None,
    csv_path: Optional[str] = None,
    include_test=True, 
    **kwargs,
):
    if aug_cfg is None:
        aug_cfg = AugmentConfig()

    # <<< MODIFICADO (recebe test_df e passa test_size) >>>
    train_df, val_df, test_df, meta = read_nbm_dataset(
        data_dir=data_dir, csv_path=csv_path,
        val_size=val_size, test_size=test_size, 
        max_per_class=max_per_class, seed=seed
    )

    train_ds, counts = make_train_ds_plus_aug(
        train_df["img_path"].values, train_df["label"].values,
        img_size=img_size, channels=channels, batch_size=batch_size,
        aug_cfg=aug_cfg, extra_fraction=extra_fraction, seed=seed
    )
    val_ds = make_val_ds(
        val_df["img_path"].values, val_df["label"].values,
        img_size=img_size, channels=channels, batch_size=batch_size
    )
    # <<< ADICIONADO (criação do test_ds) >>>
    test_ds = make_val_ds(
        test_df["img_path"].values, test_df["label"].values,
        img_size=img_size, channels=channels, batch_size=batch_size
    )

    info = dict(
        img_size=img_size,
        channels=channels,
        batch_size=batch_size,
        aug_summary=_aug_summary(aug_cfg),
        extra_fraction=float(extra_fraction),
        steps_base=counts["base_steps"],
        steps_aug=counts["aug_steps"],
        prev_aug_imgs=counts["expected_aug_images"],
    )
    # <<< MODIFICADO (retorna test_ds) >>>
    return train_ds, val_ds, test_ds, meta, info

# opcional: teste rápido se rodar diretamente
if __name__ == "__main__":
    # <<< BLOCO __main__ TOTALMENTE MODIFICADO PARA TESTAR A DIVISÃO >>>
    # Este bloco agora funciona e testa a lógica de divisão
    
    print("Iniciando teste de divisão do dataset...")
    
    # Valores de exemplo para teste:
    _IMG_SIZE = 1024
    _CHANNELS = 1
    _BATCH_SIZE = 16
    _MAX_PER_CLASS = 7000 # Use um número menor para um teste rápido
    _VAL_SIZE = 0.15     # 15% do total para validação
    _TEST_SIZE = 0.15    # 15% do total para teste
    # (Resultado esperado: 70% treino)
    
    try:
        # Vamos chamar read_nbm_dataset diretamente para inspecionar os dataframes
        # Em um treino real, você chamaria build_datasets_nbm
        
        train_df, val_df, test_df, meta = read_nbm_dataset(
            val_size=_VAL_SIZE,
            test_size=_TEST_SIZE,
            max_per_class=_MAX_PER_CLASS,
            seed=42
        )
        
        print("\n--- RESULTADO DA DIVISÃO ---")
        print(f"[META] n_train: {meta['n_train']}")
        print(f"[META] n_val:   {meta['n_val']}")
        print(f"[META] n_test:  {meta['n_test']}")
        print(f"[META] Total:   {meta['n_train'] + meta['n_val'] + meta['n_test']}")
        print(f"[META] Pesos:   {meta['class_weights']}")

        print("\n--- DISTRIBUIÇÃO (estratificada) ---")
        print("  Train por label:\n", train_df["label"].value_counts().sort_index())
        print("\n  Val   por label:\n",   val_df["label"].value_counts().sort_index())
        print("\n  Test  por label:\n",   test_df["label"].value_counts().sort_index())
        print("[PIPELINE] Info:", {**info, "extra_fraction": 2.0})
        print("\n--- TESTANDO build_datasets_nbm ---")
        # Agora testamos a função principal
        train_ds, val_ds, test_ds, meta_build, info = build_datasets_nbm(
            img_size=_IMG_SIZE, channels=_CHANNELS, batch_size=_BATCH_SIZE,
            max_per_class=_MAX_PER_CLASS,
            val_size=_VAL_SIZE,
            test_size=_TEST_SIZE,
            extra_fraction=0.5 # Teste com 50% de augment extra
        )
        
        print(f"\n[PIPELINE] Datasets criados com sucesso.")
        print(f"[PIPELINE] Info: {info}")
        print(f"  -> Train DS: {train_ds}")
        print(f"  -> Val DS:   {val_ds}")
        print(f"  -> Test DS:  {test_ds}")

    except FileNotFoundError as e:
        print(f"\n[ERRO] CSV do dataset não encontrado. Teste pulado.")
        print(f"  (Erro: {e})")
        print("  (Para testar, configure o 'data_dir' ou a variável de ambiente 'MAMMO_PATH')")
    except Exception as e:
        print(f"\n[ERRO] Falha ao construir datasets: {e}")
        import traceback
        traceback.print_exc()