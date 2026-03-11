from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

# -----------------------------------------------------------------------------
# Imports relativos ao projeto (funcionam tanto com "python -m official_model..."
# quanto rodando o arquivo diretamente)
# -----------------------------------------------------------------------------
try:
    from .data_pipeline import build_datasets_nbm
    from .model_cnn import build_model
except Exception:
    # fallback se rodar como script direto
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from official_model.data_pipeline import build_datasets_nbm
    from official_model.model_cnn import build_model


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pick_split(
    train_ds, val_ds, test_ds, split: str
) -> tf.data.Dataset:
    split = split.lower()
    if split == "train":
        return train_ds
    if split == "val":
        return val_ds
    if split == "test":
        return test_ds
    raise ValueError(f"split inválido: {split} (use 'train', 'val' ou 'test')")


def _collect_layer_stats(
    model: keras.Model,
    x_batch: tf.Tensor,
    training: bool = False,
) -> List[Dict[str, Any]]:
    """
    Cria um submodelo que retorna a saída de TODAS as camadas (exceto InputLayer)
    para um único batch, e computa estatísticas por camada.
    """
    # Ignora InputLayer(s)
    layers = [ly for ly in model.layers if not isinstance(ly, keras.layers.InputLayer)]
    if not layers:
        raise RuntimeError("Modelo não possui camadas além do InputLayer.")

    # Modelo que devolve as saídas de todas as camadas
    probe = keras.Model(
        inputs=model.inputs,
        outputs=[ly.output for ly in layers],
        name="activations_probe_model",
    )

    # Faz forward pass (sem dropout por padrão: training=False)
    outputs = probe(x_batch, training=training)

    # Caso o modelo tenha só 1 camada "real"
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    stats: List[Dict[str, Any]] = []
    for ly, out in zip(layers, outputs):
        # Pode haver camadas que não produzem Tensor (muito raro), ignorar
        if not isinstance(out, tf.Tensor):
            continue

        # Converte só para cálculo numérico, mantendo no gráfico TF
        x = tf.cast(out, tf.float32)
        numel = int(tf.size(x))

        if numel == 0:
            # Camada "vazia" (não é o caso aqui, mas por segurança)
            stats.append(
                dict(
                    name=ly.name,
                    type=ly.__class__.__name__,
                    shape=tuple(out.shape),
                    dtype=str(out.dtype.name),
                    numel=0,
                    min=np.nan,
                    max=np.nan,
                    mean=np.nan,
                    std=np.nan,
                    frac_neg=np.nan,
                    frac_zero=np.nan,
                    frac_pos=np.nan,
                )
            )
            continue

        v_min = float(tf.reduce_min(x))
        v_max = float(tf.reduce_max(x))
        v_mean = float(tf.reduce_mean(x))
        v_std = float(tf.math.reduce_std(x))

        # Frações de valores <0, =0, >0
        neg = int(tf.math.count_nonzero(x < 0.0))
        zero = int(tf.math.count_nonzero(tf.equal(x, 0.0)))
        pos = int(tf.math.count_nonzero(x > 0.0))

        frac_neg = neg / numel
        frac_zero = zero / numel
        frac_pos = pos / numel

        stats.append(
            dict(
                name=ly.name,
                type=ly.__class__.__name__,
                shape=tuple(out.shape),
                dtype=str(out.dtype.name),
                numel=numel,
                min=v_min,
                max=v_max,
                mean=v_mean,
                std=v_std,
                frac_neg=frac_neg,
                frac_zero=frac_zero,
                frac_pos=frac_pos,
            )
        )

    return stats


def _print_stats(stats: List[Dict[str, Any]]):
    print("\n[ACTIVATIONS: RESUMO POR CAMADA]")
    for i, s in enumerate(stats):
        print(f"\n#{i:02d}  Layer: {s['name']}  ({s['type']})")
        print(f"     shape     : {s['shape']}, dtype={s['dtype']}, numel={s['numel']}")
        print(
            f"     min/max   : {s['min']:.5f} / {s['max']:.5f}    "
            f"mean={s['mean']:.5f}  std={s['std']:.5f}"
        )
        print(
            "     frac(<0/==0/>0): "
            f"{s['frac_neg']:.4f} / {s['frac_zero']:.4f} / {s['frac_pos']:.4f}"
        )


def _save_csv(stats: List[Dict[str, Any]], out_path: Path):
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "idx",
        "name",
        "type",
        "shape",
        "dtype",
        "numel",
        "min",
        "max",
        "mean",
        "std",
        "frac_neg",
        "frac_zero",
        "frac_pos",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for i, s in enumerate(stats):
            w.writerow(
                [
                    i,
                    s["name"],
                    s["type"],
                    s["shape"],
                    s["dtype"],
                    s["numel"],
                    f"{s['min']:.8g}",
                    f"{s['max']:.8g}",
                    f"{s['mean']:.8g}",
                    f"{s['std']:.8g}",
                    f"{s['frac_neg']:.8g}",
                    f"{s['frac_zero']:.8g}",
                    f"{s['frac_pos']:.8g}",
                ]
            )
    print(f"\n[ACTIVATIONS] CSV salvo em: {out_path}")


def main():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    ap = argparse.ArgumentParser(
        description=(
            "Inspeciona as ativações de TODAS as camadas do modelo "
            "para um batch do dataset (train/val/test)."
        )
    )
    ap.add_argument("--img-size", type=int, default=1024)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-per-class", type=int, default=7000)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Qual split usar para o batch de entrada.",
    )
    ap.add_argument(
        "--training-mode",
        action="store_true",
        help="Se setado, executa o modelo com training=True (com Dropout ativo).",
    )
    ap.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Opcional: caminho para um modelo Keras salvo (.keras/.h5). "
             "Se vazio, usa build_model().",
    )
    ap.add_argument(
        "--save-csv",
        type=str,
        default="",
        help="Se fornecido, salva um CSV com as estatísticas. "
             "Se vazio, salva em outputs/check_activations/activations_stats.csv.",
    )
    args = ap.parse_args()

    # -------------------------------------------------------------------------
    # 1) Datasets (usa a mesma função de treino)
    # -------------------------------------------------------------------------
    print("[INFO] Construindo datasets com build_datasets_nbm...")
    train_ds, val_ds, test_ds, meta, info = build_datasets_nbm(
        img_size=args.img_size,
        channels=args.channels,
        batch_size=args.batch_size,
        max_per_class=args.max_per_class,
        val_size=args.val_size,
        test_size=args.test_size,
        extra_fraction=0.0,  # para não misturar augment extra nas estatísticas
    )

    print(
        f"[DATA] Train/Val/Test: {meta['n_train']} / {meta['n_val']} / {meta['n_test']}"
    )
    print(f"[DATA] Classes      : {meta.get('class_map')}")
    print(f"[DATA] Class weights: {meta.get('class_weights')}")

    ds = _pick_split(train_ds, val_ds, test_ds, args.split)
    xb, yb = next(iter(ds.take(1)))
    print(f"\n[INPUT BATCH] split={args.split}")
    print(f"  X shape / dtype: {xb.shape} / {xb.dtype}")
    print(f"  y shape / dtype: {yb.shape} / {yb.dtype}")
    try:
        uniq = tf.sort(tf.unique(yb).y).numpy().tolist()
        print(f"  labels únicos   : {uniq}")
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # 2) Modelo (constrói ou carrega)
    # -------------------------------------------------------------------------
    if args.model_path:
        model_path = Path(args.model_path).expanduser().resolve()
        print(f"\n[MODEL] Carregando modelo salvo de: {model_path}")
        model = keras.models.load_model(model_path, compile=False)
        num_classes = model.output_shape[-1]
    else:
        num_classes = len(meta.get("class_map", {0: 0, 1: 1, 2: 2}))
        print(
            f"\n[MODEL] Construindo modelo com build_model "
            f"(img_size={args.img_size}, ch={args.channels}, num_classes={num_classes})"
        )
        model = build_model(
            img_size=args.img_size,
            channels=args.channels,
            num_classes=num_classes,
        )

    model.summary(line_length=120)

    # -------------------------------------------------------------------------
    # 3) Coleta de estatísticas de TODAS as camadas
    # -------------------------------------------------------------------------
    print(
        f"\n[ACTIVATIONS] Coletando estatísticas de todas as camadas "
        f"(training={args.training_mode})..."
    )
    stats = _collect_layer_stats(model, xb, training=args.training_mode)
    _print_stats(stats)

    # -------------------------------------------------------------------------
    # 4) CSV opcional
    # -------------------------------------------------------------------------
    if args.save_csv:
        out_path = Path(args.save_csv).expanduser().resolve()
    else:
        repo_root = _get_repo_root()
        out_dir = repo_root / "outputs" / "check_activations"
        out_path = out_dir / "activations_stats.csv"

    _save_csv(stats, out_path)


if __name__ == "__main__":
    # Exemplo de uso:
    #   python -m official_model.check_activations --img-size 1024 --batch-size 8 --split val
    # ou:
    #   python official_model/check_activations.py --img-size 1024 --batch-size 8 --split val
    main()
