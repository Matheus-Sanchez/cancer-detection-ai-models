import sys, json, time, math, shutil
from pathlib import Path

from email import parser
import argparse
import numpy as np
import datetime
from datetime import datetime

# ======= Minimiza ruído de logs ANTES de importar TF =======
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"            # ERROR/WARN
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"           # comport. numérico estável
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"  # desliga XLA JIT
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

# === imports padrão ===
import sys
from pathlib import Path

# ---- GPU allocator / growth (ajuda a evitar fragmentação e crash pós-OOM) ----
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    CSVLogger,
    ReduceLROnPlateau,
    LearningRateScheduler,
)
from tensorflow.keras.callbacks import TensorBoard

# --- Configuração de Caminhos para Importação ---
# Garante que a pasta atual (official_model) e a raiz (mamo_cnn) estejam no PATH
BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
for p in (BASE_DIR, REPO_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import logging
import absl.logging
absl.logging.set_verbosity('error')

# Importa módulos locais (que devem estar na mesma pasta official_model/)
from data_pipeline import build_datasets_nbm, AugmentConfig
from model_cnn import build_model, build_keras_app_model
from metrics_logs import make_basic_callbacks, save_history, finalize_and_save

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
# Desliga JIT via API também (backup) e silencia logger
tf.config.optimizer.set_jit(False)
tf.get_logger().setLevel("ERROR")


class MultiClassAUC(tf.keras.metrics.Metric):
    def __init__(self, num_classes=3, name="auc_ovr", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self._auc = tf.keras.metrics.AUC(
            multi_label=True, num_labels=num_classes, curve="ROC", name=name
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true chega como inteiros [0..C-1]; convertemos para one-hot
        y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), depth=self.num_classes)
        return self._auc.update_state(y_true_oh, y_pred, sample_weight=sample_weight)

    def result(self):
        return self._auc.result()

    def reset_state(self):
        self._auc.reset_state()


# ======= Args =======

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img-size", type=int, default=1024)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--mixed-precision", action="store_true",
                   help="Ativa mixed precision (float16) para reduzir VRAM e acelerar em GPU.")
    p.add_argument("--epochs", type=int, default=100)

    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--extra-fraction", type=float, default=2.0)  # 100% de aug extra
    p.add_argument("--max-per-class", type=int, default=7000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-root", type=str, default=None,
                   help="Diretório raiz para salvar outputs (default: diretório atual).")
    p.add_argument("--compare-root", type=str, default=None,
                   help="(compare) Usa um diretório existente para salvar/retomar a comparação.")

    p.add_argument("--lr", type=float, default=3e-4)

    # --- DEFINIÇÃO DE PADRÕES PARA O AMBIENTE WSL ---
    # Caminhos relativos à raiz do projeto (mamo_cnn)
    default_data_dir = "dataset/Mammo_Data/Mammo-Bench"
    default_csv_path = "dataset/Mammo_Data/Mammo-Bench/CSV_Files/mammo-bench_nbm_classification.csv"

    p.add_argument("--data-dir", type=str, default=default_data_dir,
                   help=f"Caminho para o dataset (default: {default_data_dir})")
    p.add_argument("--csv-path", type=str, default=default_csv_path,
                   help=f"Caminho para o CSV (default: {default_csv_path})")

    # Flag de checagem
    p.add_argument("--check-pipeline", action="store_true", help="Só valida o pipeline e sai")

    # ======= escolha de arquitetura =======
    p.add_argument("--arch", type=str, default="custom",
                   choices=["custom", "keras_app"],
                   help="custom = sua CNN; keras_app = modelo pronto do Keras Applications")

    p.add_argument("--backbone", type=str, default="efficientnetv2b0",
                   help="Backbone do Keras Applications (ex: efficientnetv2b0, resnet50, mobilenetv2, densenet121)")
    p.add_argument("--weights", type=str, default="imagenet",
                   help="Use 'imagenet' para pré-treinado, ou 'none' para treinar do zero")
    p.add_argument("--backbone-trainable", action="store_true",
                   help="Se setado, libera treino do backbone (fine-tuning). Se não, treina só a cabeça.")

    # ======= modo automático de comparação =======
    p.add_argument("--compare", action="store_true",
                   help="Roda automaticamente: custom + (1 ou mais) backbones do Keras Applications")
    p.add_argument("--compare-backbones", type=str, default=None,
                   help="Lista separada por vírgula para rodar vários backbones no --compare (ex: resnet50,mobilenetv2)")
    # ajustes finos do modo compare
    p.add_argument("--compare-epochs-custom", type=int, default=None,
                   help="(compare) nº de epochs só para o modelo custom (default: usa --epochs)")
    p.add_argument("--compare-epochs-app", type=int, default=None,
                   help="(compare) nº de epochs só para o modelo keras_app (default: usa --epochs)")
    p.add_argument("--compare-lr-custom", type=float, default=None,
                   help="(compare) sobrescreve --lr apenas para o modelo custom")
    p.add_argument("--compare-lr-app", type=float, default=None,
                   help="(compare) sobrescreve --lr apenas para o modelo keras_app")
    p.add_argument("--compare-auto-retry", action="store_true",
                   help="(compare) se der OOM (ResourceExhaustedError), tenta automaticamente reduzir batch / congelar backbone / reduzir img_size")
    p.add_argument("--compare-extra-fraction", type=float, default=None,
                   help="(compare) sobrescreve --extra-fraction apenas no modo compare")
    p.add_argument("--compare-max-per-class", type=int, default=None,
                   help="(compare) sobrescreve --max-per-class apenas no modo compare")
    p.add_argument("--compare-batch-custom", type=int, default=None,
                   help="(compare) sobrescreve --batch-size apenas para o modelo custom")
    p.add_argument("--compare-batch-app", type=int, default=None,
                   help="(compare) sobrescreve --batch-size apenas para o modelo keras_app")
    p.add_argument("--compare-img-size-custom", type=int, default=None,
                   help="(compare) sobrescreve --img-size apenas para o modelo custom")
    p.add_argument("--compare-img-size-app", type=int, default=None,
                   help="(compare) sobrescreve --img-size apenas para o modelo keras_app")
    p.add_argument("--compare-skip-custom", action="store_true",
                   help="(compare) pula o treino do modelo custom (útil para retomar após crash)")
    p.add_argument("--compare-skip-app", action="store_true",
                   help="(compare) pula o(s) modelo(s) keras_app")

    # Nome da pasta de saída
    p.add_argument("--experiment-name", type=str, default="run",
                   help="Nome/Prefixo para a pasta criada dentro de outputs/ (ex: 'meu_teste').")

    return p.parse_args()

# ======= Util =======
def make_outdir(prefix: str = "run", root: str | None = None) -> str:
    run_name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # Se root não for passado, usa CWD (que esperamos ser mamo_cnn)
    root_dir = root if root is not None else os.getcwd()
    out_dir = os.path.join(root_dir, "outputs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_model_artifacts(model: tf.keras.Model, out_dir: str):
    # summary.txt
    sum_path = os.path.join(out_dir, "model_summary.txt")
    with open(sum_path, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    # model.json
    try:
        with open(os.path.join(out_dir, "model.json"), "w", encoding="utf-8") as f:
            f.write(model.to_json())
    except Exception:
        pass  # alguns layers custom podem bloquear to_json()


def make_basic_callbacks(out_dir: str, num_classes: int | None = None, val_ds=None):
    """Callbacks padrão: best/last checkpoints, EarlyStopping, ReduceLROnPlateau, CSVLogger e TensorBoard."""
    ckpt_best = os.path.join(out_dir, "best.keras")
    ckpt_last = os.path.join(out_dir, "last.keras")
    tb_dir    = os.path.join(out_dir, "tb")

    cb_best = ModelCheckpoint(
        ckpt_best, monitor="val_auc_ovr", mode="max",
        save_best_only=True, save_weights_only=False, verbose=1
    )
    cb_last = ModelCheckpoint(
        ckpt_last, monitor="val_loss", mode="min",
        save_best_only=False, save_weights_only=False, verbose=0
    )
    cb_es = EarlyStopping(
        monitor="val_auc_ovr", mode="max", patience=10, restore_best_weights=True, verbose=1
    )
    cb_rlrop = ReduceLROnPlateau(
        monitor="val_loss", mode="min", factor=0.3, patience=3, min_lr=1e-6, verbose=1
    )
    cb_csv = CSVLogger(os.path.join(out_dir, "history.csv"))
    cb_tb  = TensorBoard(log_dir=tb_dir, update_freq="epoch")

    return [cb_best, cb_last, cb_es, cb_rlrop, cb_csv, cb_tb]



def _normalize_weights_arg(w: str | None):
    """Normaliza argumento de weights.

    - 'imagenet' -> 'imagenet'
    - 'none'/'null'/'0'/'' -> None
    - None -> None
    """
    if w is None:
        return None
    ww = str(w).lower().strip()
    return None if ww in ("none", "null", "0", "") else w

def _clone_args(args):
    return argparse.Namespace(**vars(args))

def _apply_compare_overrides(args, *, epochs=None, extra_fraction=None, max_per_class=None, batch_size=None, img_size=None, lr=None):
    a = _clone_args(args)
    if epochs is not None:
        a.epochs = int(epochs)
    if extra_fraction is not None:
        a.extra_fraction = float(extra_fraction)
    if max_per_class is not None:
        a.max_per_class = int(max_per_class)
    if batch_size is not None:
        a.batch_size = int(batch_size)
    if img_size is not None:
        a.img_size = int(img_size)
    if lr is not None:
        a.lr = float(lr)
    return a

def _run_experiment_with_retries(args, out_dir, *, arch, backbone=None, weights=None, backbone_trainable=False, auto_retry=False):
    """Executa run_experiment; se auto_retry=True, tenta recuperar de OOM ajustando batch/img_size/trainable."""
    attempts = []
    a0 = _clone_args(args)

    batch = int(a0.batch_size)
    img_size = int(a0.img_size)
    trainable = bool(backbone_trainable)

    while True:
        a_try = _clone_args(a0)
        a_try.batch_size = batch
        a_try.img_size = img_size

        try:
            res = run_experiment(
                a_try,
                out_dir,
                arch=arch,
                backbone=backbone,
                weights=weights,
                backbone_trainable=trainable,
            )
            res["effective_config"] = {
                "img_size": int(img_size),
                "batch_size": int(batch),
                "backbone_trainable": bool(trainable),
            }
            if attempts:
                res["oom_retries"] = attempts
            return res

        except tf.errors.ResourceExhaustedError as e:
            attempts.append({"img_size": int(img_size), "batch_size": int(batch), "backbone_trainable": bool(trainable)})

            print("\n[OOM] ResourceExhaustedError durante o experimento.")
            print(f"      Tentativa: img_size={img_size} batch_size={batch} trainable={trainable}")
            print("      Dica: reduza --batch-size e/ou --img-size; e para modelos prontos, treine cabeça primeiro (sem --backbone-trainable).\n")

            if not auto_retry:
                raise

            if batch > 1:
                batch = max(1, batch // 2)
                continue
            if trainable:
                trainable = False
                continue
            if img_size > 256:
                img_size = max(256, img_size // 2)
                continue
            raise

    ww = str(w).lower().strip()
    return None if ww in ("none", "null", "0", "") else w


def run_experiment(
    args,
    out_dir: str,
    arch: str,
    backbone: str | None = None,
    weights: str | None = "imagenet",
    backbone_trainable: bool = False,
):
    """
    Executa 1 experimento completo (build datasets -> build model -> fit -> eval -> salvar artefatos).
    Retorna o dict de resultados (results_final.json).
    """
    os.makedirs(out_dir, exist_ok=True)

    # limpeza de sessão / seeds para reduzir interferência entre execuções
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    aug_cfg = AugmentConfig()

    # ======= Datasets (mantém seu pipeline/normalização) =======
    train_ds, val_ds, test_ds, meta, info = build_datasets_nbm(
        img_size=args.img_size,
        channels=args.channels,
        batch_size=args.batch_size,
        max_per_class=args.max_per_class,
        val_size=args.val_size,
        test_size=args.test_size,
        extra_fraction=args.extra_fraction,
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        aug_cfg=aug_cfg,
        seed=args.seed,
    )
    print("[PIPELINE] Info:", info)

    # ======= Modelo =======
    if arch == "custom":
        model = build_model(img_size=args.img_size, channels=args.channels, num_classes=3)
        model_choice = {"arch": "custom"}
    else:
        w = _normalize_weights_arg(weights)
        model = build_keras_app_model(
            img_size=args.img_size,
            channels=args.channels,
            num_classes=3,
            backbone=backbone or args.backbone,
            weights=w,
            trainable_backbone=bool(backbone_trainable),
        )
        model_choice = {
            "arch": "keras_app",
            "backbone": backbone or args.backbone,
            "weights": w,
            "backbone_trainable": bool(backbone_trainable),
        }

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy", MultiClassAUC(num_classes=3, name="auc_ovr")],
    )
    model.summary()

    # ======= Callbacks =======
    cbs = make_basic_callbacks(out_dir, num_classes=3, val_ds=val_ds)

    # ======= Treino =======
    class_weights = meta.get("class_weights", None)
    t0 = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=cbs,
        class_weight=class_weights if class_weights else None,
        verbose=1,
    )
    train_time = time.perf_counter() - t0

    # ======= Salva artefatos =======
    save_model_artifacts(model, out_dir)
    save_history(history, out_dir)  # cria history.json

    # ======= Avaliações (val/test) =======
    print("\n[EVAL] Validação:")
    val_values = model.evaluate(val_ds, verbose=1)
    val_metrics = dict(zip(model.metrics_names, [float(x) for x in val_values]))
    print(val_metrics)

    print("\n[EVAL] Teste:")
    test_values = model.evaluate(test_ds, verbose=1)
    test_metrics = dict(zip(model.metrics_names, [float(x) for x in test_values]))
    print(test_metrics)

    # ======= Salva resultados finais =======
    results = {
        "model_choice": model_choice,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "lr": float(args.lr),
        "epochs": int(args.epochs),
        "img_size": int(args.img_size),
        "batch_size": int(args.batch_size),
        "extra_fraction": float(args.extra_fraction),
        "val_size": float(args.val_size),
        "test_size": float(args.test_size),
        "seed": int(args.seed),
        "class_weights": class_weights,
        "train_time_sec": float(train_time),
        "pipeline_info": info,
    }
    with open(os.path.join(out_dir, "results_final.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ======= Avaliação completa + artefatos (val) =======
    steps_val = math.ceil(meta["n_val"] / args.batch_size)
    extra_info = {
        "lr": float(args.lr),
        "epochs": int(args.epochs),
        "img_size": int(args.img_size),
        "batch_size": int(args.batch_size),
        "extra_fraction": float(args.extra_fraction),
        "val_size": float(args.val_size),
        "test_size": float(args.test_size),
        "seed": int(args.seed),
        "class_weights": meta.get("class_weights", {}),
        "model_choice": model_choice,
        "train_time_sec": float(train_time),
    }
    _ = finalize_and_save(
        model=model,
        ds=val_ds,
        steps=steps_val,
        num_classes=3,
        run_dir=out_dir,
        train_time_sec=train_time,
        extra_info=extra_info,
    )

    # ======= Avaliação (test) e anexar no results_final.json =======
    steps_test = math.ceil(meta["n_test"] / args.batch_size)
    test_dict = model.evaluate(test_ds, steps=steps_test, verbose=0, return_dict=True)

    rf = os.path.join(out_dir, "results_final.json")
    with open(rf, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["test_metrics"] = {k: float(v) for k, v in test_dict.items()}
    with open(rf, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    from metrics_logs import plot_training_curves
    plot_training_curves(history.history, out_dir, fname="training_curves.png")

    return data


def main():
    args = parse_args()

    # GPU memory growth (evita reservar 100% e reduz chance de crash pós-OOM)
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


    if not hasattr(args, "check_pipeline"):
        args.check_pipeline = False

    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(8)

    # Semente (global)
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    # ======= (Opcional) Modo de checagem do pipeline =======
    if args.check_pipeline:
        out_dir = make_outdir(prefix="check")
        print(f"[INFO] Salvando saídas em: {out_dir}")
        aug_cfg = AugmentConfig()
        print("Iniciando teste de divisão do dataset...")
        train_ds, val_ds, test_ds, meta, info = build_datasets_nbm(
            img_size=args.img_size,
            channels=args.channels,
            batch_size=args.batch_size,
            max_per_class=args.max_per_class,
            val_size=args.val_size,
            test_size=args.test_size,
            extra_fraction=args.extra_fraction,
            data_dir=args.data_dir,
            csv_path=args.csv_path,
            aug_cfg=aug_cfg,
            seed=args.seed,
        )
        print("\n--- RESULTADO DA DIVISÃO ---")
        print(f"[META] n_train: {meta['n_train']}")
        print(f"[META] n_val:   {meta['n_val']}")
        print(f"[META] n_test:  {meta['n_test']}")
        print(f"[META] Pesos:   {meta['class_weights']}")
        print("\n[PIPELINE] Datasets criados com sucesso.")
        print("[PIPELINE] Info:", info)
        sys.exit(0)

    # ======= Modo automático: compara custom vs modelos prontos =======
    if args.compare:
        root_dir = args.compare_root if args.compare_root else make_outdir(prefix="compare", root=args.out_root)
        os.makedirs(root_dir, exist_ok=True)
        print(f"[INFO] Salvando comparação em: {root_dir}")

        # backbones a rodar
        backbones = [args.backbone]
        if args.compare_backbones:
            backbones = [b.strip() for b in args.compare_backbones.split(",") if b.strip()]

        weights = _normalize_weights_arg(args.weights)
        runs = []

        # 1) Custom
        out_custom = os.path.join(root_dir, "custom")
        print("\n==============================")
        print("[COMPARE] Rodando: custom")
        print("==============================")

        if args.compare_skip_custom:
            res_custom = {"skipped": True}
        else:
            args_custom = _apply_compare_overrides(
                args,
                epochs=args.compare_epochs_custom if args.compare_epochs_custom is not None else args.epochs,
                extra_fraction=args.compare_extra_fraction,
                max_per_class=args.compare_max_per_class,
                batch_size=args.compare_batch_custom,
                img_size=args.compare_img_size_custom,
            )
            try:
                res_custom = _run_experiment_with_retries(
                    args_custom,
                    out_custom,
                    arch="custom",
                    backbone=None,
                    weights=None,
                    backbone_trainable=False,
                    auto_retry=args.compare_auto_retry,
                )
            except tf.errors.ResourceExhaustedError as e:
                res_custom = {"error": "ResourceExhaustedError", "message": str(e)}

        runs.append({"name": "custom", "dir": "custom", "results": res_custom})

        # 2) Keras Applications (1 ou mais backbones)
        if args.compare_skip_app:
            for bb in backbones:
                tag = f"keras_app_{bb}"
                runs.append({"name": tag, "dir": tag, "results": {"skipped": True}})
        else:
            for bb in backbones:
                tag = f"keras_app_{bb}"
                out_bb = os.path.join(root_dir, tag)
                print("\n==============================")
                print(f"[COMPARE] Rodando: {tag}")
                print("==============================")

                args_app = _apply_compare_overrides(
                    args,
                    epochs=args.compare_epochs_app if args.compare_epochs_app is not None else args.epochs,
                    extra_fraction=args.compare_extra_fraction,
                    max_per_class=args.compare_max_per_class,
                    batch_size=args.compare_batch_app,
                    img_size=args.compare_img_size_app,
                )
                try:
                    res_bb = _run_experiment_with_retries(
                        args_app,
                        out_bb,
                        arch="keras_app",
                        backbone=bb,
                        weights=weights,
                        backbone_trainable=args.backbone_trainable,
                        auto_retry=args.compare_auto_retry,
                    )
                except tf.errors.ResourceExhaustedError as e:
                    res_bb = {"error": "ResourceExhaustedError", "message": str(e)}

                runs.append({"name": tag, "dir": tag, "results": res_bb})

        summary = {
            "created_at": datetime.now().isoformat(),
            "seed": int(args.seed),
            "base_args": {
                "img_size": int(args.img_size),
                "channels": int(args.channels),
                "batch_size": int(args.batch_size),
                "epochs": int(args.epochs),
                "lr": float(args.lr),
                "extra_fraction": float(args.extra_fraction),
                "max_per_class": int(args.max_per_class),
                "mixed_precision": bool(getattr(args, "mixed_precision", False)),
            },
            "compare_overrides": {
                "compare_epochs_custom": args.compare_epochs_custom,
                "compare_epochs_app": args.compare_epochs_app,
                "compare_extra_fraction": args.compare_extra_fraction,
                "compare_max_per_class": args.compare_max_per_class,
                "compare_batch_custom": args.compare_batch_custom,
                "compare_batch_app": args.compare_batch_app,
                "compare_img_size_custom": args.compare_img_size_custom,
                "compare_img_size_app": args.compare_img_size_app,
                "compare_auto_retry": bool(args.compare_auto_retry),
                "compare_skip_custom": bool(args.compare_skip_custom),
                "compare_skip_app": bool(args.compare_skip_app),
            },
            "keras_app": {
                "weights": weights,
                "backbone_trainable": bool(args.backbone_trainable),
                "backbones": backbones,
            },
            "runs": [
                {
                    "name": r["name"],
                    "dir": r["dir"],
                    "skipped": bool(r["results"].get("skipped", False)) if isinstance(r["results"], dict) else False,
                    "val_metrics": r["results"].get("val_metrics", {}) if isinstance(r["results"], dict) else {},
                    "test_metrics": r["results"].get("test_metrics", {}) if isinstance(r["results"], dict) else {},
                    "effective_config": r["results"].get("effective_config", {}) if isinstance(r["results"], dict) else {},
                    "error": r["results"].get("error") if isinstance(r["results"], dict) else None,
                }
                for r in runs
            ],
        }
        with open(os.path.join(root_dir, "compare_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Comparação concluída. Veja: {os.path.join(root_dir, 'compare_summary.json')}")
        return

    # ======= Execução normal: roda 1 único experimento =======
    out_dir = make_outdir(prefix=args.experiment_name, root=args.out_root)
    print(f"[INFO] Salvando saídas em: {out_dir}")

    weights = _normalize_weights_arg(args.weights)
    _ = run_experiment(
        args,
        out_dir,
        arch=args.arch,
        backbone=args.backbone,
        weights=weights,
        backbone_trainable=args.backbone_trainable,
    )

    print(f"\n✅ Artefatos salvos em: {out_dir}")
if __name__ == "__main__":
    main()