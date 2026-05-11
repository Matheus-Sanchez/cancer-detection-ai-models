 # official_model/main.py
# -*- coding: utf-8 -*-
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


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    CSVLogger,
    ReduceLROnPlateau,
    LearningRateScheduler,
)
from tensorflow.keras.callbacks import TensorBoard

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import logging
import absl.logging
absl.logging.set_verbosity('error')

from data_pipeline import build_datasets_nbm, AugmentConfig
from model_cnn import build_model, build_keras_app_model
from train import train
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
    p.add_argument("--epochs", type=int, default=100)

    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--extra-fraction", type=float, default=2.0)  # 100% de aug extra
    p.add_argument("--max-per-class", type=int, default=7000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--csv-path", type=str, default=None)

    # ✅ coloque a flag no mesmo parser
    p.add_argument("--check-pipeline", action="store_true", help="Só valida o pipeline e sai")

        # ======= escolha de arquitetura (custom vs modelo pronto) =======
    p.add_argument("--arch", type=str, default="custom",
                   choices=["custom", "keras_app"],
                   help="custom = sua CNN; keras_app = modelo pronto do Keras Applications")

    p.add_argument("--backbone", type=str, default="efficientnetv2b0",
                   help="Backbone do Keras Applications (ex: efficientnetv2b0, resnet50, mobilenetv2, densenet121)")

    p.add_argument("--weights", type=str, default="imagenet",
                   help="Use 'imagenet' para pré-treinado, ou 'none' para treinar do zero")

    p.add_argument("--backbone-trainable", action="store_true",
                   help="Se setado, libera treino do backbone (fine-tuning). Se não, treina só a cabeça.")


    return p.parse_args()

# ======= Util =======
def make_outdir() -> str:
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(os.getcwd(), "outputs", run_name)
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

def main():
    args = parse_args()

    if not hasattr(args, "check_pipeline"):
        args.check_pipeline = False

    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(8)

    # Semente
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    # Diretório de saída
    out_dir = make_outdir()
    print(f"[INFO] Salvando saídas em: {out_dir}")

    # Config de augment (use a padrão do pipeline; personalize aqui se quiser)
    aug_cfg = AugmentConfig()

   

    # ======= (Opcional) Modo de checagem do pipeline =======
    if args.check_pipeline:
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
        )
        print("\n--- RESULTADO DA DIVISÃO ---")
        print(f"[META] n_train: {meta['n_train']}")
        print(f"[META] n_val:   {meta['n_val']}")
        print(f"[META] n_test:  {meta['n_test']}")
        print(f"[META] Pesos:   {meta['class_weights']}")
        print("\n[PIPELINE] Datasets criados com sucesso.")
        print("[PIPELINE] Info:", info)
        sys.exit(0)

    # ======= Datasets =======
    train_ds, val_ds, test_ds, meta, info = build_datasets_nbm(
        img_size=args.img_size,
        channels=args.channels,
        batch_size=args.batch_size,
        max_per_class=args.max_per_class,
        val_size=args.val_size,
        test_size=args.test_size,           # novo split de teste
        extra_fraction=args.extra_fraction, # 200% de augment extra
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        aug_cfg=aug_cfg,
    )
    print("[PIPELINE] Info:", info)

        # ======= Modelo =======
    if args.arch == "custom":
        model = build_model(img_size=args.img_size, channels=args.channels, num_classes=3)
        chosen = {"arch": "custom", "backbone": None, "weights": None, "backbone_trainable": None}
    else:
        w = None if str(args.weights).lower().strip() in ("none", "null", "0", "") else args.weights
        model = build_keras_app_model(
            img_size=args.img_size,
            channels=args.channels,
            num_classes=3,
            backbone=args.backbone,
            weights=w,
            trainable_backbone=args.backbone_trainable,
        )
        chosen = {"arch": "keras_app", "backbone": args.backbone, "weights": w, "backbone_trainable": args.backbone_trainable}

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[
            "accuracy",
            MultiClassAUC(num_classes=3, name="auc_ovr")
        ]
    )
    model.summary()

    #

    cbs = make_basic_callbacks(out_dir, num_classes=3, val_ds=val_ds)

    # ======= Callbacks =======
    ckpt_last = os.path.join(out_dir, "last.keras")
    ckpt_best = os.path.join(out_dir, "best.keras")
    csv_log   = os.path.join(out_dir, "history.csv")

    # ======= Treino =======
    class_weights = meta.get("class_weights", None)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=cbs,
        class_weight=class_weights if class_weights else None,
        verbose=1,
    )

    # ======= Salva artefatos do modelo =======
    save_model_artifacts(model, out_dir)
    save_history(history, out_dir)  # cria history.json

    # ======= Avaliações =======
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
        "model_choice": chosen,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "lr": float(args.lr),
        "epochs": int(args.epochs),
        "img_size": int(args.img_size),
        "batch_size": int(args.batch_size),
        "extra_fraction": float(args.extra_fraction),
        "val_size": float(args.val_size),
        "test_size": float(args.test_size),
        "class_weights": class_weights,
        "train_time_sec": getattr(history, "model", None) and None,  # placeholder
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
        "class_weights": meta.get("class_weights", {}),
    }
    _ = finalize_and_save(
        model=model,
        ds=val_ds,
        steps=steps_val,
        num_classes=3,
        run_dir=out_dir,
        train_time_sec=None,
        extra_info=extra_info,
    )

    # ======= Avaliação (test) e anexar no results_final.json =======
    steps_test = math.ceil(meta["n_test"] / args.batch_size)
    test_dict = model.evaluate(test_ds, steps=steps_test, verbose=0, return_dict=True)
    # atualiza o results_final.json já criado pelo finalize_and_save:
    rf = os.path.join(out_dir, "results_final.json")
    with open(rf, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["test_metrics"] = {k: float(v) for k, v in test_dict.items()}
    with open(rf, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Artefatos salvos em: {out_dir}")

    print("\n[RESULTS]")
    for k, v in results.items():
        if isinstance(v, dict):
            print(" ", k + ":")
            for kk, vv in v.items():
                print(f"   - {kk}: {vv}")
        else:
            print(f"  {k}: {v}")
    

    from metrics_logs import plot_training_curves
    plot_training_curves(history.history, out_dir, fname="training_curves.png")

    print(f"\n✅ Artefatos salvos em: {out_dir}")


if __name__ == "__main__":
    main()
