from __future__ import annotations
import os
import json
import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

# --- DESATIVAR XLA / JIT PARA NÃO QUEBRAR A PyFunc DA QNN ---
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_JIT_PROFILING"] = "false"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Desativa otimizações globais do Keras que poderiam tentar compilar o grafo
tf.config.optimizer.set_jit(False)

from data_pipeline import build_datasets_nbm
from qmetrics import (
    make_basic_callbacks,
    QubitActivationsMonitor,
    QGradNormMonitor,
    QubitPurityMonitor,
    PredictionEntropyMonitor,
    finalize_and_save,
)
from model_qnn import build_model, compile_model

tf.get_logger().setLevel("ERROR")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-6)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--csv-path", type=str, default=None)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--extra-fraction", type=float, default=0.0)
    p.add_argument("--n-qubits", type=int, default=6)
    p.add_argument("--q-reps", type=int, default=1)
    p.add_argument("--q-shots", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    # 0. Diretório de saída
    base_output = Path.cwd() / "outputs"
    timestamp = int(time.time())
    run_dir = base_output / f"run_qnn_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Salvando saídas em: {str(run_dir.resolve())}")

    # 1. Carrega datasets
    train_ds, val_ds, test_ds, meta, info = build_datasets_nbm(
        img_size=args.img_size,
        channels=args.channels,
        batch_size=args.batch,  # <- compatível com seu data_pipeline
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        val_size=args.val_size,
        test_size=args.test_size,
        max_per_class=args.max_per_class,
        extra_fraction=args.extra_fraction,
    )

    print(
        f"[INFO] Dados carregados: "
        f"{meta['n_train']} Treino, {meta['n_val']} Validação, {meta['n_test']} Teste"
    )

    steps_base = info.get("steps_base", info.get("base_steps", 0))
    steps_aug = info.get("steps_aug", info.get("aug_steps", 0))
    steps_per_epoch = int(steps_base + steps_aug)
    val_steps = int(np.ceil(meta["n_val"] / args.batch))

    # Número de classes
    num_classes = len(meta.get("class_map", {}))
    if num_classes == 0:
        print(
            "[WARN] class_map vazio no meta. Assumindo 3 classes "
            "(Normal, Benign, Malignant)."
        )
        num_classes = 3

    print(
        f"[INFO] Configurando modelo para {num_classes} classes com "
        f"{args.n_qubits} qubits."
    )

    # 2. Constrói modelo híbrido
    model = build_model(
        img_size=args.img_size,
        channels=args.channels,
        num_classes=num_classes,
        n_qubits=args.n_qubits,
        reps=args.q_reps,
        shots=args.q_shots,
    )

    # Congela o tronco CNN para esta fase (apenas proj_to_qubits, QNN e densas finais treinam)
    # for layer in model.layers:
    #     if layer.name not in ["proj_to_qubits", "qiskit_qnn", "dense", "dense_1"]:
    #         layer.trainable = False

    model = compile_model(model, lr=args.lr, weight_decay=args.wd)
    model.summary()

    # 3. Callbacks básicos
    cbs = make_basic_callbacks(run_dir, monitor="val_accuracy", mode="max")

    # 3.1. Monitores quânticos
    try:
        q_layer = model.get_layer("qiskit_qnn")
        probe_q = keras.Model(model.input, q_layer.output)
        proj_to_q = keras.Model(
            model.input, model.get_layer("proj_to_qubits").output
        )

        x_probe, y_probe = next(iter(val_ds))

        cbs.extend(
            [
                QubitActivationsMonitor(probe_q, x_probe, run_dir, freq=1),
                QGradNormMonitor(
                    model,
                    x_probe,
                    y_probe,
                    q_layer_name="qiskit_qnn",
                    run_dir=run_dir,
                    freq=1,
                ),
                QubitPurityMonitor(
                    q_layer,
                    proj_to_q,
                    x_probe,
                    run_dir,
                    max_samples=8,
                    freq=1,
                ),
                PredictionEntropyMonitor(
                    val_ds, run_dir, max_batches=5, freq=1
                ),
            ]
        )
        print("[INFO] Monitores quânticos ativados.")
    except Exception as e:
        print(f"[WARN] Falha ao configurar monitores quânticos: {e}")

    # 4. Treinamento
    print("[INFO] Iniciando treino...")
    t0 = time.time()
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=cbs,
        verbose=1,
    )
    train_time = time.time() - t0

    # Salvar histórico
    with open(run_dir / "history.json", "w") as f:
        json.dump(
            {k: [float(x) for x in v] for k, v in history.history.items()},
            f,
            indent=2,
        )

    # 4.1. (Opcional) gerar confusion matrix, classification report etc.
    try:
        finalize_and_save(
            model=model,
            val_ds=val_ds,
            val_steps=val_steps,
            num_classes=num_classes,
            run_dir=run_dir,
            train_time_sec=train_time,
            extra_info={
                "epochs": int(args.epochs),
                "lr": float(args.lr),
                "weight_decay": float(args.wd),
            },
        )
    except Exception as e:
        print(f"[WARN] finalize_and_save falhou: {e}")

    # 5. Teste final
    if meta["n_test"] > 0:
        print("\n[INFO] Avaliando performance final no conjunto de TESTE...")
        test_res = model.evaluate(test_ds, verbose=1)
        test_metrics = {
            name: float(val) for name, val in zip(model.metrics_names, test_res)
        }
        print(f"[TEST RESULT] {test_metrics}")
        with open(run_dir / "test_results.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

    print("[DONE] Treino e teste finalizados.")


if __name__ == "__main__":
    main()
