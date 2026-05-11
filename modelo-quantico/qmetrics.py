from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.stats import entropy  # entropia de Shannon

try:
    from qiskit.quantum_info import SparsePauliOp
except Exception:
    SparsePauliOp = None

# sklearn é opcional: se não tiver, finalize_and_save vira no-op
try:
    from sklearn.metrics import confusion_matrix, classification_report
except Exception:
    confusion_matrix = None
    classification_report = None


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_numpy(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def _predict_eager(model, x, training: bool = False):
    """Chamada direta ao modelo em eager (sem .predict/.fit graph)."""
    y = model(x, training=training)
    return _to_numpy(y)


# ---------- 1) Ativações dos qubits ----------
class QubitActivationsMonitor(keras.callbacks.Callback):
    def __init__(
        self,
        probe_q_model: keras.Model,
        x_probe,
        run_dir: Path,
        freq: int = 1,
        sat_thr: float = 0.95,
    ):
        super().__init__()
        self.probe = probe_q_model
        self.x_probe = x_probe
        self.run_dir = Path(run_dir)
        self.freq = int(freq)
        self.sat_thr = float(sat_thr)
        _ensure_dir(self.run_dir)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.freq) != 0:
            return
        z = _predict_eager(self.probe, self.x_probe, training=False)
        z = _to_numpy(z)
        mean = z.mean(axis=0).tolist()
        std = z.std(axis=0).tolist()
        sat_frac = (np.abs(z) >= self.sat_thr).mean(axis=0).tolist()
        payload = {
            "epoch": int(epoch),
            "mean": mean,
            "std": std,
            "sat_frac": sat_frac,
        }
        with open(self.run_dir / "q_activations.jsonl", "a") as f:
            f.write(json.dumps(payload) + "\n")


# ---------- 2) Norma do gradiente da QNN ----------
class QGradNormMonitor(keras.callbacks.Callback):
    def __init__(
        self,
        model: keras.Model,
        x_batch,
        y_batch,
        q_layer_name: str,
        run_dir: Path,
        freq: int = 1,
    ):
        super().__init__()
        self.model_ref = model
        self.xb = x_batch
        self.yb = y_batch
        self.q_layer_name = q_layer_name
        self.run_dir = Path(run_dir)
        self.freq = int(freq)
        _ensure_dir(self.run_dir)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.freq) != 0:
            return
        q_layer = self.model_ref.get_layer(self.q_layer_name)
        theta = q_layer.theta
        with tf.GradientTape() as tape:
            preds = self.model_ref(self.xb, training=True)
            loss = self.model_ref.compiled_loss(self.yb, preds)
        grads = tape.gradient(loss, [theta])
        g = grads[0]
        val = (
            float(tf.linalg.global_norm([g]).numpy())
            if g is not None
            else 0.0
        )
        with open(self.run_dir / "q_gradnorm.txt", "a") as f:
            f.write(f"{epoch}\t{val:.6e}\n")


# ---------- 3) Pureza dos qubits via Bloch ----------
class QubitPurityMonitor(keras.callbacks.Callback):
    def __init__(
        self,
        q_layer,
        proj_model: keras.Model,
        x_batch,
        run_dir: Path,
        max_samples: int = 8,
        freq: int = 1,
    ):
        super().__init__()
        self.q_layer = q_layer
        self.proj = proj_model
        self.xb = x_batch
        self.max_samples = int(max_samples)
        self.run_dir = Path(run_dir)
        self.freq = int(freq)
        _ensure_dir(self.run_dir)

        if SparsePauliOp is None:
            raise ImportError("QubitPurityMonitor requer Qiskit instalado.")

        n = self.q_layer.n_qubits
        self.obsX = [
            SparsePauliOp("I" * i + "X" + "I" * (n - 1 - i))
            for i in range(n)
        ]
        self.obsY = [
            SparsePauliOp("I" * i + "Y" + "I" * (n - 1 - i))
            for i in range(n)
        ]
        self.obsZ = [
            SparsePauliOp("I" * i + "Z" + "I" * (n - 1 - i))
            for i in range(n)
        ]

    def _bloch_for(self, q_input_vec: np.ndarray, theta: np.ndarray):
        alpha = self.q_layer.data_scale * q_input_vec
        qc = self.q_layer._build_numeric_circuit(alpha, theta)
        est = self.q_layer.estimator

        valsX = np.asarray(
            est.run(
                circuits=[qc] * len(self.obsX),
                observables=self.obsX,
            ).result().values,
            dtype=np.float32,
        )
        valsY = np.asarray(
            est.run(
                circuits=[qc] * len(self.obsY),
                observables=self.obsY,
            ).result().values,
            dtype=np.float32,
        )
        valsZ = np.asarray(
            est.run(
                circuits=[qc] * len(self.obsZ),
                observables=self.obsZ,
            ).result().values,
            dtype=np.float32,
        )

        r2 = valsX**2 + valsY**2 + valsZ**2
        return (1.0 + r2) / 2.0  # pureza por qubit

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.freq) != 0:
            return

        x = self.xb[0] if isinstance(self.xb, (list, tuple)) else self.xb
        x = x[: self.max_samples]

        # Chamada em eager, sem predict()
        q_in = _predict_eager(self.proj, x, training=False)

        theta = _to_numpy(self.q_layer.theta)
        purities = [self._bloch_for(q_in[k], theta) for k in range(q_in.shape[0])]
        mean_purity = np.mean(purities, axis=0).tolist()
        with open(self.run_dir / "q_purity.jsonl", "a") as f:
            f.write(
                json.dumps(
                    {"epoch": int(epoch), "mean_purity": mean_purity}
                )
                + "\n"
            )


# ---------- 4) Entropia das predições ----------
class PredictionEntropyMonitor(keras.callbacks.Callback):
    """
    Monitora a entropia das predições do modelo (em validação).
    Baixa entropia -> modelo mais confiante.
    """

    def __init__(
        self, val_ds, run_dir: Path, max_batches: int = 5, freq: int = 1
    ):
        super().__init__()
        self.val_ds = val_ds
        self.run_dir = Path(run_dir)
        self.max_batches = int(max_batches)
        self.freq = int(freq)
        _ensure_dir(self.run_dir)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.freq) != 0:
            return

        all_probs = []
        for i, (xb, _) in enumerate(self.val_ds):
            if i >= self.max_batches:
                break
            probs = _predict_eager(self.model, xb, training=False)
            all_probs.append(probs)

        if not all_probs:
            return

        all_probs = np.concatenate(all_probs, axis=0)
        entropies = entropy(all_probs, axis=1)
        avg_entropy = float(np.mean(entropies))

        print(f"\n[ENTROPY] epoch={epoch} avg_entropy={avg_entropy:.4f}")
        with open(self.run_dir / "q_entropy.txt", "a") as f:
            f.write(f"{epoch}\t{avg_entropy:.6f}\n")


# ---------- 5) Callbacks básicos / history / finalize ----------
def make_basic_callbacks(
    run_dir: Path, monitor: str = "val_accuracy", mode: str = "max"
):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ckpt = keras.callbacks.ModelCheckpoint(
        filepath=str(run_dir / "best_model.keras"),
        monitor=monitor,
        mode=mode,
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    csv = keras.callbacks.CSVLogger(str(run_dir / "training_log.csv"))

    return [ckpt, csv]


def save_history(history: keras.callbacks.History, run_dir: Path):
    run_dir = Path(run_dir)
    hist_dict = {
        k: [float(v) for v in vals] for k, vals in history.history.items()
    }
    (run_dir / "history.json").write_text(json.dumps(hist_dict, indent=2))


def finalize_and_save(
    model,
    val_ds,
    val_steps,
    num_classes,
    run_dir,
    train_time_sec: float | None = None,
    extra_info: dict | None = None,
):
    """
    Gera confusion_matrix.csv, classification_report.txt e metrics_summary.json,
    usando os dados de validação.

    Se scikit-learn não estiver disponível, vira no-op silenciosa.
    """
    if confusion_matrix is None or classification_report is None:
        # sklearn não instalado; não faz nada
        return {}

    run_dir = Path(run_dir)
    _ensure_dir(run_dir)

    y_true, y_prob = [], []
    for i, (xb, yb) in enumerate(val_ds):
        if val_steps is not None and i >= val_steps:
            break
        probs = _predict_eager(model, xb, training=False)
        y_prob.append(probs)
        y_true.append(_to_numpy(yb))

    if not y_true:
        return {}

    y_true = np.concatenate(y_true, axis=0)
    y_prob = np.concatenate(y_prob, axis=0)
    y_pred = np.argmax(y_prob, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    np.savetxt(
        run_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d"
    )

    # Classification report
    rep_str = classification_report(
        y_true, y_pred, labels=list(range(num_classes)), digits=4
    )
    (run_dir / "classification_report.txt").write_text(rep_str)

    # Resumo de métricas
    metrics = {
        "val_accuracy_manual": float((y_pred == y_true).mean()),
        "train_time_sec": float(train_time_sec or 0.0),
    }
    if extra_info:
        metrics.update(extra_info)

    (run_dir / "metrics_summary.json").write_text(
        json.dumps(metrics, indent=2)
    )

    return metrics
