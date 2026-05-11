from __future__ import annotations
import json, datetime as dt
from pathlib import Path
from typing import Dict, Optional
from xml.parsers.expat import model

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, recall_score, precision_score, f1_score
)
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Callback: métricas por classe a cada época ----------
class PerClassMetricsCallback(keras.callbacks.Callback):
    """
    Grava por época: class_{k}_recall, class_{k}_precision e class_{k}_auc (one-vs-rest) a partir do val_ds.
    """
    def __init__(self, val_ds, num_classes: int, save_dir: Path):
        super().__init__()
        self.val_ds = val_ds
        self.K = int(num_classes)
        self.save_dir = Path(save_dir)
        self.hist = {"epoch": []}
        for k in range(self.K):
            self.hist[f"class_{k}_recall"]    = []
            self.hist[f"class_{k}_precision"] = []
            self.hist[f"class_{k}_auc"]       = []

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_prob = [], []
        for xb, yb in self.val_ds:
            pb = self.model.predict(xb, verbose=0)
            y_true.append(yb.numpy())
            y_prob.append(pb)
        y_true = np.concatenate(y_true, axis=0)
        y_prob = np.concatenate(y_prob, axis=0)
        y_pred = y_prob.argmax(1)

        rec = recall_score(
            y_true, y_pred, average=None,
            labels=list(range(self.K)), zero_division=0
        )
        prec = precision_score(
            y_true, y_pred, average=None,
            labels=list(range(self.K)), zero_division=0
        )

        # AUC one-vs-rest por classe
        y_true_oh = tf.one_hot(y_true, self.K).numpy()
        aucs = []
        for k in range(self.K):
            try:
                aucs.append(roc_auc_score(y_true_oh[:, k], y_prob[:, k]))
            except ValueError:
                aucs.append(float("nan"))

        self.hist["epoch"].append(epoch + 1)
        for k in range(self.K):
            self.hist[f"class_{k}_recall"].append(float(rec[k]) if k < len(rec) else float("nan"))
            self.hist[f"class_{k}_precision"].append(float(prec[k]) if k < len(prec) else float("nan"))
            self.hist[f"class_{k}_auc"].append(float(aucs[k]))

        # salva/atualiza CSV acumulado por época
        pd.DataFrame(self.hist).to_csv(self.save_dir / "metrics_history.csv", index=False)

# ---------- util: salvar history ----------
def save_history(history: keras.callbacks.History, run_dir: Path):
    run_dir = Path(run_dir)
    # CSVLogger já grava history.csv; aqui garantimos o JSON também:
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    (run_dir / "history.json").write_text(json.dumps(hist_dict, indent=2))

# ---------- avaliação final + artefatos ----------
def finalize_and_save(
    model: keras.Model,
    ds,                  # <<< ALTERADO: de 'val_ds' para 'ds' (genérico)
    steps: int,           # <<< ALTERADO: de 'val_steps' para 'steps'
    num_classes: int,
    run_dir: Path,
    train_time_sec: Optional[float] = None,
    extra_info: Optional[Dict] = None,
) -> Dict:
    run_dir = Path(run_dir)

     # Avaliação Keras (robusta a n métricas)
     # <<< ALTERADO: usa ds e steps >>>
    eval_out = model.evaluate(ds, steps=steps, verbose=0, return_dict=True)
    val_loss = float(eval_out.get("loss", float("nan")))
    val_acc  = float(eval_out.get("accuracy", float("nan")))
    val_auc_keras = float(eval_out.get("auc_ovr", float("nan")))
    # Predições para métricas detalhadas
    y_true, y_prob = [], []
     # <<< ALTERADO: usa ds e steps >>>
    for xb, yb in ds.take(steps):
            pb1 = model.predict(xb, verbose=0)
            pb2 = model.predict(tf.image.flip_left_right(xb), verbose=0)
            pb  = (pb1 + pb2) / 2.0
            y_true.append(yb.numpy()); y_prob.append(pb)
    y_true = np.concatenate(y_true, axis=0)
    y_prob = np.concatenate(y_prob, axis=0)
    y_pred = y_prob.argmax(1)
    y_true_oh = tf.one_hot(y_true, num_classes).numpy()

    # AUC OVR
    try:
        val_auc_ovr = float(roc_auc_score(y_true_oh, y_prob, multi_class="ovr"))
    except Exception:
        val_auc_ovr = float("nan")

    # Per-class recall/AUC
    recalls = recall_score(y_true, y_pred, average=None,
                           labels=list(range(num_classes)), zero_division=0)
    per_class = {f"class_{k}_recall": float(recalls[k]) for k in range(num_classes)}
    for k in range(num_classes):
        try:
            per_class[f"class_{k}_auc"] = float(roc_auc_score(y_true_oh[:, k], y_prob[:, k]))
        except Exception:
            per_class[f"class_{k}_auc"] = float("nan")

    # Classification report
    report_txt = classification_report(
        y_true, y_pred,
        target_names=[f"Class_{k}" for k in range(num_classes)],
        zero_division=0
    )
    (run_dir / "classification_report.txt").write_text(report_txt)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    disp = ConfusionMatrixDisplay(cm, display_labels=[f"C{k}" for k in range(num_classes)])
    plt.figure()
    disp.plot(values_format="d")
    plt.title("Matriz de Confusão - Avaliação Final") # Título generalizado
    plt.savefig(run_dir / "cm.png", bbox_inches="tight")
    plt.close()

    # Sumário/arquitetura do modelo
    from io import StringIO
    import contextlib
    sio = StringIO()
    with contextlib.redirect_stdout(sio):
        model.summary(line_length=120)
    architecture_summary = sio.getvalue()
    (run_dir / "model_summary.txt").write_text(architecture_summary)
    (run_dir / "model.json").write_text(model.to_json())

    # JSON de resultados “consolidado”
    results = {
        "final_loss": float(val_loss),      # Renomeado de val_loss
        "final_accuracy": float(val_acc),   # Renomeado de val_accuracy
        "final_auc_ovr": val_auc_ovr,       # Renomeado de val_auc_ovr
        "final_auc_keras": val_auc_keras,   # Renomeado de val_auc_keras
        **per_class,
        "train_time_sec": float(train_time_sec) if train_time_sec is not None else None,
        **(extra_info or {}),
    }
    (run_dir / "results_final.json").write_text(json.dumps(results, indent=2))
    # --- Grad-CAMs (opcional, pós-avaliação) ---
    try:
         # <<< ALTERADO: usa ds >>>
        ds_oh = ds.map(
            lambda x, y: (x, tf.one_hot(y, num_classes)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        save_gradcams_for_errors(
            model, ds_oh, run_dir / "gradcam",
            max_images=40, last_name=last_conv_name(model)
        )
        print("✅ Grad-CAM salvo em:", run_dir / "gradcam")
    except Exception as e:
        print("[Aviso] Falha ao gerar Grad-CAM:", e)
    return results

# =========================
# Grad-CAM utils
# =========================
def last_conv_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (L.Conv2D, L.SeparableConv2D, L.DepthwiseConv2D)):
            return layer.name
    return None

def grad_cam(model, img_tensor, class_index, last_name=None):
    if last_name is None:
        last_name = last_conv_name(model)
    conv_layer = model.get_layer(last_name)
    grad_model = keras.Model([model.inputs], [conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        img_tensor = tf.cast(img_tensor, tf.float32)
        conv_out, preds = grad_model(tf.expand_dims(img_tensor, 0))
        loss = preds[:, class_index]
    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_out * tf.reshape(weights, (1, 1, 1, -1)), axis=-1)[0]
    cam = tf.maximum(cam, 0)
    cam = (cam - tf.reduce_min(cam)) / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()

def overlay_heatmap(image_np, heatmap, alpha=0.4):
    import matplotlib.cm as cm
    # Reconstruir para visualização (dataset está normalizado em [0,1])
    base = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
    hm = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    colored = cm.get_cmap('jet')(hm / 255.0)[..., :3]
    overlay = (1 - alpha) * (base / 255.0) + alpha * colored
    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

def save_gradcams_for_errors(model, ds_oh, out_dir, max_images=40, last_name=None): # <<< 'val_ds' -> 'ds_oh'
    """
    Salva Grad-CAMs em duas subpastas:
      - erros: amostras onde y_true != y_pred
      - acertos: amostras onde y_true == y_pred
    Espera ds_oh no formato (x, y_onehot). Limita cada grupo a max_images.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    erros_dir = out_dir / "erros"
    acertos_dir = out_dir / "acertos"
    erros_dir.mkdir(parents=True, exist_ok=True)
    acertos_dir.mkdir(parents=True, exist_ok=True)

    if last_name is None:
        last_name = last_conv_name(model)

    images_saved_err = 0
    images_saved_ok = 0
    print(f"[INFO] Gerando Grad-CAMs (até {max_images}) para erros e acertos...")

    for xb, yb_oh in ds_oh: # <<< 'val_ds' -> 'ds_oh'
        if images_saved_err >= max_images and images_saved_ok >= max_images:
            break

        pb = model.predict(xb, verbose=0)
        y_true = np.argmax(yb_oh.numpy(), axis=1)
        y_pred = np.argmax(pb, axis=1)

        err_idx = np.where(y_true != y_pred)[0]
        ok_idx = np.where(y_true == y_pred)[0]

        # Erros
        for i in err_idx:
            if images_saved_err >= max_images:
                break
            img_tensor = xb[i]
            t = int(y_true[i]); p = int(y_pred[i])
            try:
                hm = grad_cam(model, img_tensor, p, last_name)
                hm_resized = tf.image.resize(hm[..., None], img_tensor.shape[:2])[..., 0].numpy()
                ov = overlay_heatmap(img_tensor.numpy(), hm_resized)
                plt.imsave(erros_dir / f"err_{images_saved_err:02d}_true{t}_pred{p}.png", ov)
                images_saved_err += 1
            except Exception as e:
                print(f"[Aviso] Grad-CAM (erro) falhou: {e}")

        # Acertos
        for i in ok_idx:
            if images_saved_ok >= max_images:
                break
            img_tensor = xb[i]
            cls = int(y_true[i])  # igual ao y_pred
            try:
                hm = grad_cam(model, img_tensor, cls, last_name)
                hm_resized = tf.image.resize(hm[..., None], img_tensor.shape[:2])[..., 0].numpy()
                ov = overlay_heatmap(img_tensor.numpy(), hm_resized)
                plt.imsave(acertos_dir / f"ok_{images_saved_ok:02d}_class{cls}.png", ov)
                images_saved_ok += 1
            except Exception as e:
                print(f"[Aviso] Grad-CAM (acerto) falhou: {e}")

    print(f"[INFO] Salvos: erros={images_saved_err} em {erros_dir}, acertos={images_saved_ok} em {acertos_dir}")




def plot_training_curves(history_dict, out_dir, fname="training_curves.png"):
    """
    Salva um PNG com curvas de treino/val das métricas encontradas no history.
    - Procura por pares (métrica, val_métrica) automaticamente.
    """
    import matplotlib
    matplotlib.use("Agg")  # garante backend não-interativo
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hist = dict(history_dict)  # copy (pode vir como History.history)
    keys = set(hist.keys())

    # Descobre pares metric/val_metric que existem
    pairs = []
    for k in sorted(keys):
        if k.startswith("val_"):
            continue
        vk = f"val_{k}"
        if vk in keys:
            pairs.append((k, vk))

    # Se não achou nada, tenta pelo menos loss/val_loss
    if not pairs and "loss" in keys and "val_loss" in keys:
        pairs = [("loss", "val_loss")]

    # Monta figura
    n = max(1, min(len(pairs), 6))  # evita subplots demais
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(6*cols, 4*rows))

    for i, (tr, va) in enumerate(pairs, 1):
        ax = plt.subplot(rows, cols, i)
        ax.plot(hist[tr], label=tr)
        ax.plot(hist[va], label=va)
        ax.set_xlabel("Epoch")
        ax.set_title(tr.replace("_", " / "))
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Extras úteis, se existirem
    for extra in ["lr"]:
        if extra in hist:
            plt.figure(figsize=(6,3.2))
            plt.plot(hist[extra])
            plt.title("Learning rate")
            plt.xlabel("Epoch")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "lr_curve.png", dpi=150, bbox_inches="tight")

    plt.tight_layout()
    out_path = out_dir / fname
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✅ Curvas salvas em: {out_path}")

class ValMacroF1(tf.keras.callbacks.Callback):
    # <<< Adicionado construtor para receber val_ds >>>
    def __init__(self, val_ds, **kwargs):
        super().__init__(**kwargs)
        self.val_ds = val_ds

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []
        # <<< Usa self.val_ds >>>
        for x,y in self.val_ds: 
            p = tf.argmax(self.model(x, training=False), axis=-1)
            y_true.extend(y.numpy()); y_pred.extend(p.numpy())
        logs['val_macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

# ---------- atalhos para checkpoints ----------
def make_basic_callbacks(run_dir: Path, num_classes: int, val_ds):
    run_dir = Path(run_dir)

    f1_callback = ValMacroF1(val_ds=val_ds)

    return [
        keras.callbacks.ModelCheckpoint(str(run_dir / "best.keras"),
                                        monitor="val_loss", mode="min",
                                        save_best_only=True, verbose=0),
        keras.callbacks.ModelCheckpoint(str(run_dir / "best_auc.keras"),
                                        monitor="val_auc_ovr", mode="max",     # <<< AQUI
                                        save_best_only=True, verbose=0),
        keras.callbacks.ModelCheckpoint(str(run_dir / "last.keras"),
                                        save_best_only=False, verbose=1),
        f1_callback,
        keras.callbacks.EarlyStopping(monitor='val_macro_f1', mode='max',
                                      patience=10, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc_ovr", mode="max",   # <<< AQUI
                                          factor=0.3, patience=4, min_lr=1e-7, verbose=0),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv"), append=True),
        PerClassMetricsCallback(val_ds=val_ds, num_classes=num_classes, save_dir=run_dir),
    ]

