"""
mamo_data5.py — CustomSimpleMammoCNN_v2 (integrado com ideias do mamo_cnn4.py)

- Lê o dataset (Mammo-Bench)
- Separa no máximo 7000 imagens por classe
- Augmentations leves (flip, brilho, contraste)
- Salva métricas: history.csv/json, per-classe por época (recall/AUC), cm.png,
  classification_report.txt, results_<run>.json (com AUC), best.keras/last.keras
- Salva arquitetura do modelo: model_summary.txt e model.json (e inclui summary no JSON de resultados)
"""

import os, json, uuid, time, datetime as dt
from pathlib import Path

# Verbosidade do TF e estabilidade
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_auc_score, recall_score
)
import matplotlib.pyplot as plt

# =========================
# Seeds e GPU
# =========================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print("[WARN] Memory growth não pôde ser habilitado:", e)

# =========================
# Caminhos
# =========================
REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR  = (REPO_ROOT / "dataset" / "Mammo_Data" / "Mammo-Bench").resolve()
CSV_FILE  = BASE_DIR / "CSV_Files" / "mammo-bench_nbm_classification.csv"
IMG_ROOT  = BASE_DIR / "Preprocessed_Dataset"
assert CSV_FILE.exists(), f"CSV não encontrado: {CSV_FILE}"

REPO_ROOT = Path(__file__).resolve().parents[1]  # raiz do repo (pai de official_model/)
OUT_ROOT  = Path(os.environ.get("MAMMO_OUT", REPO_ROOT / "outputs")).resolve()
OUT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_NAME = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
RUN_DIR = OUT_ROOT / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Salvando saídas em: {RUN_DIR}")

# =========================
# Hiperparâmetros simples
# =========================
IMG_SIZE = 1024            # 1024x1024 grayscale (seguro para ~12GB)
CHANNELS = 1
NUM_CLASSES = 3           # Normal/Benign/Malignant
BATCH_SIZE = 8
EPOCHS = 30
VAL_SIZE = 0.15
MAX_PER_CLASS = 7000      # <= pedido: separar até 7000 por classe

# Augmentações leves
AUG_BRIGHTNESS = 0.03
AUG_CONTRAST_L = 0.92
AUG_CONTRAST_U = 1.08
# Quantidade extra de amostras AUGMENTADAS por época (fração do tamanho do treino)
AUG_EXTRA_FRACTION = 0.50  # ex.: 0.50 => +50% amostras augmentadas além das 17.850 “limpas”

# =========================
# Carrega CSV, limita 7000 por classe e faz split
# =========================
df = pd.read_csv(CSV_FILE)
label_col = "classification"
label_map = {"Normal": 0, "Benign": 1, "Malignant": 2}
assert label_col in df.columns, f"Coluna '{label_col}' não encontrada no CSV."

# manter apenas classes de interesse e mapear para ids
df = df[df[label_col].isin(label_map)].copy()
df["label"] = df[label_col].map(label_map).astype(np.int32)

# coluna do caminho pré-processado (igual seus scripts)
img_col = "preprocessed_image_path"
assert img_col in df.columns, f"Coluna '{img_col}' não encontrada no CSV."
df["img_path"] = df[img_col].apply(lambda p: str((BASE_DIR / p).resolve()))

# filtra arquivos existentes
exists_mask = df["img_path"].map(lambda p: os.path.exists(p))
missing = int((~exists_mask).sum())
if missing:
    print(f"[AVISO] {missing} arquivos ausentes serão ignorados.")
df = df[exists_mask].reset_index(drop=True)

# === Limitar a até 7000 por classe ===
df_bal = (
    df.groupby("label", group_keys=False, sort=False)
      .apply(lambda g: g.sample(n=min(len(g), MAX_PER_CLASS), random_state=SEED))
      .reset_index(drop=True)
)

# split estratificado
train_df, val_df = train_test_split(
    df_bal, test_size=VAL_SIZE, stratify=df_bal["label"], random_state=SEED
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print(f"[INFO] Train: {len(train_df)} | Val: {len(val_df)}")
print("[INFO] Dist. treino por classe:", train_df["label"].value_counts().to_dict())
print("[INFO] Dist. val por classe:", val_df["label"].value_counts().to_dict())

# class weights
classes_sorted = np.array(sorted(train_df["label"].unique()))
class_weights_arr = compute_class_weight(
    class_weight="balanced", classes=classes_sorted, y=train_df["label"].values
)
CLASS_WEIGHTS = {int(c): float(w) for c, w in zip(classes_sorted, class_weights_arr)}
print("[INFO] Class weights:", CLASS_WEIGHTS)

# =========================
# tf.data pipeline (decode + aug leve)
# =========================
def _decode_resize(path, label):
    img = tf.io.read_file(path)
    # robusto a JPG/PNG/TIFF
    img = tf.io.decode_image(img, channels=CHANNELS, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)   # [0,1]
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], antialias=True)
    return img, label

def _augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=AUG_BRIGHTNESS)
    img = tf.image.random_contrast(img, lower=AUG_CONTRAST_L, upper=AUG_CONTRAST_U)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label

def make_ds(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_decode_resize, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.repeat()  # permite steps_per_epoch
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def make_train_ds_plus_aug(paths, labels, extra_fraction: float):
    """
    Uma passada 'limpa' (sem augment) + fatia finita de um stream infinito augmentado.
    Garante ver exatamente N limpas e extra_fraction*N augmentadas por época.
    """
    N = len(paths)
    autotune = tf.data.AUTOTUNE

    base = tf.data.Dataset.from_tensor_slices((paths, labels))
    base = base.shuffle(buffer_size=N, seed=SEED, reshuffle_each_iteration=True)
    base = base.map(_decode_resize, num_parallel_calls=autotune)
    base = base.batch(BATCH_SIZE)

    num_extra = int(N * max(0.0, float(extra_fraction)))
    extra_steps = int(np.ceil(num_extra / BATCH_SIZE)) if num_extra > 0 else 0

    aug = tf.data.Dataset.from_tensor_slices((paths, labels))
    aug = aug.shuffle(buffer_size=N, seed=SEED, reshuffle_each_iteration=True)
    aug = aug.map(_decode_resize, num_parallel_calls=autotune)
    aug = aug.map(_augment,        num_parallel_calls=autotune)
    aug = aug.repeat().batch(BATCH_SIZE).prefetch(autotune)

    ds = base if extra_steps == 0 else base.concatenate(aug.take(extra_steps))
    return ds.prefetch(autotune)

train_ds = make_train_ds_plus_aug(train_df["img_path"].values, train_df["label"].values, AUG_EXTRA_FRACTION)
val_ds   = make_ds(val_df["img_path"].values,   val_df["label"].values,   False)

# =========================
# Modelo simples (conv -> BN -> ReLU) x2 por bloco + MaxPool, fecha com GAP
# =========================
inputs = L.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS))
x = inputs

# Bloco 1
x = L.Conv2D(32, 5, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
x = L.BatchNormalization()(x); x = L.ReLU()(x)
x = L.Conv2D(32, 5, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
x = L.BatchNormalization()(x); x = L.ReLU()(x)
x = L.MaxPooling2D(2)(x)

# Bloco 2
x = L.Conv2D(64, 5, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
x = L.BatchNormalization()(x); x = L.ReLU()(x)
x = L.Conv2D(64, 5, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
x = L.BatchNormalization()(x); x = L.ReLU()(x)
x = L.MaxPooling2D(2)(x)

# Bloco 3
x = L.Conv2D(128, 5, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
x = L.BatchNormalization()(x); x = L.ReLU()(x)
x = L.Conv2D(128, 5, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
x = L.BatchNormalization()(x); x = L.ReLU()(x)
x = L.MaxPooling2D(2)(x)

# # Bloco 4
# x = L.Conv2D(256, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
# x = L.BatchNormalization()(x); x = L.ReLU()(x)
# x = L.Conv2D(256, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
# x = L.BatchNormalization()(x); x = L.ReLU()(x)
# x = L.MaxPooling2D(2)(x)

# Cabeça
GAP = L.GlobalAveragePooling2D()(x) 
GMP = L.GlobalMaxPooling2D()(x)
x = L.Concatenate()([GAP, GMP])
x = L.Dropout(0.3)(x)

x = L.Dense(256, activation="silu")(x)
x = L.Dropout(0.3)(x)
x = L.Dense(256, activation="silu")(x)
x = L.Dropout(0.3)(x)
outputs = L.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs, name="CustomSimpleMammoCNN")
model.summary()

# =========================
# Compilação e callbacks
# =========================
optimizer = keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-5)
loss = keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

ckpt_best = keras.callbacks.ModelCheckpoint(
    filepath=str(RUN_DIR / "best.keras"),
    monitor="val_loss", mode="min", save_best_only=True, verbose=1
)
ckpt_last = keras.callbacks.ModelCheckpoint(
    filepath=str(RUN_DIR / "last.keras"),
    save_best_only=False, verbose=1
)
es = keras.callbacks.EarlyStopping(
    monitor="val_loss", mode="min", patience=8, restore_best_weights=True, verbose=1
)
csvlog = keras.callbacks.CSVLogger(str(RUN_DIR / "history.csv"))

# =============== Per-class metrics por época (recall e AUC) ===============
class PerClassMetrics(keras.callbacks.Callback):
    def __init__(self, val_ds, save_dir: Path):
        super().__init__()
        self.val_ds = val_ds
        self.save_dir = Path(save_dir)
        self.hist = {"epoch": []}
        for k in range(NUM_CLASSES):
            self.hist[f"class_{k}_recall"] = []
            self.hist[f"class_{k}_auc"] = []

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_prob = [], []
        for xb, yb in self.val_ds:
            pb = self.model.predict(xb, verbose=0)
            y_true.append(yb.numpy()); y_prob.append(pb)
        y_true = np.concatenate(y_true)
        y_prob = np.concatenate(y_prob)
        y_pred = y_prob.argmax(1)

        # recalls por classe
        rec = recall_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
        # AUC por classe (one-vs-rest)
        y_true_oh = tf.one_hot(y_true, NUM_CLASSES).numpy()
        aucs = []
        for k in range(NUM_CLASSES):
            try:
                aucs.append(roc_auc_score(y_true_oh[:, k], y_prob[:, k]))
            except ValueError:
                aucs.append(np.nan)

        self.hist["epoch"].append(epoch + 1)
        for k in range(NUM_CLASSES):
            self.hist[f"class_{k}_recall"].append(float(rec[k]) if k < len(rec) else np.nan)
            self.hist[f"class_{k}_auc"].append(float(aucs[k]))

        pd.DataFrame(self.hist).to_csv(self.save_dir / "metrics_history.csv", index=False)

perclass = PerClassMetrics(val_ds, RUN_DIR)

# =========================
# Treino
# =========================
print("[INFO] Iniciando treino...")
t0 = time.perf_counter()

# steps por época: N limpas + frac*N augmentadas
N_train = len(train_df)
steps_per_epoch = int(np.ceil((N_train + int(N_train * AUG_EXTRA_FRACTION)) / BATCH_SIZE))
val_steps = int(np.ceil(len(val_df) / BATCH_SIZE))

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    callbacks=[ckpt_best, ckpt_last, es, csvlog, perclass],
    class_weight=CLASS_WEIGHTS,
    verbose=1
)

t1 = time.perf_counter()
total_train_s = t1 - t0
print(f"[INFO] Tempo total: {total_train_s/60:.2f} min")

# salvar history.json
hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(RUN_DIR / "history.json", "w") as f:
    json.dump(hist_dict, f, indent=2)

# =========================
# Avaliação + métricas finais
# =========================
print("[INFO] Avaliando no conjunto de validação...")
val_loss, val_acc = model.evaluate(val_ds, steps=val_steps, verbose=1)
print(f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

# y_true / y_prob
y_true, y_prob = [], []
for xb, yb in val_ds:
    pb = model.predict(xb, verbose=0)
    y_true.append(yb.numpy()); y_prob.append(pb)
y_true = np.concatenate(y_true)
y_prob = np.concatenate(y_prob)
y_pred = y_prob.argmax(1)
y_true_oh = tf.one_hot(y_true, NUM_CLASSES).numpy()

# AUC OVR final
try:
    val_auc_ovr = float(roc_auc_score(y_true_oh, y_prob, multi_class="ovr"))
except Exception:
    val_auc_ovr = float("nan")

# CM
cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
cmd = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Benign", "Malignant"])
plt.figure()
cmd.plot(values_format="d")
plt.title("Matriz de Confusão - Validação")
plt.savefig(RUN_DIR / "cm.png", bbox_inches="tight")
plt.close()

# Classification report
rep = classification_report(y_true, y_pred, target_names=["Normal","Benign","Malignant"], zero_division=0)
with open(RUN_DIR / "classification_report.txt", "w") as f:
    f.write(rep)

# =========================
# Salvar modelo e arquitetura
# =========================
final_path = RUN_DIR / "final_model.keras"
model.save(final_path)
print("✅ Modelo final salvo:", final_path)

# summary (texto) e json da arquitetura
from io import StringIO
import contextlib
sio = StringIO()
with contextlib.redirect_stdout(sio):
    model.summary(line_length=120)
architecture_summary = sio.getvalue()
with open(RUN_DIR / "model_summary.txt", "w") as f:
    f.write(architecture_summary)

with open(RUN_DIR / "model.json", "w") as f:
    f.write(model.to_json())

# results_<run>.json com hiperparâmetros, dataset, métricas e arquitetura
total_params = int(model.count_params())
run_id = f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
results = {
    "run_id": run_id,
    "save_dir": str(RUN_DIR),
    "hyperparameters": {
        "img_size": IMG_SIZE, "channels": CHANNELS,
        "batch_size": BATCH_SIZE, "epochs": EPOCHS,
        "aug_brightness": AUG_BRIGHTNESS, "aug_contrast": [AUG_CONTRAST_L, AUG_CONTRAST_U],
        "optimizer": "AdamW(3e-4, wd=1e-5)"
    },
    "dataset": {
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "max_per_class": int(MAX_PER_CLASS),
        "class_weights": {str(k): float(v) for k, v in CLASS_WEIGHTS.items()}
    },
    "model": {
        "name": model.name,
        "total_params": total_params,
        "files": {
            "final_model": str(final_path),
            "model_summary_txt": str(RUN_DIR / "model_summary.txt"),
            "model_json": str(RUN_DIR / "model.json")
        },
        "architecture_summary": architecture_summary
    },
    "final_metrics": {
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "val_auc_ovr": val_auc_ovr,
        "train_time_sec": float(total_train_s)
    }
}
with open(RUN_DIR / f"results_{run_id}.json", "w") as f:
    json.dump(results, f, indent=2)
print("📄 Resultados salvos:", RUN_DIR / f"results_{run_id}.json")

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

def save_gradcams_for_errors(model, val_ds, out_dir, max_images=40, last_name=None):
    """
    Salva Grad-CAMs em duas subpastas:
      - erros: amostras onde y_true != y_pred
      - acertos: amostras onde y_true == y_pred
    Espera val_ds no formato (x, y_onehot). Limita cada grupo a max_images.
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

    for xb, yb_oh in val_ds:
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

# Após salvar results e artefatos, gerar Grad-CAMs
try:
    # converter rótulos do val_ds para one-hot para compatibilidade da função
    val_ds_oh = val_ds.map(lambda x, y: (x, tf.one_hot(y, NUM_CLASSES)), num_parallel_calls=tf.data.AUTOTUNE)
    save_gradcams_for_errors(model, val_ds_oh, RUN_DIR / "gradcam", max_images=40, last_name=last_conv_name(model))
    print("✅ Grad-CAM salvo em:", RUN_DIR / "gradcam")
except Exception as e:
    print("[Aviso] Falha ao gerar Grad-CAM:", e)

print("\n✅ FIM. Artefatos em:", RUN_DIR)






# # Bloco 1
# x = L.Conv2D(32, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
# x = L.BatchNormalization()(x); x = L.ReLU()(x)
# x = L.Conv2D(32, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
# x = L.BatchNormalization()(x); x = L.ReLU()(x)
# x = L.MaxPooling2D(2)(x)

# # Bloco 2
# x = L.Conv2D(64, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
# x = L.BatchNormalization()(x); x = L.ReLU()(x)
# x = L.Conv2D(64, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
# x = L.BatchNormalization()(x); x = L.ReLU()(x)
# x = L.MaxPooling2D(2)(x)

# # Bloco 3
# x = L.Conv2D(128, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
# x = L.BatchNormalization()(x); x = L.ReLU()(x)
# x = L.Conv2D(128, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
# x = L.BatchNormalization()(x); x = L.ReLU()(x)
# x = L.MaxPooling2D(2)(x)

# # Bloco 4
# x = L.Conv2D(256, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
# x = L.BatchNormalization()(x); x = L.ReLU()(x)
# x = L.Conv2D(256, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
# x = L.BatchNormalization()(x); x = L.ReLU()(x)
# x = L.MaxPooling2D(2)(x)

# Cabeça
# GAP = L.GlobalAveragePooling2D()(x) 
# GMP = L.GlobalMaxPooling2D()(x)
# x = L.Concatenate()([GAP, GMP])

# x = L.Dense(256, activation="relu")(x)
# x = L.Dropout(0.3)(x)
# x = L.Dense(256, activation="relu")(x)
# x = L.Dropout(0.3)(x)
# outputs = L.Dense(NUM_CLASSES, activation="softmax")(x)

# model = keras.Model(inputs, outputs, name="CustomSimpleMammoCNN")
# model.summary()
# [INFO] Train: 17850 | Val: 3150
# [INFO] Dist. treino por classe: {2: 5950, 0: 5950, 1: 5950}
# [INFO] Dist. val por classe: {1: 1050, 2: 1050, 0: 1050}
# [INFO] Class weights: {0: 1.0, 1: 1.0, 2: 1.0}
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1761431316.508379  691456 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 9712 MB memory:  -> device: 0, name: NVIDIA RTX A2000 12GB, pci bus id: 0000:01:00.0, compute capability: 8.6
# Model: "CustomSimpleMammoCNN"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │ input_layer (InputLayer)      │ (None, 512, 512, 1)       │               0 │ -                          │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ conv2d (Conv2D)               │ (None, 512, 512, 32)      │             288 │ input_layer[0][0]          │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ batch_normalization           │ (None, 512, 512, 32)      │             128 │ conv2d[0][0]               │
# │ (BatchNormalization)          │                           │                 │                            │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ re_lu (ReLU)                  │ (None, 512, 512, 32)      │               0 │ batch_normalization[0][0]  │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ conv2d_1 (Conv2D)             │ (None, 512, 512, 32)      │           9,216 │ re_lu[0][0]                │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ batch_normalization_1         │ (None, 512, 512, 32)      │             128 │ conv2d_1[0][0]             │
# │ (BatchNormalization)          │                           │                 │                            │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ re_lu_1 (ReLU)                │ (None, 512, 512, 32)      │               0 │ batch_normalization_1[0][… │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ max_pooling2d (MaxPooling2D)  │ (None, 256, 256, 32)      │               0 │ re_lu_1[0][0]              │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ conv2d_2 (Conv2D)             │ (None, 256, 256, 64)      │          18,432 │ max_pooling2d[0][0]        │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ batch_normalization_2         │ (None, 256, 256, 64)      │             256 │ conv2d_2[0][0]             │
# │ (BatchNormalization)          │                           │                 │                            │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ re_lu_2 (ReLU)                │ (None, 256, 256, 64)      │               0 │ batch_normalization_2[0][… │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ conv2d_3 (Conv2D)             │ (None, 256, 256, 64)      │          36,864 │ re_lu_2[0][0]              │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ batch_normalization_3         │ (None, 256, 256, 64)      │             256 │ conv2d_3[0][0]             │
# │ (BatchNormalization)          │                           │                 │                            │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ re_lu_3 (ReLU)                │ (None, 256, 256, 64)      │               0 │ batch_normalization_3[0][… │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ max_pooling2d_1               │ (None, 128, 128, 64)      │               0 │ re_lu_3[0][0]              │
# │ (MaxPooling2D)                │                           │                 │                            │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ conv2d_4 (Conv2D)             │ (None, 128, 128, 128)     │          73,728 │ max_pooling2d_1[0][0]      │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ batch_normalization_4         │ (None, 128, 128, 128)     │             512 │ conv2d_4[0][0]             │
# │ (BatchNormalization)          │                           │                 │                            │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ re_lu_4 (ReLU)                │ (None, 128, 128, 128)     │               0 │ batch_normalization_4[0][… │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ conv2d_5 (Conv2D)             │ (None, 128, 128, 128)     │         147,456 │ re_lu_4[0][0]              │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ batch_normalization_5         │ (None, 128, 128, 128)     │             512 │ conv2d_5[0][0]             │
# │ (BatchNormalization)          │                           │                 │                            │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ re_lu_5 (ReLU)                │ (None, 128, 128, 128)     │               0 │ batch_normalization_5[0][… │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ max_pooling2d_2               │ (None, 64, 64, 128)       │               0 │ re_lu_5[0][0]              │
# │ (MaxPooling2D)                │                           │                 │                            │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ conv2d_6 (Conv2D)             │ (None, 64, 64, 256)       │         294,912 │ max_pooling2d_2[0][0]      │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ batch_normalization_6         │ (None, 64, 64, 256)       │           1,024 │ conv2d_6[0][0]             │
# │ (BatchNormalization)          │                           │                 │                            │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ re_lu_6 (ReLU)                │ (None, 64, 64, 256)       │               0 │ batch_normalization_6[0][… │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ conv2d_7 (Conv2D)             │ (None, 64, 64, 256)       │         589,824 │ re_lu_6[0][0]              │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ batch_normalization_7         │ (None, 64, 64, 256)       │           1,024 │ conv2d_7[0][0]             │
# │ (BatchNormalization)          │                           │                 │                            │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ re_lu_7 (ReLU)                │ (None, 64, 64, 256)       │               0 │ batch_normalization_7[0][… │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ max_pooling2d_3               │ (None, 32, 32, 256)       │               0 │ re_lu_7[0][0]              │
# │ (MaxPooling2D)                │                           │                 │                            │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ global_average_pooling2d      │ (None, 256)               │               0 │ max_pooling2d_3[0][0]      │
# │ (GlobalAveragePooling2D)      │                           │                 │                            │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ global_max_pooling2d          │ (None, 256)               │               0 │ max_pooling2d_3[0][0]      │
# │ (GlobalMaxPooling2D)          │                           │                 │                            │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ concatenate (Concatenate)     │ (None, 512)               │               0 │ global_average_pooling2d[… │
# │                               │                           │                 │ global_max_pooling2d[0][0] │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ dense (Dense)                 │ (None, 512)               │         262,656 │ concatenate[0][0]          │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ dropout (Dropout)             │ (None, 512)               │               0 │ dense[0][0]                │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ dense_1 (Dense)               │ (None, 256)               │         131,328 │ dropout[0][0]              │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ dropout_1 (Dropout)           │ (None, 256)               │               0 │ dense_1[0][0]              │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ dense_2 (Dense)               │ (None, 3)                 │             771 │ dropout_1[0][0]            │
# └───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
#  Total params: 1,569,315 (5.99 MB)
#  Trainable params: 1,567,395 (5.98 MB)
#  Non-trainable params: 1,920 (7.50 KB)
# [INFO] Iniciando treino...
# Epoch 1/30
# I0000 00:00:1761431347.577929  691601 device_compiler.h:196] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 0s 165ms/step - accuracy: 0.5234 - loss: 1.11832025-10-25 19:35:36.804598: E external/local_xla/xla/stream_executor/cuda/cuda_timer.cc:86] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.
# 2025-10-25 19:35:37.019160: E external/local_xla/xla/stream_executor/cuda/cuda_timer.cc:86] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.
# 2025-10-25 19:35:38.043343: E external/local_xla/xla/stream_executor/cuda/cuda_timer.cc:86] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.
# 2025-10-25 19:35:38.315409: E external/local_xla/xla/stream_executor/cuda/cuda_timer.cc:86] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.

# Epoch 1: val_loss improved from None to 0.80909, saving model to /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/best.keras

# Epoch 1: saving model to /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/last.keras
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 452s 189ms/step - accuracy: 0.5595 - loss: 0.9155 - val_accuracy: 0.5908 - val_loss: 0.8091
# Epoch 2/30
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 0s 167ms/step - accuracy: 0.5962 - loss: 0.8170  
# Epoch 2: val_loss improved from 0.80909 to 0.78748, saving model to /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/best.keras

# Epoch 2: saving model to /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/last.keras
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 415s 186ms/step - accuracy: 0.5990 - loss: 0.8063 - val_accuracy: 0.5975 - val_loss: 0.7875
# Epoch 3/30
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 0s 167ms/step - accuracy: 0.6087 - loss: 0.7832  
# Epoch 3: val_loss improved from 0.78748 to 0.75364, saving model to /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/best.keras

# Epoch 3: saving model to /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/last.keras
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 415s 186ms/step - accuracy: 0.6093 - loss: 0.7829 - val_accuracy: 0.6044 - val_loss: 0.7536
# Epoch 4/30
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 0s 167ms/step - accuracy: 0.6174 - loss: 0.7629  
# Epoch 4: val_loss improved from 0.75364 to 0.73702, saving model to /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/best.keras

# Epoch 4: saving model to /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/last.keras
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 415s 186ms/step - accuracy: 0.6188 - loss: 0.7655 - val_accuracy: 0.6200 - val_loss: 0.7370
# Epoch 5/30
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 0s 167ms/step - accuracy: 0.6194 - loss: 0.7546  
# Epoch 5: val_loss did not improve from 0.73702

# Epoch 5: saving model to /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/last.keras
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 416s 187ms/step - accuracy: 0.6205 - loss: 0.7540 - val_accuracy: 0.6302 - val_loss: 0.7407
# Epoch 6/30
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 0s 167ms/step - accuracy: 0.6244 - loss: 0.7485  
# Epoch 6: val_loss did not improve from 0.73702

# Epoch 6: saving model to /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/last.keras
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 416s 187ms/step - accuracy: 0.6242 - loss: 0.7414 - val_accuracy: 0.6330 - val_loss: 0.7385
# Epoch 7/30
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 0s 167ms/step - accuracy: 0.6288 - loss: 0.7323  
# Epoch 7: val_loss improved from 0.73702 to 0.71008, saving model to /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/best.keras

# Epoch 7: saving model to /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/last.keras
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 416s 187ms/step - accuracy: 0.6340 - loss: 0.7279 - val_accuracy: 0.6429 - val_loss: 0.7101
# Epoch 8/30
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 0s 167ms/step - accuracy: 0.6345 - loss: 0.7215  
# Epoch 8: val_loss improved from 0.71008 to 0.69213, saving model to /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/best.keras

# Epoch 8: saving model to /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/last.keras
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 417s 187ms/step - accuracy: 0.6383 - loss: 0.7186 - val_accuracy: 0.6413 - val_loss: 0.6921
# Epoch 9/30
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 0s 167ms/step - accuracy: 0.6488 - loss: 0.7114  
# Epoch 9: val_loss did not improve from 0.69213

# Epoch 9: saving model to /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/last.keras
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 416s 186ms/step - accuracy: 0.6454 - loss: 0.7081 - val_accuracy: 0.6305 - val_loss: 0.7231
# Epoch 10/30
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 0s 167ms/step - accuracy: 0.6488 - loss: 0.7027  
# Epoch 10: val_loss did not improve from 0.69213

# Epoch 10: saving model to /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/last.keras
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 418s 187ms/step - accuracy: 0.6482 - loss: 0.7027 - val_accuracy: 0.6394 - val_loss: 0.7085
# Epoch 11/30
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 0s 167ms/step - accuracy: 0.6403 - loss: 0.7043  
# Epoch 11: val_loss did not improve from 0.69213

# Epoch 11: saving model to /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/last.keras
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 417s 187ms/step - accuracy: 0.6457 - loss: 0.7007 - val_accuracy: 0.6660 - val_loss: 0.7062
# Epoch 12/30
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 0s 167ms/step - accuracy: 0.6468 - loss: 0.6903  
# Epoch 12: val_loss did not improve from 0.69213

# Epoch 12: saving model to /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/last.keras
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 417s 187ms/step - accuracy: 0.6472 - loss: 0.6943 - val_accuracy: 0.6508 - val_loss: 0.7208
# Epoch 13/30
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 0s 167ms/step - accuracy: 0.6558 - loss: 0.6829  
# Epoch 13: val_loss did not improve from 0.69213

# Epoch 13: saving model to /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/last.keras
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 417s 187ms/step - accuracy: 0.6548 - loss: 0.6883 - val_accuracy: 0.6495 - val_loss: 0.7018
# Epoch 14/30
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 0s 167ms/step - accuracy: 0.6635 - loss: 0.6779  
# Epoch 14: val_loss did not improve from 0.69213


# Epoch 14: saving model to /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/last.keras
# Epoch 14: saving model to /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/last.keras
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 417s 187ms/step - accuracy: 0.6602 - loss: 0.6815 - val_accuracy: 0.6667 - val_loss: 0.6931
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 417s 187ms/step - accuracy: 0.6602 - loss: 0.6815 - val_accuracy: 0.6667 - val_loss: 0.6931
# Epoch 15/30
# Epoch 15/30
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 0s 167ms/step - accuracy: 0.6626 - loss: 0.6864
# Epoch 15: val_loss did not improve from 0.69213
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 0s 167ms/step - accuracy: 0.6626 - loss: 0.6864
# Epoch 15: val_loss did not improve from 0.69213

# Epoch 15: saving model to /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/last.keras
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 417s 187ms/step - accuracy: 0.6630 - loss: 0.6812 - val_accuracy: 0.6502 - val_loss: 0.7129
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 417s 187ms/step - accuracy: 0.6630 - loss: 0.6812 - val_accuracy: 0.6502 - val_loss: 0.7129
# Epoch 16/30
# Epoch 16/30
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 0s 167ms/step - accuracy: 0.6616 - loss: 0.6796
# Epoch 16: val_loss did not improve from 0.69213

# Epoch 16: saving model to /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/last.keras
# 2232/2232 ━━━━━━━━━━━━━━━━━━━━ 417s 187ms/step - accuracy: 0.6631 - loss: 0.6765 - val_accuracy: 0.6641 - val_loss: 0.6971
# Epoch 16: early stopping
# Restoring model weights from the end of the best epoch: 8.
# [INFO] Tempo total: 111.65 min
# [INFO] Avaliando no conjunto de validação...
# 394/394 ━━━━━━━━━━━━━━━━━━━━ 14s 35ms/step - accuracy: 0.6413 - loss: 0.6921
# Val loss: 0.6921 | Val acc: 0.6413
# ✅ Modelo final salvo: /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/final_model.keras
# 📄 Resultados salvos: /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240/results_20251025_212101_6318f5.json

# ✅ FIM. Artefatos em: /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251025_192240