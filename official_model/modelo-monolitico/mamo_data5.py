# -*- coding: utf-8 -*-
"""
CustomSimpleMammoCNN_v1

Uma CNN **simples** (camada sobre camada) para classificação 3 classes (Normal/Benign/Malignant)
em mamografias pré-processadas (grayscale), com **comentários detalhados** e salvando tudo
em `./outputs/<run>/`.

✔ Sem transfer learning
✔ Sem criar muitas funções (apenas o essencial para pipeline)
✔ Compatível com GPU de 12GB (RTX A200*)
✔ Mantém o caminho do dataset no mesmo formato usado nos seus scripts anteriores

(*) Se a sua GPU for a RTX A200 com 12GB (ou A2000 12GB), esta configuração deve caber bem.

Como rodar:
    # Opcional: aponte explicitamente o dataset (igual seus códigos anteriores)
    # export MAMMO_PATH=/caminho/para/Mammo-Bench
    # No Windows (PowerShell): $env:MAMMO_PATH="C:\\caminho\\Mammo-Bench"

    python CustomSimpleMammoCNN_v1.py

Ajustes rápidos:
- BATCH_SIZE: 8 por padrão (seguro p/ 12GB a 512x512). Se sobrar VRAM, tente 10–12.
- IMG_SIZE: 512 é um bom meio-termo. Se quiser treinos mais rápidos, 384 também funciona.

Saídas salvas em: ./outputs/<timestamp>/
- best.keras, last.keras, history.json, history.csv
- plots: acc.png, loss.png, cm.png
- logs: training.log
"""

# ============================
# Imports e configuração básica
# ============================
import os, sys, json, time, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

# Reduz verbosidade do TensorFlow **antes** do import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"        # 0=all, 1=info, 2=warning, 3=error
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"       # estabilidade numérica

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

# ============================
# Seeds e GPU (12GB VRAM)
# ============================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Habilita memory growth: impede o TF de alocar toda VRAM de uma vez
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print("[WARN] Memory growth não pôde ser habilitado:", e)

# ============================
# Caminhos do dataset e saída
# ============================
# Mantém a mesma lógica dos códigos anteriores: usa variável de ambiente MAMMO_PATH
# e, se não existir, usa um caminho padrão relativo ao script (ajuste se necessário).
BASE_DIR = Path(os.environ.get("MAMMO_PATH", Path(__file__).resolve().parent / "dataset" / "Mammo_Data" / "Mammo-Bench")).resolve()
CSV_FILE = BASE_DIR / "CSV_Files" / "mammo-bench_nbm_classification.csv"
IMG_ROOT = BASE_DIR / "Preprocessed_Dataset"

assert CSV_FILE.exists(), f"CSV não encontrado: {CSV_FILE}"

# Pasta de outputs no mesmo diretório do script
OUT_ROOT = Path(__file__).resolve().parent / "./outputs"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_NAME = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
RUN_DIR = OUT_ROOT / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Salvando saídas em: {RUN_DIR}")

# ============================
# Hiperparâmetros principais (simples)
# ============================
IMG_SIZE = 1024         # 512x512 grayscale
CHANNELS = 1            # imagens pré-processadas em escala de cinza
NUM_CLASSES = 3         # Normal/Benign/Malignant
BATCH_SIZE = 8          # seguro p/ 12GB
EPOCHS = 30             # early stopping vai parar antes se melhorar rápido
VAL_SIZE = 0.15         # 15% para validação

# Aumentações **simples** (manter coerência com seus treinos anteriores)
AUG_BRIGHTNESS = 0.03   # variação de brilho
AUG_CONTRAST_L = 0.92   # contraste mínimo
AUG_CONTRAST_U = 1.08   # contraste máximo

# ============================
# Carrega CSV e prepara split
# ============================
df = pd.read_csv(CSV_FILE)

# Mapeia rótulos de string para inteiros (0..2), igual aos seus scripts
label_col = "classification"
label_map = {"Normal":0, "Benign":1, "Malignant":2}
assert label_col in df.columns, f"Coluna '{label_col}' não encontrada no CSV. Colunas: {df.columns.tolist()}"

df = df[df[label_col].isin(label_map)].copy()
df['label'] = df[label_col].map(label_map)

# Caminho absoluto das imagens pré-processadas (coluna padrão dos seus scripts)
img_col = "preprocessed_image_path"
assert img_col in df.columns, f"Coluna '{img_col}' não encontrada no CSV. Colunas: {df.columns.tolist()}"

df['img_path'] = df[img_col].apply(lambda p: str((BASE_DIR / p).resolve()))

# Split estratificado (reprodutível)
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(
    df, test_size=VAL_SIZE, stratify=df['label'], random_state=SEED
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
print(f"[INFO] Train: {len(train_df)}, Val: {len(val_df)}")

# Class weights para compensar eventual desbalanceamento
from sklearn.utils.class_weight import compute_class_weight
classes = np.array(sorted(train_df['label'].unique()))
class_weights_arr = compute_class_weight(class_weight='balanced', classes=classes, y=train_df['label'].values)
CLASS_WEIGHTS = {int(c): float(w) for c,w in zip(classes, class_weights_arr)}
print("[INFO] Class weights:", CLASS_WEIGHTS)

# ============================
# tf.data: loader + aumentação simples
# ============================
@tf.function
def _decode_resize(path, label):
    """Lê PNG em escala de cinza, converte para float32 [0,1] e redimensiona para IMG_SIZE."""
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=CHANNELS)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], antialias=True)
    return img, label

@tf.function
def _augment(img, label):
    """Aumentações **simples** para robustez: flip horizontal, brilho e contraste."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=AUG_BRIGHTNESS)
    img = tf.image.random_contrast(img, lower=AUG_CONTRAST_L, upper=AUG_CONTRAST_U)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label

# Constrói datasets de treino/validação (mínimo de funções)
train_ds = tf.data.Dataset.from_tensor_slices((train_df['img_path'].values, train_df['label'].astype(np.int32).values))
train_ds = train_ds.shuffle(buffer_size=len(train_df), seed=SEED)
train_ds = train_ds.map(_decode_resize, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_df['img_path'].values, val_df['label'].astype(np.int32).values))
val_ds = val_ds.map(_decode_resize, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

# ============================
# Modelo **simples** (camada sobre camada)
# ============================
# Estratégia: blocos Conv2D -> BN -> ReLU, com pooling ocasional, terminando em GAP + Dense.
# Usamos filtros moderados para caber bem na VRAM e evitar overfitting em excesso.

inputs = L.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS))
x = inputs

# Bloco 1
x = L.Conv2D(32, 3, padding='same', use_bias=False)(x)  # conv leve
x = L.BatchNormalization()(x)
x = L.ReLU()(x)
x = L.Conv2D(32, 3, padding='same', use_bias=False)(x)  # empilha mais uma conv
x = L.BatchNormalization()(x)
x = L.ReLU()(x)
x = L.MaxPooling2D(2)(x)  # reduz metade (512->256)

# Bloco 2
x = L.Conv2D(64, 3, padding='same', use_bias=False)(x)
x = L.BatchNormalization()(x)
x = L.ReLU()(x)
x = L.Conv2D(64, 3, padding='same', use_bias=False)(x)
x = L.BatchNormalization()(x)
x = L.ReLU()(x)
x = L.MaxPooling2D(2)(x)  # 256->128

# Bloco 3
x = L.Conv2D(128, 3, padding='same', use_bias=False)(x)
x = L.BatchNormalization()(x)
x = L.ReLU()(x)
x = L.Conv2D(128, 3, padding='same', use_bias=False)(x)
x = L.BatchNormalization()(x)
x = L.ReLU()(x)
x = L.MaxPooling2D(2)(x)  # 128->64

# Bloco 4
x = L.Conv2D(256, 3, padding='same', use_bias=False)(x)
x = L.BatchNormalization()(x)
x = L.ReLU()(x)
x = L.Conv2D(256, 3, padding='same', use_bias=False)(x)
x = L.BatchNormalization()(x)
x = L.ReLU()(x)
x = L.MaxPooling2D(2)(x)  # 64->32

# Cabeça final (Global Average Pooling reduz muito os params)
x = L.GlobalAveragePooling2D()(x)
x = L.Dropout(0.3)(x)               # regularização simples
x = L.Dense(256, activation='relu')(x)
x = L.Dropout(0.3)(x)
outputs = L.Dense(NUM_CLASSES, activation='softmax')(x)

model = keras.Model(inputs, outputs, name='CustomSimpleMammoCNN')
model.summary()

# ============================
# Compilação e callbacks
# ============================
# AdamW costuma funcionar bem; LR 3e-4 é um bom ponto de partida.
optimizer = keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-5)
loss = keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Callbacks para salvar tudo em ./outputs/<run>/
ckpt_best = keras.callbacks.ModelCheckpoint(
    filepath=str(RUN_DIR / 'best.keras'), monitor='val_accuracy', mode='max',
    save_best_only=True, verbose=1
)
ckpt_last = keras.callbacks.ModelCheckpoint(
    filepath=str(RUN_DIR / 'last.keras'), save_best_only=False, verbose=1
)
es = keras.callbacks.EarlyStopping(
    monitor='val_accuracy', mode='max', patience=8, restore_best_weights=True, verbose=1
)
csvlog = keras.callbacks.CSVLogger(str(RUN_DIR / 'history.csv'))

# ============================
# Treino
# ============================
print("[INFO] Iniciando treino...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[ckpt_best, ckpt_last, es, csvlog],
    class_weight=CLASS_WEIGHTS,
    verbose=1
)

# Salva histórico
hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(RUN_DIR / 'history.json', 'w') as f:
    json.dump(hist_dict, f, indent=2)

# ============================
# Avaliação, plots e métricas básicas
# ============================
print("[INFO] Avaliando no conjunto de validação...")
val_loss, val_acc = model.evaluate(val_ds, verbose=1)
print(f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

# Plots simples (sem estilos)
import matplotlib.pyplot as plt

def _plot_curve(values, title, ylabel, outfile):
    plt.figure()
    plt.plot(values)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()

_plot_curve(hist_dict.get('accuracy', []), 'Treino - Accuracy', 'Accuracy', RUN_DIR / 'acc.png')
_plot_curve(hist_dict.get('val_accuracy', []), 'Validação - Accuracy', 'Accuracy', RUN_DIR / 'val_acc.png')
_plot_curve(hist_dict.get('loss', []), 'Treino - Loss', 'Loss', RUN_DIR / 'loss.png')
_plot_curve(hist_dict.get('val_loss', []), 'Validação - Loss', 'Loss', RUN_DIR / 'val_loss.png')

# Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Coleta rótulos e predições da validação
y_true = []
y_pred = []
for xb, yb in val_ds:
    pb = model.predict(xb, verbose=1)
    y_true.append(yb.numpy())
    y_pred.append(np.argmax(pb, axis=1))
y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
cmd = ConfusionMatrixDisplay(cm, display_labels=['Normal','Benign','Malignant'])
plt.figure()
cmd.plot(values_format='d')
plt.title('Matriz de Confusão - Validação')
plt.savefig(RUN_DIR / 'cm.png', bbox_inches='tight')
plt.close()

print("[INFO] Finalizado. Artefatos salvos em:", RUN_DIR)



