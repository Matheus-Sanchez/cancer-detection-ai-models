from __future__ import annotations
from pyexpat import model
from rich import inspect
from sklearn import metrics
from sklearn.metrics import auc
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
from tensorflow import keras
from tensorflow.keras import layers as L
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')

def build_model(img_size: int, channels: int, num_classes: int) -> keras.Model:
    """
    CNN simples e estável: (Conv->BN->ReLU)x2 + MaxPool por bloco; cabeça GAP+GMP+densas.
    """
    inputs = L.Input(shape=(img_size, img_size, channels))
    x = inputs


# SeparableConv2D(32, 5, strides = 1,  padding="same", use_bias=False,
#                 depthwise_initializer="he_normal", pointwise_initializer="he_normal")(x)
# batch_normalization
# ReLU
# eparableConv2D(32, 5, strides = 1,  padding="same", use_bias=False,
#                 depthwise_initializer="he_normal", pointwise_initializer="he_normal")(x)
# batch_normalization
# ReLU
# max_pooling2


    for i, f in enumerate([32, 48, 64, 96, 128,]):
        x = L.SeparableConv2D(f, 5, strides=(2 if i == 0 else 1), padding="same", use_bias=False,
                depthwise_initializer="he_normal", pointwise_initializer="he_normal")(x)
        # x = L.SeparableConv2D(f, 5, strides = 1,  padding="same", use_bias=False,
        #         depthwise_initializer="he_normal", pointwise_initializer="he_normal")(x)
        # x = L.BatchNormalization()(x); 
        x = L.GroupNormalization(groups=16)(x);
        x = L.Activation("swish")(x)
        # x = L.ReLU()(x)
        # x = L.SpatialDropout2D(0.10)(x)
        # x = L.SeparableConv2D(f, 3, strides=(1), padding="same", dilation_rate=(1 if i == 0 else (2 if i == 1 else 3)),
        #               use_bias=False, depthwise_initializer="he_normal", pointwise_initializer="he_normal")(x)
        x = L.SeparableConv2D(f, 3, strides = 1,  padding="same", use_bias=False,
                depthwise_initializer="he_normal", pointwise_initializer="he_normal")(x)
        # x = L.BatchNormalization()(x); 
        x = L.GroupNormalization(groups=16)(x);
        x = L.Activation("swish")(x)
        # x = L.ReLU()(x)
        # x = L.SpatialDropout2D(0.10)(x)
        x = L.MaxPooling2D(2)(x)

    gap = L.GlobalAveragePooling2D()(x)
    gmp = L.GlobalMaxPooling2D()(x)
    x = L.Concatenate()([gap, gmp])

    # x = L.Flatten()(x)
    x = L.Dropout(0.4)(x)
    x = L.Dense(256, activation="silu")(x)
    x = L.Dropout(0.4)(x)
    x = L.Dense(256, activation="silu")(x)
    x = L.Dropout(0.4)(x)

    outputs = L.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return keras.Model(inputs, outputs, name="CustomSimpleMammoCNN")


# =========================
# Keras Applications (modelo pronto) mantendo seu pipeline (imagens em [0,1])
# =========================
def _resolve_keras_backbone(backbone: str):
    b = (backbone or "").lower().strip()

    if b in ("efficientnetv2b0", "effnetv2b0"):
        from tensorflow.keras.applications import EfficientNetV2B0
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
        return EfficientNetV2B0, preprocess_input

    if b in ("resnet50",):
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.applications.resnet import preprocess_input
        return ResNet50, preprocess_input

    if b in ("mobilenetv2", "mobilenet_v2"):
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        return MobileNetV2, preprocess_input

    if b in ("densenet121",):
        from tensorflow.keras.applications import DenseNet121
        from tensorflow.keras.applications.densenet import preprocess_input
        return DenseNet121, preprocess_input

    raise ValueError(
        f"Backbone '{backbone}' não suportado. Use: "
        "efficientnetv2b0 | resnet50 | mobilenetv2 | densenet121"
    )


def build_keras_app_model(
    img_size: int,
    channels: int,
    num_classes: int,
    backbone: str = "efficientnetv2b0",
    weights: str | None = "imagenet",
    trainable_backbone: bool = False,
    dropout: float = 0.3,
    head_units: int = 256,
) -> keras.Model:
    """
    Modelo pronto (Keras Applications) SEM mexer no seu dataset:
      - dataset continua grayscale e normalizado em [0,1]
      - aqui dentro: converte 1->3 canais + aplica preprocess_input correto
    """
    BackboneCls, preprocess_input = _resolve_keras_backbone(backbone)

    inputs = L.Input(shape=(img_size, img_size, channels), name="input")
    x = inputs

    # 1) garantir RGB (modelos prontos esperam 3 canais)
    if channels == 1:
        x = L.Lambda(lambda t: tf.image.grayscale_to_rgb(t), name="to_rgb")(x)
    elif channels != 3:
        # fallback: projeta para 3 canais sem alterar o pipeline
        x = L.Conv2D(3, kernel_size=1, padding="same", name="to_rgb_conv")(x)

    # 2) manter pipeline [0,1], mas adaptar para o preprocess do ImageNet
    #    Muitos preprocess_input assumem [0,255], então escalamos aqui.
    if weights is not None:
        x = L.Lambda(lambda t: preprocess_input(t * 255.0), name="preprocess")(x)
    else:
        x = L.Lambda(lambda t: tf.cast(t, tf.float32), name="cast_float32")(x)

    # 3) backbone
    import inspect as _inspect
    base_kwargs = {}
    # EfficientNetV2 (e alguns outros) podem ter preprocess interno; desligamos para não duplicar.
    if "include_preprocessing" in _inspect.signature(BackboneCls).parameters:
        base_kwargs["include_preprocessing"] = False

    base = BackboneCls(
        include_top=False,
        weights=weights,
        input_shape=(img_size, img_size, 3),
        **base_kwargs,
    )
    base.trainable = bool(trainable_backbone)

    feat = base(x)
    feat = L.GlobalAveragePooling2D(name="gap")(feat)

    # 4) cabeça de classificação
    feat = L.Dropout(dropout, name="drop1")(feat)
    feat = L.Dense(head_units, activation="swish", name="head_dense")(feat)
    feat = L.Dropout(dropout, name="drop2")(feat)

    outputs = L.Dense(num_classes, activation="softmax", dtype="float32", name="pred")(feat)
    name = f"{backbone}_{'imagenet' if weights else 'scratch'}"
    return keras.Model(inputs, outputs, name=name)

# AUC macro (OvR) para multi-classe (labels inteiros)
class MacroAUC(tf.keras.metrics.Metric):
    """
    AUC macro (OvR) para multi-classe usando rótulos inteiros.
    Calcula AUC por classe em one-vs-rest e faz a média.
    """
    def __init__(self, num_classes: int, name="auc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self._aucs = [tf.keras.metrics.AUC(num_thresholds=1000, name=f"auc_{k}")
                      for k in range(self.num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=self.num_classes)
        for k in range(self.num_classes):
            self._aucs[k].update_state(y_true_oh[:, k], y_pred[:, k], sample_weight)

    def result(self):
        vals = [m.result() for m in self._aucs]
        return tf.reduce_mean(tf.stack(vals))

    def reset_states(self):
        for m in self._aucs:
            m.reset_states()

def make_sparse_ce_with_label_smoothing(num_classes: int, label_smoothing: float = 0.1):
    """
    Implementa label smoothing para rótulo inteiro:
    converte y_true -> one-hot e aplica CategoricalCrossentropy(label_smoothing).
    Compatível com versões do Keras que não aceitam 'label_smoothing' em SparseCE.
    """
    ce = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    num_classes = int(num_classes)
    def _loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=num_classes)
        return ce(y_true_oh, y_pred)
    return _loss

def compile_model(model: keras.Model, lr=3e-4, weight_decay=1e-5):
    opt = keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    num_classes = model.output_shape[-1]
    # Loss esparsa COM smoothing (compatível)
    loss = make_sparse_ce_with_label_smoothing(num_classes=num_classes, label_smoothing=0.05)    # AUC macro OvR com rótulos inteiros
    # AUC macro OvR com rótulos inteiros
    auc_macro = MacroAUC(num_classes=num_classes, name="auc")
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy", auc_macro], jit_compile=False, steps_per_execution=64)
    return model
