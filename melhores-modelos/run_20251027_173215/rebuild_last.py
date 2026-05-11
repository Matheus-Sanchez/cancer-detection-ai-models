#!/usr/bin/env python3
# Auto-generated: exact rebuild helper for "last"
# Usage:
#   python rebuild_last.py --weights /path/to/last.keras  (opcional, para carregar pesos)
#   python rebuild_last.py --print-summary
import json, argparse, os
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow import keras

# --- TFA opcional ---
custom_objects = {}
try:
    import tensorflow_addons as tfa
    custom_objects.update({
        "GroupNormalization": tfa.layers.GroupNormalization,
        "InstanceNormalization": getattr(tfa.layers, "InstanceNormalization", None),
        "GELU": getattr(tfa.activations, "gelu", None),
    })
    custom_objects = {k: v for k, v in custom_objects.items() if v is not None}
except Exception:
    pass

# --- Compat Layer ---
def load_model_compat(path, custom_objects=None):
    try:
        import keras as k
        if hasattr(k, "saving"):
            return k.saving.load_model(path, custom_objects=custom_objects)
    except Exception:
        pass
    from tensorflow.keras.models import load_model as tf_load_model
    return tf_load_model(path, custom_objects=custom_objects)

def serialize_compat(obj):
    try:
        import keras as k
        if hasattr(k, "saving"):
            return k.saving.serialize_keras_object(obj)
    except Exception:
        pass
    try:
        from tensorflow.keras.utils import serialize_keras_object
        return serialize_keras_object(obj)
    except Exception:
        try:
            return obj.get_config()
        except Exception:
            return {"__unserializable__": str(obj), "type": type(obj).__name__}

def deserialize_compat(payload):
    try:
        import keras as k
        if hasattr(k, "saving"):
            return k.saving.deserialize_keras_object(payload)
    except Exception:
        pass
    try:
        from tensorflow.keras.utils import deserialize_keras_object
        return deserialize_keras_object(payload)
    except Exception:
        return payload
# ----------------------------------------

def load_json(name):
    with open(name, "r", encoding="utf-8") as f:
        return json.load(f)

def build_model_from_config():
    cfg = load_json("last_architecture.json")
    # Observação: Model.from_config existe tanto em Keras 3 quanto em tf.keras
    model = keras.Model.from_config(cfg, custom_objects=custom_objects)
    return model

def maybe_compile(model):
    try:
        comp = load_json("last_compile.json")

        # Optimizer
        opt = None
        if comp.get("optimizer") is not None:
            try:
                opt = deserialize_compat(comp["optimizer"])
            except Exception:
                opt = None

        # Loss (string ou objeto serializado)
        loss = None
        if comp.get("loss") is not None:
            try:
                loss = deserialize_compat(comp["loss"])
            except Exception:
                loss = comp["loss"]

        # Métricas
        mets = []
        for m in comp.get("metrics", []) or []:
            try:
                mets.append(deserialize_compat(m))
            except Exception:
                mets.append(m)

        # Métricas ponderadas
        wmets = []
        for m in comp.get("weighted_metrics", []) or []:
            try:
                wmets.append(deserialize_compat(m))
            except Exception:
                wmets.append(m)

        if opt is not None or loss is not None or mets or wmets:
            model.compile(optimizer=opt, loss=loss, metrics=mets, weighted_metrics=wmets)
    except FileNotFoundError:
        pass
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default=None, help="Path to last.keras (optional)")
    ap.add_argument("--print-summary", dest="print_summary", action="store_true")
    args = ap.parse_args()

    model = build_model_from_config()
    model = maybe_compile(model)

    if args.weights:
        loaded = load_model_compat(args.weights, custom_objects=custom_objects)
        model.set_weights(loaded.get_weights())

    if args.print_summary:
        model.summary()

    # Totais de parâmetros (útil para sanity-check)
    trainable_count = sum(v.numpy().size for v in model.trainable_variables)
    non_trainable_count = sum(v.numpy().size for v in model.non_trainable_variables)
    total_params = trainable_count + non_trainable_count
    print(f"Model name: {model.name} | total_params={total_params} | trainable={trainable_count} | non_trainable={non_trainable_count}")

if __name__ == "__main__":
    main()
