#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Introspect & Rebuild Keras Models (.keras)
------------------------------------------
Usage:
  python introspect_and_rebuild_keras.py --dir /path/to/run_dir

Gera, para cada {best,last}.keras encontrado:
  <tag>_summary.txt
  <tag>_architecture.json
  <tag>_optimizer.json
  <tag>_compile.json
  rebuild_<tag>.py
  <tag>_sanity.txt
"""
import json, argparse, sys, os
from pathlib import Path

# Silenciar logs ruidosos do TF
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# --- Imports base ---
import tensorflow as tf
from tensorflow import keras

# --- TFA opcional (GroupNormalization etc.) ---
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

# --- KERAS/TENSORFLOW COMPAT LAYER ------------------------------------------
# Usa keras.saving.* (Keras 3) se existir; senão, cai para tf.keras.* (TF 2.x)
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
    # Serializa optimizer/loss/metric (Keras 3 ou tf.keras). Se não der, tenta get_config ou string.
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
    # Reconstrói um objeto serializado (otimizador, loss, métrica)
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
        # Quando a loss vier como string (ex.: "categorical_crossentropy") já é usável
        return payload
# ---------------------------------------------------------------------------

def dump_summary(model, out_txt):
    with open(out_txt, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda s: f.write(s + "\n"))

def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def build_rebuild_script(target_py, arch_json_name, compile_json_name, tag):
    # Gera um script de reconstrução com a mesma compat layer embutida
    code = f'''#!/usr/bin/env python3
# Auto-generated: exact rebuild helper for "{tag}"
# Usage:
#   python rebuild_{tag}.py --weights /path/to/{tag}.keras  (opcional, para carregar pesos)
#   python rebuild_{tag}.py --print-summary
import json, argparse, os
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow import keras

# --- TFA opcional ---
custom_objects = {{}}
try:
    import tensorflow_addons as tfa
    custom_objects.update({{
        "GroupNormalization": tfa.layers.GroupNormalization,
        "InstanceNormalization": getattr(tfa.layers, "InstanceNormalization", None),
        "GELU": getattr(tfa.activations, "gelu", None),
    }})
    custom_objects = {{k: v for k, v in custom_objects.items() if v is not None}}
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
            return {{"__unserializable__": str(obj), "type": type(obj).__name__}}

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
    cfg = load_json("{arch_json_name}")
    # Observação: Model.from_config existe tanto em Keras 3 quanto em tf.keras
    model = keras.Model.from_config(cfg, custom_objects=custom_objects)
    return model

def maybe_compile(model):
    try:
        comp = load_json("{compile_json_name}")

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
    ap.add_argument("--weights", type=str, default=None, help="Path to {tag}.keras (optional)")
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
    print(f"Model name: {{model.name}} | total_params={{total_params}} | trainable={{trainable_count}} | non_trainable={{non_trainable_count}}")

if __name__ == "__main__":
    main()
'''
    with open(target_py, "w", encoding="utf-8") as f:
        f.write(code)

def process_one(model_path: Path):
    tag = model_path.stem  # "best" ou "last"
    print(f"[INFO] Loading: {model_path}")
    model = load_model_compat(str(model_path), custom_objects=custom_objects)
    out_prefix = model_path.parent / tag

    # 1) Summary legível
    dump_summary(model, f"{out_prefix}_summary.txt")

    # 2) Arquitetura exata (topologia)
    arch = model.get_config()
    write_json(f"{out_prefix}_architecture.json", arch)

    # 3) Infos de compile (optimizer/loss/metrics)
    compile_info = {}
    try:
        opt = getattr(model, "optimizer", None)
        compile_info["optimizer"] = serialize_compat(opt) if opt is not None else None
    except Exception:
        compile_info["optimizer"] = None

    try:
        compile_info["loss"] = serialize_compat(getattr(model, "loss", None))
    except Exception:
        compile_info["loss"] = None

    try:
        compile_info["metrics"] = [serialize_compat(m) for m in getattr(model, "metrics", []) if m is not None]
    except Exception:
        compile_info["metrics"] = []

    try:
        compile_info["weighted_metrics"] = [serialize_compat(m) for m in getattr(model, "weighted_metrics", []) if m is not None]
    except Exception:
        compile_info["weighted_metrics"] = []

    write_json(f"{out_prefix}_compile.json", compile_info)

    # 4) Optimizer.get_config (detalhado) quando existir
    opt_cfg = {}
    try:
        if getattr(model, "optimizer", None) is not None:
            opt_cfg = model.optimizer.get_config()
    except Exception:
        pass
    write_json(f"{out_prefix}_optimizer.json", opt_cfg or {"optimizer": None})

    # 5) Script de reconstrução
    rebuild_py = model_path.parent / f"rebuild_{tag}.py"
    build_rebuild_script(
        rebuild_py,
        arch_json_name=f"{tag}_architecture.json",
        compile_json_name=f"{tag}_compile.json",
        tag=tag
    )

    # 6) Sanidade: totais de params
    trainable_count = sum(v.numpy().size for v in model.trainable_variables)
    non_trainable_count = sum(v.numpy().size for v in model.non_trainable_variables)
    total_params = trainable_count + non_trainable_count
    with open(f"{out_prefix}_sanity.txt", "w", encoding="utf-8") as f:
        f.write(f"model_name={model.name}\n")
        f.write(f"total_params={total_params}\n")
        f.write(f"trainable_params={trainable_count}\n")
        f.write(f"non_trainable_params={non_trainable_count}\n")

    print(f"[OK] Wrote: {out_prefix}_summary.txt, {out_prefix}_architecture.json, {out_prefix}_optimizer.json, {out_prefix}_compile.json, rebuild_{tag}.py, {out_prefix}_sanity.txt")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, required=True, help="Directory that contains best.keras and/or last.keras")
    args = ap.parse_args()

    run_dir = Path(args.dir).expanduser().resolve()
    if not run_dir.exists():
        print(f"[ERR] Directory not found: {run_dir}", file=sys.stderr)
        sys.exit(2)

    any_found = False
    for fname in ("best.keras", "last.keras"):
        p = run_dir / fname
        if p.exists():
            any_found = True
            process_one(p)
        else:
            print(f"[WARN] Not found: {p}")

    if not any_found:
        print("[ERR] No .keras files found (expected best.keras and/or last.keras).", file=sys.stderr)
        sys.exit(3)

if __name__ == "__main__":
    main()
