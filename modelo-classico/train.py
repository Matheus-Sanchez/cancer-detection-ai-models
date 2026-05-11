# --- TOPO DO train.py ---
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from model_cnn import build_model, compile_model
from metrics_logs import make_basic_callbacks, save_history, finalize_and_save

def train(
    train_ds,
    val_ds,
    test_ds,             # <<< ADICIONADO
    steps_per_epoch: int,
    val_steps: int,
    test_steps: int,      # <<< ADICIONADO
    run_dir: Path,
    img_size: int,
    channels: int,
    num_classes: int,
    epochs: int = 30,
    lr: float = 3e-4,
    weight_decay: float = 3e-5,
    class_weights: Optional[Dict[int, float]] = None,
    resume_ckpt: Optional[str] = None,
    initial_epoch: int = 0,
) -> Dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # build or load model
    if resume_ckpt and Path(resume_ckpt).exists():
        print(f"[INFO] Carregando checkpoint: {resume_ckpt}")
        model = keras.models.load_model(resume_ckpt, compile=False)
    else:
        model = build_model(img_size=img_size, channels=channels, num_classes=num_classes)
    # (re)compila com config atual
    model = compile_model(model, lr=lr, weight_decay=weight_decay)
    model.summary()

    # <<< Callbacks AINDA usam val_ds (correto, para monitoramento) >>>
    callbacks = make_basic_callbacks(run_dir=run_dir, num_classes=num_classes, val_ds=val_ds)



    steps_train = int(steps_per_epoch)
    steps_val   = int(val_steps)
    steps_test  = int(test_steps)

    t0 = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_train,
        validation_steps=steps_val,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
        initial_epoch=initial_epoch
    )
    train_time = time.perf_counter() - t0
    save_history(history, run_dir)

    # salvar modelo final
    model.save(run_dir / "final_model.keras")

    # avaliação + artefatos finais
    # <<< ALTERADO: Avaliação final usa test_ds e test_steps >>>
    results = finalize_and_save(
        model=model, ds=test_ds, steps=test_steps, num_classes=num_classes,
        run_dir=run_dir, train_time_sec=train_time,
        extra_info={"epochs": int(epochs), "lr": float(lr), "weight_decay": float(weight_decay)}
    )
    return results