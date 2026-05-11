from __future__ import annotations
import time, datetime as dt
from pathlib import Path
from typing import Optional, Dict

import numpy as np
from tensorflow import keras

from .model_cnn import build_model, compile_model
from .metrics_logs import make_basic_callbacks, save_history, finalize_and_save

def train(
    train_ds,
    val_ds,
    steps_per_epoch: int,
    val_steps: int,
    run_dir: Path,
    img_size: int,
    channels: int,
    num_classes: int,
    epochs: int = 30,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    class_weights: Optional[Dict[int, float]] = None,
) -> Dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(img_size=img_size, channels=channels, num_classes=num_classes)
    model = compile_model(model, lr=lr, weight_decay=weight_decay)
    model.summary()

    callbacks = make_basic_callbacks(run_dir=run_dir, num_classes=num_classes, val_ds=val_ds)

    t0 = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    train_time = time.perf_counter() - t0
    save_history(history, run_dir)

    # salvar modelo final
    model.save(run_dir / "final_model.keras")

    # avaliação + artefatos finais
    results = finalize_and_save(
        model=model, val_ds=val_ds, val_steps=val_steps, num_classes=num_classes,
        run_dir=run_dir, train_time_sec=train_time,
        extra_info={"epochs": int(epochs), "lr": float(lr), "weight_decay": float(weight_decay)}
    )
    return results