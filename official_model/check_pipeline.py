# official_model/check_pipeline.py
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

import tensorflow as tf
import numpy as np

# Importa SOMENTE do data_pipeline (check não monta modelo)
try:
    from .data_pipeline import build_datasets_nbm, AugmentConfig
except Exception:
    # fallback se rodar como script
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from official_model.data_pipeline import build_datasets_nbm, AugmentConfig


def _tensor_stats(x: tf.Tensor) -> str:
    """Resumo rápido de um batch de imagens."""
    return (
        f"shape={tuple(x.shape)}, dtype={x.dtype}, "
        f"min={float(tf.reduce_min(x)):.3f}, "
        f"max={float(tf.reduce_max(x)):.3f}, "
        f"mean={float(tf.reduce_mean(x)):.4f}, "
        f"std={float(tf.math.reduce_std(x)):.4f}"
    )


def _count_batch_size(xb: tf.Tensor) -> int:
    """Pega o batch_size real de um tensor (funciona mesmo com shape[0]=None)."""
    if xb.shape.rank == 0:
        return 0
    if xb.shape[0] is not None:
        return int(xb.shape[0])
    return int(tf.shape(xb)[0])


def main():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    ap = argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=1024)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)

    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--max-per-class", type=int, default=7000)

    # Augment (mantendo os nomes antigos da CLI)
    ap.add_argument("--aug-extra-fraction", type=float, default=0.50)
    ap.add_argument("--aug-brightness", type=float, default=0.03)
    ap.add_argument("--aug-contrast-low", type=float, default=0.92)
    ap.add_argument("--aug-contrast-high", type=float, default=1.08)

    # Caminhos opcionais
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--csv-path", type=str, default=None)

    ap.add_argument(
        "--strict",
        action="store_true",
        help="faz exit(1) se alguma verificação importante falhar",
    )
    args = ap.parse_args()

    # Monta o AugmentConfig a partir da CLI
    aug_cfg = AugmentConfig(
        brightness_delta=args.aug_brightness,
        contrast_lower=args.aug_contrast_low,
        contrast_upper=args.aug_contrast_high,
        # demais campos (flip, translate, zoom, noise, cutout) ficam como default
    )

    # 1) pega TUDO pronto do reader (datasets + meta + info de contagem)
    train_ds, val_ds, test_ds, meta, info = build_datasets_nbm(
        img_size=args.img_size,
        channels=args.channels,
        batch_size=args.batch_size,
        max_per_class=args.max_per_class,
        val_size=args.val_size,
        test_size=args.test_size,
        extra_fraction=args.aug_extra_fraction,
        aug_cfg=aug_cfg,
        data_dir=args.data_dir,
        csv_path=args.csv_path,
    )

    print("\n[DATA]")
    print(f"  Train / Val / Test : {meta['n_train']} / {meta['n_val']} / {meta['n_test']}")
    print(f"  Classes            : {meta['class_map']}")
    print(f"  Class weights      : {meta['class_weights']}")
    print(f"  CSV                : {meta.get('csv', '<desconhecido>')}")
    print(f"  Base dir           : {meta.get('base_dir', '<desconhecido>')}")

    print("\n[PIPELINE]")
    print(
        f"  img_size / ch / batch : {info['img_size']} / "
        f"{info['channels']} / {info['batch_size']}"
    )
    print(f"  aug summary         : {info.get('aug_summary', 'none')}")
    print(f"  extra_fraction      : {info['extra_fraction']}")
    print(
        f"  steps base / aug    : {info['steps_base']} / {info['steps_aug']}  "
        f"(prev_aug_imgs≈{info['prev_aug_imgs']})"
    )

    ok = True

    # 2) Estatística de 1 batch de TREINO (deve estar em float16, vindo da GPU pipeline)
    print("\n[CHECK: TRAIN BATCH]")
    try:
        train_x, train_y = next(iter(train_ds.take(1)))
        print("  X:", _tensor_stats(train_x))
        print("  y shape:", tuple(train_y.shape), "| uniques:", tf.sort(tf.unique(train_y).y).numpy().tolist())

        if train_x.dtype != tf.float16:
            ok = False
            print("  ! esperado dtype float16 no treino (por causa de mixed_precision/copy_to_device)")
        if train_x.shape[-1] != info["channels"]:
            ok = False
            print(f"  ! channels não bate: tensor tem {train_x.shape[-1]}, info['channels']={info['channels']}")
    except StopIteration:
        ok = False
        print("  ! train_ds vazio (nenhum batch retornado)")

    # 3) Estatística de 1 batch de VALIDAÇÃO (float32, sem aug)
    print("\n[CHECK: VAL BATCH]")
    try:
        val_x, val_y = next(iter(val_ds.take(1)))
        print("  X:", _tensor_stats(val_x))
        print("  y shape:", tuple(val_y.shape), "| uniques:", tf.sort(tf.unique(val_y).y).numpy().tolist())

        if val_x.dtype != tf.float32:
            ok = False
            print("  ! dtype esperado para VAL: float32")
        if val_x.shape[-1] != info["channels"]:
            ok = False
            print(f"  ! channels não bate em VAL: tensor tem {val_x.shape[-1]}, info['channels']={info['channels']}")
        # labels inteiros [0..2]
        uniq = tf.sort(tf.unique(val_y).y).numpy().tolist()
        if len(uniq) > 0 and (min(uniq) < 0 or max(uniq) > 2):
            ok = False
            print("  ! labels de VAL fora de [0..2]")
    except StopIteration:
        ok = False
        print("  ! val_ds vazio (nenhum batch retornado)")

    # 4) Estatística de 1 batch de TESTE (float32, sem aug)
    print("\n[CHECK: TEST BATCH]")
    try:
        test_x, test_y = next(iter(test_ds.take(1)))
        print("  X:", _tensor_stats(test_x))
        print("  y shape:", tuple(test_y.shape), "| uniques:", tf.sort(tf.unique(test_y).y).numpy().tolist())

        if test_x.dtype != tf.float32:
            ok = False
            print("  ! dtype esperado para TEST: float32")
        if test_x.shape[-1] != info["channels"]:
            ok = False
            print(
                f"  ! channels não bate em TEST: tensor tem {test_x.shape[-1]}, "
                f"info['channels']={info['channels']}"
            )
        uniq_t = tf.sort(tf.unique(test_y).y).numpy().tolist()
        if len(uniq_t) > 0 and (min(uniq_t) < 0 or max(uniq_t) > 2):
            ok = False
            print("  ! labels de TEST fora de [0..2]")
    except StopIteration:
        ok = False
        print("  ! test_ds vazio (nenhum batch retornado)")

    # 5) Contar uma “época” de treino (base + aug) usando steps calculados no reader
    print("\n[CHECK: CONTAGEM TRAIN (base + aug)]")
    counted_base, counted_aug = 0, 0
    total_steps = info["steps_base"] + info["steps_aug"]

    step_iter = train_ds.take(total_steps)
    for step, (xb, yb) in enumerate(step_iter):
        n = _count_batch_size(xb)
        if step < info["steps_base"]:
            counted_base += n
        else:
            counted_aug += n

    print(f"  base imgs (real) : {counted_base}")
    print(f"  aug  imgs (real) : {counted_aug}  (prev≈{info['prev_aug_imgs']})")

    if counted_base <= 0 or (info["steps_aug"] > 0 and counted_aug <= 0):
        ok = False
        print("  ! contagem de imagens inesperada (zero)")

    # 6) Resultado final
    print("\n[CHECK: RESULTADO GERAL]")
    if ok:
        print("  ✅ Pipeline OK (tipos, splits e contagens parecem consistentes)")
    else:
        print("  ❌ Foram encontrados problemas na checagem do pipeline.")
        if args.strict:
            sys.exit(1)


if __name__ == "__main__":
    # Exemplo:
    #   python -m official_model.check_pipeline \
    #       --img-size 1024 --channels 1 --batch-size 8 \
    #       --aug-extra-fraction 0.5 --val-size 0.15 --test-size 0.15
    main()
