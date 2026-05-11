[DATA] Dist. por label: {0: 28896, 2: 7244, 1: 7050}
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1763332673.117504   65934 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 9712 MB memory:  -> device: 0, name: NVIDIA RTX A2000 12GB, pci bus id: 0000:01:00.0, compute capability: 8.6

[DATA]
  Train / Val / Test : 14700 / 3150 / 3150
  Classes            : {'Normal': 0, 'Benign': 1, 'Malignant': 2}
  Class weights      : {0: 1.0, 1: 1.0, 2: 1.0}
  CSV                : /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/dataset/Mammo_Data/Mammo-Bench/CSV_Files/mammo-bench_nbm_classification.csv
  Base dir           : /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/dataset/Mammo_Data/Mammo-Bench

[PIPELINE]
  img_size / ch / batch : 1024 / 1 / 8
  aug summary         : flip LR + translate ±5% + zoom [0.96,1.04] + brightness ±0.03 + contrast [0.92,1.08] + noise σ=0.005 + cutout p=0.25 a≤8%
  extra_fraction      : 2.0
  steps base / aug    : 1838 / 3675  (prev_aug_imgs≈29400)


         [[{{node IteratorGetNext}}]]
  X: shape=(8, 1024, 1024, 1), dtype=<dtype: 'float16'>, min=0.000, max=1.000, mean=0.2595, std=0.2448
  y shape: (8,) | uniques: [0, 1, 2]

[CHECK: VAL BATCH]
  X: shape=(8, 1024, 1024, 1), dtype=<dtype: 'float32'>, min=0.000, max=1.000, mean=0.2346, std=0.2034
  y shape: (8,) | uniques: [1, 2]

[CHECK: TEST BATCH]
  X: shape=(8, 1024, 1024, 1), dtype=<dtype: 'float32'>, min=0.000, max=1.000, mean=0.2143, std=0.2014
  y shape: (8,) | uniques: [0, 1, 2]

[CHECK: CONTAGEM TRAIN (base + aug)]

  base imgs (real) : 14700
  aug  imgs (real) : 29400  (prev≈29400)








  [MODEL] Carregando modelo salvo de: /mnt/c/Users/matheus.sduda/source/repos/mamo_cnn/outputs/run_20251115_194832/best.keras
Model: "CustomSimpleMammoCNN"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                      ┃ Output Shape                 ┃           Param # ┃ Connected to                  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)          │ (None, 1024, 1024, 1)        │                 0 │ -                             │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ separable_conv2d                  │ (None, 1024, 1024, 32)       │                57 │ input_layer[0][0]             │
│ (SeparableConv2D)                 │                              │                   │                               │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ group_normalization               │ (None, 1024, 1024, 32)       │                64 │ separable_conv2d[0][0]        │
│ (GroupNormalization)              │                              │                   │                               │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ re_lu (ReLU)                      │ (None, 1024, 1024, 32)       │                 0 │ group_normalization[0][0]     │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ spatial_dropout2d                 │ (None, 1024, 1024, 32)       │                 0 │ re_lu[0][0]                   │
│ (SpatialDropout2D)                │                              │                   │                               │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ separable_conv2d_1                │ (None, 1024, 1024, 32)       │             1,312 │ spatial_dropout2d[0][0]       │
│ (SeparableConv2D)                 │                              │                   │                               │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ group_normalization_1             │ (None, 1024, 1024, 32)       │                64 │ separable_conv2d_1[0][0]      │
│ (GroupNormalization)              │                              │                   │                               │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ re_lu_1 (ReLU)                    │ (None, 1024, 1024, 32)       │                 0 │ group_normalization_1[0][0]   │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ spatial_dropout2d_1               │ (None, 1024, 1024, 32)       │                 0 │ re_lu_1[0][0]                 │
│ (SpatialDropout2D)                │                              │                   │                               │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ max_pooling2d (MaxPooling2D)      │ (None, 512, 512, 32)         │                 0 │ spatial_dropout2d_1[0][0]     │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ separable_conv2d_2                │ (None, 512, 512, 64)         │             2,848 │ max_pooling2d[0][0]           │
│ (SeparableConv2D)                 │                              │                   │                               │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ group_normalization_2             │ (None, 512, 512, 64)         │               128 │ separable_conv2d_2[0][0]      │
│ (GroupNormalization)              │                              │                   │                               │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ re_lu_2 (ReLU)                    │ (None, 512, 512, 64)         │                 0 │ group_normalization_2[0][0]   │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ spatial_dropout2d_2               │ (None, 512, 512, 64)         │                 0 │ re_lu_2[0][0]                 │
│ (SpatialDropout2D)                │                              │                   │                               │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ separable_conv2d_3                │ (None, 512, 512, 64)         │             4,672 │ spatial_dropout2d_2[0][0]     │
│ (SeparableConv2D)                 │                              │                   │                               │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ group_normalization_3             │ (None, 512, 512, 64)         │               128 │ separable_conv2d_3[0][0]      │
│ (GroupNormalization)              │                              │                   │                               │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ re_lu_3 (ReLU)                    │ (None, 512, 512, 64)         │                 0 │ group_normalization_3[0][0]   │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ spatial_dropout2d_3               │ (None, 512, 512, 64)         │                 0 │ re_lu_3[0][0]                 │
│ (SpatialDropout2D)                │                              │                   │                               │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ max_pooling2d_1 (MaxPooling2D)    │ (None, 256, 256, 64)         │                 0 │ spatial_dropout2d_3[0][0]     │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ separable_conv2d_4                │ (None, 256, 256, 128)        │             9,792 │ max_pooling2d_1[0][0]         │
│ (SeparableConv2D)                 │                              │                   │                               │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ group_normalization_4             │ (None, 256, 256, 128)        │               256 │ separable_conv2d_4[0][0]      │
│ (GroupNormalization)              │                              │                   │                               │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ re_lu_4 (ReLU)                    │ (None, 256, 256, 128)        │                 0 │ group_normalization_4[0][0]   │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ spatial_dropout2d_4               │ (None, 256, 256, 128)        │                 0 │ re_lu_4[0][0]                 │
│ (SpatialDropout2D)                │                              │                   │                               │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ separable_conv2d_5                │ (None, 256, 256, 128)        │            17,536 │ spatial_dropout2d_4[0][0]     │
│ (SeparableConv2D)                 │                              │                   │                               │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ group_normalization_5             │ (None, 256, 256, 128)        │               256 │ separable_conv2d_5[0][0]      │
│ (GroupNormalization)              │                              │                   │                               │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ re_lu_5 (ReLU)                    │ (None, 256, 256, 128)        │                 0 │ group_normalization_5[0][0]   │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ spatial_dropout2d_5               │ (None, 256, 256, 128)        │                 0 │ re_lu_5[0][0]                 │
│ (SpatialDropout2D)                │                              │                   │                               │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ max_pooling2d_2 (MaxPooling2D)    │ (None, 128, 128, 128)        │                 0 │ spatial_dropout2d_5[0][0]     │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ global_average_pooling2d          │ (None, 128)                  │                 0 │ max_pooling2d_2[0][0]         │
│ (GlobalAveragePooling2D)          │                              │                   │                               │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ global_max_pooling2d              │ (None, 128)                  │                 0 │ max_pooling2d_2[0][0]         │
│ (GlobalMaxPooling2D)              │                              │                   │                               │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ concatenate (Concatenate)         │ (None, 256)                  │                 0 │ global_average_pooling2d[0][… │
│                                   │                              │                   │ global_max_pooling2d[0][0]    │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ dropout (Dropout)                 │ (None, 256)                  │                 0 │ concatenate[0][0]             │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ dense (Dense)                     │ (None, 256)                  │            65,792 │ dropout[0][0]                 │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ dropout_1 (Dropout)               │ (None, 256)                  │                 0 │ dense[0][0]                   │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ dense_1 (Dense)                   │ (None, 256)                  │            65,792 │ dropout_1[0][0]               │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ dropout_2 (Dropout)               │ (None, 256)                  │                 0 │ dense_1[0][0]                 │
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────────┤
│ dense_2 (Dense)                   │ (None, 3)                    │               771 │ dropout_2[0][0]               │
└───────────────────────────────────┴──────────────────────────────┴───────────────────┴───────────────────────────────┘
 Total params: 169,468 (661.98 KB)
 Trainable params: 169,468 (661.98 KB)
 Non-trainable params: 0 (0.00 B)

[ACTIVATIONS] Coletando estatísticas de todas as camadas (training=False)...
2025-11-16 20:27:57.350790: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:473] Loaded cuDNN version 91200

[ACTIVATIONS: RESUMO POR CAMADA]

#00  Layer: separable_conv2d  (SeparableConv2D)
     shape     : (1, 1024, 1024, 32), dtype=float16, numel=33554432
     min/max   : -0.40161 / 0.42432    mean=-0.00193  std=0.01726
     frac(<0/==0/>0): 0.5195 / 0.2223 / 0.2582

#01  Layer: group_normalization  (GroupNormalization)
     shape     : (1, 1024, 1024, 32), dtype=float16, numel=33554432
     min/max   : -9.82031 / 9.27344    mean=0.01929  std=0.36161
     frac(<0/==0/>0): 0.5257 / 0.0000 / 0.4743

#02  Layer: re_lu  (ReLU)
     shape     : (1, 1024, 1024, 32), dtype=float16, numel=33554432
     min/max   : 0.00000 / 9.27344    mean=0.12921  std=0.23352
     frac(<0/==0/>0): 0.0000 / 0.5257 / 0.4743

#03  Layer: spatial_dropout2d  (SpatialDropout2D)
     shape     : (1, 1024, 1024, 32), dtype=float32, numel=33554432
     min/max   : 0.00000 / 9.27344    mean=0.12921  std=0.23352
     frac(<0/==0/>0): 0.0000 / 0.5257 / 0.4743

#04  Layer: separable_conv2d_1  (SeparableConv2D)
     shape     : (1, 1024, 1024, 32), dtype=float16, numel=33554432
     min/max   : -2.55273 / 1.77539    mean=-0.01446  std=0.07160
     frac(<0/==0/>0): 0.6081 / 0.0000 / 0.3919

#05  Layer: group_normalization_1  (GroupNormalization)
     shape     : (1, 1024, 1024, 32), dtype=float16, numel=33554432
     min/max   : -37.28125 / 30.59375    mean=-0.03448  std=0.89723
     frac(<0/==0/>0): 0.4963 / 0.0000 / 0.5037

#06  Layer: re_lu_1  (ReLU)
     shape     : (1, 1024, 1024, 32), dtype=float16, numel=33554432
     min/max   : 0.00000 / 30.59375    mean=0.30305  std=0.49105
     frac(<0/==0/>0): 0.0000 / 0.4963 / 0.5037

#07  Layer: spatial_dropout2d_1  (SpatialDropout2D)
     shape     : (1, 1024, 1024, 32), dtype=float32, numel=33554432
     min/max   : 0.00000 / 30.59375    mean=0.30305  std=0.49105
     frac(<0/==0/>0): 0.0000 / 0.4963 / 0.5037

#08  Layer: max_pooling2d  (MaxPooling2D)
     shape     : (1, 512, 512, 32), dtype=float16, numel=8388608
     min/max   : 0.00000 / 30.59375    mean=0.55045  std=0.63134
     frac(<0/==0/>0): 0.0000 / 0.2761 / 0.7239

#09  Layer: separable_conv2d_2  (SeparableConv2D)
     shape     : (1, 512, 512, 64), dtype=float16, numel=16777216
     min/max   : -8.00781 / 6.63672    mean=-0.04519  std=0.59821
     frac(<0/==0/>0): 0.5603 / 0.0000 / 0.4397

#10  Layer: group_normalization_2  (GroupNormalization)
     shape     : (1, 512, 512, 64), dtype=float16, numel=16777216
     min/max   : -18.40625 / 15.33594    mean=-0.00980  std=1.02561
     frac(<0/==0/>0): 0.5161 / 0.0000 / 0.4839

#11  Layer: re_lu_2  (ReLU)
     shape     : (1, 512, 512, 64), dtype=float16, numel=16777216
     min/max   : 0.00000 / 15.33594    mean=0.40664  std=0.58979
     frac(<0/==0/>0): 0.0000 / 0.5161 / 0.4839

#12  Layer: spatial_dropout2d_2  (SpatialDropout2D)
     shape     : (1, 512, 512, 64), dtype=float32, numel=16777216
     min/max   : 0.00000 / 15.33594    mean=0.40664  std=0.58979
     frac(<0/==0/>0): 0.0000 / 0.5161 / 0.4839

#13  Layer: separable_conv2d_3  (SeparableConv2D)
     shape     : (1, 512, 512, 64), dtype=float16, numel=16777216
     min/max   : -2.71484 / 3.00000    mean=-0.02723  std=0.21153
     frac(<0/==0/>0): 0.5503 / 0.0000 / 0.4497

#14  Layer: group_normalization_3  (GroupNormalization)
     shape     : (1, 512, 512, 64), dtype=float16, numel=16777216
     min/max   : -18.21875 / 18.53125    mean=-0.00019  std=1.02024
     frac(<0/==0/>0): 0.4998 / 0.0000 / 0.5002

#15  Layer: re_lu_3  (ReLU)
     shape     : (1, 512, 512, 64), dtype=float16, numel=16777216
     min/max   : 0.00000 / 18.53125    mean=0.39055  std=0.60492
     frac(<0/==0/>0): 0.0000 / 0.4998 / 0.5002

#16  Layer: spatial_dropout2d_3  (SpatialDropout2D)
     shape     : (1, 512, 512, 64), dtype=float32, numel=16777216
     min/max   : 0.00000 / 18.53125    mean=0.39055  std=0.60492
     frac(<0/==0/>0): 0.0000 / 0.4998 / 0.5002

#17  Layer: max_pooling2d_1  (MaxPooling2D)
     shape     : (1, 256, 256, 64), dtype=float16, numel=4194304
     min/max   : 0.00000 / 18.53125    mean=0.66916  std=0.75314
     frac(<0/==0/>0): 0.0000 / 0.3135 / 0.6865

#18  Layer: separable_conv2d_4  (SeparableConv2D)
     shape     : (1, 256, 256, 128), dtype=float16, numel=8388608
     min/max   : -10.00781 / 8.21875    mean=-0.45935  std=1.04865
     frac(<0/==0/>0): 0.6917 / 0.0000 / 0.3083

#19  Layer: group_normalization_4  (GroupNormalization)
     shape     : (1, 256, 256, 128), dtype=float16, numel=8388608
     min/max   : -11.05469 / 7.31641    mean=-0.03173  std=1.06087
     frac(<0/==0/>0): 0.4834 / 0.0000 / 0.5166

#20  Layer: re_lu_4  (ReLU)
     shape     : (1, 256, 256, 128), dtype=float16, numel=8388608
     min/max   : 0.00000 / 7.31641    mean=0.41060  std=0.55289
     frac(<0/==0/>0): 0.0000 / 0.4834 / 0.5166

#21  Layer: spatial_dropout2d_4  (SpatialDropout2D)
     shape     : (1, 256, 256, 128), dtype=float32, numel=8388608
     min/max   : 0.00000 / 7.31641    mean=0.41060  std=0.55289
     frac(<0/==0/>0): 0.0000 / 0.4834 / 0.5166

#22  Layer: separable_conv2d_5  (SeparableConv2D)
     shape     : (1, 256, 256, 128), dtype=float16, numel=8388608
     min/max   : -3.56641 / 2.91602    mean=-0.03985  std=0.55916
     frac(<0/==0/>0): 0.5284 / 0.0000 / 0.4716

#23  Layer: group_normalization_5  (GroupNormalization)
     shape     : (1, 256, 256, 128), dtype=float16, numel=8388608
     min/max   : -8.17188 / 15.83594    mean=-0.28848  std=0.93635
     frac(<0/==0/>0): 0.5450 / 0.0000 / 0.4550

#24  Layer: re_lu_5  (ReLU)
     shape     : (1, 256, 256, 128), dtype=float16, numel=8388608
     min/max   : 0.00000 / 15.83594    mean=0.25539  std=0.38302
     frac(<0/==0/>0): 0.0000 / 0.5450 / 0.4550

#25  Layer: spatial_dropout2d_5  (SpatialDropout2D)
     shape     : (1, 256, 256, 128), dtype=float32, numel=8388608
     min/max   : 0.00000 / 15.83594    mean=0.25539  std=0.38302
     frac(<0/==0/>0): 0.0000 / 0.5450 / 0.4550

#26  Layer: max_pooling2d_2  (MaxPooling2D)
     shape     : (1, 128, 128, 128), dtype=float16, numel=2097152
     min/max   : 0.00000 / 15.83594    mean=0.29732  std=0.42331
     frac(<0/==0/>0): 0.0000 / 0.5141 / 0.4859

#27  Layer: global_average_pooling2d  (GlobalAveragePooling2D)
     shape     : (1, 128), dtype=float16, numel=128
     min/max   : 0.00000 / 1.10352    mean=0.29732  std=0.32996
     frac(<0/==0/>0): 0.0000 / 0.1328 / 0.8672

#28  Layer: global_max_pooling2d  (GlobalMaxPooling2D)
     shape     : (1, 128), dtype=float16, numel=128
     min/max   : 0.00000 / 15.83594    mean=2.42630  std=2.61568
     frac(<0/==0/>0): 0.0000 / 0.1328 / 0.8672

#29  Layer: concatenate  (Concatenate)
     shape     : (1, 256), dtype=float16, numel=256
     min/max   : 0.00000 / 15.83594    mean=1.36181  std=2.14674
     frac(<0/==0/>0): 0.0000 / 0.1328 / 0.8672

#30  Layer: dropout  (Dropout)
     shape     : (1, 256), dtype=float16, numel=256
     min/max   : 0.00000 / 15.83594    mean=1.36181  std=2.14674
     frac(<0/==0/>0): 0.0000 / 0.1328 / 0.8672

#31  Layer: dense  (Dense)
     shape     : (1, 256), dtype=float16, numel=256
     min/max   : -0.27783 / 16.45312    mean=0.74803  std=2.24828
     frac(<0/==0/>0): 0.2656 / 0.5195 / 0.2148

#32  Layer: dropout_1  (Dropout)
     shape     : (1, 256), dtype=float16, numel=256
     min/max   : -0.27783 / 16.45312    mean=0.74803  std=2.24828
     frac(<0/==0/>0): 0.2656 / 0.5195 / 0.2148

#33  Layer: dense_1  (Dense)
     shape     : (1, 256), dtype=float16, numel=256
     min/max   : -0.27808 / 9.64062    mean=0.80019  std=1.96407
     frac(<0/==0/>0): 0.6367 / 0.1445 / 0.2188

#34  Layer: dropout_2  (Dropout)
     shape     : (1, 256), dtype=float16, numel=256
     min/max   : -0.27808 / 9.64062    mean=0.80019  std=1.96407
     frac(<0/==0/>0): 0.6367 / 0.1445 / 0.2188

#35  Layer: dense_2  (Dense)
     shape     : (1, 3), dtype=float32, numel=3
     min/max   : 0.00019 / 0.86346    mean=0.33333  std=0.37895
     frac(<0/==0/>0): 0.0000 / 0.0000 / 1.0000