I0000 00:00:1761695262.044175    3032 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 9712 MB memory:  -> device: 0, name: NVIDIA RTX A2000 12GB, pci bus id: 0000:01:00.0, compute capability: 8.6

[DATA]
  Train/Val        : 17850 / 3150
  Classes          : {'Normal': 0, 'Benign': 1, 'Malignant': 2}
  Class weights    : {0: 1.0, 1: 1.0, 2: 1.0}

[PIPELINE]
  img_size         : 1024 | channels: 1 | batch: 8
  aug              : flip LR + brightness ±0.03 + contrast [0.92, 1.08]
  extra_fraction   : 0.5
  steps base/aug   : 2232 / 1116  (prev_aug_imgs≈8925)

[CHECK: VAL BATCH]
  shape            : (8, 1024, 1024, 1)
  dtype            : <dtype: 'float32'>
  range            : min=0.000  max=1.000
  mean/std         : 0.2113 / 0.1939
  labels únicos    : [0, 1, 2]
2025-10-28 20:49:08.825903: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence

[CHECK: CONTAGEM]
  base imgs (real) : 17850
  aug  imgs (real) : 8928  (prev≈8925)






  pip install qiskit qiskit-aer pylatexenc


  python quantum_model/main_qnn.py \
  --img-size 512 --channels 1 \
  --batch 16 \
  --epochs 20 \
  --n-qubits 6 --q-reps 1 --q-shots 0 \
  --max-per-class 2000 \
  --extra-fraction 0.0


  # 1) Limita / configura threads usadas pelas libs numéricas (OpenMP / MKL)
export OMP_NUM_THREADS=14
export MKL_NUM_THREADS=14

# (opcional, mas pode ajudar a manter tudo consistente)
export NUMEXPR_MAX_THREADS=14

# 2) Agora roda o treino normalmente
python quantum_model/main_qnn.py \
  --img-size 512 --channels 1 \
  --batch 32 \
  --epochs 20 \
  --n-qubits 8 --q-reps 1 --q-shots 0 \
  --max-per-class 2000 \
  --extra-fraction 0.0