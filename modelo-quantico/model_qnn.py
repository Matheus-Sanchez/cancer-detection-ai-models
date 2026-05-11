from __future__ import annotations
import os
import math

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("QISKIT_TELEMETRY_OPT_OUT", "1")
from qiskit_aer.primitives import Estimator

# ============================
# Qiskit imports (com fallback)
# ============================
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    try:
        from qiskit_aer.primitives import Estimator as AerEstimator

        _HAVE_AER_ESTIMATOR = True
    except Exception:
        _HAVE_AER_ESTIMATOR = False
        from qiskit.primitives import Estimator
except Exception as e:
    raise ImportError(
        "Não consegui importar Qiskit. Instale:\n"
        "  pip install qiskit qiskit-aer\n"
        f"Erro original: {e}"
    )


# ============================
# 1. Quantum Layer
# ============================
class QuantumLayer(L.Layer):
    """
    Layer híbrida Keras + Qiskit.

    - Encoding: RY(alpha_i) em cada qubit.
    - Blocos variacionais: RZ(theta[l, i]) + CNOT ring.
    - Suporta:
        * train_inputs = False  -> NÃO propaga gradiente para o encoding (alpha),
                                  só para theta (mais leve).
        * Error mitigation simples via "XX" dynamical decoupling.
    """
class QuantumLayer(keras.layers.Layer):
    def __init__(
        self,
        n_qubits: int = 4,
        reps: int = 1,
        data_scale: float = math.pi,
        shots: int = 0,
        error_mitigation: bool = True,
        seed: int | None = 1234,
        name: str = "quantum_layer",
        train_inputs: bool = True,
        n_threads: int | None = None,   # <-- NOVO
    ):
        super().__init__(name=name)
        assert n_qubits >= 1
        self.n_qubits = int(n_qubits)
        self.reps = int(reps)
        self.data_scale = float(data_scale)
        self.shots = int(shots)
        self.error_mitigation = error_mitigation
        self.train_inputs = bool(train_inputs)

        # ----------------------------
        # Configuração de threads CPU
        # ----------------------------
        if n_threads is None:
            # Usa quase todos os núcleos, deixando 2 pro sistema
            cpu_cores = os.cpu_count() or 4
            n_threads = max(1, cpu_cores - 2)
        self.n_threads = int(n_threads)

        # ----------------------------
        # Configura Estimator
        # ----------------------------
        self._estimator = None
        if "AerEstimator" in globals() and _HAVE_AER_ESTIMATOR:
            max_threads = min(self.n_threads, 4)  # nunca mais do que 4 aqui
            max_experiments = 1                   # nada de paralelizar 4 de uma vez em WSL
            
            backend_opts = {
                "method": "statevector",
                "max_parallel_threads": max_threads,
                "max_parallel_experiments": max_experiments,
                # deixa o threshold alto pra NÃO paralelizar o statevector de 8 qubits
                "statevector_parallel_threshold": 12,
            }
            
            run_opts = {}
            if self.shots > 0:
                run_opts["shots"] = int(self.shots)
            
            self._estimator = AerEstimator(
                backend_options=backend_opts,
                run_options=run_opts,
            )
            
            print(
                f"[QNN] AerEstimator com {max_threads} threads "
                f"(n_qubits={self.n_qubits}, shots={self.shots})"
            )
        else:
            # Fallback pro Estimator "puro" do Qiskit (sem Aer)
            self._estimator = Estimator()
            print(
                "[QNN] Usando Estimator padrão (sem AerEstimator); "
                "controle de threads pode ser limitado."
            )

        # Observáveis Z em cada qubit para expectativa
        self._obs = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.n_qubits - 1 - i))
            for i in range(self.n_qubits)
        ]

    @property
    def estimator(self):
        """Compatibilidade com QubitPurityMonitor (qmetrics)."""
        return self._estimator

    # ----- construção de circuitos numéricos -----
    def _build_numeric_circuit(
        self, alpha_vec: np.ndarray, theta_mat: np.ndarray
    ) -> QuantumCircuit:
        n = self.n_qubits
        qc = QuantumCircuit(n)

        # 1. Encoding RY
        for i in range(n):
            qc.ry(float(alpha_vec[i]), i)

        # 2. Error mitigation simples (dynamical decoupling "XX")
        if self.error_mitigation:
            qc.barrier()
            for i in range(n):
                qc.x(i)
                qc.x(i)
            qc.barrier()

        # 3. Blocos variacionais: RZ + CNOT ring
        for l in range(self.reps):
            for i in range(n):
                qc.rz(float(theta_mat[l, i]), i)
            for i in range(n):
                qc.cx(i, (i + 1) % n)

        return qc

    def _batch_expectations_np(self, alpha, theta):
        """
        alpha: (B, n_qubits)
        theta: (reps, n_qubits)
        Retorna: (B, n_qubits) com <Z_i>.
        """
        B = alpha.shape[0]
        circuits = []
        observables = []
        for b in range(B):
            qc = self._build_numeric_circuit(alpha[b], theta)
            circuits.extend([qc] * self.n_qubits)
            observables.extend(self._obs)

        job = self._estimator.run(circuits=circuits, observables=observables)
        vals = np.asarray(job.result().values, dtype=np.float32)
        return vals.reshape(B, self.n_qubits)

    def _batch_grads_np(self, alpha, theta, upstream):
        """
        Parameter-shift em batch.
        upstream: dL/dy, shape (B, n_qubits)
        Retorna (g_alpha, g_theta).
        Se train_inputs=False, g_alpha = 0 (não propagamos grad para o encoding).
        """
        shift = math.pi / 2.0
        B, N = alpha.shape
        g_alpha = np.zeros_like(alpha, dtype=np.float32)
        g_theta = np.zeros_like(theta, dtype=np.float32)

        # Gradiente em theta (sempre)
        for l in range(self.reps):
            for i in range(N):
                t_p, t_m = theta.copy(), theta.copy()
                t_p[l, i] += shift
                t_m[l, i] -= shift
                f_p = self._batch_expectations_np(alpha, t_p)
                f_m = self._batch_expectations_np(alpha, t_m)
                # soma sobre batch e qubits
                g_theta[l, i] = np.sum(0.5 * (f_p - f_m) * upstream)

        # Gradiente em alpha: só se train_inputs=True
        if self.train_inputs:
            for j in range(N):
                a_p, a_m = alpha.copy(), alpha.copy()
                a_p[:, j] += shift
                a_m[:, j] -= shift
                f_p = self._batch_expectations_np(a_p, theta)
                f_m = self._batch_expectations_np(a_m, theta)
                g_alpha[:, j] = np.sum(
                    0.5 * (f_p - f_m) * upstream, axis=1
                )

        return g_alpha, g_theta

    def build(self, input_shape):
        self.theta = self.add_weight(
            name="theta",
            shape=(self.reps, self.n_qubits),
            initializer="random_uniform",
            trainable=True,
            dtype=tf.float32,
        )
        super().build(input_shape)

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        theta = tf.identity(self.theta)

        # Shapes dinâmicos
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]

        @tf.custom_gradient
        def _op(x, th):
            alpha = self.data_scale * x

            # Forward (numpy_function chama código NumPy/Qiskit no host)
            def _fwd(a_np, t_np):
                return self._batch_expectations_np(a_np, t_np)

            y = tf.numpy_function(_fwd, [alpha, th], tf.float32)

            # Shape estático + dinâmico
            y.set_shape((None, self.n_qubits))
            y = tf.reshape(y, (batch_size, self.n_qubits))

            # Backward
            def _grad(dy):
                dy = tf.convert_to_tensor(dy, dtype=tf.float32)

                def _g(a_np, t_np, u_np):
                    return self._batch_grads_np(a_np, t_np, u_np)

                ga, gt = tf.numpy_function(
                    _g, [alpha, th, dy], [tf.float32, tf.float32]
                )

                # Shapes dos grads
                ga.set_shape(inputs.shape)  # estático
                ga = tf.reshape(ga, tf.shape(inputs))  # dinâmico
                gt.set_shape(self.theta.shape)

                # Se train_inputs=False, ga já veio zero de _batch_grads_np
                return ga * self.data_scale, gt

            return y, _grad

        return _op(inputs, theta)


# ============================
# 2. Tronco CNN (réplica da sua SimpleMammoCNN, versão separable)
# ============================
def _cnn_trunk(inputs):
    x = inputs

    # --- Bloco 1 ---
    x = L.SeparableConv2D(
        32,
        5,
        strides=2,
        padding="same",
        use_bias=False,
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x)
    x = L.GroupNormalization(groups=32)(x)
    x = L.ReLU()(x)
    x = L.SeparableConv2D(
        32,
        3,
        strides=1,
        padding="same",
        dilation_rate=1,
        use_bias=False,
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x)
    x = L.GroupNormalization(groups=32)(x)
    x = L.ReLU()(x)
    x = L.MaxPooling2D(2)(x)

    # --- Bloco 2 ---
    x = L.SeparableConv2D(
        64,
        5,
        strides=1,
        padding="same",
        use_bias=False,
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x)
    x = L.GroupNormalization(groups=32)(x)
    x = L.ReLU()(x)
    x = L.SeparableConv2D(
        64,
        3,
        strides=1,
        padding="same",
        dilation_rate=2,
        use_bias=False,
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x)
    x = L.GroupNormalization(groups=32)(x)
    x = L.ReLU()(x)
    x = L.MaxPooling2D(2)(x)

    # --- Bloco 3 ---
    x = L.SeparableConv2D(
        128,
        5,
        strides=1,
        padding="same",
        use_bias=False,
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x)
    x = L.GroupNormalization(groups=32)(x)
    x = L.ReLU()(x)
    x = L.SeparableConv2D(
        128,
        3,
        strides=1,
        padding="same",
        dilation_rate=3,
        use_bias=False,
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x)
    x = L.GroupNormalization(groups=32)(x)
    x = L.ReLU()(x)
    x = L.MaxPooling2D(2)(x)

    # Head de features: concat(GAP, GMP) -> 256
    gap = L.GlobalAveragePooling2D()(x)
    gmp = L.GlobalMaxPooling2D()(x)
    return L.Concatenate()([gap, gmp])


# ============================
# 3. Modelo híbrido
# ============================
def build_model(
    img_size: int,
    channels: int,
    num_classes: int,
    n_qubits: int = 4,
    reps: int = 1,
    shots: int = 0,
) -> keras.Model:
    inputs = L.Input(shape=(img_size, img_size, channels))

    # 1. Tronco CNN
    x = _cnn_trunk(inputs)  # (None, 256)
    x = L.Dropout(0.3)(x)

    # 2. Projeção clássica -> espaço quântico
    q_in = L.Dense(
        n_qubits, activation="tanh", name="proj_to_qubits"
    )(x)

    # 3. Camada quântica
    q_out = QuantumLayer(
        n_qubits=n_qubits,
        reps=reps,
        shots=shots,
        error_mitigation=True,
        train_inputs=False,  # << não propaga grad para o encoding (mais leve)
        name="qiskit_qnn",
    )(q_in)

    # 4. Classificador final
    x = L.Dropout(0.3)(q_out)
    x = L.Dense(256, activation="swish", name="dense")(x)
    x = L.Dropout(0.4)(x)
    outputs = L.Dense(
        num_classes, activation="softmax", dtype="float32", name="dense_1"
    )(x)

    return keras.Model(inputs, outputs, name="Hybrid_MammoCNN_Robust")


def compile_model(
    model: keras.Model, lr: float = 1e-4, weight_decay: float = 1e-6
):
    opt = keras.optimizers.AdamW(
        learning_rate=lr, weight_decay=weight_decay
    )
    try:
        loss = keras.losses.SparseCategoricalCrossentropy(
            label_smoothing=0.05
        )
    except TypeError:
        loss = keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=["accuracy"],
        run_eagerly=True,  # seguro com PyFunc + monitores
        jit_compile=False,
    )
    return model
