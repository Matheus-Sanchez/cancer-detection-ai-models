# Metodologia

O projeto compara duas famílias de modelos para classificação de mamografias digitais:

- Uma CNN clássica customizada, usada como baseline principal.
- Uma arquitetura híbrida quântico-clássica, usada como experimento exploratório.

O dataset e parte do protocolo de comparação foram definidos a partir do projeto original [Gaurav2543/Mammo-Bench](https://github.com/Gaurav2543/Mammo-Bench) e do artigo associado [Mammo-Bench: A Large-Scale Benchmark Dataset of Mammography Images](https://link.springer.com/chapter/10.1007/978-3-032-02489-3_11).

## Pipeline Geral

O fluxo experimental usado na pesquisa pode ser resumido como:

1. Leitura do CSV de anotações N/B/M do Mammo-Bench.
2. Normalização dos rótulos para `Normal`, `Benign` e `Malignant`.
3. Mapeamento das classes para os valores numéricos `0`, `1` e `2`.
4. Balanceamento por classe com limite de 7.000 imagens.
5. Divisão estratificada em treino, validação e teste.
6. Carregamento das imagens em escala de cinza.
7. Redimensionamento e normalização para `[0, 1]`.
8. Treinamento dos modelos.
9. Avaliação quantitativa por métricas clássicas.
10. Avaliação qualitativa por Grad-CAM nos modelos CNN.

## Modelo CNN Clássico

A CNN clássica foi desenvolvida em TensorFlow/Keras e treinada do zero. A versão atual usa:

- `SeparableConv2D`.
- `GroupNormalization`.
- Ativações `swish`/`silu`.
- `MaxPooling2D`.
- `GlobalAveragePooling2D`.
- `GlobalMaxPooling2D`.
- Camadas densas com dropout.
- Saída `softmax` com três classes.

O modelo foi avaliado com acurácia, perda, AUC one-vs-rest, matriz de confusão, precision, recall e F1-score.

## Modelo Híbrido Quântico-Clássico

O modelo quântico é uma arquitetura híbrida implementada com TensorFlow/Keras e Qiskit.

Sua estrutura geral é:

1. Tronco CNN para extração de características.
2. Projeção densa para um vetor do tamanho do número de qubits.
3. Camada quântica baseada em Qiskit.
4. Classificador clássico final.

A camada quântica usa:

- Encoding com rotações `RY`.
- Parâmetros treináveis em rotações `RZ`.
- Emaranhamento com CNOT em anel.
- Valores esperados de observáveis Pauli-Z como saída.

Os experimentos QNN atuais são preliminares e servem como base para investigações futuras.

## Interpretabilidade

Para a CNN clássica, foram gerados mapas Grad-CAM em exemplos de acertos e erros.

Esses mapas ajudam a visualizar quais regiões da imagem influenciaram a decisão do modelo, mas não devem ser interpretados como laudo médico nem como garantia de correspondência com estruturas patológicas reais.
