# cancer-detection-ai-models

Projeto de pesquisa comparando redes neurais convolucionais clássicas (CNNs) e redes neurais híbridas quântico-clássicas (QNNs) para classificação de mamografias digitais.

O objetivo é avaliar modelos de inteligência artificial para classificar mamografias em três categorias:

- `Normal`
- `Benign`
- `Malignant`

Este repositório preserva o código, os modelos treinados, os resultados e os artefatos usados na comparação experimental. O projeto tem finalidade acadêmica e de pesquisa; não é um sistema de diagnóstico clínico.

## Visão Geral

A pesquisa compara duas abordagens:

- Uma CNN clássica customizada, implementada com TensorFlow/Keras e treinada do zero.
- Uma arquitetura híbrida quântico-clássica, combinando um tronco CNN com uma camada quântica implementada com Qiskit.

O artigo completo escrito sobre a pesquisa está no arquivo:

`analide-cnn-qnn.docx`

## Dataset

Os experimentos usam o benchmark Mammo-Bench, especificamente a tarefa N/B/M baseada no arquivo:

`mammo-bench_nbm_classification.csv`

O dataset e o protocolo de referência foram encontrados a partir do projeto original [Gaurav2543/Mammo-Bench](https://github.com/Gaurav2543/Mammo-Bench), associado ao trabalho [Mammo-Bench: A Large-Scale Benchmark Dataset of Mammography Images](https://link.springer.com/chapter/10.1007/978-3-032-02489-3_11), de Gaurav Bhole, S. Suba e Nita Parekh.

O dataset completo não está incluído no GitHub porque é grande demais para o repositório. Um link externo com instruções de acesso será adicionado posteriormente.

Nos experimentos principais, a base foi balanceada para:

| Classe | Rótulo | Amostras |
| --- | ---: | ---: |
| Normal | 0 | 7.000 |
| Benign | 1 | 7.000 |
| Malignant | 2 | 7.000 |

O conjunto balanceado possui 21.000 mamografias pré-processadas em escala de cinza. A divisão usada foi:

| Subconjunto | Percentual | Amostras |
| --- | ---: | ---: |
| Treino | 70% | 14.700 |
| Validação | 15% | 3.150 |
| Teste | 15% | 3.150 |

Mais detalhes: [docs/dataset.md](docs/dataset.md).

## Estrutura

```text
modelo-classico/
  Pipeline, arquitetura, treinamento e avaliação da CNN clássica.

modelo-quantico/
  Modelo híbrido quântico-clássico usando TensorFlow/Keras e Qiskit.

melhores-modelos/
  Runs selecionados, modelos salvos, métricas, curvas, matrizes de confusão e Grad-CAMs.

docs/
  Documentação resumida sobre dataset, metodologia, resultados e organização.
```

Os arquivos `.keras` e demais artefatos de treinamento foram mantidos porque fazem parte da comparação descrita na pesquisa.

## CNN Clássica

O pipeline clássico inclui:

- Normalização dos rótulos `Normal`, `Benign` e `Malignant`.
- Divisão estratificada em treino, validação e teste.
- Balanceamento com limite de 7.000 imagens por classe.
- Leitura de imagens em escala de cinza.
- Redimensionamento e normalização para `[0, 1]`.
- Data augmentation com flip horizontal, translação, zoom, brilho/contraste, ruído e cutout.
- Avaliação com acurácia, loss, AUC, precision, recall, F1-score, matriz de confusão e Grad-CAM.

A CNN customizada usa convoluções separáveis, Group Normalization, ativações Swish/Silu, Max Pooling, Global Average Pooling, Global Max Pooling, Dropout e camadas densas com saída `softmax`.

## QNN Híbrida

O modelo quântico é uma primeira versão exploratória de arquitetura híbrida. Ele combina:

- Tronco CNN para extração de características.
- Projeção densa para um vetor com dimensão igual ao número de qubits.
- Camada quântica baseada em Qiskit.
- Classificador clássico final.

A camada quântica usa encoding `RY`, parâmetros treináveis `RZ`, emaranhamento com CNOT em anel e valores esperados de observáveis Pauli-Z.

Os resultados QNN atuais foram preservados, mas ainda não superam o baseline clássico.

## Resultados Principais

O melhor resultado CNN consolidado para apresentação está em:

`melhores-modelos/run_20251206_053622`

| Modelo | Run | Test accuracy | Test AUC OvR | Observação |
| --- | --- | ---: | ---: | --- |
| CNN clássica | `run_20251206_053622` | 0.6873 | 0.8457 | Melhor resultado consolidado com métricas completas |
| QNN híbrida | `run_qnn_1764246834` | 0.3244 | - | Experimento exploratório, próximo ao baseline aleatório |

Como referência externa, o trabalho Mammo-Bench reporta 77.8% de acurácia para classificação em três classes sem augmentation e 78.8% com augmentation das classes minoritárias.

Métricas de validação do melhor run CNN:

| Métrica | Valor |
| --- | ---: |
| Final accuracy | 0.6895 |
| Final AUC OvR | 0.8482 |
| Final loss | 0.6880 |

Desempenho por classe no mesmo run:

| Classe | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| Class 0 - Normal | 0.73 | 0.85 | 0.79 |
| Class 1 - Benign | 0.58 | 0.47 | 0.52 |
| Class 2 - Malignant | 0.74 | 0.76 | 0.75 |

A classe benigna foi a mais difícil para o modelo, especialmente em recall. Esse é um ponto importante da discussão científica.

Mais detalhes: [docs/resultados.md](docs/resultados.md).

## Figuras e Artefatos

Alguns artefatos visuais usados na análise:

- Ranking dos modelos: [melhores-modelos/ranking_melhor_val_accuracy.png](melhores-modelos/ranking_melhor_val_accuracy.png)
- Curvas de treinamento: [melhores-modelos/curvas_treinamento_todos_modelos.svg](melhores-modelos/curvas_treinamento_todos_modelos.svg)
- Matriz de confusão do melhor run CNN: [melhores-modelos/run_20251206_053622/cm.png](melhores-modelos/run_20251206_053622/cm.png)
- Exemplos Grad-CAM nas subpastas `gradcam/` dos runs selecionados.

## Reprodutibilidade

O repositório preserva os artefatos experimentais gerados durante a pesquisa. As instruções completas para reprodução serão adicionadas depois que o link externo do dataset estiver documentado.

Estado atual:

- O dataset não está incluído no GitHub.
- Os modelos salvos foram mantidos no repositório.
- O código de treinamento e avaliação está preservado.
- Comandos públicos completos de execução serão documentados em uma etapa futura.

## Limitações

- A melhor CNN atinge desempenho relevante, mas ainda limitado para contexto médico.
- A classe benigna apresenta recall menor que as classes normal e maligna.
- Os experimentos QNN ainda são preliminares.
- Os mapas Grad-CAM são apoio visual de interpretabilidade, não explicações médicas conclusivas.
- O projeto não deve ser usado para diagnóstico clínico.

## Documentação

- [Dataset](docs/dataset.md)
- [Metodologia](docs/metodologia.md)
- [Resultados](docs/resultados.md)
- [Organização do repositório](docs/organizacao.md)
- [Referências](docs/referencias.md)
