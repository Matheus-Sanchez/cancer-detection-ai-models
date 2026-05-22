# Dataset

Este projeto utiliza o benchmark Mammo-Bench para a tarefa de classificação N/B/M, isto é, classificação de mamografias digitais em três classes:

| Classe | Rótulo numérico |
| --- | ---: |
| Normal | 0 |
| Benign | 1 |
| Malignant | 2 |

O arquivo de anotações usado nos experimentos foi:

`mammo-bench_nbm_classification.csv`

## Fonte Original

O dataset foi encontrado a partir do repositório original:

[Gaurav2543/Mammo-Bench](https://github.com/Gaurav2543/Mammo-Bench)

Esse repositório acompanha o trabalho:

[Mammo-Bench: A Large-Scale Benchmark Dataset of Mammography Images](https://link.springer.com/chapter/10.1007/978-3-032-02489-3_11), de Gaurav Bhole, S. Suba e Nita Parekh.

O paper da Springer descreve o Mammo-Bench como um benchmark de grande escala para imagens de mamografia, com pipeline de pré-processamento que inclui segmentação da mama, remoção do músculo peitoral e recorte inteligente. O próprio paper também informa que o dataset é destinado à pesquisa e não deve ser usado para diagnóstico clínico direto.

## Disponibilidade

O dataset completo não está incluído neste repositório porque seu tamanho não é adequado para armazenamento direto no GitHub.

O link externo para acesso ao dataset será adicionado posteriormente. Até lá, este repositório preserva o código, os resultados, os modelos salvos e os artefatos usados na comparação experimental.

## Balanceamento

Para reduzir o viés causado pelo desbalanceamento natural entre classes em bases de mamografia, os experimentos principais limitaram cada classe a no máximo 7.000 imagens.

| Classe | Amostras usadas |
| --- | ---: |
| Normal | 7.000 |
| Benign | 7.000 |
| Malignant | 7.000 |
| Total | 21.000 |

Com esse balanceamento, os pesos de classe usados nos experimentos principais ficaram iguais a `1.0` para todas as classes.

## Divisão dos Dados

A divisão foi feita de forma estratificada, preservando a distribuição das classes:

| Subconjunto | Percentual | Amostras |
| --- | ---: | ---: |
| Treino | 70% | 14.700 |
| Validação | 15% | 3.150 |
| Teste | 15% | 3.150 |

## Pre-processamento

As imagens foram lidas a partir do conjunto pré-processado do Mammo-Bench, convertidas para escala de cinza com um canal, redimensionadas e normalizadas para o intervalo `[0, 1]`.

Nos experimentos principais da CNN clássica, a resolução usada foi `1024 x 1024 x 1`.

## Aumento de Dados

O pipeline de treino aplica aumento de dados dinâmico, incluindo:

- Flip horizontal.
- Translação.
- Zoom.
- Ajustes de brilho e contraste.
- Ruído gaussiano leve.
- Cutout.

Essas transformações são aplicadas apenas ao conjunto de treino.
