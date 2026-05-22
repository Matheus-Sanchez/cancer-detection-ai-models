# Resultados

Esta página resume os principais resultados preservados em `melhores-modelos/`.

## Ranking Geral

O arquivo consolidado com o ranking dos modelos é:

`melhores-modelos/resumo_melhores_modelos.csv`

Resumo dos melhores resultados por validação:

| Modelo | Família | Melhor val_accuracy | Val_accuracy final |
| --- | --- | ---: | ---: |
| CNN-noClean | Clássico | 0.7723 | 0.7262 |
| CNN-1031 | Clássico | 0.7143 | 0.7114 |
| CNN-1025 | Clássico | 0.7029 | 0.6959 |
| CNN-1206 | Clássico | 0.6902 | 0.6889 |
| CNN-1111 | Clássico | 0.6902 | 0.6898 |
| CNN-1027 | Clássico | 0.6886 | 0.6879 |
| CNN-1129 | Clássico | 0.6879 | 0.6876 |
| QNN-6834 | Quântico | 0.3729 | 0.3330 |
| QNN-4863 | Quântico | 0.3252 | 0.3208 |

O resultado `CNN-noClean` aparece com maior validação no ranking, mas deve ser interpretado com cuidado por representar uma configuração anterior sem a mesma limpeza/curadoria dos experimentos posteriores.

## Comparação com o Trabalho Mammo-Bench

O projeto original [Gaurav2543/Mammo-Bench](https://github.com/Gaurav2543/Mammo-Bench) e o artigo associado reportam os seguintes resultados de referência para classificação em três classes:

| Referência | Configuração | Acurácia |
| --- | --- | ---: |
| Mammo-Bench | Sem augmentation | 77.8% |
| Mammo-Bench | Com augmentation das classes minoritárias | 78.8% |

Esses valores servem como comparação externa para os resultados deste repositório. O melhor run consolidado da CNN deste projeto ficou abaixo do baseline reportado no Mammo-Bench, mas preserva uma implementação própria em TensorFlow/Keras e inclui uma comparação exploratória com QNN.

## Melhor Resultado Consolidado da CNN

O melhor resultado consolidado para apresentação científica está em:

`melhores-modelos/run_20251206_053622`

Métricas principais:

| Métrica | Valor |
| --- | ---: |
| Final accuracy | 0.6895 |
| Final AUC OvR | 0.8482 |
| Final loss | 0.6880 |
| Test accuracy | 0.6873 |
| Test AUC OvR | 0.8457 |
| Test loss | 0.6754 |

Métricas por classe:

| Classe | Precision | Recall | F1-score | Suporte |
| --- | ---: | ---: | ---: | ---: |
| Class 0 - Normal | 0.73 | 0.85 | 0.79 | 1050 |
| Class 1 - Benign | 0.58 | 0.47 | 0.52 | 1050 |
| Class 2 - Malignant | 0.74 | 0.76 | 0.75 | 1050 |

O ponto mais importante desse resultado é que a classe benigna apresentou desempenho inferior, especialmente em recall. Isso indica que o modelo ainda tem dificuldade em separar lesões benignas de outras categorias.

## Experimentos QNN

Os experimentos quânticos atuais foram preservados como parte da comparação, mas ainda não superam o baseline clássico.

Exemplo:

| Modelo | Loss | Accuracy |
| --- | ---: | ---: |
| QNN-6834 | 1.0996 | 0.3244 |

Como o problema possui três classes balanceadas, um classificador aleatório teria desempenho próximo de 33%. Portanto, os resultados QNN atuais devem ser apresentados como exploratórios.

## Figuras

Figuras principais preservadas no repositório:

- `melhores-modelos/ranking_melhor_val_accuracy.png`
- `melhores-modelos/curvas_treinamento_todos_modelos.svg`
- `melhores-modelos/run_20251206_053622/cm.png`
- Grad-CAMs em `melhores-modelos/*/gradcam/`

## Interpretação

Os resultados mostram que a CNN clássica aprendeu padrões relevantes da base, com AUC acima de 0.84 no melhor experimento consolidado. Ainda assim, a acurácia em torno de 69% e o baixo recall da classe benigna deixam claro que o modelo não está pronto para uso clínico.

A principal contribuição experimental do repositório é a comparação organizada entre uma CNN clássica funcional e uma primeira versão de arquitetura híbrida quântico-clássica para a mesma tarefa.
