# Referências

## Fonte do Dataset

O dataset usado neste projeto foi encontrado a partir do repositório:

[Gaurav2543/Mammo-Bench](https://github.com/Gaurav2543/Mammo-Bench)

O repositório disponibiliza os arquivos CSV de classificação, código de pré-processamento e instruções de acesso ao dataset completo.

## Artigo de Referência

O trabalho associado ao dataset é:

[Mammo-Bench: A Large-Scale Benchmark Dataset of Mammography Images](https://link.springer.com/chapter/10.1007/978-3-032-02489-3_11)

Autores: Gaurav Bhole, S. Suba e Nita Parekh.

Observação sobre data: a página da Springer informa publicação online em 01 de novembro de 2025, enquanto a forma de citação exibida pela própria Springer aparece como Bhole, Suba e Parekh (2026), no volume de proceedings ICCABS 2025.

## Uso Como Comparação

O artigo Mammo-Bench é usado neste projeto como:

- Fonte do dataset.
- Referência metodológica para a tarefa N/B/M.
- Baseline externo de comparação para classificação em três classes.

Resultados reportados pelo Mammo-Bench para classificação em três classes:

| Configuração | Acurácia |
| --- | ---: |
| Sem augmentation | 77.8% |
| Com augmentation das classes minoritárias | 78.8% |

## Aviso de Uso

O próprio trabalho Mammo-Bench informa que o dataset é destinado a pesquisa e não deve ser usado para diagnóstico clínico direto. Este repositório segue a mesma restrição.

