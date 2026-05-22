# Organização do Repositório

Este repositório preserva tanto o código quanto os artefatos gerados durante a pesquisa. Isso inclui modelos `.keras`, históricos de treinamento, matrizes de confusão, relatórios de classificação, curvas, arquivos de TensorBoard e exemplos Grad-CAM.

## Decisão de Curadoria

Os modelos salvos e os resultados gerados não foram removidos porque fazem parte da comparação experimental da pesquisa.

Para manter o repositório compreensível no GitHub, a documentação foi organizada em camadas:

- `README.md`: visão geral rápida para visitantes.
- `docs/dataset.md`: descrição do dataset e do protocolo de divisão.
- `docs/metodologia.md`: resumo do pipeline experimental.
- `docs/resultados.md`: consolidação dos principais resultados.
- `docs/referencias.md`: fonte do dataset, artigo Mammo-Bench e comparação externa.
- `analide-cnn-qnn.docx`: artigo completo escrito sobre a pesquisa.

## Artefatos de Treinamento

A pasta `melhores-modelos/` contém os principais experimentos selecionados. Cada subpasta de run pode conter:

- Modelos salvos (`best.keras`, `last.keras`, `final_model.keras`).
- Histórico de treinamento (`history.json`, `history.csv`).
- Resultados finais (`results_final.json`, `test_results.json`).
- Relatório de classificação.
- Matriz de confusão.
- Curvas de treinamento.
- Exemplos Grad-CAM.
- Logs auxiliares.

## Arquivos Grandes

Como os arquivos `.keras` e alguns artefatos podem crescer bastante, o `.gitignore` foi configurado para reduzir a chance de novos outputs grandes entrarem no Git por acidente.

Essa configuração não remove nem apaga os modelos que já existem no repositório. Ela apenas orienta futuros arquivos gerados.

## Próximos Passos de Organização

Melhorias futuras recomendadas:

- Adicionar link externo oficial para o dataset.
- Definir uma licença para o repositório.
- Criar metadados de citação quando a forma final de citação estiver definida.
- Separar, no futuro, modelos muito grandes em GitHub Releases, Git LFS ou armazenamento externo, mantendo no repositório apenas os resumos e figuras principais.
- Padronizar comandos públicos de reprodução quando o caminho do dataset estiver documentado.
