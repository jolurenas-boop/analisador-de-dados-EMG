# Analisador de relatórios CIVM e F50

Aplicativo Streamlit para importar em lote relatórios PDF de CIVM e fadiga, extrair campos tabulares reconhecidos e consolidar os registros por voluntário, condição e momento.

## Executar

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_streamlit_master.py
```

Para gerar o laudo diretamente em PDF, instale também `weasyprint` e as bibliotecas de sistema exigidas pela plataforma. Sem ele, o app oferece HTML para impressão pelo navegador.

## Padrão dos nomes

O nome é interpretado como `<ID>_<TIPO><CONDIÇÃO><MOMENTO>.pdf` (espaços e separadores também são aceitos):

- `002 CIVMCPRE.pdf`: voluntário 002, CIVM, controle, pré;
- `002 CIVMIPOS.pdf`: voluntário 002, CIVM, intervenção, pós;
- `003 F50IPOS.pdf`: voluntário 003, fadiga F50, intervenção, pós.

`C` = controle, `I` = intervenção; `PRE` = pré, `POS` = pós. `FAD`/`FADIGA` também são aceitos como sinônimos do tipo fadiga. Arquivos cujo nome não corresponda ao padrão ficam pendentes para preenchimento manual; nenhum campo é deduzido pelo conteúdo para substituir metadados ausentes.

## Integridade e limites

- Os campos do banco são lidos das tabelas/cabeçalhos previstos pelos parsers; a tabela mestra agrega os valores de fadiga por músculo para análise, mas a planilha mantém uma aba separada com cada linha muscular original.
- Cada PDF tem hash SHA-256, páginas de origem, contagem de tabelas, estado da extração e alertas. A auditoria JSON inclui texto e tabelas extraídos página a página para inspeção e comparação com o PDF original.
- Tabelas do PDF que não correspondem ao parser ainda ficam disponíveis no JSON de auditoria; a ausência de um campo reconhecido não é preenchida por estimativa.
- PDFs digitalizados sem camada de texto são sinalizados; OCR automático não é aplicado. Revise os documentos e os status parciais/sem dados antes de usar resultados.
- Arquivos idênticos pelo hash não são importados duas vezes. Relatórios diferentes que ocupem a mesma chave de voluntário/condição/momento são mantidos e sinalizados; pareamentos/análises inferenciais ambíguos são bloqueados em vez de fazer média silenciosa.
- A extração automatizada não garante ausência absoluta de erro. Faça conferência humana da planilha e da auditoria contra os relatórios-fonte antes de qualquer interpretação ou divulgação científica.
