# 🧬 Plataforma de Bioengenharia e Eletromiografia (sEMG)
## Análise Integrada de CIVM, Fadiga Isométrica Sustentada e Estudo Crossover

Este repositório contém a aplicação web para processamento de sinais de eletromiografia de superfície (sEMG) e dinamometria (células de carga) no movimento de supino plano, com foco em Contração Isométrica Voluntária Máxima (CIVM), taxa de desenvolvimento de força (RFD) e fadiga espectral sustentada, comparando as condições **Controle** e **Intervenção (mobilização miofascial)**, nos momentos **Pré** e **Pós**.

---

## 📁 Estrutura de Arquivos do Projeto

- **`app_streamlit_master.py`**: Aplicação principal. Faz o **upload em lote** de todos os relatórios (CIVM e Fadiga, em PDF, de todos os voluntários), reconhece automaticamente os metadados de cada arquivo pelo nome, extrai as métricas reais de cada relatório, monta a **Tabela Mestra** (uma linha por voluntário × condição × momento), compara estatisticamente Controle × Intervenção e Pré × Pós (com ANOVA de medidas repetidas via Pingouin), e emite um laudo em PDF via WeasyPrint.
- **`requirements.txt`**: Bibliotecas Python necessárias para execução local e deploy no Streamlit Community Cloud.
- **`README.md`**: este guia.

> Os demais módulos mencionados em versões anteriores deste README (scripts de processamento de sinal bruto, geração de PDF com ReportLab, pipeline de terminal/Colab) continuam fazendo parte do pipeline de aquisição/geração dos relatórios PDF, mas não são necessários para rodar o portal web — o portal consome diretamente os PDFs já gerados por eles.

---

## 📂 Convenção de nome de arquivo para o Upload em Lote

Para que o reconhecimento automático funcione, nomeie os PDFs no padrão:

```
<ID>_<CIVM|FAD><C|I><PRE|POS>.pdf
```

| Campo | Significado | Valores |
|---|---|---|
| `ID` | Identificador do voluntário | número, ex.: `001`, `002` |
| `CIVM`/`FAD` (ou `FADIGA`) | Tipo de relatório | `CIVM` = força/EMG na contração máxima; `FAD`/`FADIGA` = fadiga sustentada |
| `C`/`I` | Condição | `C` = Controle; `I` = Intervenção (mobilização miofascial) |
| `PRE`/`POS` | Momento | Pré-sessão / Pós-sessão |

**Exemplos válidos:** `001_CIVMCPRE.pdf`, `001_CIVMIPOS.pdf`, `001_FADCPRE.pdf`, `003_FADIGAIPOS.pdf`.

Arquivos que não seguirem exatamente esse padrão **não são descartados**: eles caem automaticamente em uma fila de pendências dentro do próprio módulo de upload, onde é possível completar ID/Tipo/Condição/Momento manualmente antes de confirmar o processamento.

### ✅ Verificação automática de consistência
Cada relatório de CIVM (e de Fadiga, quando aplicável) traz internamente, no seu campo "Arquivo analisado", o nome do arquivo bruto (`.txt`) que originou aquele PDF. A aplicação compara esse nome interno com a Condição indicada no nome do PDF e **sinaliza automaticamente** qualquer divergência (por exemplo, um PDF chamado `..._CIVMCPOS.pdf` cujo conteúdo interno aponta para um arquivo de Intervenção), além de qualquer alerta automático de controle de qualidade do relatório original (ruído elevado, pico fora da janela de CIVM etc.). Essas sinalizações aparecem destacadas na Tabela Mestra e no laudo PDF.

---

## 🧮 O que a Tabela Mestra contém

Uma linha por **voluntário × condição × momento** (ex.: 5 voluntários × 2 condições × 2 momentos = 20 linhas), com:

- **Mecânica da CIVM**: pico de força (D/E), força média na janela (D/E), onset, instante do pico, tempo onset-pico, RFD em 0–50/0–100/0–200 ms (D/E) e as respectivas assimetrias bilaterais.
- **Eletromiografia da CIVM**: RMS e Eficiência Neuromuscular (ENM) por músculo (Peitoral, Tríceps, Deltoide), D/E e assimetrias.
- **Fadiga (resumo)**: médias de MDF inicial/final, MDF Slope, queda de MDF (%) e RMS Slope agregadas entre os canais musculares do teste de fadiga correspondente.
- **Rastreabilidade**: arquivo PDF de origem, nome interno do arquivo bruto, consistência do nome, alerta de qualidade, SHA-256.

A partir dela, o módulo **"3. Comparação Controle × Intervenção"** gera automaticamente a tabela de média ± desvio-padrão por grupo e o teste pareado (t de Student + Wilcoxon) comparando a variação Pós−Pré entre Intervenção e Controle, para cada variável — além de gráficos de barras com trajetória individual por voluntário.

---

## 🚀 Como Executar Localmente

1. Clone o repositório ou baixe os arquivos em uma pasta:
```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute a aplicação web:
```bash
streamlit run app_streamlit_master.py
```
Acesse no seu navegador em `http://localhost:8501`.

4. Na barra lateral, use **"Carregar Estudo Demonstrativo"** para testar a interface com dados sintéticos antes de subir seus PDFs reais.

---

## ☁️ Deploy no Streamlit Community Cloud

1. Suba os arquivos deste repositório para o seu GitHub.
2. Acesse [share.streamlit.io](https://share.streamlit.io) e conecte sua conta do GitHub.
3. Selecione o repositório e configure o **Main file path** como `app_streamlit_master.py`.
4. Clique em **Deploy!**.

> Observação sobre o WeasyPrint: o Streamlit Community Cloud pode não ter as bibliotecas de sistema (Pango/Cairo) que o WeasyPrint exige. Se a geração do PDF falhar no deploy, a aplicação automaticamente oferece o download do laudo em HTML (pronto para "Imprimir → Salvar como PDF" no navegador) como alternativa.
