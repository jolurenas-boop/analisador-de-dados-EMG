# 🧬 Plataforma de Bioengenharia e Eletromiografia (sEMG)
## Análise Integrada de CVM, Fadiga Isométrica Sustentada e Estudo Crossover

Este repositório contém a suíte completa de algoritmos e aplicações web para processamento de sinais de eletromiografia de superfície (sEMG) e dinamometria (células de carga) no movimento de supino plano, com foco em contrações voluntárias máximas (CVM), taxa de desenvolvimento de força (TDF) e testes de fadiga sustentada a 50% da CVM (30 segundos a 2000 Hz).

---

## 📁 Estrutura de Arquivos do Projeto

### 1. Aplicações Web (Streamlit)
- **`app_streamlit_master.py`**: Aplicação principal completa. Realiza o upload duplo (relatórios de Fadiga em PDF e CIVM em PDF/CSV), fusão dos dados por chaves compostas (`ID_Voluntario`, `Condicao`, `Momento`, `Musculo`), modelagem estatística Crossover via ANOVA de Medidas Repetidas (Pingouin) e emissão de laudo acadêmico em PDF via WeasyPrint / HTML.
- **`app_streamlit_fadiga.py`**: Aplicação web dedicada exclusivamente à análise de fadiga espectral e recrutamento compensatório a partir de arquivos brutos `.txt` ou `.csv` de 30s @ 2000 Hz.

### 2. Módulos de Processamento de Sinal e Algoritmos
- **`protocolo_fadiga_emg.py`**: Motor matemático de sEMG. Implementa janelamento móvel (1,0s com 50% de overlap), Welch PSD com janela de Hanning, cálculo contínuo de MDF e RMS, e regressão linear por mínimos quadrados (MDF Slope e RMS Slope).
- **`gerar_pdf_fadiga.py`**: Módulo gerador de relatórios técnicos em PDF via ReportLab para a análise de fadiga isolada.
- **`protocolo_analise_supino_cvm.py`**: Algoritmo para contrações voluntárias máximas (CVM). Implementa a metodologia de **Onset Unificado** contra onsets locais, cálculo de TDF em janelas de 50ms, 100ms e 200ms, e índice de Eficiência Neuromuscular (ENM = Impulso / iEMG).
- **`protocolo_supino_estatistico_pdf.py`**: Pipeline unificado que processa dados de CVM e fadiga, roda testes de hipótese pareados (t-test / Wilcoxon) e compila relatórios com gráficos integrados.
- **`protocolo_supino_completo.py`**: Script genérico interativo para terminal ou Google Colab, com upload automático e leitor blindado contra metadados de equipamentos de aquisição.

### 3. Configuração e Dependências
- **`requirements.txt`**: Lista de bibliotecas Python necessárias para execução local e deploy no Streamlit Community Cloud.
- **`README.md`**: Guia de documentação e instruções de execução.

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

3. Execute a aplicação web principal:
```bash
streamlit run app_streamlit_master.py
```
Acesse no seu navegador em `http://localhost:8501`.

---

## ☁️ Deploy no Streamlit Community Cloud

1. Suba os arquivos deste repositório para o seu GitHub.
2. Acesse [share.streamlit.io](https://share.streamlit.io) e conecte sua conta do GitHub.
3. Selecione o repositório e configure o **Main file path** como `app_streamlit_master.py`.
4. Clique em **Deploy!**.
