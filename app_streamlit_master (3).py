"""
================================================================================
PORTAL DE BIOENGENHARIA: FUSÃO MULTI-RELATÓRIO (CIVM + FADIGA sEMG)
PROCESSAMENTO CROSSOVER, ANOVA DE MEDIDAS REPETIDAS & LAUDO WEASYPRINT
================================================================================
"""

import os
import io
import re
import base64
import tempfile
import numpy as np
import pandas as pd
import pdfplumber
import pingouin as pg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Verificação de disponibilidade da biblioteca WeasyPrint
try:
    import weasyprint
    WEASYPRINT_INSTALLED = True
except (ImportError, OSError):
    WEASYPRINT_INSTALLED = False

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DE INTERFACE STREAMLIT
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BioEng: Fusão CIVM & Fadiga sEMG",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS personalizada
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .kpi-box {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

# Inicialização de Session State
if 'df_fatigue_master' not in st.session_state:
    st.session_state['df_fatigue_master'] = pd.DataFrame()
if 'df_civm_master' not in st.session_state:
    st.session_state['df_civm_master'] = pd.DataFrame()
if 'df_merged_master' not in st.session_state:
    st.session_state['df_merged_master'] = pd.DataFrame()

# ==============================================================================
# MÓDULO 1: PADRONIZAÇÃO DE NOMENCLATURAS E AUXILIARES
# ==============================================================================

def standardize_muscle_name(name: str) -> str:
    """
    Padroniza os nomes dos músculos e canais segundo a sequência oficial do laboratório:
    1 - Peitoral maior direito (PMD)
    2 - Peitoral maior esquerdo (PME)
    3 - Tríceps braquial direito (TBD)
    4 - Tríceps braquial esquerdo (TBE)
    5 - Deltóide direito (DLD)
    6 - Deltóide esquerdo (DLE)
    7 - Célula de carga direita (CCD)
    8 - Célula de carga esquerda (CCE)
    """
    clean = str(name).strip().lower().replace('\n', ' ')
    is_left = any(w in clean for w in ['esquerdo', 'esquerda', 'pme', 'tbe', 'dle', 'cce']) or clean.endswith(' e') or ' (e)' in clean or '_e' in clean
    
    if 'peitoral' in clean or 'pm' in clean:
        return 'Peitoral maior esquerdo (PME)' if is_left else 'Peitoral maior direito (PMD)'
    elif 'tríceps' in clean or 'triceps' in clean or 'tb' in clean:
        return 'Tríceps braquial esquerdo (TBE)' if is_left else 'Tríceps braquial direito (TBD)'
    elif 'deltoide' in clean or 'deltóide' in clean or 'dl' in clean:
        return 'Deltóide esquerdo (DLE)' if is_left else 'Deltóide direito (DLD)'
    elif 'célula' in clean or 'celula' in clean or 'carga' in clean or 'cc' in clean:
        return 'Célula de carga esquerda (CCE)' if is_left else 'Célula de carga direita (CCD)'
    return str(name).strip()

def clean_float(val) -> float:
    """Converte números formatados com unidades (uV, Hz, %, vírgulas) para float."""
    if val is None or pd.isna(val):
        return np.nan
    clean = str(val).replace('%', '').replace('+', '').replace('uV/s', '').replace('Hz/s', '').replace('Hz', '').replace('uV', '').replace('kgf', '').strip().replace(',', '.')
    try:
        return float(clean)
    except (ValueError, TypeError):
        return np.nan

# ==============================================================================
# MÓDULO 2: PARSERS DE DOCUMENTOS (PDF & CSV)
# ==============================================================================


def parse_filename_metadata(filename: str) -> tuple[dict, str]:
    """Interpreta nomes como 002_CIVMC_PRE.pdf ou 002 F50 I POS.pdf.

    CIVM identifica contração máxima; F50 identifica fadiga. O sufixo C/I
    indica Controle/Intervenção e PRE/POS indica o momento da coleta.
    """
    stem = os.path.splitext(os.path.basename(filename))[0].upper()
    tokens = [t for t in re.split(r"[^A-Z0-9]+", stem) if t]
    participant = next((t for t in tokens if re.fullmatch(r"\d{1,4}", t)), None)
    if participant is None:
        raise ValueError("não encontrei o número do voluntário (ex.: 002)")
    # Busca o código completo no nome, permitindo condição/momento colados:
    # 002 CIVMCPRE, 003 F50IPOS, além das formas com separadores.
    test_match = re.search(r"(CIVM|F50)([CI])?(PRE|POS|PÓS)?", stem)
    if not test_match:
        raise ValueError("não encontrei CIVM ou F50 no nome do arquivo")
    base_test = test_match.group(1)
    cond_code = test_match.group(2)
    moment_code = test_match.group(3)

    # Complementa metadados quando os campos estiverem em tokens separados.
    if cond_code is None:
        test_token_idx = next((i for i, token in enumerate(tokens)
                               if token.startswith(base_test)), -1)
        if test_token_idx >= 0 and test_token_idx + 1 < len(tokens) and tokens[test_token_idx + 1] in ("C", "I"):
            cond_code = tokens[test_token_idx + 1]
    if cond_code is None:
        cond_code = next((token for token in tokens if token in ("C", "I")), None)
    if moment_code is None:
        moment_code = next((token for token in tokens if token in ("PRE", "POS", "PÓS")), None)
    if cond_code not in ("C", "I"):
        raise ValueError("não encontrei C (controle) ou I (intervenção) no nome")
    if moment_code is None:
        raise ValueError("não encontrei PRE ou POS no nome")
    metadata = {
        "ID_Voluntario": f"VOL_{int(participant):02d}",
        "Ordem_Sessao": 1,
        "Condicao": "Controle" if cond_code == "C" else "Intervenção",
        "Momento": "Pré" if moment_code == "PRE" else "Pós",
    }
    return metadata, base_test

def parse_fatigue_pdf(file_bytes_or_path, metadata: dict) -> pd.DataFrame:
    """
    Extrai parâmetros de fadiga eletromiográfica a partir de tabelas em relatórios PDF:
    - Músculo, MDF Inicial, MDF Final, MDF Slope, Queda MDF (%), RMS Slope, R² e p-valor.
    """
    records = []
    stream = file_bytes_or_path if isinstance(file_bytes_or_path, str) else io.BytesIO(file_bytes_or_path)
    
    with pdfplumber.open(stream) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue
                header = [str(c or '').replace('\n', ' ').strip().lower() for c in table[0]]
                has_mdf = any('mdf' in h for h in header)
                has_musc = any(any(k in h for k in ['músculo', 'musculo', 'canal']) for h in header)
                
                if has_mdf and has_musc:
                    col_map = {}
                    for idx, h in enumerate(header):
                        if any(k in h for k in ['músculo', 'musculo', 'canal']):
                            col_map['musculo'] = idx
                        elif 'mdf inicial' in h:
                            col_map['mdf_inicial'] = idx
                        elif 'mdf final' in h:
                            col_map['mdf_final'] = idx
                        elif 'mdf slope' in h or 'inclin' in h:
                            col_map['mdf_slope'] = idx
                        elif 'queda' in h or 'delta' in h:
                            col_map['queda_mdf'] = idx
                        elif 'rms slope' in h:
                            col_map['rms_slope'] = idx
                        elif 'r²' in h or 'r2' in h:
                            col_map['r2'] = idx
                        elif 'p-valor' in h or 'p_valor' in h or 'p-val' in h:
                            col_map['p_val'] = idx
                    
                    for row in table[1:]:
                        if not row or len(row) <= max(col_map.values(), default=0):
                            continue
                        musc_raw = str(row[col_map.get('musculo', 0)]).strip()
                        if not musc_raw or musc_raw.lower().startswith('músculo'):
                            continue
                        
                        records.append({
                            'ID_Voluntario': metadata.get('ID_Voluntario', 'VOL_01'),
                            'Ordem_Sessao': metadata.get('Ordem_Sessao', 1),
                            'Condicao': metadata.get('Condicao', 'Controle'),
                            'Momento': metadata.get('Momento', 'Pre'),
                            'Musculo': standardize_muscle_name(musc_raw),
                            'MDF_Inicial_Hz': clean_float(row[col_map['mdf_inicial']]) if 'mdf_inicial' in col_map else np.nan,
                            'MDF_Final_Hz': clean_float(row[col_map['mdf_final']]) if 'mdf_final' in col_map else np.nan,
                            'MDF_Slope_Hz_s': clean_float(row[col_map['mdf_slope']]) if 'mdf_slope' in col_map else np.nan,
                            'Queda_MDF_pct': clean_float(row[col_map['queda_mdf']]) if 'queda_mdf' in col_map else np.nan,
                            'RMS_Slope_uV_s': clean_float(row[col_map['rms_slope']]) if 'rms_slope' in col_map else np.nan,
                            'MDF_R2': clean_float(row[col_map['r2']]) if 'r2' in col_map else np.nan,
                            'MDF_p_valor': clean_float(row[col_map['p_val']]) if 'p_val' in col_map else np.nan,
                        })
    return pd.DataFrame(records)

def parse_civm_input(file_bytes_or_path, filename: str, metadata: dict) -> pd.DataFrame:
    """
    Extrai métricas de contração máxima a partir de arquivos CSV ou relatórios PDF:
    - Pico de RMS e RMS Médio da contração máxima.
    """
    records = []
    
    if filename.lower().endswith(('.csv', '.txt', '.tsv')):
        stream = file_bytes_or_path if isinstance(file_bytes_or_path, str) else io.BytesIO(file_bytes_or_path)
        try:
            df = pd.read_csv(stream, sep=None, engine='python')
        except Exception:
            if isinstance(file_bytes_or_path, str):
                df = pd.read_csv(file_bytes_or_path, sep=';')
            else:
                df = pd.read_csv(io.BytesIO(file_bytes_or_path), sep=';')
            
        musc_col = next((c for c in df.columns if any(k in str(c).lower() for k in ['canal', 'musculo', 'nome_coluna'])), None)
        rms_col = next((c for c in df.columns if 'rms' in str(c).lower() and 'pico' not in str(c).lower()), None)
        pico_col = next((c for c in df.columns if any(k in str(c).lower() for k in ['pico_a_pico', 'pico', 'maximo', 'max'])), None)
        
        if musc_col and (rms_col or pico_col):
            for _, row in df.iterrows():
                musc_raw = str(row[musc_col]).strip()
                if any(k in musc_raw.lower() for k in ['celula', 'célula', 'carga']):
                    continue
                records.append({
                    'ID_Voluntario': metadata.get('ID_Voluntario', 'VOL_01'),
                    'Ordem_Sessao': metadata.get('Ordem_Sessao', 1),
                    'Condicao': metadata.get('Condicao', 'Controle'),
                    'Momento': metadata.get('Momento', 'Pre'),
                    'Musculo': standardize_muscle_name(musc_raw),
                    'Pico_RMS_CIVM_uV': clean_float(row[pico_col]) if pico_col else clean_float(row[rms_col]),
                    'RMS_Medio_CIVM_uV': clean_float(row[rms_col]) if rms_col else np.nan,
                })
    elif filename.lower().endswith('.pdf'):
        stream = file_bytes_or_path if isinstance(file_bytes_or_path, str) else io.BytesIO(file_bytes_or_path)
        with pdfplumber.open(stream) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    header = [str(c or '').replace('\n', ' ').strip().lower() for c in table[0]]
                    has_musc = any(any(k in h for k in ['músculo', 'musculo', 'canal']) for h in header)
                    has_rms = any('rms' in h for h in header)
                    
                    if has_musc and has_rms:
                        col_map = {}
                        for idx, h in enumerate(header):
                            if any(k in h for k in ['músculo', 'musculo', 'canal']):
                                col_map['musculo'] = idx
                            elif any(k in h for k in ['pico', 'max', 'máx']):
                                col_map['pico'] = idx
                            elif 'rms' in h:
                                col_map['rms'] = idx
                        
                        for row in table[1:]:
                            if not row: continue
                            musc_raw = str(row[col_map.get('musculo', 0)]).strip()
                            if any(k in musc_raw.lower() for k in ['celula', 'célula', 'carga']) or not musc_raw:
                                continue
                            records.append({
                                'ID_Voluntario': metadata.get('ID_Voluntario', 'VOL_01'),
                                'Ordem_Sessao': metadata.get('Ordem_Sessao', 1),
                                'Condicao': metadata.get('Condicao', 'Controle'),
                                'Momento': metadata.get('Momento', 'Pre'),
                                'Musculo': standardize_muscle_name(musc_raw),
                                'Pico_RMS_CIVM_uV': clean_float(row[col_map['pico']]) if 'pico' in col_map else clean_float(row[col_map['rms']]),
                                'RMS_Medio_CIVM_uV': clean_float(row[col_map['rms']]) if 'rms' in col_map else np.nan,
                            })
    return pd.DataFrame(records)

# ==============================================================================
# MÓDULO 3: FUSÃO DE DADOS (PANDAS.MERGE) & FORMATO TIDY
# ==============================================================================

def merge_crossover_datasets(df_fatigue: pd.DataFrame, df_civm: pd.DataFrame) -> pd.DataFrame:
    """
    Executa a fusão dos datasets de Fadiga e CIVM utilizando as 4 chaves primárias:
    ['ID_Voluntario', 'Condicao', 'Momento', 'Musculo'].
    Retorna uma única tabela consolidada em formato longo (tidy data).
    """
    if df_fatigue.empty or df_civm.empty:
        return pd.DataFrame()
        
    merge_keys = ['ID_Voluntario', 'Condicao', 'Momento', 'Musculo']
    
    # Merge com inner join para pareamento rigoroso
    df_merged = pd.merge(
        df_fatigue,
        df_civm,
        on=merge_keys,
        how='inner',
        suffixes=('_fadiga', '_civm')
    )
    
    # Consolidação da coluna Ordem_Sessao
    if 'Ordem_Sessao_fadiga' in df_merged.columns:
        df_merged['Ordem_Sessao'] = df_merged['Ordem_Sessao_fadiga'].combine_first(df_merged.get('Ordem_Sessao_civm'))
        df_merged.drop(columns=['Ordem_Sessao_fadiga', 'Ordem_Sessao_civm'], inplace=True, errors='ignore')
    elif 'Ordem_Sessao_civm' in df_merged.columns:
        df_merged['Ordem_Sessao'] = df_merged['Ordem_Sessao_civm']
        df_merged.drop(columns=['Ordem_Sessao_civm'], inplace=True, errors='ignore')
        
    # Reordenação lógica das colunas (Tidy Data)
    desired_cols = [
        'ID_Voluntario', 'Ordem_Sessao', 'Condicao', 'Momento', 'Musculo',
        'Pico_RMS_CIVM_uV', 'RMS_Medio_CIVM_uV',
        'MDF_Inicial_Hz', 'MDF_Final_Hz', 'MDF_Slope_Hz_s', 'Queda_MDF_pct',
        'RMS_Slope_uV_s', 'MDF_R2', 'MDF_p_valor'
    ]
    existing_cols = [c for c in desired_cols if c in df_merged.columns]
    remaining = [c for c in df_merged.columns if c not in existing_cols]
    
    return df_merged[existing_cols + remaining]

# ==============================================================================
# MÓDULO 4: ANÁLISE ESTATÍSTICA (PINGOUIN ANOVA TWO-WAY RM)
# ==============================================================================

def run_repeated_measures_statistics(df_tidy: pd.DataFrame, target_muscle: str = None):
    """
    Aplica Two-Way Repeated Measures ANOVA (Condição x Momento) e correlações.
    Fator 1 (Within): Condicao (Controle vs Intervenção)
    Fator 2 (Within): Momento (Pré vs Pós)
    Sujeito: ID_Voluntario
    """
    sub_df = df_tidy.copy()
    if target_muscle and target_muscle != "Todos os Músculos (Agrupado)":
        sub_df = sub_df[sub_df['Musculo'] == target_muscle]
        
    # Agrupa por voluntário x condição x momento
    grouped = sub_df.groupby(['ID_Voluntario', 'Condicao', 'Momento'], as_index=False).agg({
        'Pico_RMS_CIVM_uV': 'mean',
        'MDF_Slope_Hz_s': 'mean',
        'RMS_Slope_uV_s': 'mean'
    })
    
    stats_out = {}
    
    # 1. ANOVA de Medidas Repetidas para Pico RMS (Força Máxima)
    try:
        aov_civm = pg.rm_anova(
            data=grouped,
            dv='Pico_RMS_CIVM_uV',
            within=['Condicao', 'Momento'],
            subject='ID_Voluntario',
            detailed=True
        )
        stats_out['aov_civm'] = aov_civm
    except Exception as e:
        stats_out['aov_civm_err'] = str(e)
        
    # 2. ANOVA de Medidas Repetidas para MDF Slope (Fadiga)
    try:
        aov_fatigue = pg.rm_anova(
            data=grouped,
            dv='MDF_Slope_Hz_s',
            within=['Condicao', 'Momento'],
            subject='ID_Voluntario',
            detailed=True
        )
        stats_out['aov_fatigue'] = aov_fatigue
    except Exception as e:
        stats_out['aov_fatigue_err'] = str(e)
        
    # 3. Post-Hoc Pairwise Tests com Correção de Bonferroni
    try:
        post_civm = pg.pairwise_tests(
            data=grouped,
            dv='Pico_RMS_CIVM_uV',
            within=['Condicao', 'Momento'],
            subject='ID_Voluntario',
            padjust='bonf'
        )
        stats_out['post_civm'] = post_civm
    except Exception:
        pass
        
    # 4. Correlação entre Ativação Máxima (CIVM) e Taxa de Fadiga (MDF Slope)
    try:
        corr_res = pg.corr(grouped['Pico_RMS_CIVM_uV'], grouped['MDF_Slope_Hz_s'], method='pearson')
        stats_out['correlation'] = corr_res
    except Exception:
        pass
        
    stats_out['processed_data'] = grouped
    return stats_out

# ==============================================================================
# MÓDULO 5: VISUALIZAÇÃO GRÁFICA & BASE64 PARA LAUDO
# ==============================================================================

def generate_statistical_plots(grouped_df: pd.DataFrame, muscle_label: str):
    """
    Gera gráficos de interação e gráfico de dispersão com reta de regressão.
    Retorna a figura Matplotlib e as imagens codificadas em base64 para o WeasyPrint.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.set_theme(style='whitegrid', palette='colorblind', font='DejaVu Sans')
    
    # Gráfico 1: Interação - Pico RMS CIVM
    ax1 = axes[0]
    sns.pointplot(
        data=grouped_df,
        x='Momento',
        y='Pico_RMS_CIVM_uV',
        hue='Condicao',
        markers=['o', 's'],
        linestyles=['-', '--'],
        capsize=0.1,
        err_kws={'linewidth': 1.5},
        ax=ax1
    )
    ax1.set_title(f"Capacidade Máxima: Pico RMS (CIVM)\n[{muscle_label}]", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Pico RMS (uV)")
    ax1.set_xlabel("Momento da Coleta")
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Interação - MDF Slope (Fadiga)
    ax2 = axes[1]
    sns.pointplot(
        data=grouped_df,
        x='Momento',
        y='MDF_Slope_Hz_s',
        hue='Condicao',
        markers=['o', 's'],
        linestyles=['-', '--'],
        capsize=0.1,
        err_kws={'linewidth': 1.5},
        ax=ax2
    )
    ax2.set_title(f"Resistência à Fadiga: MDF Slope\n[{muscle_label}]", fontsize=11, fontweight='bold')
    ax2.set_ylabel("MDF Slope (Hz/s)")
    ax2.set_xlabel("Momento da Coleta")
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Dispersão & Regressão - Ativação Máxima vs Fadiga
    ax3 = axes[2]
    sns.regplot(
        data=grouped_df,
        x='Pico_RMS_CIVM_uV',
        y='MDF_Slope_Hz_s',
        scatter_kws={'alpha': 0.7, 's': 50},
        line_kws={'color': 'darkred', 'linewidth': 2},
        ax=ax3
    )
    corr_val = np.corrcoef(grouped_df['Pico_RMS_CIVM_uV'], grouped_df['MDF_Slope_Hz_s'])[0, 1]
    ax3.set_title(f"Correlação: Força vs. Fadiga\n(r = {corr_val:.3f})", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Pico RMS CIVM (uV)")
    ax3.set_ylabel("MDF Slope (Hz/s)")
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Conversão para base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    buf.seek(0)
    b64_img = base64.b64encode(buf.read()).decode('utf-8')
    
    return fig, b64_img

# ==============================================================================
# MÓDULO 6: LAUDO ACADÊMICO WEASYPRINT (HTML + CSS PRINT)
# ==============================================================================

def build_weasyprint_html(df_tidy: pd.DataFrame, stats_results: dict, b64_chart: str, muscle_label: str) -> str:
    """Monta o documento HTML completo com formatação CSS para impressão de laudo PDF."""
    
    aov_civm_html = ""
    if 'aov_civm' in stats_results:
        df_c = stats_results['aov_civm'][['Source', 'SS', 'ddof1', 'ddof2', 'F', 'p_unc', 'ng2']].copy()
        df_c.columns = ['Fonte de Variação', 'SQ', 'GL1', 'GL2', 'F', 'p-valor', 'Eta²g']
        aov_civm_html = df_c.to_html(index=False, classes='styled-table', float_format="%.4f")
        
    aov_fat_html = ""
    if 'aov_fatigue' in stats_results:
        df_f = stats_results['aov_fatigue'][['Source', 'SS', 'ddof1', 'ddof2', 'F', 'p_unc', 'ng2']].copy()
        df_f.columns = ['Fonte de Variação', 'SQ', 'GL1', 'GL2', 'F', 'p-valor', 'Eta²g']
        aov_fat_html = df_f.to_html(index=False, classes='styled-table', float_format="%.4f")
        
    sample_table_html = df_tidy.head(8).to_html(index=False, classes='styled-table', float_format="%.2f")
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="utf-8">
        <title>Laudo Biomecânico: Crossover CIVM e Fadiga sEMG</title>
        <style>
            @page {{
                size: A4 portrait;
                margin: 1.5cm;
                @bottom-right {{
                    content: "Página " counter(page) " de " counter(pages);
                    font-size: 8pt;
                    color: #64748B;
                }}
                @bottom-left {{
                    content: "Laboratório de Biomecânica e Eletromiografia";
                    font-size: 8pt;
                    color: #64748B;
                }}
            }}
            body {{
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                color: #1E293B;
                line-height: 1.4;
                font-size: 10pt;
            }}
            .header-bar {{
                border-bottom: 3px solid #2563EB;
                padding-bottom: 8px;
                margin-bottom: 16px;
            }}
            h1 {{
                color: #1E3A8A;
                font-size: 18pt;
                margin: 0;
                font-weight: bold;
            }}
            .subtitle {{
                color: #64748B;
                font-size: 9.5pt;
                margin-top: 4px;
            }}
            h2 {{
                color: #1E3A8A;
                font-size: 12pt;
                border-bottom: 1px solid #E2E8F0;
                padding-bottom: 4px;
                margin-top: 16px;
                margin-bottom: 8px;
            }}
            .styled-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 8px 0 14px 0;
                font-size: 8pt;
            }}
            .styled-table th {{
                background-color: #2563EB;
                color: white;
                text-align: center;
                padding: 5px;
                font-weight: bold;
            }}
            .styled-table td {{
                border: 1px solid #E2E8F0;
                padding: 4px 6px;
                text-align: center;
            }}
            .styled-table tr:nth-child(even) {{
                background-color: #F8FAFC;
            }}
            .chart-img {{
                width: 100%;
                max-width: 100%;
                height: auto;
                border: 1px solid #CBD5E1;
                border-radius: 4px;
                margin-top: 8px;
            }}
            .info-card {{
                background-color: #F8FAFC;
                border-left: 4px solid #3B82F6;
                padding: 8px 12px;
                font-size: 8.5pt;
                margin-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="header-bar">
            <h1>LAUDO DE ANÁLISE BIOMECÂNICA CROSSOVER</h1>
            <div class="subtitle">Fusão Digital de Eletromiografia de Superfície: CIVM & Teste de Fadiga Isométrica Sustentada</div>
        </div>

        <div class="info-card">
            <b>Delineamento Experimental:</b> Estudo Crossover com Medidas Repetidas (Fatores: Condição [Controle vs. Intervenção] x Momento [Pré vs. Pós]).<br>
            <b>Músculo em Foco:</b> {muscle_label} | <b>Total de Registros Pareados:</b> {len(df_tidy)} linhas.
        </div>

        <h2>1. Tabela de Medidas Repetidas: ANOVA Two-Way (Pingouin)</h2>
        <p><b>A) Capacidade de Produção de Força (Pico RMS da CIVM):</b></p>
        {aov_civm_html}

        <p><b>B) Resistência à Fadiga Neuromuscular (MDF Slope):</b></p>
        {aov_fat_html}

        <h2>2. Visualização das Interações e Correlação Força-Fadiga</h2>
        <img class="chart-img" src="data:image/png;base64,{b64_chart}" alt="Gráficos de Interação e Dispersão" />

        <div style="page-break-before: always;"></div>

        <h2>3. Amostra da Tabela Mestra Consolidada (Formato Tidy Data)</h2>
        <p>Abaixo são apresentadas as primeiras linhas do conjunto de dados unificado via chaves primárias:</p>
        {sample_table_html}

        <h2>4. Síntese Interpretativa e Discussão Fisiológica</h2>
        <p style="text-align: justify; font-size: 8.5pt;">
            A análise integrada dos dados de Contração Isométrica Voluntária Máxima (CIVM) e da taxa de fadiga espectral (MDF Slope) 
            permite identificar se a mobilização miofascial induziu alterações na capacidade de recrutamento neural de alta frequência ou na resistência mecânica periférica. 
            Uma interação estatisticamente significante (Condição * Momento, p &lt; 0,05) no Pico de RMS indica modulação aguda no drive neural central pós-intervenção. 
            Concomitantemente, uma atenuação na taxa de queda da Frequência Mediana (MDF Slope menos negativo) sinaliza maior preservação da velocidade de condução do potencial de ação no sarcolema, 
            evidenciando potencial retardo da fadiga metabólica periférica.
        </p>
    </body>
    </html>
    """
    return html_content

# ==============================================================================
# GERADOR DE DADOS DEMO (CROSSOVER COMPLETO 8 VOLUNTÁRIOS)
# ==============================================================================

def generate_mock_crossover_study() -> pd.DataFrame:
    """Gera um conjunto de dados crossover realista (8 voluntários, 4 sessões, 6 músculos)."""
    records = []
    muscles = [
        'Peitoral maior direito (PMD)', 'Peitoral maior esquerdo (PME)',
        'Tríceps braquial direito (TBD)', 'Tríceps braquial esquerdo (TBE)',
        'Deltóide direito (DLD)', 'Deltóide esquerdo (DLE)'
    ]
    np.random.seed(42)
    
    for sub_id in range(1, 9):
        vol_str = f"VOL_{sub_id:02d}"
        for cond in ['Controle', 'Intervenção']:
            ordem = 1 if (sub_id % 2 != 0 and cond == 'Controle') or (sub_id % 2 == 0 and cond == 'Intervenção') else 2
            for mom in ['Pré', 'Pós']:
                for musc in muscles:
                    base_rms = 500.0 if 'PME' in musc or 'DLE' in musc else 250.0
                    base_mdf_slope = -0.60 if 'PME' in musc else -0.30
                    
                    int_boost_rms = 45.0 if cond == 'Intervenção' and mom == 'Pós' else 0.0
                    int_atten_fatigue = 0.22 if cond == 'Intervenção' and mom == 'Pós' else 0.0
                    
                    pico_rms = base_rms + int_boost_rms + np.random.normal(0, 30.0)
                    rms_medio = pico_rms * 0.75 + np.random.normal(0, 15.0)
                    
                    mdf_slope = base_mdf_slope + int_atten_fatigue + np.random.normal(0, 0.08)
                    mdf_init = 125.0 + np.random.normal(0, 4.0)
                    mdf_final = mdf_init + (mdf_slope * 28.0)
                    mdf_pct = ((mdf_final - mdf_init) / mdf_init) * 100.0
                    rms_slope = 0.35 - (0.10 if cond == 'Intervenção' and mom == 'Pós' else 0.0) + np.random.normal(0, 0.05)
                    
                    records.append({
                        'ID_Voluntario': vol_str,
                        'Ordem_Sessao': ordem,
                        'Condicao': cond,
                        'Momento': mom,
                        'Musculo': musc,
                        'Pico_RMS_CIVM_uV': round(float(pico_rms), 2),
                        'RMS_Medio_CIVM_uV': round(float(rms_medio), 2),
                        'MDF_Inicial_Hz': round(float(mdf_init), 1),
                        'MDF_Final_Hz': round(float(mdf_final), 1),
                        'MDF_Slope_Hz_s': round(float(mdf_slope), 3),
                        'Queda_MDF_pct': round(float(mdf_pct), 1),
                        'RMS_Slope_uV_s': round(float(rms_slope), 3),
                        'MDF_R2': round(float(np.random.uniform(0.70, 0.95)), 2),
                        'MDF_p_valor': round(float(np.random.uniform(0.001, 0.045)), 4),
                    })
    return pd.DataFrame(records)

# ==============================================================================
# INTERFACE DO USUÁRIO STREAMLIT
# ==============================================================================

st.markdown('<p class="main-title">🧬 Plataforma de Bioengenharia: Fusão CIVM & Fadiga sEMG</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Extração digital de relatórios PDF/CSV, alinhamento de chaves de cruzamento e modelagem estatística Crossover com Laudo WeasyPrint</p>', unsafe_allow_html=True)

# Barra Lateral: Navegação e Gestão de Demonstração
st.sidebar.header("🧭 Módulos do Sistema")
nav_choice = st.sidebar.radio(
    "Navegação:",
    [
        "1. Upload & Extração Dupla",
        "2. Tabela Mestra (Fusão Tidy)",
        "3. Análise Estatística Crossover",
        "4. Emissão de Laudo PDF (WeasyPrint)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Acesso Rápido de Teste")
if st.sidebar.button("Carregar Estudo Crossover Demonstrativo"):
    mock_df = generate_mock_crossover_study()
    st.session_state['df_merged_master'] = mock_df
    st.session_state['df_fatigue_master'] = mock_df[['ID_Voluntario', 'Ordem_Sessao', 'Condicao', 'Momento', 'Musculo', 'MDF_Inicial_Hz', 'MDF_Final_Hz', 'MDF_Slope_Hz_s', 'Queda_MDF_pct', 'RMS_Slope_uV_s', 'MDF_R2', 'MDF_p_valor']].copy()
    st.session_state['df_civm_master'] = mock_df[['ID_Voluntario', 'Ordem_Sessao', 'Condicao', 'Momento', 'Musculo', 'Pico_RMS_CIVM_uV', 'RMS_Medio_CIVM_uV']].copy()
    st.sidebar.success(f"Estudo Crossover carregado com {len(mock_df)} registros!")

# -----------------------------------------------------------------------------
# MÓDULO 1: UPLOAD E EXTRAÇÃO DUPLA
# -----------------------------------------------------------------------------
if nav_choice == "1. Upload & Extração Dupla":
    st.header("📂 Módulo 1: Upload e compilação automática em lote")
    st.markdown("Envie relatórios CIVM e F50 juntos. O aplicativo lê voluntário, teste, condição e momento diretamente do nome do arquivo.")
    st.info("Padrão recomendado: **002_CIVMC_PRE.pdf**, **002_CIVMI_POS.csv**, **002_F50C_PRE.pdf**. C = controle; I = intervenção; PRE = pré; POS = pós. Também são aceitos espaços e hífens como separadores.")
    batch_files = st.file_uploader(
        "Selecione o lote de relatórios", type=["pdf", "csv", "txt", "tsv"],
        accept_multiple_files=True, key="up_batch"
    )
    if st.button("Processar lote e adicionar ao banco", type="primary", disabled=not batch_files):
        fatigue_frames, civm_frames, errors, report = [], [], [], []
        for uploaded in batch_files or []:
            try:
                metadata, test_type = parse_filename_metadata(uploaded.name)
                if test_type == "F50":
                    if not uploaded.name.lower().endswith(".pdf"):
                        raise ValueError("relatórios F50 precisam estar em PDF")
                    extracted = parse_fatigue_pdf(uploaded.getvalue(), metadata)
                    target = fatigue_frames
                else:
                    extracted = parse_civm_input(uploaded.getvalue(), uploaded.name, metadata)
                    target = civm_frames
                if extracted.empty:
                    raise ValueError("nenhuma tabela compatível foi encontrada; confira o formato/colunas do relatório")
                extracted["Arquivo_Origem"] = uploaded.name
                target.append(extracted)
                report.append({"Arquivo": uploaded.name, "Tipo": "F50 — Fadiga" if test_type == "F50" else "CIVM",
                               "Voluntário": metadata["ID_Voluntario"], "Condição": metadata["Condicao"],
                               "Momento": metadata["Momento"], "Linhas extraídas": len(extracted), "Status": "OK"})
            except Exception as exc:
                errors.append(f"{uploaded.name}: {exc}")
                report.append({"Arquivo": uploaded.name, "Tipo": "—", "Voluntário": "—", "Condição": "—",
                               "Momento": "—", "Linhas extraídas": 0, "Status": f"ERRO: {exc}"})
        if fatigue_frames:
            st.session_state["df_fatigue_master"] = pd.concat([st.session_state["df_fatigue_master"], *fatigue_frames], ignore_index=True).drop_duplicates()
        if civm_frames:
            st.session_state["df_civm_master"] = pd.concat([st.session_state["df_civm_master"], *civm_frames], ignore_index=True).drop_duplicates()
        if report:
            st.markdown("### Resultado do processamento")
            st.dataframe(pd.DataFrame(report), use_container_width=True, hide_index=True)
            st.success(f"Lote concluído: {sum(len(x) for x in fatigue_frames)} linhas de F50 e {sum(len(x) for x in civm_frames)} linhas de CIVM adicionadas.")
            if errors:
                st.warning(f"{len(errors)} arquivo(s) não foram processados. Corrija os nomes/formatos indicados acima e envie novamente.")

    st.markdown("---")
    c_prev1, c_prev2 = st.columns(2)
    with c_prev1:
        st.markdown(f"**Banco de Fadiga:** ({len(st.session_state['df_fatigue_master'])} registros)")
        if not st.session_state["df_fatigue_master"].empty:
            st.dataframe(st.session_state["df_fatigue_master"].head(10), use_container_width=True)
    with c_prev2:
        st.markdown(f"**Banco de CIVM:** ({len(st.session_state['df_civm_master'])} registros)")
        if not st.session_state["df_civm_master"].empty:
            st.dataframe(st.session_state["df_civm_master"].head(10), use_container_width=True)
    if st.button("Limpar bancos carregados", type="secondary"):
        st.session_state["df_fatigue_master"] = pd.DataFrame()
        st.session_state["df_civm_master"] = pd.DataFrame()
        st.session_state["df_merged_master"] = pd.DataFrame()
        st.rerun()

# -----------------------------------------------------------------------------
# MÓDULO 2: FUSÃO DE DADOS (DATA MERGING) E TABELA ÚNICA
# -----------------------------------------------------------------------------
elif nav_choice == "2. Tabela Mestra (Fusão Tidy)":
    st.header("🔗 Módulo 2: Fusão de Dados e Tabela Mestra Consolidada")
    st.markdown("""
    Este módulo utiliza o algoritmo `pandas.merge()` para cruzar as informações de Fadiga e CIVM através das quatro chaves de alinhamento:
    **`['ID_Voluntario', 'Condicao', 'Momento', 'Musculo']`**.
    """)

    if st.button("Executar Fusão das Fontes (Merge)", type="primary"):
        merged = merge_crossover_datasets(st.session_state['df_fatigue_master'], st.session_state['df_civm_master'])
        if not merged.empty:
            st.session_state['df_merged_master'] = merged
            st.success(f"Fusão realizada com sucesso! Total de {len(merged)} linhas no formato Tidy Data.")
        else:
            st.error("Não foi possível cruzar os dados. Verifique se os metadados (ID_Voluntario, Condicao, Momento e Musculo) coincidem em ambas as fontes.")

    df_tidy = st.session_state['df_merged_master']
    
    if not df_tidy.empty:
        # Métricas de Diagnóstico da Fusão
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Linhas no Banco Fadiga", len(st.session_state['df_fatigue_master']))
        k2.metric("Linhas no Banco CIVM", len(st.session_state['df_civm_master']))
        k3.metric("Linhas Pareadas (Merge)", len(df_tidy))
        match_rate = (len(df_tidy) / max(1, len(st.session_state['df_fatigue_master']))) * 100.0
        k4.metric("Taxa de Pareamento", f"{match_rate:.1f}%")

        st.markdown("### 📋 Tabela Mestra Consolidada (Formato Tidy)")
        st.dataframe(df_tidy, use_container_width=True)

        # Botões de Download
        st.markdown("### 📥 Exportar Tabela Mestra")
        d_col1, d_col2 = st.columns(2)

        # 1. Exportação Excel (.xlsx)
        buf_xlsx = io.BytesIO()
        with pd.ExcelWriter(buf_xlsx, engine='openpyxl') as writer:
            df_tidy.to_excel(writer, index=False, sheet_name='Master_Tidy')
        buf_xlsx.seek(0)
        d_col1.download_button(
            label="📊 Baixar Tabela Mestra em Excel (.xlsx)",
            data=buf_xlsx,
            file_name="tabela_mestra_crossover_semg.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        # 2. Exportação CSV (UTF-8 com BOM para SPSS/JASP/R)
        csv_bytes = df_tidy.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')
        d_col2.download_button(
            label="📄 Baixar Tabela Mestra em CSV (.csv - SPSS/R)",
            data=csv_bytes,
            file_name="tabela_mestra_crossover_semg.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Nenhum dado pareado disponível. Faça o upload no Módulo 1 ou carregue o Estudo Demonstrativo na barra lateral.")

# -----------------------------------------------------------------------------
# MÓDULO 3: MÓDULO ESTATÍSTICO (PINGOUIN ANOVA TWO-WAY RM)
# -----------------------------------------------------------------------------
elif nav_choice == "3. Análise Estatística Crossover":
    st.header("📊 Módulo 3: Análise Estatística Crossover (Pingouin)")
    st.markdown("""
    Modelo estatístico de **Medidas Repetidas (Within-Subject)**:
    - **Fator 1**: Condição (`Controle` vs. `Intervenção`)
    - **Fator 2**: Momento (`Pré` vs. `Pós`)
    - **Unidade Amostral**: `ID_Voluntario`
    """)

    df_tidy = st.session_state['df_merged_master']

    if not df_tidy.empty:
        # Seletor de Músculo para análise estratificada
        muscle_options = ["Todos os Músculos (Agrupado)"] + sorted(list(df_tidy['Musculo'].unique()))
        selected_muscle = st.selectbox("Selecione o Músculo para a Modelagem Estatística:", muscle_options)

        stats_res = run_repeated_measures_statistics(df_tidy, selected_muscle)

        st.markdown("---")
        st.subheader("1. ANOVA Two-Way de Medidas Repetidas: Pico de RMS da CIVM")
        if 'aov_civm' in stats_res:
            df_c = stats_res['aov_civm'][['Source', 'SS', 'ddof1', 'ddof2', 'MS', 'F', 'p_unc', 'ng2']].copy()
            df_c.columns = ['Fonte de Variação', 'SQ', 'GL1', 'GL2', 'QM', 'F', 'p-valor', 'Eta²g']
            st.dataframe(df_c.style.format({'SQ': '{:.2f}', 'QM': '{:.2f}', 'F': '{:.3f}', 'p-valor': '{:.4f}', 'Eta²g': '{:.3f}'}), use_container_width=True)
        elif 'aov_civm_err' in stats_res:
            st.warning(f"Aviso no cálculo da ANOVA CIVM: {stats_res['aov_civm_err']}")

        st.subheader("2. ANOVA Two-Way de Medidas Repetidas: Taxa de Fadiga (MDF Slope)")
        if 'aov_fatigue' in stats_res:
            df_f = stats_res['aov_fatigue'][['Source', 'SS', 'ddof1', 'ddof2', 'MS', 'F', 'p_unc', 'ng2']].copy()
            df_f.columns = ['Fonte de Variação', 'SQ', 'GL1', 'GL2', 'QM', 'F', 'p-valor', 'Eta²g']
            st.dataframe(df_f.style.format({'SQ': '{:.4f}', 'QM': '{:.4f}', 'F': '{:.3f}', 'p-valor': '{:.4f}', 'Eta²g': '{:.3f}'}), use_container_width=True)
        elif 'aov_fatigue_err' in stats_res:
            st.warning(f"Aviso no cálculo da ANOVA Fadiga: {stats_res['aov_fatigue_err']}")

        # Gráficos de Interação e Dispersão
        st.markdown("---")
        st.subheader("3. Gráficos de Interação e Dispersão (Força vs. Fadiga)")
        fig, b64_img = generate_statistical_plots(stats_res['processed_data'], selected_muscle)
        st.pyplot(fig)
        
        # Salva para uso no laudo
        st.session_state['last_chart_b64'] = b64_img
        st.session_state['last_stats_res'] = stats_res
        st.session_state['last_muscle_label'] = selected_muscle
    else:
        st.info("Tabela mestra vazia. Execute o Módulo 2 ou carregue o Estudo Demonstrativo para visualizar a modelagem estatística.")

# -----------------------------------------------------------------------------
# MÓDULO 4: LAUDO AUTOMATIZADO WEASYPRINT
# -----------------------------------------------------------------------------
elif nav_choice == "4. Emissão de Laudo PDF (WeasyPrint)":
    st.header("📄 Módulo 4: Compilação de Laudo Automatizado (WeasyPrint)")
    st.markdown("Compilação da Tabela Única, Estatística ANOVA, Coeficientes e Gráficos de Interação em um Laudo PDF de alta resolução.")

    df_tidy = st.session_state['df_merged_master']

    if not df_tidy.empty:
        selected_muscle = st.session_state.get('last_muscle_label', 'Todos os Músculos (Agrupado)')
        stats_res = st.session_state.get('last_stats_res')
        b64_chart = st.session_state.get('last_chart_b64')

        if stats_res is None or b64_chart is None:
            stats_res = run_repeated_measures_statistics(df_tidy, selected_muscle)
            _, b64_chart = generate_statistical_plots(stats_res['processed_data'], selected_muscle)

        html_doc = build_weasyprint_html(df_tidy, stats_res, b64_chart, selected_muscle)

        # Prévia visual do laudo em HTML
        with st.expander("👁️ Visualizar Prévia do Laudo Formatado (HTML)", expanded=True):
            st.components.v1.html(html_doc, height=550, scrolling=True)

        # Geração do PDF via WeasyPrint
        st.markdown("### 📥 Download do Laudo PDF")
        
        if WEASYPRINT_INSTALLED:
            try:
                pdf_bytes = weasyprint.HTML(string=html_doc).write_pdf()
                st.download_button(
                    label="📑 Baixar Laudo Científico Completo (PDF via WeasyPrint)",
                    data=pdf_bytes,
                    file_name="laudo_crossover_civm_fadiga.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Erro ao compilar PDF com WeasyPrint: {e}")
        else:
            st.warning("⚠️ A biblioteca `weasyprint` não está instalada no ambiente Python atual.")
            st.info("Para habilitar a compilação direta do PDF no servidor, inclua `weasyprint` no `requirements.txt`.")
            
            # Disponibiliza o download do arquivo HTML estilizado
            st.download_button(
                label="🌐 Baixar Laudo Estilizado (.html - Pronto para Imprimir em PDF no Navegador)",
                data=html_doc.encode('utf-8'),
                file_name="laudo_crossover_civm_fadiga.html",
                mime="text/html",
                use_container_width=True
            )
    else:
        st.info("Tabela mestra vazia. Execute os módulos anteriores ou carregue o Estudo Demonstrativo para emitir o laudo.")
