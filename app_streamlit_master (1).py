"""
================================================================================
PORTAL DE BIOENGENHARIA: FUSÃO MULTI-RELATÓRIO (CIVM + FADIGA sEMG)
VERSÃO CIENTÍFICA APRIMORADA - EMBASADA NAS DIRETRIZES SENIAM, ISEK E CRAM'S
PROCESSAMENTO CROSSOVER, NORMALIZAÇÃO POR %CIVM, DELTAS, ASI & LAUDO WEASYPRINT
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
    page_title="BioEng Master: CIVM & Fadiga sEMG",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS personalizada
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .kpi-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #F1F5F9;
        border-radius: 6px 6px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Inicialização do Session State
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
# MÓDULO 2: PARSERS DE DOCUMENTOS (PDF & CSV) COM METADADOS
# ==============================================================================

def parse_fatigue_pdf(file_bytes_or_path, metadata: dict) -> pd.DataFrame:
    """
    Extrai parâmetros de fadiga eletromiográfica a partir de tabelas em relatórios PDF:
    - Músculo, MDF Inicial, MDF Final, MDF Slope, Queda MDF (%), RMS Slope, R² e p-valor.
    Aplica cálculo da Normalized MDF Slope (%/s) conforme Mannion et al. (1997) e Eur Spine J (2026).
    """
    records = []
    stream = file_bytes_or_path if isinstance(file_bytes_or_path, str) else io.BytesIO(file_bytes_or_path)
    
    with pdfplumber.open(stream) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue
                header = [str(c).replace('\n', ' ').strip().lower() for c in table[0] if c is not None]
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
                        
                        mdf_init = clean_float(row[col_map['mdf_inicial']]) if 'mdf_inicial' in col_map else np.nan
                        mdf_final = clean_float(row[col_map['mdf_final']]) if 'mdf_final' in col_map else np.nan
                        mdf_slope = clean_float(row[col_map['mdf_slope']]) if 'mdf_slope' in col_map else np.nan
                        queda_mdf = clean_float(row[col_map['queda_mdf']]) if 'queda_mdf' in col_map else np.nan
                        rms_slope = clean_float(row[col_map['rms_slope']]) if 'rms_slope' in col_map else np.nan
                        r2_val = clean_float(row[col_map['r2']]) if 'r2' in col_map else np.nan
                        pval = clean_float(row[col_map['p_val']]) if 'p_val' in col_map else np.nan
                        
                        # Cálculo científico: Normalized MDF Slope (%/s) [Mannion et al., 1997; Eur Spine J, 2026]
                        norm_mdf_slope = (mdf_slope / mdf_init) * 100.0 if (mdf_init and mdf_init > 0 and not np.isnan(mdf_slope)) else np.nan
                        abs_mdf_dec = (mdf_init - mdf_final) if (mdf_init is not None and mdf_final is not None) else np.nan

                        records.append({
                            'ID_Voluntario': metadata.get('ID_Voluntario', 'VOL_01'),
                            'Ordem_Sessao': metadata.get('Ordem_Sessao', 1),
                            'Condicao': metadata.get('Condicao', 'Controle'),
                            'Momento': metadata.get('Momento', 'Pre'),
                            'Musculo': standardize_muscle_name(musc_raw),
                            'MDF_Inicial_Hz': mdf_init,
                            'MDF_Final_Hz': mdf_final,
                            'MDF_Slope_Hz_s': mdf_slope,
                            'Normalized_MDF_Slope_pct_s': norm_mdf_slope,
                            'Queda_MDF_pct': queda_mdf,
                            'Absolute_MDF_Decrease_Hz': abs_mdf_dec,
                            'RMS_Slope_uV_s': rms_slope,
                            'MDF_R2': r2_val,
                            'MDF_p_valor': pval,
                        })
    return pd.DataFrame(records)

def parse_civm_input(file_bytes_or_path, filename: str, metadata: dict) -> pd.DataFrame:
    """
    Extrai métricas de contração máxima a partir de arquivos CSV ou relatórios PDF:
    - Pico de RMS e RMS Médio da contração máxima.
    """
    records = []
    
    # 1. Entrada via Planilha CSV ou TXT
    if filename.lower().endswith(('.csv', '.txt', '.tsv')):
        stream = file_bytes_or_path if isinstance(file_bytes_or_path, str) else io.BytesIO(file_bytes_or_path)
        try:
            df = pd.read_csv(stream, sep=None, engine='python')
        except Exception:
            stream.seek(0)
            df = pd.read_csv(stream, sep=';')
            
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

    # 2. Entrada via Relatório PDF
    elif filename.lower().endswith('.pdf'):
        stream = file_bytes_or_path if isinstance(file_bytes_or_path, str) else io.BytesIO(file_bytes_or_path)
        with pdfplumber.open(stream) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    header = [str(c).replace('\n', ' ').strip().lower() for c in table[0] if c is not None]
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
# MÓDULO 3: FUSÃO DE DADOS (PANDAS.MERGE) & NORMALIZAÇÃO POR %CIVM (CRAM'S / SENIAM)
# ==============================================================================

def merge_crossover_datasets(df_fatigue: pd.DataFrame, df_civm: pd.DataFrame) -> pd.DataFrame:
    """
    Executa a fusão dos datasets de Fadiga e CIVM utilizando as 4 chaves primárias:
    ['ID_Voluntario', 'Condicao', 'Momento', 'Musculo'].
    Aplica normalização eletromiográfica por %CIVM segundo Cram's Introduction to sEMG.
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
        
    # --- NORMALIZAÇÃO CIENTÍFICA POR %CIVM (%MVIC) ---
    # 1. Taxa de recrutamento compensatório normalizada por %CIVM/s
    df_merged['RMS_Slope_pct_CIVM_s'] = (df_merged['RMS_Slope_uV_s'] / df_merged['Pico_RMS_CIVM_uV']) * 100.0
    
    # 2. Ativação relativa da contração máxima (Razão RMS Médio / Pico)
    df_merged['Ativacao_Relativa_pct_CIVM'] = (df_merged['RMS_Medio_CIVM_uV'] / df_merged['Pico_RMS_CIVM_uV']) * 100.0
    
    # Reordenação estrutural das colunas (Formato Longo Tidy)
    desired_cols = [
        'ID_Voluntario', 'Ordem_Sessao', 'Condicao', 'Momento', 'Musculo',
        'Pico_RMS_CIVM_uV', 'RMS_Medio_CIVM_uV', 'Ativacao_Relativa_pct_CIVM',
        'MDF_Inicial_Hz', 'MDF_Final_Hz', 'MDF_Slope_Hz_s', 'Normalized_MDF_Slope_pct_s',
        'Queda_MDF_pct', 'Absolute_MDF_Decrease_Hz',
        'RMS_Slope_uV_s', 'RMS_Slope_pct_CIVM_s', 'MDF_R2', 'MDF_p_valor'
    ]
    existing_cols = [c for c in desired_cols if c in df_merged.columns]
    remaining = [c for c in df_merged.columns if c not in existing_cols]
    
    return df_merged[existing_cols + remaining]

# ==============================================================================
# MÓDULO 4: CÁLCULO DE EFEITOS DELTA (CROSSOVER) E ASSIMETRIA BILATERAL (ASI)
# ==============================================================================

def strip_accents(text: str) -> str:
    """Remove acentos e converte para minúsculas para comparações robustas."""
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFD', str(text)) if unicodedata.category(c) != 'Mn').lower()

def calculate_crossover_deltas(df_merged: pd.DataFrame, target_var: str = 'Normalized_MDF_Slope_pct_s') -> pd.DataFrame:
    """
    Calcula os deltas intra-sessão (Pós - Pré) e o efeito líquido da intervenção:
    Delta_Controle = Pós_Controle - Pré_Controle
    Delta_Intervencao = Pós_Intervencao - Pré_Intervencao
    Treatment_Effect = Delta_Intervencao - Delta_Controle
    """
    if df_merged.empty:
        return pd.DataFrame()
        
    piv = df_merged.pivot_table(
        index=['ID_Voluntario', 'Musculo'],
        columns=['Condicao', 'Momento'],
        values=target_var,
        aggfunc='mean'
    ).reset_index()
    
    # Achatamento de multi-index
    new_cols = []
    for c in piv.columns:
        if isinstance(c, tuple):
            joined = '_'.join(str(x) for x in c if str(x) != '')
            new_cols.append(joined)
        else:
            new_cols.append(str(c))
    piv.columns = new_cols
    
    # Normalização robusta de nomes de colunas
    for c in list(piv.columns):
        norm_c = strip_accents(c)
        if 'interven' in norm_c and 'pre' in norm_c:
            piv.rename(columns={c: 'Intervencao_Pre'}, inplace=True)
        elif 'interven' in norm_c and 'pos' in norm_c:
            piv.rename(columns={c: 'Intervencao_Pos'}, inplace=True)
        elif 'controle' in norm_c and 'pre' in norm_c:
            piv.rename(columns={c: 'Controle_Pre'}, inplace=True)
        elif 'controle' in norm_c and 'pos' in norm_c:
            piv.rename(columns={c: 'Controle_Pos'}, inplace=True)
            
    if all(c in piv.columns for c in ['Controle_Pre', 'Controle_Pos', 'Intervencao_Pre', 'Intervencao_Pos']):
        piv['Delta_Controle'] = piv['Controle_Pos'] - piv['Controle_Pre']
        piv['Delta_Intervencao'] = piv['Intervencao_Pos'] - piv['Intervencao_Pre']
        piv['Efeito_Tratamento_Liquido'] = piv['Delta_Intervencao'] - piv['Delta_Controle']
        piv['Delta_Pct_Controle'] = (piv['Delta_Controle'] / piv['Controle_Pre'].abs()) * 100.0
        piv['Delta_Pct_Intervencao'] = (piv['Delta_Intervencao'] / piv['Intervencao_Pre'].abs()) * 100.0
        return piv
    return pd.DataFrame()

def calculate_bilateral_asymmetry(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o Índice de Assimetria Bilateral (ASI %):
    ASI = ((Direito - Esquerdo) / (Direito + Esquerdo)) * 100
    Para Peitoral (PMD vs PME), Tríceps (TBD vs TBE) e Deltóide (DLD vs DLE).
    """
    if df_merged.empty:
        return pd.DataFrame()
        
    pairs = [
        ('Peitoral', 'Peitoral maior direito (PMD)', 'Peitoral maior esquerdo (PME)'),
        ('Tríceps', 'Tríceps braquial direito (TBD)', 'Tríceps braquial esquerdo (TBE)'),
        ('Deltóide', 'Deltóide direito (DLD)', 'Deltóide esquerdo (DLE)')
    ]
    
    records = []
    grouped = df_merged.groupby(['ID_Voluntario', 'Condicao', 'Momento'])
    
    for (vol, cond, mom), grp in grouped:
        for group_name, right_m, left_m in pairs:
            r_row = grp[grp['Musculo'] == right_m]
            l_row = grp[grp['Musculo'] == left_m]
            
            if not r_row.empty and not l_row.empty:
                r_civm = r_row['Pico_RMS_CIVM_uV'].values[0]
                l_civm = l_row['Pico_RMS_CIVM_uV'].values[0]
                r_slope = r_row['Normalized_MDF_Slope_pct_s'].values[0]
                l_slope = l_row['Normalized_MDF_Slope_pct_s'].values[0]
                
                asi_civm = ((r_civm - l_civm) / (r_civm + l_civm)) * 100.0 if (r_civm + l_civm) != 0 else 0.0
                asi_fatigue = ((r_slope - l_slope) / (abs(r_slope) + abs(l_slope))) * 100.0 if (abs(r_slope) + abs(l_slope)) != 0 else 0.0
                
                records.append({
                    'ID_Voluntario': vol,
                    'Condicao': cond,
                    'Momento': mom,
                    'Grupo_Muscular': group_name,
                    'ASI_CIVM_pct': asi_civm,
                    'ASI_Fadiga_pct': asi_fatigue,
                    'Direito_CIVM': r_civm,
                    'Esquerdo_CIVM': l_civm,
                    'Direito_Norm_Slope': r_slope,
                    'Esquerdo_Norm_Slope': l_slope
                })
    return pd.DataFrame(records)

# ==============================================================================
# MÓDULO 5: MODELAGEM ESTATÍSTICA (PINGOUIN ANOVA TWO-WAY RM & CORRELAÇÃO)
# ==============================================================================

def run_repeated_measures_statistics(df_tidy: pd.DataFrame, target_muscle: str = None, dep_var: str = 'Normalized_MDF_Slope_pct_s'):
    """
    Aplica Two-Way Repeated Measures ANOVA (Condição x Momento) e testes correlacionais.
    Fator 1 (Within-Subject): Condição (Controle vs. Intervenção)
    Fator 2 (Within-Subject): Momento (Pré vs. Pós)
    Unidade Amostral: ID_Voluntario
    """
    sub_df = df_tidy.copy()
    if target_muscle and target_muscle != "Todos os Músculos (Agrupado)":
        sub_df = sub_df[sub_df['Musculo'] == target_muscle]
        
    grouped = sub_df.groupby(['ID_Voluntario', 'Condicao', 'Momento'], as_index=False).agg({
        'Pico_RMS_CIVM_uV': 'mean',
        'Normalized_MDF_Slope_pct_s': 'mean',
        'MDF_Slope_Hz_s': 'mean',
        'RMS_Slope_pct_CIVM_s': 'mean',
        'RMS_Slope_uV_s': 'mean'
    })
    
    stats_out = {}
    
    # 1. ANOVA de Medidas Repetidas para a Variável Dependente Escolhida
    try:
        aov_main = pg.rm_anova(
            data=grouped,
            dv=dep_var,
            within=['Condicao', 'Momento'],
            subject='ID_Voluntario',
            detailed=True
        )
        stats_out['aov_main'] = aov_main
        stats_out['dep_var_name'] = dep_var
    except Exception as e:
        stats_out['aov_main_err'] = str(e)
        
    # 2. ANOVA para a Capacidade Máxima (Pico RMS CIVM)
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

    # 3. Post-Hoc Pairwise Tests com Correção de Bonferroni
    try:
        post_tests = pg.pairwise_tests(
            data=grouped,
            dv=dep_var,
            within=['Condicao', 'Momento'],
            subject='ID_Voluntario',
            padjust='bonf'
        )
        stats_out['post_hoc'] = post_tests
    except Exception:
        pass
        
    # 4. Correlação Linear (Pearson & Spearman): Força Máxima vs. Taxa de Fadiga Normalizada
    try:
        corr_res = pg.corr(grouped['Pico_RMS_CIVM_uV'], grouped['Normalized_MDF_Slope_pct_s'], method='pearson')
        stats_out['correlation'] = corr_res
    except Exception:
        pass
        
    stats_out['processed_data'] = grouped
    return stats_out

# ==============================================================================
# MÓDULO 6: VISUALIZAÇÃO GRÁFICA & BASE64 PARA LAUDO WEASYPRINT
# ==============================================================================

def generate_statistical_plots(grouped_df: pd.DataFrame, muscle_label: str, dep_var: str = 'Normalized_MDF_Slope_pct_s'):
    """
    Gera gráficos de interação das medidas repetidas e dispersão com reta de regressão.
    Retorna a figura Matplotlib e as imagens codificadas em base64 para o WeasyPrint.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.set_theme(style='whitegrid', palette='colorblind', font='DejaVu Sans')
    
    # Gráfico 1: Interação - Pico RMS CIVM (Força Máxima)
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
    
    # Gráfico 2: Interação - Variável Dependente (Fadiga Normalizada ou Absoluta)
    ax2 = axes[1]
    sns.pointplot(
        data=grouped_df,
        x='Momento',
        y=dep_var,
        hue='Condicao',
        markers=['o', 's'],
        linestyles=['-', '--'],
        capsize=0.1,
        err_kws={'linewidth': 1.5},
        ax=ax2
    )
    y_lbl = "Normalized MDF Slope (%/s)" if "norm" in dep_var.lower() else dep_var
    ax2.set_title(f"Resistência à Fadiga: {y_lbl}\n[{muscle_label}]", fontsize=11, fontweight='bold')
    ax2.set_ylabel(y_lbl)
    ax2.set_xlabel("Momento da Coleta")
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Dispersão e Reta de Regressão - Ativação Máxima vs. Fadiga
    ax3 = axes[2]
    sns.regplot(
        data=grouped_df,
        x='Pico_RMS_CIVM_uV',
        y=dep_var,
        scatter_kws={'alpha': 0.7, 's': 50},
        line_kws={'color': 'darkred', 'linewidth': 2},
        ax=ax3
    )
    corr_val = np.corrcoef(grouped_df['Pico_RMS_CIVM_uV'], grouped_df[dep_var])[0, 1]
    ax3.set_title(f"Correlação: Força vs. Fadiga\n(r = {corr_val:.3f})", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Pico RMS CIVM (uV)")
    ax3.set_ylabel(y_lbl)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Conversão para string base64 para injeção HTML
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    buf.seek(0)
    b64_img = base64.b64encode(buf.read()).decode('utf-8')
    
    return fig, b64_img

# ==============================================================================
# MÓDULO 7: LAUDO ACADÊMICO WEASYPRINT (HTML + CSS PRINT)
# ==============================================================================

def build_weasyprint_html(df_tidy: pd.DataFrame, stats_results: dict, b64_chart: str, muscle_label: str) -> str:
    """Monta o documento HTML completo com formatação CSS para impressão de laudo PDF via WeasyPrint."""
    
    aov_main_html = ""
    if 'aov_main' in stats_results:
        df_m = stats_results['aov_main'][['Source', 'SS', 'ddof1', 'ddof2', 'MS', 'F', 'p_unc', 'ng2']].copy()
        df_m.columns = ['Fonte de Variação', 'SQ', 'GL1', 'GL2', 'QM', 'F', 'p-valor', 'Eta²g']
        aov_main_html = df_m.to_html(index=False, classes='styled-table', float_format="%.4f")
        
    aov_civm_html = ""
    if 'aov_civm' in stats_results:
        df_c = stats_results['aov_civm'][['Source', 'SS', 'ddof1', 'ddof2', 'MS', 'F', 'p_unc', 'ng2']].copy()
        df_c.columns = ['Fonte de Variação', 'SQ', 'GL1', 'GL2', 'QM', 'F', 'p-valor', 'Eta²g']
        aov_civm_html = df_c.to_html(index=False, classes='styled-table', float_format="%.4f")
        
    sample_table_html = df_tidy.head(8).to_html(index=False, classes='styled-table', float_format="%.2f")
    dep_label = stats_results.get('dep_var_name', 'Fadiga Eletromiográfica')
    
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
            <div class="subtitle">Eletromiografia de Superfície: Fusão Digital CIVM & Fadiga Isométrica a 50% (SENIAM/ISEK)</div>
        </div>

        <div class="info-card">
            <b>Delineamento Experimental:</b> Estudo Cruzado com Medidas Repetidas (Condição [Controle vs. Intervenção] x Momento [Pré vs. Pós]).<br>
            <b>Músculo em Foco:</b> {muscle_label} | <b>Total de Registros Pareados:</b> {len(df_tidy)} linhas.<br>
            <b>Normalização Aplicada:</b> Normalized MDF Slope (%/s) [Mannion et al., 1997] e RMS Normalizado por %CIVM [Cram's sEMG].
        </div>

        <h2>1. Tabela de Medidas Repetidas: ANOVA Two-Way (Pingouin)</h2>
        <p><b>A) Variável Primária de Fadiga ({dep_label}):</b></p>
        {aov_main_html}

        <p><b>B) Capacidade Máxima de Força (Pico RMS da CIVM):</b></p>
        {aov_civm_html}

        <h2>2. Visualização das Interações e Correlação Força-Fadiga</h2>
        <img class="chart-img" src="data:image/png;base64,{b64_chart}" alt="Gráficos de Interação e Dispersão" />

        <div style="page-break-before: always;"></div>

        <h2>3. Amostra da Tabela Mestra Consolidada (Formato Tidy Data)</h2>
        <p>Abaixo são apresentadas as primeiras linhas do conjunto de dados unificado via chaves primárias:</p>
        {sample_table_html}

        <h2>4. Síntese Interpretativa e Fundamentação Científica</h2>
        <p style="text-align: justify; font-size: 8.5pt;">
            A análise integrada dos dados de Contração Isométrica Voluntária Máxima (CIVM) e da taxa de fadiga espectral 
            (Normalized MDF Slope, %/s) avalia se a mobilização miofascial induziu alterações na capacidade de recrutamento neuromuscular ou na resistência metabólica periférica. 
            Uma interação estatisticamente significante (Condição * Momento, p &lt; 0,05) no Pico de RMS indica modulação no drive neural central pós-intervenção. 
            Concomitantemente, uma atenuação na taxa de queda relativa da Frequência Mediana (Normalized MDF Slope mais próximo de zero) 
            sinaliza maior preservação da velocidade de condução do potencial de ação no sarcolema, decorrente de melhor perfusão tecidual ou menor acidose intramuscular local.
        </p>
    </body>
    </html>
    """
    return html_content

# ==============================================================================
# MÓDULO 8: GERADOR DE DADOS DEMO (CROSSOVER COMPLETO 8 VOLUNTÁRIOS)
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
                    
                    int_boost_rms = 48.0 if cond == 'Intervenção' and mom == 'Pós' else 0.0
                    int_atten_fatigue = 0.22 if cond == 'Intervenção' and mom == 'Pós' else 0.0
                    
                    pico_rms = base_rms + int_boost_rms + np.random.normal(0, 30.0)
                    rms_medio = pico_rms * 0.75 + np.random.normal(0, 15.0)
                    
                    mdf_slope = base_mdf_slope + int_atten_fatigue + np.random.normal(0, 0.08)
                    mdf_init = 125.0 + np.random.normal(0, 4.0)
                    mdf_final = mdf_init + (mdf_slope * 28.0)
                    mdf_pct = ((mdf_final - mdf_init) / mdf_init) * 100.0
                    norm_mdf_slope = (mdf_slope / mdf_init) * 100.0
                    
                    rms_slope = 0.35 - (0.10 if cond == 'Intervenção' and mom == 'Pós' else 0.0) + np.random.normal(0, 0.05)
                    rms_slope_pct_civm = (rms_slope / pico_rms) * 100.0
                    
                    records.append({
                        'ID_Voluntario': vol_str,
                        'Ordem_Sessao': ordem,
                        'Condicao': cond,
                        'Momento': mom,
                        'Musculo': musc,
                        'Pico_RMS_CIVM_uV': round(float(pico_rms), 2),
                        'RMS_Medio_CIVM_uV': round(float(rms_medio), 2),
                        'Ativacao_Relativa_pct_CIVM': round(float((rms_medio / pico_rms) * 100.0), 1),
                        'MDF_Inicial_Hz': round(float(mdf_init), 1),
                        'MDF_Final_Hz': round(float(mdf_final), 1),
                        'MDF_Slope_Hz_s': round(float(mdf_slope), 3),
                        'Normalized_MDF_Slope_pct_s': round(float(norm_mdf_slope), 3),
                        'Queda_MDF_pct': round(float(mdf_pct), 1),
                        'Absolute_MDF_Decrease_Hz': round(float(mdf_init - mdf_final), 1),
                        'RMS_Slope_uV_s': round(float(rms_slope), 3),
                        'RMS_Slope_pct_CIVM_s': round(float(rms_slope_pct_civm), 3),
                        'MDF_R2': round(float(np.random.uniform(0.70, 0.95)), 2),
                        'MDF_p_valor': round(float(np.random.uniform(0.001, 0.045)), 4),
                    })
    return pd.DataFrame(records)

# ==============================================================================
# INTERFACE DO USUÁRIO STREAMLIT
# ==============================================================================

st.markdown('<p class="main-title">🧬 Plataforma Master: Fusão CIVM & Fadiga sEMG</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Processamento Biomecânico Crossover com Normalização (%CIVM & %/s), Índices de Assimetria Bilateral (ASI) e Laudo Acadêmico</p>', unsafe_allow_html=True)

# Barra Lateral: Navegação e Gestão de Demonstração
st.sidebar.header("🧭 Módulos do Sistema")
nav_choice = st.sidebar.radio(
    "Navegação:",
    [
        "1. Upload & Extração Dupla",
        "2. Tabela Mestra (Fusão Tidy)",
        "3. Análise Estatística Crossover",
        "4. Efeitos Delta (Tratamento Líquido)",
        "5. Assimetria Bilateral (ASI %)",
        "6. Emissão de Laudo PDF (WeasyPrint)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Acesso Rápido de Teste")
if st.sidebar.button("Carregar Estudo Crossover Demonstrativo"):
    mock_df = generate_mock_crossover_study()
    st.session_state['df_merged_master'] = mock_df
    st.session_state['df_fatigue_master'] = mock_df[['ID_Voluntario', 'Ordem_Sessao', 'Condicao', 'Momento', 'Musculo', 'MDF_Inicial_Hz', 'MDF_Final_Hz', 'MDF_Slope_Hz_s', 'Normalized_MDF_Slope_pct_s', 'Queda_MDF_pct', 'Absolute_MDF_Decrease_Hz', 'RMS_Slope_uV_s', 'MDF_R2', 'MDF_p_valor']].copy()
    st.session_state['df_civm_master'] = mock_df[['ID_Voluntario', 'Ordem_Sessao', 'Condicao', 'Momento', 'Musculo', 'Pico_RMS_CIVM_uV', 'RMS_Medio_CIVM_uV']].copy()
    st.sidebar.success(f"Estudo Crossover carregado com {len(mock_df)} registros!")

# -----------------------------------------------------------------------------
# MÓDULO 1: UPLOAD E EXTRAÇÃO DUPLA
# -----------------------------------------------------------------------------
if nav_choice == "1. Upload & Extração Dupla":
    st.header("📂 Módulo 1: Upload e Extração Dupla de Relatórios")
    st.markdown("Faça o upload dos documentos e atribua os metadados da sessão correspondente. O sistema extrairá as métricas automaticamente.")

    col_up1, col_up2 = st.columns(2)

    with col_up1:
        st.subheader("📑 Relatórios de Fadiga (PDF)")
        files_fatigue = st.file_uploader(
            "Selecione um ou mais PDFs de Fadiga",
            type=["pdf"],
            accept_multiple_files=True,
            key="up_fatigue"
        )
        
        with st.expander("Metadados para os Relatórios de Fadiga", expanded=True):
            meta_fat_id = st.text_input("ID do Voluntário (Fadiga):", value="VOL_01", key="mf_id")
            meta_fat_ordem = st.number_input("Ordem da Sessão (Fadiga):", min_value=1, max_value=4, value=1, key="mf_ord")
            meta_fat_cond = st.selectbox("Condição (Fadiga):", ["Controle", "Intervenção"], key="mf_cond")
            meta_fat_mom = st.selectbox("Momento (Fadiga):", ["Pré", "Pós"], key="mf_mom")

        if st.button("Extrair e Adicionar Fadiga ao Banco", type="primary"):
            if files_fatigue:
                new_frames = []
                meta_dict = {'ID_Voluntario': meta_fat_id, 'Ordem_Sessao': meta_fat_ordem, 'Condicao': meta_fat_cond, 'Momento': meta_fat_mom}
                for f in files_fatigue:
                    df_extracted = parse_fatigue_pdf(f.getvalue(), meta_dict)
                    if not df_extracted.empty:
                        new_frames.append(df_extracted)
                if new_frames:
                    st.session_state['df_fatigue_master'] = pd.concat([st.session_state['df_fatigue_master']] + new_frames, ignore_index=True).drop_duplicates()
                    st.success(f"Sucesso! {len(st.session_state['df_fatigue_master'])} linhas de fadiga no banco.")
            else:
                st.warning("Selecione pelo menos um arquivo PDF de fadiga.")

    with col_up2:
        st.subheader("📊 Relatórios de CIVM (PDF ou CSV)")
        files_civm = st.file_uploader(
            "Selecione um ou mais arquivos de CIVM",
            type=["pdf", "csv", "txt"],
            accept_multiple_files=True,
            key="up_civm"
        )
        
        with st.expander("Metadados para os Relatórios de CIVM", expanded=True):
            meta_civm_id = st.text_input("ID do Voluntário (CIVM):", value="VOL_01", key="mc_id")
            meta_civm_ordem = st.number_input("Ordem da Sessão (CIVM):", min_value=1, max_value=4, value=1, key="mc_ord")
            meta_civm_cond = st.selectbox("Condição (CIVM):", ["Controle", "Intervenção"], key="mc_cond")
            meta_civm_mom = st.selectbox("Momento (CIVM):", ["Pré", "Pós"], key="mc_mom")

        if st.button("Extrair e Adicionar CIVM ao Banco", type="primary"):
            if files_civm:
                new_frames = []
                meta_dict = {'ID_Voluntario': meta_civm_id, 'Ordem_Sessao': meta_civm_ordem, 'Condicao': meta_civm_cond, 'Momento': meta_civm_mom}
                for f in files_civm:
                    df_extracted = parse_civm_input(f.getvalue(), f.name, meta_dict)
                    if not df_extracted.empty:
                        new_frames.append(df_extracted)
                if new_frames:
                    st.session_state['df_civm_master'] = pd.concat([st.session_state['df_civm_master']] + new_frames, ignore_index=True).drop_duplicates()
                    st.success(f"Sucesso! {len(st.session_state['df_civm_master'])} linhas de CIVM no banco.")
            else:
                st.warning("Selecione pelo menos um arquivo de CIVM.")

    # Visualização dos dados brutos carregados
    st.markdown("---")
    c_prev1, c_prev2 = st.columns(2)
    with c_prev1:
        st.markdown(f"**Prévia do Banco de Fadiga:** ({len(st.session_state['df_fatigue_master'])} registros)")
        if not st.session_state['df_fatigue_master'].empty:
            st.dataframe(st.session_state['df_fatigue_master'].head(6), use_container_width=True)
    with c_prev2:
        st.markdown(f"**Prévia do Banco de CIVM:** ({len(st.session_state['df_civm_master'])} registros)")
        if not st.session_state['df_civm_master'].empty:
            st.dataframe(st.session_state['df_civm_master'].head(6), use_container_width=True)

# -----------------------------------------------------------------------------
# MÓDULO 2: FUSÃO DE DADOS (DATA MERGING) E TABELA ÚNICA
# -----------------------------------------------------------------------------
elif nav_choice == "2. Tabela Mestra (Fusão Tidy)":
    st.header("🔗 Módulo 2: Fusão de Dados e Tabela Mestra Consolidada")
    st.markdown("""
    Este módulo cruza as informações de Fadiga e CIVM através das quatro chaves de alinhamento:
    **`['ID_Voluntario', 'Condicao', 'Momento', 'Musculo']`** e calcula automaticamente as variáveis normalizadas:
    - **`Normalized_MDF_Slope_pct_s`**: Taxa relativa de declínio espectral (%/s) [Mannion et al., 1997; Eur Spine J, 2026].
    - **`RMS_Slope_pct_CIVM_s`**: Taxa de recrutamento neural expressa em % da CIVM/s [Cram's Introduction to sEMG].
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
        st.markdown("### 📥 Exportar Tabela Mestra Multisseções")
        d_col1, d_col2 = st.columns(2)

        # 1. Exportação Excel com Múltiplas Abas
        buf_xlsx = io.BytesIO()
        deltas_df = calculate_crossover_deltas(df_tidy)
        asi_df = calculate_bilateral_asymmetry(df_tidy)
        
        with pd.ExcelWriter(buf_xlsx, engine='openpyxl') as writer:
            df_tidy.to_excel(writer, index=False, sheet_name='Tabela_Tidy_Master')
            if not deltas_df.empty:
                deltas_df.to_excel(writer, index=False, sheet_name='Efeitos_Delta_Crossover')
            if not asi_df.empty:
                asi_df.to_excel(writer, index=False, sheet_name='Assimetria_Bilateral_ASI')
        buf_xlsx.seek(0)
        
        d_col1.download_button(
            label="📊 Baixar Livro Completo em Excel (.xlsx)",
            data=buf_xlsx,
            file_name="estudo_crossover_semg_completo.xlsx",
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
# MÓDULO 3: ANÁLISE ESTATÍSTICA (PINGOUIN ANOVA TWO-WAY RM)
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
        c_sel1, c_sel2 = st.columns(2)
        with c_sel1:
            muscle_options = ["Todos os Músculos (Agrupado)"] + sorted(list(df_tidy['Musculo'].unique()))
            selected_muscle = st.selectbox("Selecione o Músculo para a Modelagem Estatística:", muscle_options)
        with c_sel2:
            dep_var_options = [
                'Normalized_MDF_Slope_pct_s',
                'MDF_Slope_Hz_s',
                'RMS_Slope_pct_CIVM_s',
                'Pico_RMS_CIVM_uV'
            ]
            selected_dep_var = st.selectbox("Variável Dependente Primária:", dep_var_options, index=0)

        stats_res = run_repeated_measures_statistics(df_tidy, selected_muscle, dep_var=selected_dep_var)

        st.markdown("---")
        st.subheader(f"1. ANOVA Two-Way de Medidas Repetidas: {selected_dep_var}")
        if 'aov_main' in stats_res:
            df_m = stats_res['aov_main'][['Source', 'SS', 'ddof1', 'ddof2', 'MS', 'F', 'p_unc', 'ng2']].copy()
            df_m.columns = ['Fonte de Variação', 'SQ', 'GL1', 'GL2', 'QM', 'F', 'p-valor', 'Eta²g']
            st.dataframe(df_m.style.format({'SQ': '{:.4f}', 'QM': '{:.4f}', 'F': '{:.3f}', 'p-valor': '{:.4f}', 'Eta²g': '{:.3f}'}), use_container_width=True)
        elif 'aov_main_err' in stats_res:
            st.warning(f"Aviso no cálculo da ANOVA: {stats_res['aov_main_err']}")

        st.subheader("2. ANOVA Two-Way de Medidas Repetidas: Capacidade Máxima (Pico RMS da CIVM)")
        if 'aov_civm' in stats_res:
            df_c = stats_res['aov_civm'][['Source', 'SS', 'ddof1', 'ddof2', 'MS', 'F', 'p_unc', 'ng2']].copy()
            df_c.columns = ['Fonte de Variação', 'SQ', 'GL1', 'GL2', 'QM', 'F', 'p-valor', 'Eta²g']
            st.dataframe(df_c.style.format({'SQ': '{:.2f}', 'QM': '{:.2f}', 'F': '{:.3f}', 'p-valor': '{:.4f}', 'Eta²g': '{:.3f}'}), use_container_width=True)

        if 'post_hoc' in stats_res and not stats_res['post_hoc'].empty:
            st.subheader("3. Testes Post-Hoc Pareados (Correção de Bonferroni)")
            st.dataframe(stats_res['post_hoc'], use_container_width=True)

        # Gráficos de Interação e Dispersão
        st.markdown("---")
        st.subheader("4. Gráficos de Interação e Dispersão (Força vs. Fadiga)")
        fig, b64_img = generate_statistical_plots(stats_res['processed_data'], selected_muscle, dep_var=selected_dep_var)
        st.pyplot(fig)
        
        # Salva para uso no laudo
        st.session_state['last_chart_b64'] = b64_img
        st.session_state['last_stats_res'] = stats_res
        st.session_state['last_muscle_label'] = selected_muscle
    else:
        st.info("Tabela mestra vazia. Execute o Módulo 2 ou carregue o Estudo Demonstrativo para visualizar a modelagem estatística.")

# -----------------------------------------------------------------------------
# MÓDULO 4: ANÁLISE DE EFEITOS DELTA (CROSSOVER TREATMENT EFFECT)
# -----------------------------------------------------------------------------
elif nav_choice == "4. Efeitos Delta (Tratamento Líquido)":
    st.header("🎯 Módulo 4: Análise de Efeitos Delta (Tratamento Líquido)")
    st.markdown("""
    No delineamento crossover (2x2), a eficácia da **Mobilização Miofascial** é quantificada pela diferença das mudanças:
    $$\\Delta_{\\text{Líquido}} = \\Delta_{\\text{Intervenção}} - \\Delta_{\\text{Controle}} = (\\text{Pós}_{\\text{Int}} - \\text{Pré}_{\\text{Int}}) - (\\text{Pós}_{\\text{Ctrl}} - \\text{Pré}_{\\text{Ctrl}})$$
    """)

    df_tidy = st.session_state['df_merged_master']
    if not df_tidy.empty:
        target_delta_var = st.selectbox(
            "Selecione a Variável para a Análise de Deltas:",
            ['Normalized_MDF_Slope_pct_s', 'MDF_Slope_Hz_s', 'Pico_RMS_CIVM_uV', 'RMS_Slope_pct_CIVM_s']
        )
        
        deltas_df = calculate_crossover_deltas(df_tidy, target_var=target_delta_var)
        
        if not deltas_df.empty:
            st.dataframe(deltas_df, use_container_width=True)
            
            # Resumo por Músculo
            st.markdown("### 📊 Efeito Líquido Médio da Intervenção por Grupo Muscular")
            summary_deltas = deltas_df.groupby('Musculo').agg({
                'Delta_Controle': ['mean', 'std'],
                'Delta_Intervencao': ['mean', 'std'],
                'Efeito_Tratamento_Liquido': ['mean', 'std']
            }).reset_index()
            st.dataframe(summary_deltas, use_container_width=True)
            
            # Gráfico de Barras dos Deltas
            fig_delta, ax_delta = plt.subplots(figsize=(12, 5))
            melted_deltas = deltas_df.melt(
                id_vars=['Musculo'],
                value_vars=['Delta_Controle', 'Delta_Intervencao'],
                var_name='Condicao',
                value_name='Variacao_Delta'
            )
            melted_deltas['Condicao'] = melted_deltas['Condicao'].replace({
                'Delta_Controle': 'Controle (Pós - Pré)',
                'Delta_Intervencao': 'Intervenção (Pós - Pré)'
            })
            sns.barplot(data=melted_deltas, x='Musculo', y='Variacao_Delta', hue='Condicao', ax=ax_delta, errorbar='se')
            ax_delta.set_title(f"Comparativo de Variação Intra-Sessão (Delta) | {target_delta_var}", fontweight='bold')
            ax_delta.set_ylabel(f"Delta ({target_delta_var})")
            ax_delta.set_xticklabels(ax_delta.get_xticklabels(), rotation=25, ha='right')
            ax_delta.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            plt.tight_layout()
            st.pyplot(fig_delta)
        else:
            st.warning("Não foi possível calcular os deltas. Certifique-se de que os 4 momentos (Controle Pré/Pós e Intervenção Pré/Pós) estejam presentes para os mesmos voluntários.")
    else:
        st.info("Tabela mestra vazia. Execute os módulos anteriores ou carregue o Estudo Demonstrativo.")

# -----------------------------------------------------------------------------
# MÓDULO 5: ÍNDICE DE ASSIMETRIA BILATERAL (ASI %)
# -----------------------------------------------------------------------------
elif nav_choice == "5. Assimetria Bilateral (ASI %)":
    st.header("⚖️ Módulo 5: Índice de Assimetria Bilateral (ASI %)")
    st.markdown("""
    Avalia o grau de simetria neuromuscular entre os lados direito e esquerdo do corpo durante o supino:
    $$\\text{ASI} (\\%) = \\left( \\frac{\\text{Direito} - \\text{Esquerdo}}{\\text{Direito} + \\text{Esquerdo}} \\right) \\times 100$$
    - **Sinal (+)**: Dominância do lado direito.
    - **Sinal (-)**: Dominância do lado esquerdo.
    - **0%**: Perfeita simetria funcional.
    """)

    df_tidy = st.session_state['df_merged_master']
    if not df_tidy.empty:
        asi_df = calculate_bilateral_asymmetry(df_tidy)
        if not asi_df.empty:
            st.dataframe(asi_df, use_container_width=True)
            
            # Gráfico de Boxplot / Stripplot de Assimetria
            fig_asi, (ax_asi1, ax_asi2) = plt.subplots(1, 2, figsize=(14, 5))
            
            sns.boxplot(data=asi_df, x='Grupo_Muscular', y='ASI_CIVM_pct', hue='Condicao', ax=ax_asi1)
            ax_asi1.set_title("Assimetria de Força Máxima (Pico RMS CIVM)", fontweight='bold')
            ax_asi1.set_ylabel("ASI (%)")
            ax_asi1.axhline(0, color='red', linestyle='--', linewidth=1)
            
            sns.boxplot(data=asi_df, x='Grupo_Muscular', y='ASI_Fadiga_pct', hue='Condicao', ax=ax_asi2)
            ax_asi2.set_title("Assimetria de Resistência à Fadiga (MDF Slope)", fontweight='bold')
            ax_asi2.set_ylabel("ASI (%)")
            ax_asi2.axhline(0, color='red', linestyle='--', linewidth=1)
            
            plt.tight_layout()
            st.pyplot(fig_asi)
        else:
            st.warning("Não foram encontrados pares bilaterais completos no banco de dados.")
    else:
        st.info("Tabela mestra vazia. Execute os módulos anteriores ou carregue o Estudo Demonstrativo.")

# -----------------------------------------------------------------------------
# MÓDULO 6: LAUDO AUTOMATIZADO WEASYPRINT
# -----------------------------------------------------------------------------
elif nav_choice == "6. Emissão de Laudo PDF (WeasyPrint)":
    st.header("📄 Módulo 6: Compilação de Laudo Automatizado (WeasyPrint)")
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
            
            st.download_button(
                label="🌐 Baixar Laudo Estilizado (.html - Pronto para Imprimir em PDF no Navegador)",
                data=html_doc.encode('utf-8'),
                file_name="laudo_crossover_civm_fadiga.html",
                mime="text/html",
                use_container_width=True
            )
    else:
        st.info("Tabela mestra vazia. Execute os módulos anteriores ou carregue o Estudo Demonstrativo para emitir o laudo.")
