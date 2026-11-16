import os
import re
import numpy as np
import pandas as pd
import spacy
from scipy import stats

# 1. File paths configuration
INPUT_FILE = "text_abstract.xlsx"
OUTPUT_FILE = "AcademicWeight_KKSON_v3_Report.xlsx"

print("Loading spaCy language model...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# 2. Text cleaning function to remove OpenAlex parsing artifacts
def clean_abstract(text):
    if pd.isna(text):
        return ""
    t = str(text).strip()
    
    # Remove leading non-word characters and language tags like [EN]
    t = re.sub(r'^\[[A-Z]{2}\]\s*', '', t)
    t = re.sub(r'^[^\w\(\["\']+', '', t)
    
    # Strip Elsevier-style bullet highlights starting with '•'
    if '•' in t:
        t = t.split('•')[0].strip()
        
    # Strip trailing keywords and editorial notes
    cut_markers = [
        r'\bKeywords\s*:', 
        r'\bKey words\s*:', 
        r'\bWHAT THIS PAPER ADDS\s*:',
        r'\bSupplemental data for this article',
        r'\bSUPPLEMENTAL MATERIAL\s*:',
        r'\bCopyright\s*©'
    ]
    for marker in cut_markers:
        t = re.split(marker, t, flags=re.IGNORECASE)[0].strip()
        
    t = re.sub(r'\s+', ' ', t)
    return t

# 3. Model parameters (calibrated for Humanities / Philology profile)
syntax_labels = {"acl", "advcl", "parataxis", "dislocated", "appos", "conj"}

MU_MDD, SIGMA_MDD = 2.42, 0.35
MU_MR, SIGMA_MR = 0.088, 0.038
TARGET_PROD = 0.565
TARGET_STYL = 0.135
TARGET_MAT  = 0.185

def analyze_abstract(raw_text):
    text = clean_abstract(raw_text)
    if len(text.split()) < 20:
        return [np.nan] * 12
        
    doc = nlp(text)
    words = [t for t in doc if not t.is_punct and not t.is_space]
    total_words = len(words)
    if total_words == 0:
        return [np.nan] * 12

    # --- 7 Types of Linguistic Markedness ---
    m_mat = sum(1 for t in words if len(t.text) > 8) / total_words
    m_sem = sum(1 for t in words if t.pos_ == "NOUN") / total_words
    m_imp = sum(1 for t in words if t.head != t and any(not c.is_punct for c in t.children)) / total_words
    m_prod = len(set(t.lemma_.lower() for t in words)) / total_words
    m_styl = sum(1 for t in words if t.pos_ in {"ADJ", "ADV"}) / total_words
    m_stat = sum(1 for t in words if len(t.text) > 10) / total_words

    # --- Exact MDD, MR, and Msyn calculation based on word tokens per sentence ---
    total_dist = 0
    num_rel = 0
    syntax_marked_cnt = 0

    for sent in doc.sents:
        sent_words = [t for t in sent if not t.is_punct and not t.is_space]
        word_idx_map = {t.i: idx for idx, t in enumerate(sent_words)}
        
        for t in sent_words:
            if t.head != t and t.head.i in word_idx_map:
                dist = abs(word_idx_map[t.i] - word_idx_map[t.head.i])
                total_dist += dist
                num_rel += 1
                if t.dep_ in syntax_labels:
                    syntax_marked_cnt += 1

    mdd = round(total_dist / num_rel, 3) if num_rel > 0 else 0.0
    mr = (syntax_marked_cnt / num_rel) if num_rel > 0 else 0.0
    m_syn = mr

    # --- Two-Component PASI Calculation (aligned with apa.py) ---
    # Component A: Syntactic Dependency Equilibrium (40% weight)
    syn_equilibrium = np.exp(- ( ((mdd - MU_MDD) ** 2) / (2 * (SIGMA_MDD ** 2)) + ((mr - MU_MR) ** 2) / (2 * (SIGMA_MR ** 2)) ))
    
    # Component B: Lexical-Stylistic & Terminological Density (60% weight)
    s_prod = 1.0 / (1.0 + np.exp(-22.0 * (m_prod - TARGET_PROD)))
    s_styl = 1.0 / (1.0 + np.exp(-30.0 * (m_styl - TARGET_STYL)))
    s_mat  = 1.0 / (1.0 + np.exp(-20.0 * (m_mat  - TARGET_MAT)))
    markedness_density = 0.50 * s_prod + 0.30 * s_styl + 0.20 * s_mat
    
    pasi_raw = 100.0 * (0.40 * syn_equilibrium + 0.60 * markedness_density)
    pasi_score = round(max(5.0, min(pasi_raw, 100.0)), 1)

    return [
        pasi_score,
        round(syn_equilibrium * 100, 1),
        round(markedness_density * 100, 1),
        mdd,
        round(mr * 100, 2),
        round(m_mat * 100, 2),
        round(m_sem * 100, 2),
        round(m_imp * 100, 2),
        round(m_syn * 100, 2),
        round(m_prod * 100, 2),
        round(m_styl * 100, 2),
        round(m_stat * 100, 2)
    ]

# 4. Batch Execution & Statistical Report Generation
if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File '{INPUT_FILE}' not found. Check the filename in INPUT_FILE.")
        exit()

    df = pd.read_excel(INPUT_FILE)
    text_col = 'Text Abstract' if 'Text Abstract' in df.columns else df.columns[-1]
    
    cols = [
        'PASI_v3 (%)', 'Syn_Balance(%)', 'Mark_Density (%)',
        'MDD_clean (tokens)', 'MR_clean (%)', 
        'Mmat (%)', 'Msem (%)', 'Mimp (%)', 'Msyn (%)', 'Mprod (%)', 'Mstyl (%)', 'Mstat (%)'
    ]
    
    results = df[text_col].apply(analyze_abstract)
    df[cols] = pd.DataFrame(results.tolist(), index=df.index)
    
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Report successfully saved: {OUTPUT_FILE}\n")

    if 'Group (Factor)' in df.columns:
        valid_df = df.dropna(subset=['PASI_v3 (%)'])
        high = valid_df[valid_df['Group (Factor)'] == 'High_Status']
        low = valid_df[valid_df['Group (Factor)'] == 'Low_Status']
        
        print(f"=== GROUP COMPARISON (High n={len(high)} vs Low n={len(low)}) ===")
        print(f"{'Metric':22s} | {'High (M ± SD)':16s} | {'Low (M ± SD)':16s} | {'Mann-Whitney p':14s} | {'Levene p (Variance)'}")
        print("-" * 92)
        for col in cols:
            u_stat, p_u = stats.mannwhitneyu(high[col], low[col], alternative='two-sided')
            lev_stat, p_lev = stats.levene(high[col], low[col])
            print(f"{col:22s} | {high[col].mean():6.2f} ± {high[col].std():5.2f} | "
                  f"{low[col].mean():6.2f} ± {low[col].std():5.2f} | p = {p_u:8.4f}   | p = {p_lev:8.4f}")
