import os
import re
import numpy as np
import pandas as pd
import spacy
from scipy import stats

# 1. File configuration (supports .csv and .xlsx)
INPUT_FILE = "text_abstract.xlsx"  # Specify your input file (e.g., "corpus.csv" or "text_abstract.xlsx")
OUTPUT_REPORT_XLSX = "AcademicWeight_Report.xlsx"
OUTPUT_CORPUS_CSV = "Updated_Corpus_240.csv"

print("Loading spaCy language model...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# 2. Function to clean abstracts from OpenAlex / Elsevier parsing artifacts (from paper.py)
def clean_abstract(text):
    if pd.isna(text):
        return ""
    t = str(text).strip()

    # Remove language tags such as [EN] and leading special characters
    t = re.sub(r'^\[[A-Z]{2}\]\s*', '', t)
    t = re.sub(r'^[^\w\(\["\']+', '', t)

    # Remove Elsevier bullet points ('•')
    if '•' in t:
        t = t.split('•')[0].strip()

    # Remove trailing keywords and editorial notes
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


# 3. Benjamini-Hochberg (FDR) multiple comparison correction function
def benjamini_hochberg(p_values):
    p_arr = np.asarray(p_values, dtype=float)
    n = len(p_arr)
    ranked_indices = np.argsort(p_arr)
    sorted_p = p_arr[ranked_indices]
    
    # BH formula: p_adj = p * n / rank
    ranks = np.arange(1, n + 1)
    adjusted = sorted_p * n / ranks
    
    # Enforce monotonicity from right to left and cap at 1.0
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    
    result = np.empty(n, dtype=float)
    result[ranked_indices] = adjusted
    return result


# 4. Calibration constants for the Humanities / Philology profile (synchronized with app.py)
SYNTAX_LABELS = {"acl", "advcl", "parataxis", "dislocated", "appos", "conj"}

MU_MDD, SIGMA_MDD = 2.42, 0.35
MU_SYN, SIGMA_SYN = 0.088, 0.038
MU_IMP, SIGMA_IMP = 0.485, 0.050

TARGET_PROD = 0.565
TARGET_STYL = 0.135
TARGET_MAT  = 0.105
TARGET_STAT = 0.090
MU_SEM, SIGMA_SEM = 0.295, 0.050


def analyze_abstract_7metrics(raw_text):
    text = clean_abstract(raw_text)
    if len(text.split()) < 20:
        return [np.nan] * 11

    doc = nlp(text)
    tokens_no_punct = [t for t in doc if not t.is_punct and not t.is_space]
    total_tokens = len(tokens_no_punct)

    if total_tokens == 0:
        return [np.nan] * 11

    # --- Extraction of 7 non-overlapping types of markedness ---
    # 1. Material: medium-length terms (strictly 9–10 characters)
    m_material = sum(1 for t in tokens_no_punct if 9 <= len(t.text) <= 10) / total_tokens
    # 2. Semantic: nominal density (NOUN)
    m_semantic = sum(1 for t in tokens_no_punct if t.pos_ == "NOUN") / total_tokens
    # 3. Implicational: hierarchical depth of dependent nodes with new valencies
    m_implicational = sum(
        1 for t in tokens_no_punct if t.head != t and any(not c.is_punct for c in t.children)
    ) / total_tokens

    # 4. Abstract-Syntactic (Msyn) and Mean Dependency Distance (MDD)
    syntax_marked_cnt = 0
    total_dist = 0
    num_rel = 0

    for sent in doc.sents:
        sent_words = [t for t in sent if not t.is_punct and not t.is_space]
        word_idx_map = {t.i: idx for idx, t in enumerate(sent_words)}

        for token in sent_words:
            if token.head != token and token.head.i in word_idx_map:
                total_dist += abs(word_idx_map[token.i] - word_idx_map[token.head.i])
                num_rel += 1
                if token.dep_ in SYNTAX_LABELS:
                    syntax_marked_cnt += 1

    mdd = round(total_dist / num_rel, 3) if num_rel > 0 else 0.0
    m_syn = syntax_marked_cnt / num_rel if num_rel > 0 else 0.0

    # 5. Productive: Lemma Type-Token Ratio
    m_productive = len(set(t.lemma_.lower() for t in tokens_no_punct)) / total_tokens
    # 6. Stylistic: density of qualifiers (ADJ, ADV)
    m_stylistic = sum(1 for t in tokens_no_punct if t.pos_ in {"ADJ", "ADV"}) / total_tokens
    # 7. Statistical: extra-long compounds (>10 characters)
    m_statistical = sum(1 for t in tokens_no_punct if len(t.text) > 10) / total_tokens

    # --- Two-component PASI calculation (from app.py) ---
    # Component 1: Syntactic equilibrium (MDD + Msyn + Mimp, weight 40%)
    syn_equilibrium = np.exp(-(
        ((mdd - MU_MDD) ** 2) / (2 * (SIGMA_MDD ** 2)) +
        ((m_syn - MU_SYN) ** 2) / (2 * (SIGMA_SYN ** 2)) +
        ((m_implicational - MU_IMP) ** 2) / (2 * (SIGMA_IMP ** 2))
    ))

    # Component 2: Lexical-stylistic and statistical markedness density (weight 60%)
    s_prod = 1.0 / (1.0 + np.exp(-22.0 * (m_productive - TARGET_PROD)))
    s_styl = 1.0 / (1.0 + np.exp(-30.0 * (m_stylistic - TARGET_STYL)))
    s_mat  = 1.0 / (1.0 + np.exp(-25.0 * (m_material - TARGET_MAT)))
    s_sem  = np.exp(-((m_semantic - MU_SEM) ** 2) / (2 * (SIGMA_SEM ** 2)))
    s_stat = 1.0 / (1.0 + np.exp(-25.0 * (m_statistical - TARGET_STAT)))

    markedness_density = (
        0.35 * s_prod +
        0.25 * s_styl +
        0.15 * s_mat +
        0.15 * s_sem +
        0.10 * s_stat
    )

    pasi_raw = 100.0 * (0.40 * syn_equilibrium + 0.60 * markedness_density)
    pasi_score = round(max(5.0, min(pasi_raw, 100.0)), 1)

    return [
        pasi_score,
        round(syn_equilibrium * 100, 1),
        round(markedness_density * 100, 1),
        mdd,
        round(m_material * 100, 2),
        round(m_semantic * 100, 2),
        round(m_implicational * 100, 2),
        round(m_syn * 100, 2),
        round(m_productive * 100, 2),
        round(m_stylistic * 100, 2),
        round(m_statistical * 100, 2)
    ]


# 5. Batch execution and statistical report generation
if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File '{INPUT_FILE}' not found. Check the path in INPUT_FILE.")
        exit()

    df = pd.read_csv(INPUT_FILE) if INPUT_FILE.endswith(".csv") else pd.read_excel(INPUT_FILE)
    text_col = "Text Abstract" if "Text Abstract" in df.columns else df.columns[-1]

    # Remove duplicate MR (%) column if left over from an older annotation version
    if "MR (%)" in df.columns:
        df = df.drop(columns=["MR (%)"])

    metric_cols = [
        "PASI (%)", "Syn_Equilibrium (%)", "Mark_Density (%)",
        "MDD (tokens)",
        "Mmat (%)", "Msem (%)", "Mimp (%)", "Msyn (%)", "Mprod (%)", "Mstyl (%)", "Mstat (%)"
    ]

    print(f"Running 7-factor analysis for {len(df)} abstracts...")
    results = df[text_col].apply(analyze_abstract_7metrics)
    df[metric_cols] = pd.DataFrame(results.tolist(), index=df.index)

    # Save updated corpus to CSV and XLSX
    df.to_csv(OUTPUT_CORPUS_CSV, index=False)

    # --- STATISTICAL ANALYSIS ---
    if "Group (Factor)" in df.columns:
        valid_df = df.dropna(subset=["PASI (%)"])
        high = valid_df[valid_df["Group (Factor)"] == "High_Status"]
        low = valid_df[valid_df["Group (Factor)"] == "Low_Status"]

        # 1. Between-group comparison: Levene + Mann-Whitney + FDR (Benjamini-Hochberg)
        mw_rows = []
        raw_p_values = []

        for col in metric_cols:
            u_stat, p_u = stats.mannwhitneyu(high[col], low[col], alternative="two-sided")
            lev_stat, p_lev = stats.levene(high[col], low[col])
            raw_p_values.append(p_u)
            mw_rows.append({
                "Metric": col,
                "High_Mean": round(high[col].mean(), 2),
                "High_SD": round(high[col].std(), 2),
                "High_Median": round(high[col].median(), 2),
                "High_IQR": round(high[col].quantile(0.75) - high[col].quantile(0.25), 2),
                "Low_Mean": round(low[col].mean(), 2),
                "Low_SD": round(low[col].std(), 2),
                "Low_Median": round(low[col].median(), 2),
                "Low_IQR": round(low[col].quantile(0.75) - low[col].quantile(0.25), 2),
                "Levene_F": round(lev_stat, 4),
                "Levene_p": round(p_lev, 4),
                "MannWhitney_U": round(u_stat, 1),
                "MannWhitney_p_raw": p_u
            })

        fdr_p_values = benjamini_hochberg(raw_p_values)
        for idx, row in enumerate(mw_rows):
            row["MannWhitney_p_FDR"] = fdr_p_values[idx]
            row["Significant_FDR_0.05"] = "Yes" if fdr_p_values[idx] < 0.05 else "No"

        df_group_stats = pd.DataFrame(mw_rows)

        print(f"\n=== BETWEEN-GROUP COMPARISON (High n={len(high)} vs Low n={len(low)}) ===")
        print(f"{'Metric':22s} | {'High (M ± SD)':15s} | {'Low (M ± SD)':15s} | {'Levene p':9s} | {'U-test p':9s} | {'FDR p':9s}")
        print("-" * 96)
        for _, r in df_group_stats.iterrows():
            sig_mark = "*" if r["MannWhitney_p_FDR"] < 0.05 else " "
            print(f"{r['Metric']:22s} | {r['High_Mean']:6.2f} ± {r['High_SD']:5.2f} | "
                  f"{r['Low_Mean']:6.2f} ± {r['Low_SD']:5.2f} | {r['Levene_p']:9.4f} | "
                  f"{r['MannWhitney_p_raw']:9.4f} | {r['MannWhitney_p_FDR']:8.4f}{sig_mark}")

        # 2. Spearman correlation analysis within the High_Status group
        corr_targets = [c for c in ["Citation", "FWCI"] if c in high.columns]
        corr_rows = []

        if corr_targets:
            print(f"\n=== SPEARMAN CORRELATIONS WITHIN HIGH_STATUS (n={len(high)}) ===")
            for target in corr_targets:
                target_p_raw = []
                temp_rows = []
                for col in metric_cols:
                    sub = high[[col, target]].dropna()
                    rho, p_val = stats.spearmanr(sub[col], sub[target])
                    target_p_raw.append(p_val)
                    temp_rows.append({
                        "Target_Indicator": target,
                        "Linguistic_Metric": col,
                        "Spearman_rho": round(rho, 4),
                        "p_value_raw": p_val
                    })
                target_p_fdr = benjamini_hochberg(target_p_raw)
                for idx, tr in enumerate(temp_rows):
                    tr["p_value_FDR"] = target_p_fdr[idx]
                    tr["Significant_raw_0.05"] = "Yes" if tr["p_value_raw"] < 0.05 else "No"
                    corr_rows.append(tr)
                    
            df_spearman = pd.DataFrame(corr_rows)
            for _, r in df_spearman.iterrows():
                sig_mark = "*" if r["p_value_raw"] < 0.05 else " "
                print(f"{r['Target_Indicator']:10s} vs {r['Linguistic_Metric']:22s} | "
                    f"rho = {r['Spearman_rho']:+7.4f} | p = {r['p_value_raw']:8.4f}{sig_mark} | p_FDR = {r['p_value_FDR']:8.4f}")
        else:
            df_spearman = pd.DataFrame()
        # Save complete multi-sheet report to Excel
        with pd.ExcelWriter(OUTPUT_REPORT_XLSX, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Corpus_Metrics", index=False)
            df_group_stats.to_excel(writer, sheet_name="MannWhitney_FDR_Stats", index=False)
            if not df_spearman.empty:
                df_spearman.to_excel(writer, sheet_name="Spearman_HighStatus", index=False)
        print(f"\nDone! Results saved to '{OUTPUT_REPORT_XLSX}' (3 sheets) and '{OUTPUT_CORPUS_CSV}'.")
    else:
        df.to_excel(OUTPUT_REPORT_XLSX, index=False)
        print(f"\nDone! Metrics saved to '{OUTPUT_REPORT_XLSX}' and '{OUTPUT_CORPUS_CSV}'.")
