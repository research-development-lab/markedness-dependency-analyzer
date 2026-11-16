import streamlit as st
import spacy
from spacy.cli import download
import pandas as pd
import numpy as np

# 1. Reliable caching & auto-download of language models
@st.cache_resource
def load_spacy_model(lang_code):
    model_name = "en_core_web_sm" if lang_code == "en" else "ru_core_news_md"
    try:
        return spacy.load(model_name)
    except OSError:
        with st.spinner(f"Downloading language model '{model_name}' (first run only)..."):
            download(model_name)
        return spacy.load(model_name)

# 2. UI Configuration
st.set_page_config(page_title="AcademicWeight.AI", layout="wide")
st.title("AcademicWeight.AI: Multi-Factor Text Markedness Analysis")
st.write("Analytical system for academic discourse alignment, syntactic dependency balance, and lexical-stylistic density scoring.")

text_input = st.text_area(
    "Paste the abstract, introduction or  full text of your research paper here:", 
    "This paper investigates impoliteness and value homophily in the context of YouTube-based ideological discussions beneath the videos critical towards environmental activism. Drawing on the idea of rapport management, the study finds a remarkable scale of homophily as the postings follow recurrent patterns of face and sociality rights attacks echoing the same point of view.",
    height=150
)

col_config_1, col_config_2 = st.columns(2)
with col_config_1:
    language_opt = st.selectbox(
        "Select the language of your text:", 
        ["en", "ru"], 
        format_func=lambda x: "English" if x == "en" else "Russian"
    )

with col_config_2:
    profile = st.selectbox(
        "Select your research field for model calibration:",
        [
            "1. Humanities (Philology, History, Philosophy, Arts)",
            "2. Natural Sciences (Mathematics, Physics, Chemistry, Earth Sciences)",
            "3. Engineering and IT (Engineering, Computer Science, Architecture)",
            "4. Medical and Health Sciences (Clinical Medicine, Pharmacy)",
            "5. Agricultural Sciences (Agronomy, Veterinary, Biotechnology)",
            "6. Social Sciences (Economics, Law, Sociology)"
        ]
    )

# 3. Main Analysis Execution Engine
if st.button("Calculate 7 Types of Markedness & Academic Weight"):
    nlp = load_spacy_model(language_opt)
    if nlp is None:
        st.error("Failed to initialize the NLP pipeline.")
        st.stop()
        
    doc = nlp(text_input)
    
    # Base tokens excluding punctuation and whitespace
    tokens_no_punct = [t for t in doc if not t.is_punct and not t.is_space]
    total_tokens = len(tokens_no_punct)
    
    if total_tokens < 20:
        st.warning("Please enter at least 20 words for reliable syntactic and lexical analysis.")
        st.stop()
    
    # --- Extraction of 7 Core Metrics ---
    m_material = sum(1 for t in tokens_no_punct if len(t.text) > 8) / total_tokens
    m_semantic = sum(1 for t in tokens_no_punct if t.pos_ == "NOUN") / total_tokens
    m_implicational = sum(1 for t in tokens_no_punct if t.head != t and any(not c.is_punct for c in t.children)) / total_tokens
    
    # Exact MDD and MR calculation strictly per word within each sentence (without punctuation distortion)
    syntax_labels = {"acl", "advcl", "parataxis", "dislocated", "appos", "conj"}
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
                if token.dep_ in syntax_labels:
                    syntax_marked_cnt += 1
                
    mdd = round(total_dist / num_rel, 3) if num_rel > 0 else 0.0
    mr = syntax_marked_cnt / num_rel if num_rel > 0 else 0.0
    m_abstract_syntax = mr
    
    m_productive = len(set(t.lemma_.lower() for t in tokens_no_punct)) / total_tokens
    m_stylistic = sum(1 for t in tokens_no_punct if t.pos_ in ["ADJ", "ADV"]) / total_tokens
    m_statistical = sum(1 for t in tokens_no_punct if len(t.text) > 10) / total_tokens
    
    # 4. Empirically calibrated discipline profiles (2021-2023 corpus)
    if "Natural Sciences" in profile:
        mu_mdd, mu_mr = 2.15, 0.075
        opt_mdd_min, opt_mdd_max = 1.85, 2.45
        target_prod, target_styl, target_mat = 0.54, 0.125, 0.21
        prof_name = "natural sciences"
    elif "Engineering" in profile:
        mu_mdd, mu_mr = 2.10, 0.070
        opt_mdd_min, opt_mdd_max = 1.80, 2.40
        target_prod, target_styl, target_mat = 0.53, 0.120, 0.20
        prof_name = "engineering and IT"
    elif "Medical" in profile:
        mu_mdd, mu_mr = 2.20, 0.080
        opt_mdd_min, opt_mdd_max = 1.90, 2.50
        target_prod, target_styl, target_mat = 0.55, 0.130, 0.22
        prof_name = "medical and health sciences"
    elif "Agricultural" in profile:
        mu_mdd, mu_mr = 2.25, 0.080
        opt_mdd_min, opt_mdd_max = 1.95, 2.55
        target_prod, target_styl, target_mat = 0.54, 0.125, 0.19
        prof_name = "agricultural sciences"
    elif "Social Sciences" in profile:
        mu_mdd, mu_mr = 2.38, 0.085
        opt_mdd_min, opt_mdd_max = 2.08, 2.68
        target_prod, target_styl, target_mat = 0.56, 0.135, 0.19
        prof_name = "social sciences"
    else: # Humanities / Philology (Empirical benchmark based on OpenAlex 2021-2023 corpus)
        mu_mdd, mu_mr = 2.42, 0.088
        opt_mdd_min, opt_mdd_max = 2.12, 2.72
        target_prod, target_styl, target_mat = 0.565, 0.135, 0.185
        prof_name = "humanities (Philology)"

    # 5. Two-component PASI calculation:
    # Component A: Syntactic arc balance (Gaussian norm corridor, 40% weight)
    sigma_mdd = 0.35
    sigma_mr = 0.038
    syn_balance = np.exp(- ( ((mdd - mu_mdd) ** 2) / (2 * (sigma_mdd ** 2)) + ((mr - mu_mr) ** 2) / (2 * (sigma_mr ** 2)) ))
    
    # Component B: Lexical-stylistic density (Sigmoids on significant predictors, 60% weight)
    s_prod = 1.0 / (1.0 + np.exp(-22.0 * (m_productive - target_prod)))
    s_styl = 1.0 / (1.0 + np.exp(-30.0 * (m_stylistic - target_styl)))
    s_mat  = 1.0 / (1.0 + np.exp(-20.0 * (m_material - target_mat)))
    markedness_density = 0.50 * s_prod + 0.30 * s_styl + 0.20 * s_mat
    
    pasi_raw = 100.0 * (0.40 * syn_balance + 0.60 * markedness_density)
    pasi_score = round(max(5.0, min(pasi_raw, 100.0)), 1)
    
    # 6. Smart diagnostics and actionable recommendations for the author
    issues = []
    if mdd < opt_mdd_min:
        issues.append(f"Syntactic arcs are too short (MDD = {mdd} < {opt_mdd_min}): sentences appear fragmented.")
    elif mdd > opt_mdd_max:
        issues.append(f"Cognitive overload detected (MDD = {mdd} > {opt_mdd_max}): break down overly long dependency chains.")
        
    if m_productive < target_prod:
        issues.append(f"Low productive markedness (Mprod = {round(m_productive*100, 1)}%): reduce lexical repetition and diversify vocabulary.")
    if m_stylistic < target_styl:
        issues.append(f"Low stylistic markedness (Mstyl = {round(m_stylistic*100, 1)}%): add precise analytical qualifiers (adjectives/adverbs).")
    if m_material < target_mat:
        issues.append(f"Low terminological density (Mmat = {round(m_material*100, 1)}%): strengthen domain-specific academic terminology.")

    if pasi_score >= 72.0 and len(issues) == 0:
        status = f"✓ High Academic Standard (Optimal Balance & Rich Markedness for {prof_name})"
        color_box = st.success
        recommendation = "Excellent balance between syntactic dependency mapping and lexical-stylistic density. The text meets top-tier journal expectations."
    elif pasi_score >= 58.0:
        status = f"⚠️ Moderate Academic Weight (Partial Imbalance in {prof_name})"
        color_box = st.warning
        recommendation = "To increase scholarly impact before submission: " + " ".join(issues) if issues else "Refine lexical diversity and keep syntactic arcs tightly balanced."
    else:
        status = f"❌ Low Academic Weight (Significant Structural or Lexical Deficit)"
        color_box = st.error
        recommendation = "Substantial revision recommended: " + " ".join(issues)

    # 7. UI Metrics & Report Rendering
    st.subheader("📊 Multi-Factor Analysis Results")
    col_res_1, col_res_2, col_res_3 = st.columns([2, 1, 1])
    
    with col_res_1:
        st.metric("Predictive Academic Standard Index (PASI)", f"{pasi_score}%")
        color_box(f"Current Status: {status}")
        st.info(f"💡 **Actionable Feedback:** {recommendation}")
        
    with col_res_2:
        st.metric("Mean Dependency Distance (MDD)", f"{mdd} tokens", help="Optimal corridor: 2.12 - 2.72")
        st.metric("Markedness Ratio (MR)", f"{round(mr * 100, 1)}%", help="Optimal corridor: 5.0% - 12.6%")
        
    with col_res_3:
        st.metric("Syntactic Balance Score", f"{round(syn_balance * 100, 1)}%")
        st.metric("Markedness Density Score", f"{round(markedness_density * 100, 1)}%")

    # Syntactic & Semantic Passport
        # Syntactic & Semantic Passport
    st.subheader("📋 Syntactic & Semantic Markedness Profile (7 Dimensions)")
    markedness_data = [
        {
            "Markedness Type": "1. Material (Mmat)", 
            "Value": f"{round(m_material * 100, 1)}%", 
            "Target": f"≥ {round(target_mat * 100, 1)}%", 
            "Linguistic Indicator": "Density of long polysyllabic terms (>8 characters)"
        },
        {
            "Markedness Type": "2. Semantic (Msem)", 
            "Value": f"{round(m_semantic * 100, 1)}%", 
            "Target": "~29.5%", 
            "Linguistic Indicator": "Discourse nominal density (Substantivity ratio / NOUN tag)"
        },
        {
            "Markedness Type": "3. Implicational (Mimp)", 
            "Value": f"{round(m_implicational * 100, 1)}%", 
            "Target": "~48.5%", 
            "Linguistic Indicator": "Hierarchical depth of dependent nodes opening new valencies"
        },
        {
            "Markedness Type": "4. Abstract-Syntactic (Msyn)", 
            "Value": f"{round(m_abstract_syntax * 100, 1)}%", 
            "Target": f"{round(mu_mr * 100, 1)}%", 
            "Linguistic Indicator": "Complex UD dependency edges (acl, advcl, parataxis, appos, conj)"
        },
        {
            "Markedness Type": "5. Productive (Mprod)", 
            "Value": f"{round(m_productive * 100, 1)}%", 
            "Target": f"≥ {round(target_prod * 100, 1)}%", 
            "Linguistic Indicator": "Lexical word-formative variation (Lemma Type-Token Ratio)"
        },
        {
            "Markedness Type": "6. Stylistic (Mstyl)", 
            "Value": f"{round(m_stylistic * 100, 1)}%", 
            "Target": f"≥ {round(target_styl * 100, 1)}%", 
            "Linguistic Indicator": "Density of qualifying modifiers (ADJ and ADV tags)"
        },
        {
            "Markedness Type": "7. Statistical (Mstat)", 
            "Value": f"{round(m_statistical * 100, 1)}%", 
            "Target": "≥ 9.0%", 
            "Linguistic Indicator": "Concentration of extra-long specialized composites (>10 characters)"
        }
    ]
    st.table(pd.DataFrame(markedness_data))

