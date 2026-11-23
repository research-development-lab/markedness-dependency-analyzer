import streamlit as st
import spacy
import urllib.request
import zipfile
import os
import pandas as pd
import numpy as np

# 1. Reliable caching and direct extraction to /tmp
@st.cache_resource
def load_spacy_model(lang_code):
    model_name = "en_core_web_sm" if lang_code == "en" else "ru_core_news_md"
    version = "3.8.0"

    extract_dir = f"/tmp/{model_name}"
    model_path = os.path.join(extract_dir, model_name, f"{model_name}-{version}")

    if not os.path.exists(model_path):
        with st.spinner(f"Downloading {model_name} (one-time setup)..."):
            whl_path = f"/tmp/{model_name}.whl"
            parts = [
                "https://", "github", ".com/", "explosion/", "spacy-models/",
                "releases/", "download/", f"{model_name}-{version}/",
                f"{model_name}-{version}-py3-none-any.whl",
            ]
            url = "".join(parts)
            urllib.request.urlretrieve(url, whl_path)
            with zipfile.ZipFile(whl_path, "r") as zf:
                zf.extractall(extract_dir)
            os.remove(whl_path)

    return spacy.load(model_path)

# 2. UI Configuration
st.set_page_config(page_title="AcademicWeight.AI", layout="wide")
st.title("AcademicWeight.AI: Multi-Factor Text Markedness Analysis")
st.write("Analytical system for academic discourse alignment, syntactic dependency balance, and 7-dimensional markedness scoring.")

text_input = st.text_area(
    "Paste the abstract, introduction & Literature Review or full text of your research paper here:",
    value="",
    placeholder="Paste your academic abstract, introduction, or manuscript excerpt here (in English or Russian)...",
    height=180
)

col_config_1, col_config_2, col_config_3 = st.columns([1, 1.5, 1.3])
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

with col_config_3:
    text_scope = st.selectbox(
        "Select text section / scope:",
        [
            "Abstract (150–350 words)",
            "Introduction & Literature Review (700–3,000 words)",
            "Full Text / Clean Sections (700–7,000 words)"
        ]
    )

# 3. Main Analysis Execution Engine
if st.button("Calculate 7 Types of Markedness & Academic Weight"):
    if len(text_input.strip().split()) < 20:
        st.warning("Please enter at least 20 words for reliable syntactic and lexical analysis.")
        st.stop()

    nlp = load_spacy_model(language_opt)
    doc = nlp(text_input)
    
    # Filter strictly alphabetic word tokens (excludes numbers, URLs, DOIs, citations)
    tokens_words = [t for t in doc if t.is_alpha]
    total_tokens = len(tokens_words)
    
    if total_tokens < 20:
        st.warning("Please enter at least 20 valid words for reliable syntactic and lexical analysis.")
        st.stop()

    # Soft word-count validation against empirical corpus ranges
    if "Abstract" in text_scope and total_tokens > 400:
        st.warning(
            f"⚠️ Word count notice: You entered {total_tokens} words, which exceeds a typical Abstract (150–350 words). "
            "Consider switching the scope to 'Introduction / Literature Review' or 'Full Text' for accurate calibration."
        )
    elif "Introduction" in text_scope and (total_tokens < 500 or total_tokens > 3500):
        st.warning(
            f"⚠️ Word count notice: Your text has {total_tokens} words (corpus benchmark for Intro/LitReview is 700–3,000 words, centered at ~1,200 words)."
        )
    elif "Full Text" in text_scope and (total_tokens < 700 or total_tokens > 7000):
        st.warning(
            f"⚠️ Word count notice: Your text has {total_tokens} words (recommended Full Text scope is 700–7,000 words)."
        )

    # Language-specific word length thresholds
    if language_opt == "en":
        mat_min, mat_max = 9, 10
        stat_min = 10
        mat_label = "9–10 characters"
        stat_label = ">10 characters"
    else:
        mat_min, mat_max = 10, 12
        stat_min = 12
        mat_label = "10–12 characters"
        stat_label = ">12 characters"
    
    # --- Extraction of 7 Core Non-Overlapping Metrics ---
    # 1. Material: medium-long academic terms
    m_material = sum(1 for t in tokens_words if mat_min <= len(t.text) <= mat_max) / total_tokens
    # 2. Semantic: nominal density (NOUN tag)
    m_semantic = sum(1 for t in tokens_words if t.pos_ == "NOUN") / total_tokens
    # 3. Implicational: hierarchical valency depth
    m_implicational = sum(
        1 for t in tokens_words if t.head != t and any(c.is_alpha for c in t.children)
    ) / total_tokens
    
    # 4. Abstract-Syntactic (Msyn / MR) and MDD
    syntax_labels = {"acl", "advcl", "parataxis", "dislocated", "appos", "conj"}
    syntax_marked_cnt = 0
    total_dist = 0
    num_rel = 0
    
    for sent in doc.sents:
        sent_words = [t for t in sent if t.is_alpha]
        word_idx_map = {t.i: idx for idx, t in enumerate(sent_words)}
        
        for token in sent_words:
            if token.head != token and token.head.i in word_idx_map:
                total_dist += abs(word_idx_map[token.i] - word_idx_map[token.head.i])
                num_rel += 1
                if token.dep_ in syntax_labels:
                    syntax_marked_cnt += 1
                
    mdd = round(total_dist / num_rel, 3) if num_rel > 0 else 0.0
    m_abstract_syntax = syntax_marked_cnt / num_rel if num_rel > 0 else 0.0
    
    # 5. Productive: Lemma MATTR (window=50 for short Abstracts; window=200 for Sections & Full Text)
    lemmas = [t.lemma_.lower() for t in tokens_words]
    window_size = 50 if "Abstract" in text_scope else 200

    if total_tokens <= window_size:
        m_productive = len(set(lemmas)) / total_tokens
    else:
        ttr_windows = [
            len(set(lemmas[i : i + window_size])) / window_size
            for i in range(total_tokens - window_size + 1)
        ]
        m_productive = sum(ttr_windows) / len(ttr_windows)

    # 6. Stylistic: qualifying modifiers (ADJ, ADV)
    m_stylistic = sum(1 for t in tokens_words if t.pos_ in ["ADJ", "ADV"]) / total_tokens
    # 7. Statistical: extra-long specialized composites
    m_statistical = sum(1 for t in tokens_words if len(t.text) > stat_min) / total_tokens
    
    # 4. Empirically calibrated discipline profiles (Base = Full Text / Clean Sections)
    if "Natural Sciences" in profile:
        mu_mdd, mu_syn, mu_imp = 2.15, 0.075, 0.470
        opt_mdd_min, opt_mdd_max = 1.85, 2.45
        target_prod, target_styl, target_mat, target_stat, mu_sem = 0.54, 0.125, 0.115, 0.095, 0.310
        prof_name = "natural sciences"
    elif "Engineering" in profile:
        mu_mdd, mu_syn, mu_imp = 2.10, 0.070, 0.465
        opt_mdd_min, opt_mdd_max = 1.80, 2.40
        target_prod, target_styl, target_mat, target_stat, mu_sem = 0.53, 0.120, 0.110, 0.090, 0.315
        prof_name = "engineering and IT"
    elif "Medical" in profile:
        mu_mdd, mu_syn, mu_imp = 2.20, 0.080, 0.475
        opt_mdd_min, opt_mdd_max = 1.90, 2.50
        target_prod, target_styl, target_mat, target_stat, mu_sem = 0.55, 0.130, 0.120, 0.100, 0.305
        prof_name = "medical and health sciences"
    elif "Agricultural" in profile:
        mu_mdd, mu_syn, mu_imp = 2.25, 0.080, 0.475
        opt_mdd_min, opt_mdd_max = 1.95, 2.55
        target_prod, target_styl, target_mat, target_stat, mu_sem = 0.54, 0.125, 0.105, 0.085, 0.300
        prof_name = "agricultural sciences"
    elif "Social Sciences" in profile:
        mu_mdd, mu_syn, mu_imp = 2.38, 0.085, 0.480
        opt_mdd_min, opt_mdd_max = 2.08, 2.68
        target_prod, target_styl, target_mat, target_stat, mu_sem = 0.56, 0.135, 0.105, 0.085, 0.295
        prof_name = "social sciences"
    else:
        mu_mdd, mu_syn, mu_imp = 2.42, 0.088, 0.485
        opt_mdd_min, opt_mdd_max = 2.12, 2.72
        target_prod, target_styl, target_mat, target_stat, mu_sem = 0.565, 0.135, 0.105, 0.090, 0.295
        prof_name = "humanities (Philology)"

    # 4b. Calibration adjustments based on Text Scope & Empirical Corpus Benchmarks
    if "Introduction" in text_scope:
        scope_label = "Introduction / Literature Review"
        if "Humanities" in profile:
            # Exact empirical calibration from OpenAlex_Philology_IntroLitReview_EN_2021_2023 (Cited M)
            mu_mdd = 2.540
            opt_mdd_min, opt_mdd_max = 2.25, 2.83
            mu_syn = 0.1116       # Cited M = 11.16%
            mu_imp = 0.4683       # Cited M = 46.83%
            mu_sem = 0.2517       # Cited M = 25.17%
            target_prod = 0.585   # Threshold calibrated against Cited M (60.42%) vs Uncited M (57.86%)
            target_styl = 0.125   # Threshold calibrated against Cited M (12.96%) vs Uncited M (12.12%)
            target_mat = 0.100    # Threshold calibrated against Cited M (10.21%) vs Uncited M (9.85%)
            target_stat = 0.072   # Threshold calibrated against Cited M (7.33%) vs Uncited M (6.92%)
        else:
            mu_mdd += 0.12
            opt_mdd_min += 0.12
            opt_mdd_max += 0.12
            mu_syn += 0.020
            mu_sem -= 0.035
            target_prod += 0.020
            target_stat -= 0.015

    elif "Abstract" in text_scope:
        scope_label = "Abstract"
        # Realistic calibration for Abstracts (150-350 words) with MATTR window=50
        mu_sem += 0.015
        target_prod = 0.650
    else:
        scope_label = "Full Text / Clean Sections"
   
    # 4c. Language-specific calibration (Russian exhibits longer MDD arcs and higher Msyn)
    if language_opt == "ru":
        mu_mdd += 0.20
        opt_mdd_min += 0.20
        opt_mdd_max += 0.20
        mu_syn += 0.015
        mu_sem += 0.015  # Higher share of deverbal nouns in Russian academic prose

    # 5. Full 7-Metric Two-Component PASI Calculation
    sigma_mdd = 0.35
    sigma_syn = 0.038
    sigma_imp = 0.050
    sigma_sem = 0.050

    # Component 1: Syntactic Balance (MDD + Msyn + Mimp)
    syn_balance = np.exp(-(
        ((mdd - mu_mdd) ** 2) / (2 * (sigma_mdd ** 2)) +
        ((m_abstract_syntax - mu_syn) ** 2) / (2 * (sigma_syn ** 2)) +
        ((m_implicational - mu_imp) ** 2) / (2 * (sigma_imp ** 2))
    ))
    
    # Component 2: Markedness Density (Mprod + Mstyl + Mmat + Msem + Mstat)
    # Upper cognitive thresholds to penalize lexical overload / nominalization clutter
    max_styl = target_styl + 0.08
    max_stat = target_stat + 0.07

    s_prod = 1.0 / (1.0 + np.exp(-22.0 * (m_productive - target_prod)))
    s_mat  = 1.0 / (1.0 + np.exp(-25.0 * (m_material - target_mat)))
    s_sem  = np.exp(-((m_semantic - mu_sem) ** 2) / (2 * (sigma_sem ** 2)))

    # Asymmetric plateau for Stylistic Markedness (penalizes both deficit and modifier overload)
    if m_stylistic < target_styl:
        s_styl = 1.0 / (1.0 + np.exp(-30.0 * (m_stylistic - target_styl)))
    elif m_stylistic > max_styl:
        s_styl = np.exp(-((m_stylistic - max_styl) ** 2) / (2 * (0.04 ** 2)))
    else:
        s_styl = 1.0

    # Asymmetric plateau for Statistical Markedness (penalizes excessive heavy composites)
    if m_statistical < target_stat:
        s_stat = 1.0 / (1.0 + np.exp(-25.0 * (m_statistical - target_stat)))
    elif m_statistical > max_stat:
        s_stat = np.exp(-((m_statistical - max_stat) ** 2) / (2 * (0.035 ** 2)))
    else:
        s_stat = 1.0

    markedness_density = (
        0.35 * s_prod +
        0.25 * s_styl +
        0.15 * s_mat +
        0.15 * s_sem +
        0.10 * s_stat
    )
    
    pasi_raw = 100.0 * (0.40 * syn_balance + 0.60 * markedness_density)
    pasi_score = round(max(5.0, min(pasi_raw, 100.0)), 1)
    
    # 6. Smart diagnostics and actionable recommendations (All 7 Metrics + MDD)
    issues = []

    # 1. Mean Dependency Distance (MDD)
    if mdd < opt_mdd_min:
        issues.append(f"Syntactic arcs are too short (MDD = {mdd} < {round(opt_mdd_min, 2)}): sentences appear fragmented; consider combining related clauses.")
    elif mdd > opt_mdd_max:
        issues.append(f"Cognitive overload detected (MDD = {mdd} > {round(opt_mdd_max, 2)}): break down overly long dependency chains.")

    # 2. Abstract-Syntactic Markedness (Msyn)
    syn_low = max(0.0, mu_syn - sigma_syn)
    syn_high = mu_syn + sigma_syn
    if m_abstract_syntax < syn_low:
        issues.append(f"Low abstract-syntactic complexity (Msyn = {round(m_abstract_syntax * 100, 1)}% < {round(syn_low * 100, 1)}%): incorporate more subordinate clauses, appositions, or participial constructions.")
    elif m_abstract_syntax > syn_high:
        issues.append(f"Excessive syntactic subordination (Msyn = {round(m_abstract_syntax * 100, 1)}% > {round(syn_high * 100, 1)}%): simplify heavily nested clauses to improve readability.")

    # 3. Implicational Markedness (Mimp)
    imp_low = max(0.0, mu_imp - sigma_imp)
    imp_high = mu_imp + sigma_imp
    if m_implicational < imp_low:
        issues.append(f"Flat syntactic hierarchy (Mimp = {round(m_implicational * 100, 1)}% < {round(imp_low * 100, 1)}%): expand phrase valency by adding dependent modifiers to key nodes.")
    elif m_implicational > imp_high:
        issues.append(f"Over-branched syntactic tree (Mimp = {round(m_implicational * 100, 1)}% > {round(imp_high * 100, 1)}%): reduce multi-level branching within single sentences.")

    # 4. Semantic Markedness (Msem)
    sem_low = max(0.0, mu_sem - sigma_sem)
    sem_high = mu_sem + sigma_sem
    if m_semantic < sem_low:
        issues.append(f"Low nominal density (Msem = {round(m_semantic * 100, 1)}% < {round(sem_low * 100, 1)}%): increase conceptual density by using precise domain nouns.")
    elif m_semantic > sem_high:
        issues.append(f"Heavy nominalization detected (Msem = {round(m_semantic * 100, 1)}% > {round(sem_high * 100, 1)}%): replace bulky noun chains with active verbs to make the prose more dynamic.")

    # 5. Productive Markedness (Mprod)
    if m_productive < target_prod:
        issues.append(f"Low productive markedness (Mprod = {round(m_productive * 100, 1)}% < {round(target_prod * 100, 1)}%): reduce lexical repetition and diversify vocabulary.")

    # 6. Stylistic Markedness (Mstyl)
    if m_stylistic < target_styl:
        issues.append(f"Low stylistic markedness (Mstyl = {round(m_stylistic * 100, 1)}% < {round(target_styl * 100, 1)}%): add precise analytical qualifiers (adjectives/adverbs).")
    elif m_stylistic > max_styl:
        issues.append(f"Excessive modifier density (Mstyl = {round(m_stylistic * 100, 1)}% > {round(max_styl * 100, 1)}%): reduce redundant adjectives and adverbs to prevent wordiness.")

    # 7. Material Markedness (Mmat)
    if m_material < target_mat:
        issues.append(f"Low medium-term density (Mmat = {round(m_material * 100, 1)}% < {round(target_mat * 100, 1)}%): enrich academic terminology ({mat_label}).")

    # 8. Statistical Markedness (Mstat)
    if m_statistical < target_stat:
        issues.append(f"Low composite density (Mstat = {round(m_statistical * 100, 1)}% < {round(target_stat * 100, 1)}%): include domain-specific compound terms ({stat_label}).")
    elif m_statistical > max_stat:
        issues.append(f"Overload of extra-long words (Mstat = {round(m_statistical * 100, 1)}% > {round(max_stat * 100, 1)}%): replace overly heavy morphological composites with clearer phrasing.")

    # --- Status & Recommendation Assembly ---
    if pasi_score >= 72.0:
        status = f"✓ High Academic Standard (Optimal Balance & Rich 7-Factor Markedness for {prof_name} — {scope_label})"
        color_box = st.success
        if len(issues) == 0:
            recommendation = "Excellent balance across all 7 markedness dimensions and syntactic dependency mapping."
        else:
            formatted_issues = "\n".join(f"- {issue}" for issue in issues)
            recommendation = f"Strong overall academic weight, with minor areas for further lexical refinement:\n{formatted_issues}"
    elif pasi_score >= 58.0:
        status = f"⚠️ Moderate Academic Weight (Partial Imbalance in {prof_name} — {scope_label})"
        color_box = st.warning
        if issues:
            formatted_issues = "\n".join(f"- {issue}" for issue in issues)
            recommendation = f"To increase scholarly impact before submission, address the following:\n{formatted_issues}"
        else:
            recommendation = "Refine lexical diversity and keep syntactic arcs tightly balanced."
    else:
        status = f"❌ Low Academic Weight (Significant Structural or Lexical Deficit — {scope_label})"
        color_box = st.error
        formatted_issues = "\n".join(f"- {issue}" for issue in issues)
        recommendation = f"Substantial revision recommended:\n{formatted_issues}"


    # 7. UI Metrics & Report Rendering
    st.subheader(f"📊 Multi-Factor Analysis Results ({total_tokens} valid words | MATTR window = {window_size})")
    col_res_1, col_res_2, col_res_3 = st.columns([2, 1, 1])
    
    syn_min = round(max(0.0, (mu_syn - sigma_syn) * 100), 1)
    syn_max = round((mu_syn + sigma_syn) * 100, 1)

    with col_res_1:
        st.metric("Predictive Academic Standard Index (PASI)", f"{pasi_score}%")
        color_box(f"Current Status: {status}")
        st.info(f"💡 **Actionable Feedback:** {recommendation}")
        
    with col_res_2:
        st.metric("Mean Dependency Distance (MDD)", f"{mdd} tokens", help=f"Optimal corridor for {prof_name} ({scope_label}): {round(opt_mdd_min, 2)} - {round(opt_mdd_max, 2)}")
        st.metric("Abstract-Syntactic Ratio (Msyn / MR)", f"{round(m_abstract_syntax * 100, 1)}%", help=f"Optimal corridor for {prof_name} ({scope_label}): {syn_min}% - {syn_max}%")
    with col_res_3:
        st.metric("Syntactic Balance Index", f"{round(syn_balance * 100, 1)}%")
        st.metric("Markedness Density Index", f"{round(markedness_density * 100, 1)}%")

    # Syntactic & Semantic Profile
    st.subheader("📋 7-Dimensional Syntactic & Semantic Markedness Profile")

    def get_range_status(val, low, high):
        if val < low:
            return "⚠️ Low"
        elif val > high:
            return "⚠️ High"
        return "✓ Optimal"

    def get_min_status(val, target):
        return "✓ Optimal" if val >= target else "⚠️ Below target"

    markedness_data = [
        {
            "Markedness Type": "1. Material (Mmat)",
            "Value": f"{round(m_material * 100, 1)}%",
            "Target": f"≥ {round(target_mat * 100, 1)}%",
            "Status": get_min_status(m_material, target_mat),
            "Linguistic Indicator": f"Medium-long academic terms ({mat_label})"
        },
        {
            "Markedness Type": "2. Semantic (Msem)",
            "Value": f"{round(m_semantic * 100, 1)}%",
            "Target": f"{round(sem_low * 100, 1)}% – {round(sem_high * 100, 1)}%",
            "Status": get_range_status(m_semantic, sem_low, sem_high),
            "Linguistic Indicator": "Discourse nominal density (Substantivity ratio / NOUN tag)"
        },
        {
            "Markedness Type": "3. Implicational (Mimp)",
            "Value": f"{round(m_implicational * 100, 1)}%",
            "Target": f"{round(imp_low * 100, 1)}% – {round(imp_high * 100, 1)}%",
            "Status": get_range_status(m_implicational, imp_low, imp_high),
            "Linguistic Indicator": "Hierarchical depth of dependent nodes opening new valencies"
        },
        {
            "Markedness Type": "4. Abstract-Syntactic (Msyn)",
            "Value": f"{round(m_abstract_syntax * 100, 1)}%",
            "Target": f"{round(syn_low * 100, 1)}% – {round(syn_high * 100, 1)}%",
            "Status": get_range_status(m_abstract_syntax, syn_low, syn_high),
            "Linguistic Indicator": "Complex UD dependency edges (acl, advcl, parataxis, appos, conj)"
        },
        {
            "Markedness Type": "5. Productive (Mprod)",
            "Value": f"{round(m_productive * 100, 1)}%",
            "Target": f"≥ {round(target_prod * 100, 1)}%",
            "Status": get_min_status(m_productive, target_prod),
            "Linguistic Indicator": f"Lexical word-formative variation (Lemma MATTR, window={window_size})"
        },
        {
            "Markedness Type": "6. Stylistic (Mstyl)",
            "Value": f"{round(m_stylistic * 100, 1)}%",
            "Target": f"{round(target_styl * 100, 1)}% – {round(max_styl * 100, 1)}%",
            "Status": get_range_status(m_stylistic, target_styl, max_styl),
            "Linguistic Indicator": "Density of qualifying modifiers (ADJ and ADV tags)"
        },
        {
            "Markedness Type": "7. Statistical (Mstat)",
            "Value": f"{round(m_statistical * 100, 1)}%",
            "Target": f"{round(target_stat * 100, 1)}% – {round(max_stat * 100, 1)}%",
            "Status": get_range_status(m_statistical, target_stat, max_stat),
            "Linguistic Indicator": f"Concentration of extra-long specialized composites ({stat_label})"
        }

    ]
    st.table(pd.DataFrame(markedness_data))
