# AcademicWeight.AI: Multi-Factor Text Markedness & Academic Discourse Analysis

**AcademicWeight.AI** is an analytical NLP toolkit and web application designed to evaluate the syntactic dependency balance, lexical-stylistic density, and academic discourse alignment of research texts (abstracts, introductions, and full papers).

The repository includes an interactive **Streamlit** application for authors, as well as a complete research pipeline for harvesting scholarly metadata from **OpenAlex**, computing 7 dimensions of linguistic markedness, and running non-parametric statistical validations.

---

## Key Features & Linguistic Metrics

The system evaluates English (`en_core_web_sm`) and Russian (`ru_core_news_md`) texts across **7 core dimensions of linguistic markedness** alongside syntactic dependency metrics:

1. **Material Markedness (`Mmat`)**: Density of long polysyllabic terms (`> 8` characters).
2. **Semantic Markedness (`Msem`)**: Nominal density (proportion of `NOUN` tokens).
3. **Implicational Markedness (`Mimp`)**: Hierarchical syntactic depth (proportion of dependent tokens opening new valencies).
4. **Abstract-Syntactic Markedness (`Msyn` / `MR`)**: Ratio of complex Universal Dependencies relations (`acl`, `advcl`, `parataxis`, `dislocated`, `appos`, `conj`).
5. **Productive Markedness (`Mprod`)**: Lexical variation measured via Lemma Type-Token Ratio (TTR).
6. **Stylistic Markedness (`Mstyl`)**: Density of qualifying modifiers (`ADJ` and `ADV` tags).
7. **Statistical Markedness (`Mstat`)**: Concentration of extra-long specialized composites (`> 10` characters).

### Predictive Academic Standard Index (PASI)
**PASI** (`5.0%` to `100.0%`) is a composite index calibrated across 6 academic discipline profiles (Humanities, Natural Sciences, Engineering & IT, Medical Sciences, Agricultural Sciences, and Social Sciences):
* **Syntactic Balance (40% weight)**: Gaussian norm corridor evaluating Mean Dependency Distance (`MDD`) and Markedness Ratio (`MR`) excluding punctuation distortion.
* **Markedness Density (60% weight)**: Sigmoid-weighted score combining productive (`50%`), stylistic (`30%`), and material (`20%`) markedness against empirical discipline benchmarks.

---

## Repository Structure

* `app.py` — Interactive Streamlit web application for real-time text evaluation.
* `paper.py` — OpenAlex API data harvesting script for philological abstracts.
* `process_all.py` — NLP processing pipeline for computing markedness and syntactic features.
* `levene.py`, `mann_whitney.py`, `spirman.py` — Statistical validation scripts (Levene's test, Mann-Whitney U test, Spearman correlation).
* `*.xlsx` — Empirical datasets and statistical evaluation reports.
