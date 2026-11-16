{\rtf1\ansi\ansicpg1251\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\paperw11900\paperh16840\margl1440\margr1440\vieww31680\viewh21220\viewkind0
\deftab720
\pard\pardeftab720\sl368\sa213\partightenfactor0

\f0\fs32 \cf2 \expnd0\expndtw0\kerning0
# AcademicWeight.AI: Multi-Factor Text Markedness & Academic Discourse Analysis\
\
**AcademicWeight.AI** is an analytical NLP toolkit and web application designed to evaluate the syntactic dependency balance, lexical-stylistic density, and academic discourse alignment of research texts (abstracts, introductions, and papers). \
\
The repository includes an interactive **Streamlit** application for authors, as well as a complete research pipeline for harvesting scholarly metadata from **OpenAlex**, computing 7 dimensions of linguistic markedness, and running non-parametric statistical validation.\
\
---\
\
## \uc0\u55357 \u56523  Key Features & Linguistic Metrics\
\
The system evaluates English (`en_core_web_sm`) and Russian (`ru_core_news_md`) texts across **7 core dimensions of linguistic markedness** alongside syntactic dependency metrics:\
\
1. **Material Markedness (`Mmat`)**: Density of long polysyllabic terms (`> 8` characters).\
2. **Semantic Markedness (`Msem`)**: Nominal density (proportion of `NOUN` tokens).\
3. **Implicational Markedness (`Mimp`)**: Hierarchical syntactic depth (proportion of dependent tokens opening new valencies).\
4. **Abstract-Syntactic Markedness (`Msyn` / `MR`)**: Ratio of complex Universal Dependencies relations (`acl`, `advcl`, `parataxis`, `dislocated`, `appos`, `conj`).\
5. **Productive Markedness (`Mprod`)**: Lexical variation measured via Lemma Type-Token Ratio (TTR).\
6. **Stylistic Markedness (`Mstyl`)**: Density of qualifying modifiers (`ADJ` and `ADV` tags).\
7. **Statistical Markedness (`Mstat`)**: Concentration of extra-long specialized composites (`> 10` characters).\
\
### Predictive Academic Standard Index (PASI)\
**PASI** (`5.0% \'96 100.0%`) is a composite index calibrated across 6 academic discipline profiles (Humanities, Natural Sciences, Engineering & IT, Medical Sciences, Agricultural Sciences, and Social Sciences):\
* **Syntactic Balance (40% weight):** Gaussian norm corridor evaluating Mean Dependency Distance (`MDD`) and Markedness Ratio (`MR`) excluding punctuation distortion.\
* **Markedness Density (60% weight):** Sigmoid-weighted score combining productive (`50%`), stylistic (`30%`), and material (`20%`) markedness against empirical discipline benchmarks.\
\
---\
\
## \uc0\u55357 \u56770 \u65039  Repository Structure\
\
* **`app.py`** *(Web Application)*: Interactive Streamlit interface for real-time text evaluation, discipline calibration, and actionable author diagnostics.\
* **`fetch_openalex.py`** *(Data Collection)*: Queries the OpenAlex API (`2021\'962023` corpus) to build balanced datasets of high-impact (`Citations \uc0\u8805  15`, `FWCI \u8805  1.5`) and uncited (`Citations = 0`) philological articles.\
* **`batch_analysis.py`** *(Batch Processing)*: Cleans raw abstracts, calculates all 7 markedness dimensions, `MDD`, `MR`, and `PASI` scores across Excel datasets, and exports results to `AcademicWeight_KKSON_v3_Report.xlsx`.\
* **`test_levene.py`** *(Statistical Validation)*: Performs median-centered Levene's tests (`levene_results.xlsx`) to check variance homogeneity between cited and uncited groups.\
* **`test_mann_whitney.py`** *(Statistical Validation)*: Runs two-sided Mann-Whitney U tests with rank-biserial correlation (`r_rb`) effect sizes and Benjamini-Hochberg FDR correction (`mann_whitney_results.xlsx`).\
* **`test_spearman.py`** *(Correlation Analysis)*: Computes Spearman's rank correlation (`rho`) with FDR adjustment between linguistic metrics and citation impact (`Citation`, `FWCI`) (`spearman_results.xlsx`).\
* **`requirements.txt`**: Python dependencies required to run the application and analytical scripts.\
\
---\
\
## \uc0\u9881 \u65039  Installation\
\
### 1. Clone the repository and install dependencies\
```bash\
pip install -r requirements.txt\
```\
*(Note: `statsmodels` is also required for FDR multiple-testing corrections in the statistical scripts: `pip install statsmodels`)*\
\
### 2. Download spaCy language models\
```bash\
python -m spacy download en_core_web_sm\
python -m spacy download ru_core_news_md\
```\
\
---\
\
## \uc0\u55357 \u56960  Usage\
\
### 1. Run the Interactive Web App\
Launch the Streamlit dashboard locally:\
```bash\
streamlit run app.py\
```\
Paste an abstract or paper excerpt (minimum 20 words), select the language (`English` or `Russian`) and academic discipline, and click **Calculate 7 Types of Markedness & Academic Weight**.\
\
### 2. Run the Research & Statistical Pipeline\
1. **Collect Corpus Data:** Set your free OpenAlex API key as an environment variable (`export OPENALEX_API_KEY="your_key"`) and run the collector to generate `OpenAlex_Philology_2021_2023.xlsx`:\
   ```bash\
   python fetch_openalex.py\
   ```\
2. **Compute Batch Metrics:** Place your input file (`text_abstract.xlsx`) in the root directory and run:\
   ```bash\
   python batch_analysis.py\
   ```\
3. **Run Statistical Tests:** Ensure your processed dataset is named `data.xlsx` (with standardized column headers matching the statistical scripts) and execute:\
   ```bash\
   python test_levene.py\
   python test_mann_whitney.py\
   python test_spearman.py\
   ```\
}