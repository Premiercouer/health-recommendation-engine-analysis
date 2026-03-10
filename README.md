# Health Recommendation Engine Analysis

Reverse-engineering a health recommendation rule engine through EDA, text matching, rule decomposition, and multi-label classification (XGBoost / RF / LR).

---

## Project Overview

This project analyzes a proprietary health recommendation dataset where an unknown rule engine maps patients' health dimension values to personalized text recommendations. The goal is to fully reconstruct the hidden logic — without access to the original rule source code — using data analysis and machine learning techniques.

**The dataset contains 8 health dimensions:**
`Nutrition` · `Obesity` · `Sleep` · `Depression` · `Wellness` · `Anti-Stress` · `Anti-Smoke` · `Movement`

For each combination of dimension values, the system outputs:
- A `recommendations` column — concatenated health tips
- A `status_assessment` column — a natural language summary of the patient's health status

> ⚠️ **Data Notice:** The original dataset is proprietary and cannot be shared. A mock sample (`data_sample.xlsx`) with identical column structure is provided for reproduction. Notebook outputs were produced on the full dataset (746,496 rows) and are preserved for reference.

---

## Repository Structure

```
├── data_sample.xlsx          # Mock dataset (identical structure, 100 rows)
├── final_base_tips.pkl       # 29 extracted base tip sentences
├── EDA.ipynb                 # Exploratory data analysis
├── text_match.ipynb          # Tip extraction & rule decomposition pipeline
├── Rule_restoration.ipynb    # Standalone rule restoration system (ABCD classification)
└── model_comparison.ipynb    # Multi-label classification: XGBoost vs RF vs LR
```

---

## Notebooks

### 1 · EDA.ipynb
Exploratory analysis of the dataset structure, revealing that the data is a **deterministic combinatorial lookup table** — not sampled data. Key findings:
- 5 of 16 range columns are constant across all rows
- The remaining 11 variable columns form a complete combinatorial expansion (3×4×3×... = 746,496 rows)
- Status distribution, improving rates, and dimension correlations are visualized across all 8 health dimensions

### 2 · text_match.ipynb
Extracts and cleans atomic recommendation sentences from raw free text, then maps each tip to its controlling dimension(s).

- **Tip Extraction:** Regex end-anchor approach to handle tips containing internal periods (e.g. `"ex. walking"`), followed by graph-based fragment merging using NetworkX
- **Rule Decomposition:** Frequency-variance analysis — if a tip's appearance rate varies significantly (variance > 10) across a dimension's value ranges, that dimension controls the tip
- **Coupling Analysis:** FacetGrid heatmaps to identify tips jointly controlled by two dimensions

### 3 · Rule_restoration.ipynb
A standalone rule restoration system that classifies all 29 base tips into four types based on their cross-dimension trigger patterns:

| Type | Description |
|------|-------------|
| **A** | Triggered exclusively by one dimension (background freq ≈ 0) |
| **B** | Modulated by one dimension but present at baseline |
| **C** | Jointly controlled by two or more dimensions |
| **D** | Constant tip — appears regardless of all dimension values (~50% rate) |

### 4 · model_comparison.ipynb
Frames recommendation prediction as a **multi-label classification** problem and compares three models:

| Model | Hamming Loss | F1 (Micro) | Training Time |
|-------|-------------|------------|---------------|
| **XGBoost** | **0.1170** | **0.7681** | — |
| Random Forest | 0.1237 | 0.7611 | 1698.5s |
| Logistic Regression | 0.1466 | 0.6984 | 188.0s |

**Feature matrix X:** 11 numeric features (midpoint-encoded health dimension values)  
**Label matrix Y:** 53 binary labels — 29 tip presence flags + 24 dimension status flags

---

## Key Technical Contributions

- Identified the dataset as a **deterministic rule-based system** rather than sampled data — a critical insight that shaped the entire analysis approach
- Designed a **regex end-anchor + graph merge pipeline** to correctly parse concatenated tip sentences with internal punctuation
- Developed a **frequency-variance framework** to reverse-engineer which health dimension controls each recommendation tip
- Built an **ABCD classification system** that fully reconstructs the rule engine's logic from output text alone

---

## Setup

```bash
pip install pandas numpy scikit-learn xgboost networkx matplotlib seaborn openpyxl scipy
```

Run notebooks in order: `EDA` → `text_match` → `Rule_restoration` → `model_comparison`

> Replace `data_sample.xlsx` with the full dataset path to reproduce the original outputs.
