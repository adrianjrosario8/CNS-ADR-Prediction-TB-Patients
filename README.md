# TB Neurological ADR Risk Predictor

> A clinical ML system predicting CNS adverse drug reaction risk in tuberculosis patients from patient-level clinical data - trained on a Ghanaian TB cohort, validated with three convergent methods, and deployed as a live Streamlit application.

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://adrianjrosario8-cns-adr-prediction-tb-patients-app-c0svjo.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python)](https://python.org)

---

## The Problem This Solves

Prior ML models built on this dataset focused exclusively on hepatotoxicity - liver-related adverse drug reactions. Neurological ADR prediction models in the literature traditionally rely on drug-level features, which limits real-world clinical applicability.

This project addresses a dual literature gap: predicting CNS-specific ADRs in TB patients using patient-level clinical variables, making it directly applicable for clinical decision support during treatment.

---

## Model Performance

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.951 |
| Nested CV AUC | 0.91 (SD: ±0.03) |
| Repeated Stratified K-Fold AUC | 0.91 |
| Stratified K-Fold AUC | 0.90 |
| Recall (High Risk) | 0.833 |
| F1 Score | 0.889 |
| Average Precision | 0.92 |
| Brier Score | 0.132 |

**Why Recall is the headline metric:** In clinical pharmacovigilance, missing a high-risk patient is far more costly than a false positive. The model is explicitly optimised for recall on high-risk cases - the appropriate trade-off for a safety-critical ADR screening tool.

**Validation approach:** Despite a small dataset (311 patients), AUC 0.951 was confirmed across three independent methods - Nested CV, Stratified K-Fold, and Repeated Stratified K-Fold - with consistent standard deviations. Convergence across three methods on a small clinical cohort is strong evidence of a stable, non-overfit model.

**Small dataset context:** 311 samples is common in clinical pharmacovigilance research due to data access restrictions and the cost of clinical data collection. Rigorous multi-method validation was specifically chosen to address this constraint.

---

## Example Output

```
Risk Classification:    HIGH RISK
CNS ADR Risk Probability:  78.3%

Clinical Risk Factors Identified:
- HIV-positive status may increase susceptibility to CNS adverse reactions
- Gastrointestinal side effects may indicate broader medication intolerance
- Hearing-related side effects may reflect broader neurological toxicity
- Longer treatment duration may increase cumulative neurotoxicity exposure
```

---

## Clinical Features

| Feature | Clinical Rationale |
|---------|-------------------|
| Gastrointestinal ADRs | Systemic drug intolerance signal - associated with broader ADR risk |
| Genitourinary reactions | Broad drug sensitivity pattern linked to CNS ADR risk |
| Alcohol consumption | Potentiates drug neurotoxicity, impairs hepatic drug metabolism |
| HIV status | Immune dysregulation increases CNS ADR susceptibility |
| Weight change | Nutritional deficiency proxy affecting drug pharmacokinetics |
| Skin reactions | Hypersensitivity response that may extend to neurological tissue |
| Audiologic reactions | Early indicator of aminoglycoside neurotoxicity |
| Total treatment duration | Cumulative drug exposure and neurological ADR risk |
| Continuation phase duration | Extended TB treatment phase exposure |
| Age | Physiological risk modifier for drug clearance |
| TB diagnostic test | Clinical severity and disease classification indicator |
| Baseline PCS-12 | Physical health reserve at treatment initiation |

---

## SHAP Analysis

SHAP analysis identified the most influential features driving CNS ADR predictions:

- Gastrointestinal ADRs
- Genitourinary reactions
- Alcohol consumption
- Treatment duration

These findings reinforce that patient-level clinical factors - not drug properties alone - are critical predictors of neurological adverse drug reactions during TB treatment.

---

## Run Locally

```bash
git clone https://github.com/adrianjrosario8/CNS-ADR-Prediction-TB-Patients.git
cd CNS-ADR-Prediction-TB-Patients
pip install -r requirements.txt
streamlit run app.py
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Model | AdaBoost (tuned hyperparameters), scikit-learn |
| Data Processing | pandas, NumPy |
| Interpretation | SHAP |
| Validation | Nested CV, Stratified K-Fold, Repeated Stratified K-Fold, Brier Score |
| Frontend | Streamlit |

---

## Model Selection

Multiple models were evaluated including Logistic Regression, Random Forest, XGBoost, and Gradient Boosting. AdaBoost with tuned hyperparameters was selected as the final model for its best balance between recall and precision, strongest generalisation performance, and minimal overfitting on the small clinical cohort.

---

## What Differentiates This Project

- **Dual literature gap:** First ML model predicting CNS-specific ADRs in TB patients from patient-level EHR features - prior work focused on hepatotoxicity, not neurological outcomes.
- **Patient-level features:** Unlike existing neurological ADR models that rely on drug properties, this system uses clinical variables directly available in patient records - making it actionable in real-world settings.
- **Rigorous validation on small data:** Three-method convergence at AUC 0.951 on 311 patients demonstrates robustness specifically designed to address small clinical cohort constraints.
- **Clinical reasoning layer:** Predictions are accompanied by structured clinical explanations grounded in pharmacovigilance evidence - not black-box scores.
- **Recall-optimised design:** Model architecture reflects clinical priorities, not just benchmark metrics.

---

## Clinical and Business Outcomes

- Early identification of high-risk TB patients for neurological adverse drug reactions
- Proactive patient monitoring and targeted intervention during treatment
- Reduced long-term treatment costs by preventing severe adverse events requiring hospitalisation
- Improved resource allocation in resource-limited clinical settings
- Data-driven decision support that limits unnecessary interventions

---

## Limitations and Future Work

- Dataset size (311 patients) is a known constraint common in clinical pharmacovigilance research - multi-method validation was specifically chosen to address this
- External validation on larger, multi-centre TB cohorts would strengthen generalisability
- Future improvements: FastAPI inference endpoint, MLflow experiment tracking, FHIR-compatible data ingestion, batch patient cohort processing

---

## App Preview

<img width="1920" height="795" alt="Screenshot (48)" src="https://github.com/user-attachments/assets/07a0ebd9-192f-49ae-bd40-ddffe60183d9" />

<img width="1920" height="753" alt="Screenshot (49)" src="https://github.com/user-attachments/assets/6be9baef-bc29-4122-ad43-34af9c71feca" />

---

## Clinical Note

This tool is intended for **clinical decision support only** and does not replace professional medical judgment. All predictions should be interpreted in the context of full patient assessment by a qualified clinician. Model trained on a Ghanaian TB patient cohort - validate against your local patient population before clinical use.

---

## Author

**Adrian Jacob Rosario**
MS Pharmaceutical Sciences - Pharmacometrics & Systems Pharmacology, University of Pittsburgh

Building end-to-end pharmacovigilance ML systems at the intersection of pharmaceutical research and production ML engineering.

[GitHub Portfolio](https://github.com/adrianjrosario8) | [LinkedIn](https://www.linkedin.com/in/adrian-jacob-rosario-330a47235/)
