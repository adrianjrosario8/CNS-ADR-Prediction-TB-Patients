# TB Neurological ADR Risk Predictor

> A clinical ML system that predicts CNS adverse drug reaction risk in tuberculosis patients from patient-level EHR data. Trained on a Ghanaian TB cohort, validated with three convergent methods, extended with a RAG-powered evidence layer, and deployed as a live Streamlit application.

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://adrianjrosario8-cns-adr-prediction-tb-patients-app-ywbjlu.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python)](https://python.org)

---

## What This Project Does

Most ML work on this dataset focused exclusively on hepatotoxicity. Neurological ADR prediction models in the literature typically rely on drug-level features, which limits how useful they are in real clinical settings.

This project fills a dual gap. It predicts CNS-specific ADRs in TB patients using patient-level clinical variables that are directly available in patient records. Then, for every high-risk prediction, a RAG pipeline retrieves relevant biomedical literature and generates a source-cited clinical rationale. The end result is an auditable decision-support tool rather than a black-box score.

---

## System Architecture

The application runs in two layers.

**Layer 1 - ML Risk Model:** An AdaBoost classifier trained on patient-level EHR features predicts CNS ADR probability. High-recall optimisation was the explicit design goal because in pharmacovigilance, missing a high-risk patient is far more costly than a false positive.

**Layer 2 - RAG Evidence Pipeline:** When a prediction crosses the high-risk threshold, a LangChain + FAISS retrieval chain queries a PubMed-embedded TB/CNS ADR corpus using Sentence Transformers embeddings. The retrieved abstracts are passed to Groq/LLaMA 3, which synthesises a literature-grounded clinical rationale with PMID citations. The risk score becomes a reasoned, traceable output.

```
Patient Input
     |
AdaBoost Risk Model -> Risk Score + Clinical Risk Factors
     | (if High Risk)
FAISS Retrieval (PubMed TB/CNS ADR corpus)
     |
LLaMA 3 via Groq -> Literature-grounded rationale + PMID citations
     |
Structured Clinical Output
```

---

## Model Performance

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.951 |
| Nested CV AUC | 0.91 (SD: +-0.03) |
| Repeated Stratified K-Fold AUC | 0.91 |
| Stratified K-Fold AUC | 0.90 |
| Recall (High Risk) | 0.833 |
| F1 Score | 0.889 |
| Average Precision | 0.92 |
| Brier Score | 0.132 |

**Why Recall is the headline metric:** Missing a high-risk patient is the failure mode that matters in pharmacovigilance. The model is explicitly optimised for recall on high-risk cases, which is the right clinical trade-off for an ADR screening tool.

**On validation with small data:** 311 patients is a realistic constraint in clinical pharmacovigilance research. Data access restrictions and collection costs make large cohorts rare in this space. AUC 0.951 confirmed across three independent validation methods with consistent standard deviations is strong evidence of a stable, non-overfit model, not an artefact of a single train/test split.

---

## Example Output

The output below is a real prediction for a 68-year-old HIV-positive patient with alcohol use, multi-system ADR history, and 12 months of TB treatment:

```
CNS ADR Risk Score:     80.54%
Classification:         HIGH RISK

Clinical Risk Factors:
- HIV infection associated with increased ADR susceptibility
- High alcohol intake associated with neurotoxicity risk
- Digestive symptoms indicate systemic intolerance
- Skin reactions suggest hypersensitivity risk
- Hearing-related symptoms suggest possible neurotoxicity
- Longer treatment increases cumulative toxicity risk
- Advanced age increases ADR susceptibility

Retrieved Evidence (PubMed):
- PMID 16725084: Tuberculous meningitis - comparative study re concurrent HIV infection
- PMID 28233512: Tuberculosis Associated with HIV Infection

Clinical Interpretation (LLaMA 3, grounded in retrieved literature):
HIV coinfection is associated with higher risk of extrapulmonary and disseminated TB,
including extrameningeal TB. HIV-infected patients are more likely to receive complex
TB treatment regimens. Regular neurological assessments and close monitoring are
recommended given this patient's risk profile.
```

The RAG layer retrieved HIV-specific TB/CNS literature for a patient whose top risk features were HIV status and alcohol use. The retrieval is semantically coherent with the clinical risk profile, not keyword matching.

---

## Clinical Features

| Feature | Clinical Rationale |
|---------|-------------------|
| Gastrointestinal ADRs | Systemic drug intolerance signal, associated with broader ADR risk |
| Genitourinary reactions | Broad drug sensitivity pattern linked to CNS ADR risk |
| Alcohol consumption | Potentiates drug neurotoxicity and impairs hepatic drug metabolism |
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

These findings reinforce that patient-level clinical factors, not drug properties alone, are what matter for predicting neurological ADRs during TB treatment. It also validates the decision to build around patient-level features rather than drug-level inputs.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| ML Model | AdaBoost (tuned), scikit-learn |
| RAG Retrieval | LangChain, FAISS |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| LLM Backend | Groq / LLaMA 3 |
| Corpus | PubMed TB/CNS ADR abstracts |
| Interpretation | SHAP |
| Validation | Nested CV, Stratified K-Fold, Repeated Stratified K-Fold, Brier Score |
| Frontend | Streamlit |
| Data Processing | pandas, NumPy |

---

## Model Selection

Logistic Regression, Random Forest, XGBoost, Gradient Boosting, and AdaBoost were all evaluated. AdaBoost with tuned hyperparameters was selected for its best balance of recall and precision, strongest generalisation on a small clinical cohort, and minimal overfitting, confirmed by convergent multi-method validation.

---

## What Differentiates This Project

**Dual literature gap:** This is the first ML model predicting CNS-specific ADRs in TB patients from patient-level EHR features. Prior work on this dataset focused on hepatotoxicity, not neurological outcomes.

**Patient-level features:** Most existing neurological ADR models rely on drug properties. This system uses clinical variables that are directly available in patient records, so it's actionable without drug-level data infrastructure.

**RAG-powered explainability:** High-risk predictions trigger a retrieval pipeline over a PubMed-indexed TB/CNS ADR corpus. The LLM rationale is grounded in source-cited literature, not generated from scratch. This was designed with regulatory defensibility in mind.

**Rigorous validation on small data:** Three-method convergence at AUC 0.951 on 311 patients. The validation approach was specifically chosen to address the small cohort constraint rather than paper over it.

**Recall-optimised design:** The model architecture reflects clinical priorities. Benchmark metrics come second.

---

## Run Locally

```bash
git clone https://github.com/adrianjrosario8/CNS-ADR-Prediction-TB-Patients.git
cd CNS-ADR-Prediction-TB-Patients
pip install -r requirements.txt
streamlit run app.py
```

A `.env` file with your Groq API key is required for the RAG layer:
```
GROQ_API_KEY=your_key_here
```

---

## Limitations and Future Work

- Dataset size (311 patients) is a known constraint common in clinical pharmacovigilance. Multi-method validation was specifically chosen to address this rather than treat it as a footnote.
- External validation on larger, multi-centre TB cohorts would strengthen generalisability.
- The RAG corpus currently covers PubMed abstracts. Full-text retrieval would improve evidence depth.
- Planned additions: FastAPI inference endpoint, MLflow experiment tracking, FHIR-compatible data ingestion, batch cohort processing.

---

## App Preview

<img width="1920" height="796" alt="image" src="https://github.com/user-attachments/assets/fca07fe8-70f6-4424-90ac-e67db1375245" />

<img width="1920" height="797" alt="image" src="https://github.com/user-attachments/assets/eb411540-bbbc-42ab-898b-8a449eb0969f" />

<img width="1920" height="698" alt="image" src="https://github.com/user-attachments/assets/7c64a615-39bf-4e4e-a7a8-c50499690938" />

<img width="1920" height="770" alt="image" src="https://github.com/user-attachments/assets/a816a5ec-adff-4981-a0b1-4315da46e088" />

---

## Clinical Note

This tool is for research and educational purposes only. Not for clinical decision making. All predictions should be interpreted in the context of full patient assessment by a qualified clinician. The model was trained on a Ghanaian TB patient cohort, so validate against your local patient population before any clinical use.

---

## Author

**Adrian Jacob Rosario**
MS Pharmaceutical Sciences (Pharmacometrics and Systems Pharmacology), University of Pittsburgh

Building end-to-end pharmacovigilance AI systems at the intersection of pharmaceutical research and production ML engineering.

[GitHub Portfolio](https://github.com/adrianjrosario8) | [LinkedIn](https://www.linkedin.com/in/adrian-jacob-rosario-330a47235/)
