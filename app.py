import streamlit as st
import pickle
import pandas as pd


# Load the trained Model

with open("final_model_ab.pkl", "rb") as f:
    model = pickle.load(f)

with open("columns_f.pkl", "rb") as f:
    model_columns = pickle.load(f)


# Default Values for Low-Importance Features

DEFAULTS = {
    "Gender": 1,
    "Marital_Status": 1,
    "Occupation": 1,
    "Education": 2,
    "Recreational_drug_use": 0,
    "Type_Recreational_drug": 0,
    "Cigarette_Smoking": 0,
    "Pack_year_Smoking": 0.0,
    "Alcohol_use": 0,
    "Type_of_TB": 1,
    "Baseline_MCS12": 45.0,
    "Patient_onHAART": 0,
    "WHO_HIV_stage": 1,
    "HAARTstarted_beforeTB": 0,
    "Regimen_of_HAART": 0,
    "Other_comorbidity": 0,
    "Hepatotoxic_Effects": 0,
    "GIT_Severity": 0,
    "Skin_rxn_severity": 0,
    "Genitourinary_rxn_severity": 0,
    "Hepatotoxic_severity": 0,
    "Ophthalmic_effects": 0,
    "Duration_of_InitiationP_months": 2.0,
    "Weight_IP": 0.0,
}

# App Title

st.title("🧠 TB Neurological ADR Risk Predictor")

st.write(
    "Predict the risk of CNS adverse drug reactions "
    "in tuberculosis patients during treatment."
)
# Patient Information Section

st.subheader("👤 Patient Information")

age = st.number_input(
    "Age",
    min_value=0,
    max_value=100,
    value=30
)

hiv = st.selectbox( "HIV Status", ["Negative", "Positive"])

hiv = 1 if hiv == "Positive" else 0

tb_test = st.selectbox(
    "TB Diagnostic Test Result",
    ["Culture", "GeneXpert", "Sputum Smear"]
)

tb_map = {
    "Culture": 1,
    "GeneXpert": 2,
    "Sputum Smear": 3
}

tb_test = tb_map[tb_test]

baseline_pcs = st.number_input(
    "Baseline Physical Health Score (0-100)",
    min_value=0.0,
    max_value=100.0,
    value=45.0,
    step=0.5
)

alcohol = st.number_input(
    "Alcohol Units Per Week",
    min_value=0.0,
    value=0.0
)

weight = st.number_input(
    "Weight Change Since Treatment Start (kg)",
    value=0.0
)

# Treatment Details Section

st.subheader("💊 Treatment Details")

duration_cont = st.number_input(
    "Treatment Continuation Duration (Months)",
    min_value=0.0,
    value=4.0
)

total_duration = st.number_input(
    "Total Treatment Duration (Months)",
    min_value=0.0,
    value=6.0
)

# ADR Section

st.subheader("⚠️ Adverse Reactions Observed During Treatment")

git = st.selectbox(
    "Gastrointestinal Side Effects?",
    ["No", "Yes"]
)

git = 1 if git == "Yes" else 0

genito = st.selectbox(
    "Urinary/Reproductive Side Effects?",
    ["No", "Yes"]
)

genito = 1 if genito == "Yes" else 0

skin = st.selectbox(
    "Skin Reactions (rash, itching)?",
    ["No", "Yes"]
)

skin = 1 if skin == "Yes" else 0

audio = st.selectbox(
    "Hearing-Related Side Effects?",
    ["No", "Yes"]
)

audio = 1 if audio == "Yes" else 0

# Prediction Button

if st.button("🔍 Predict CNS ADR Risk"):


    input_data = DEFAULTS.copy()

    input_data.update({
        "Age": age,
        "HIV_status": hiv,
        "TB_Diagnostic_test": tb_test,
        "Baseline_PCS12": baseline_pcs,
        "Units_Alcohol_per_week": alcohol,
        "Weight_Change": weight,
        "Duration_of_continuationP_months": duration_cont,
        "Total_Treatment_Duration_month": total_duration,
        "GIT_ADRs": git,
        "Genitourinary_reaction": genito,
        "Skin_reactions": skin,
        "Audiologic_Rxn": audio,
    })

    # Convert to DataFrame

    input_df = pd.DataFrame([input_data])

    # Ensure Correct Feature Order

    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Predict Probability

    prob = model.predict_proba(input_df)[0][1]

    # Prediction Output

    st.subheader("📊 Prediction Result")

    st.metric(
        "CNS ADR Risk Probability",
        f"{prob:.2%}"
    )

    # Risk Stratification

    if prob < 0.35:
        st.success(
            "✅ Low Risk- standard monitoring sufficient."
        )

    elif prob < 0.65:
        st.warning(
            "⚠️ Moderate Risk- monitor patient closely for neurological symptoms."
        )

    else:
        st.error(
            "🚨 High Risk- consider closer monitoring and clinical evaluation."
        )

    # Clinical Explanation Section

    st.subheader("🩺 Clinical Risk Factors Identified")

    explanations = []

    if alcohol > 14:
        explanations.append(
            "Higher alcohol consumption may increase neurological toxicity risk."
        )

    if hiv == 1:
        explanations.append(
            "HIV-positive status may increase susceptibility to CNS adverse reactions."
        )

    if git == 1:
        explanations.append(
            "Gastrointestinal side effects may indicate broader medication intolerance."
        )

    if genito == 1:
        explanations.append(
            "Urinary/reproductive side effects may reflect systemic adverse drug reactions."
        )

    if skin == 1:
        explanations.append(
            "Skin reactions may indicate increased drug sensitivity."
        )

    if audio == 1:
        explanations.append(
            "Hearing-related side effects may reflect broader neurological toxicity."
        )

    if total_duration > 6:
        explanations.append(
            "Longer treatment duration may increase cumulative neurotoxicity exposure."
        )

    if age > 60:
        explanatiopns.append(
            "Older age may increase vulnerability to adverse drug reactions."
        )

    if len(explanations) == 0:
        explanations.append(
            "No major high-risk clinical indicators were detected."
        )

    for e in explanations:
        st.write("• " + e)

# Sidebar

with st.sidebar:

    st.header("About")

    st.write(
        "This app predicts the risk of CNS adverse drug reactions "
        "in tuberculosis patients using machine learning."
    )

    st.write("**Model:** AdaBoost (Tuned)")
    st.write("**ROC-AUC:** 0.951")
    st.write("**Recall (High Risk):** 0.833")
    st.write("**Brier Score:** 0.132")
    st.write("**Nested CV AUC:** ~0.91")
    st.write("**Dataset:** Ghanaian TB patient cohort")

# Disclaimer

st.info(
    "This tool is intended for clinical decision support only "
    "and should not replace professional medical judgment."
)