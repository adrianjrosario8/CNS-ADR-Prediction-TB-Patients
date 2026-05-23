import streamlit as st
import pickle
import pandas as pd

from scripts.rag_pipeline import generate_clinical_rationale



# MODEL LOAD

with open("final_model_ab.pkl", "rb") as f:
    model = pickle.load(f)

with open("columns_f.pkl", "rb") as f:
    model_columns = pickle.load(f)


# DEFAULT FEATURES

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

# SESSION STATE


if "results_ready" not in st.session_state:
    st.session_state.results_ready = False


# UI HEADER


st.title("🧠 TB Neurological ADR Risk Predictor")

st.write(
    "ML + Retrieval-Augmented Evidence System for CNS Adverse Drug Reaction risk in TB patients."
)



# INPUT SECTION


st.subheader("👤 Patient Information")

age = st.number_input("Age", 0, 100, 30)

hiv = st.selectbox("HIV Status", ["Negative", "Positive"])
hiv = 1 if hiv == "Positive" else 0

tb_test = st.selectbox("TB Diagnostic Method", ["Culture", "GeneXpert", "Sputum Smear"])
tb_test = {"Culture": 1, "GeneXpert": 2, "Sputum Smear": 3}[tb_test]

baseline_pcs = st.number_input("Baseline Health Score", 0.0, 100.0, 45.0)

alcohol = st.number_input("Alcohol Units Per Week", 0.0, value=0.0)

weight = st.number_input("Weight Change (kg)", value=0.0)



# TREATMENT


st.subheader("💊 Treatment Details")

duration_cont = st.number_input("Treatment Continuation (months)", 0.0, value=4.0)

total_duration = st.number_input("Total Treatment Duration (months)", 0.0, value=6.0)


# ADR INPUTS


st.subheader("⚠️ Adverse Reactions")

git = st.selectbox("Digestive Symptoms (nausea, vomiting, discomfort)?", ["No", "Yes"])
git = 1 if git == "Yes" else 0

genito = st.selectbox("Urinary/Reproductive Symptoms?", ["No", "Yes"])
genito = 1 if genito == "Yes" else 0

skin = st.selectbox("Skin Reactions?", ["No", "Yes"])
skin = 1 if skin == "Yes" else 0

audio = st.selectbox("Hearing-Related Symptoms?", ["No", "Yes"])
audio = 1 if audio == "Yes" else 0


# PREDICTION


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

    df = pd.DataFrame([input_data])
    df = df.reindex(columns=model_columns, fill_value=0)

    prob = model.predict_proba(df)[0][1]

    st.session_state.prob = prob
    st.session_state.input_data = input_data
    st.session_state.results_ready = True


# RESULTS SECTION


if st.session_state.results_ready:

    prob = st.session_state.prob

    st.subheader("📊 Prediction Result")

    st.metric("CNS ADR Risk Score", f"{prob:.2%}")

    st.caption("Model-derived score (not clinically calibrated probability)")

    if prob < 0.35:
        st.success("Low Risk")
    elif prob < 0.65:
        st.warning("Moderate Risk")
    else:
        st.error("High Risk")


    # CLINICAL FACTORS


    st.subheader("🩺 Clinical Risk Factors")

    risks = []

    if alcohol > 14:
        risks.append("High alcohol intake associated with neurotoxicity risk.")

    if hiv:
        risks.append("HIV infection associated with increased ADR susceptibility.")

    if git:
        risks.append("Digestive symptoms indicate systemic intolerance.")

    if genito:
        risks.append("Urinary/reproductive symptoms indicate systemic sensitivity.")

    if skin:
        risks.append("Skin reactions suggest hypersensitivity risk.")

    if audio:
        risks.append("Hearing-related symptoms suggest possible neurotoxicity.")

    if total_duration > 6:
        risks.append("Longer treatment increases cumulative toxicity risk.")

    if age > 60:
        risks.append("Advanced age increases ADR susceptibility.")

    if not risks:
        risks.append("No major clinical risk factors detected.")

    for r in risks:
        st.write("• " + r)


# RAG SECTION 


if st.session_state.results_ready and st.session_state.prob >= 0.65:

    st.subheader("📚 Evidence-Based Clinical Insights")

    st.info(
        "Literature-based associations only. No diagnostic interpretation."
    )

    data = st.session_state.input_data
    prob = st.session_state.prob

    patient_summary = f"""
TB CNS ADR Risk Case:
Age {data['Age']}
HIV {data['HIV_status']}
Alcohol {data['Units_Alcohol_per_week']}
GI ADR {data['GIT_ADRs']}
Genitourinary ADR {data['Genitourinary_reaction']}
Skin ADR {data['Skin_reactions']}
Audio ADR {data['Audiologic_Rxn']}
Duration {data['Total_Treatment_Duration_month']}
Score {prob:.2%}
"""

    answer, sources = generate_clinical_rationale(patient_summary)

    with st.expander("🧾 Patient Summary"):
        st.code(patient_summary, language="text")

    with st.expander("🧠 Retrieved Evidence"):
        for s in sources:
            st.write(f"• {s}")

    with st.expander("📊 Clinical Interpretation"):
        st.write(answer)


# SIDEBAR


with st.sidebar:
    st.header("System Info")
    st.write("Model: AdaBoost Ensemble")
    st.write("Architecture: ML + FAISS + Groq RAG")
    st.write("Purpose: Clinical Decision Support Demo")


# DISCLAIMER

st.info(
    "For research/educational use only. Not for clinical decision making."
)