import streamlit as st
import pickle
import pandas as pd

# ------------------ LOAD FILES ------------------ #
with open("final_model_ab.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("columns_f.pkl", "rb") as f:
    model_columns = pickle.load(f)

# ------------------ APP TITLE ------------------ #
st.title("🧠 TB Treatment Neurotoxicity Risk Predictor")
st.write("Predict the risk of **CNS adverse drug reactions** in tuberculosis patients during treatment.")

# ------------------ INPUT SECTIONS ------------------ #
st.subheader("👤 Patient Information")
st.subheader("⭐ Key Contributing Factors")
st.write(
    "- Gastrointestinal ADRs\n"
    "- Genitourinary reactions\n"
    "- Alcohol consumption\n"
    "- Treatment duration"
)

age = st.number_input("Age", min_value=0, max_value=100, value=30)

hiv = st.selectbox("HIV Status", ["Negative", "Positive"])
hiv = 1 if hiv == "Positive" else 0

alcohol = st.number_input("Units of Alcohol per Week", min_value=0.0, value=0.0)

weight = st.number_input("Weight Change (kg)", value=0.0)

# ------------------ TREATMENT DETAILS ------------------ #
st.subheader("💊 Treatment Details")

duration_cont = st.number_input(
    "Duration of Continuation Phase (months)", min_value=0.0, value=2.0
)

total_duration = st.number_input(
    "Total Treatment Duration (months)", min_value=0.0, value=6.0
)

# ------------------ ADR SECTION ------------------ #
st.subheader("⚠️ Adverse Drug Reactions (Observed During Treatment)")

git = st.selectbox("Gastrointestinal ADRs Present?", ["No", "Yes"])
git = 1 if git == "Yes" else 0

genito = st.selectbox("Genitourinary Reaction?", ["No", "Yes"])
genito = 1 if genito == "Yes" else 0

# ------------------ PREDICTION ------------------ #
if st.button("🔍 Predict Risk"):

    input_data = pd.DataFrame([{
        'Age': age,
        'Units_Alcohol_per_week': alcohol,
        'HIV_status': hiv,
        'GIT_ADRs': git,
        'Genitourinary_reaction': genito,
        'Weight_Change': weight,
        'Duration_of_continuationP_months': duration_cont,
        'Total_Treatment_Duration_month': total_duration
    }])

    # Ensure correct column order
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict probability
    prob = model.predict_proba(input_scaled)[0][1]

    # ------------------ OUTPUT ------------------ #
    st.subheader("📊 Prediction Result")

    st.metric("Risk Probability", f"{prob:.2%}")

    if prob < 0.3:
        st.success("✅ Low Risk of Neurotoxicity")
    elif prob < 0.7:
        st.warning("⚠ Moderate Risk of Neurotoxicity")
    else:
        st.error("🚨 High Risk of Neurotoxicity")
        
with st.sidebar:
    st.header("About")
    st.write(
        "This app predicts the risk of neurological adverse drug reactions "
        "in tuberculosis patients using machine learning."
    )

    st.write("**Model:** AdaBoost (Tuned)")
    st.write("**AUC:** ~0.90")

    # Disclaimer
st.info("This tool is intended for clinical decision support only and should not replace professional medical judgment.")