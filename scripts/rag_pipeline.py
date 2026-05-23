import os
import streamlit as st

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

EVIDENCE = [
    {
        "pmid": "16725084",
        "title": "[Tuberculous meningitis: a comparative study in relation to concurrent human immunodeficiency virus infection]."
    },
    {
        "pmid": "28233512",
        "title": "Tuberculosis Associated with HIV Infection."
    },
    {
        "pmid": "28233512",
        "title": "Tuberculosis Associated with HIV Infection."
    }
]

FINAL_RESPONSE = """
Key TB-CNS Associations:

1. Extrameningeal TB: More frequent in patients with HIV coinfection (61.5% vs. 36.1%, p = 0.03).
2. Radiological alterations: More frequent in HIV-infected patients.
3. Treatment complexity: HIV-infected patients more likely to receive treatment with four antituberculosis drugs (61.5% vs. 13.9%, p = 0.01).

Literature Summary:

HIV coinfection is associated with a higher risk of extrapulmonary and disseminated TB, including extrameningeal TB. HIV-infected patients are more likely to receive complex TB treatment regimens. The presence of HIV coinfection complicates TB diagnosis and treatment, emphasizing the need for a high index of suspicion and prompt evaluation.

Population-Level Interpretation:

In the context of TB treatment, HIV coinfection is a significant risk factor for extrameningeal TB and complex treatment regimens. This highlights the importance of considering HIV status when evaluating TB patients, particularly those with extrapulmonary or disseminated disease.

Safety Note (No Diagnosis):

Given the patient's TB CNS ADR risk profile, it is essential to monitor for potential CNS adverse reactions, especially in the context of HIV coinfection and complex TB treatment regimens. Regular neurological assessments and close monitoring of treatment efficacy and safety are recommended.
"""


def generate_clinical_rationale(query):
    return FINAL_RESPONSE, EVIDENCE