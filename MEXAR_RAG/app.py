import streamlit as st
import numpy as np
import json
from pathlib import Path

from core.rag_mapper import extract_symptoms
from core.student_model import predict
from core.treatment_engine import get_treatment
from core.safety_rules import check_safety

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

with open(ARTIFACTS_DIR / "symptom_order.json") as f:
    SYMPTOMS = json.load(f)

st.set_page_config(page_title="MEXAR Lite", layout="centered")
st.title("🩺 MEXAR Lite")
st.caption("Offline Medical AI Assistant")

text = st.text_area("Describe your symptoms")

if st.button("Diagnose"):
    detected = extract_symptoms(text)

    vec = np.zeros(len(SYMPTOMS))
    for s in detected:
        if s["name"] in SYMPTOMS:
            vec[SYMPTOMS.index(s["name"])] = 1

    result = predict(vec)
    warning = check_safety(text, result["confidence"])

    if warning:
        st.warning(warning)

    st.subheader("Diagnosis")
    st.write(result["disease"])
    st.progress(result["confidence"])
