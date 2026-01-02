import streamlit as st
import torch
import numpy as np

st.set_page_config(page_title="MEXAR Nano Diagnosis", layout="wide")

from model import MEXAR_Nano_V2, ExplainabilityEngine


INPUT_DIM = 132
DEVICE = "cpu"


@st.cache_data
def load_metadata():
    import pandas as pd
    df = pd.read_csv("training.csv")
    symptoms = list(df.columns[:-1])
    diseases = sorted(df.iloc[:, -1].str.strip().str.lower().unique())
    return symptoms, diseases


symptom_names, disease_names = load_metadata()


@st.cache_resource
def load_model():
    device = torch.device("cpu")

    model = MEXAR_Nano_V2(
        input_dim=INPUT_DIM,
        num_classes=len(disease_names)
    ).to(device)

    state = torch.load(
        "student_model.pth",
        map_location=device
    )
    model.load_state_dict(state, strict=False)
    model.eval()

    temperature = torch.load(
        "temperature.pth",
        map_location=device
    )

    return model, temperature


model, temperature = load_model()

explainer = ExplainabilityEngine(
    model,
    symptom_names,
    disease_names
)


st.title("🧠 MEXAR Nano Medical Diagnosis System")
st.caption("Explainable AI for Symptom-Based Disease Prediction")

st.markdown("### Select patient symptoms:")

cols = st.columns(4)
symptom_vector = np.zeros(len(symptom_names))

for i, symptom in enumerate(symptom_names):
    if cols[i % 4].checkbox(symptom.replace("_", " ")):
        symptom_vector[i] = 1


if st.button("🔍 Diagnose"):
    input_tensor = torch.tensor(symptom_vector, dtype=torch.float32)

    with torch.no_grad():
        logits, conf, _ = model(input_tensor.unsqueeze(0))
        logits = logits / temperature
        probs = torch.softmax(logits, dim=1).squeeze()

    topk = torch.topk(probs, 3)

    st.markdown("## 🩺 Diagnosis Results")

    for rank, (idx, prob) in enumerate(zip(topk.indices, topk.values), 1):
        st.write(
            f"**{rank}. {disease_names[idx]}** — Confidence: `{prob.item():.2%}`"
        )

    disease, explanation = explainer.diagnose_and_explain(input_tensor)

    st.markdown("## 🧠 Explanation")
    st.info(explanation)
