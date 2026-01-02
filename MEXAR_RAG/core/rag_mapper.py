import numpy as np
import torch
import pickle
import spacy
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

nlp = spacy.load("en_core_web_sm")

with open(ARTIFACTS_DIR / "symptom_list.pkl", "rb") as f:
    SYMPTOMS = pickle.load(f)

SYMPTOM_EMB = torch.tensor(
    np.load(ARTIFACTS_DIR / "symptom_embeddings.npy"),
    dtype=torch.float32
)

encoder = SentenceTransformer("all-MiniLM-L6-v2")

def extract_symptoms(text, threshold=0.45):
    text = text.lower()
    doc = nlp(text)

    negated = {t.head.text for t in doc if t.dep_ == "neg"}
    detected = []

    for sent in doc.sents:
        emb = encoder.encode(sent.text, convert_to_tensor=True)
        scores = util.cos_sim(emb, SYMPTOM_EMB)[0]

        for idx, score in enumerate(scores):
            if score >= threshold:
                name = SYMPTOMS[idx]
                if name not in negated:
                    detected.append({
                        "name": name,
                        "score": float(score)
                    })

    return detected
