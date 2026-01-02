import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------
# Gated Symptom Unit
# ---------------------------
class GatedSymptomUnit(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gate = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        mask = torch.sigmoid(self.gate(x))
        return x * mask


# ---------------------------
# Residual Block
# ---------------------------
class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.block(x)


# ---------------------------
# MEXAR Nano V2
# ---------------------------
class MEXAR_Nano_V2(nn.Module):
    def __init__(self, input_dim=132, num_classes=41, hidden_dim=64):
        super().__init__()

        self.gsu = GatedSymptomUnit(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.res1 = ResBlock(hidden_dim)
        self.res2 = ResBlock(hidden_dim)

        self.bottleneck = nn.Linear(hidden_dim, 32)

        self.classifier = nn.Linear(32, num_classes)
        self.confidence_head = nn.Linear(32, 1)

    def forward(self, x):
        gated = self.gsu(x)
        x = F.relu(self.input_proj(gated))

        x = self.res1(x)
        x = self.res2(x)

        latent = F.relu(self.bottleneck(x))

        logits = self.classifier(latent)
        confidence = torch.sigmoid(self.confidence_head(latent))

        return logits, confidence, latent


# ---------------------------
# Explainability Engine
# ---------------------------
class ExplainabilityEngine:
    def __init__(self, model, symptom_names, disease_names):
        self.model = model
        self.symptom_names = symptom_names
        self.disease_names = disease_names

    def diagnose_and_explain(self, input_vector):
        self.model.eval()

        x = input_vector.unsqueeze(0).requires_grad_(True)
        logits, conf, _ = self.model(x)

        pred_idx = torch.argmax(logits, dim=1).item()
        pred_disease = self.disease_names[pred_idx]

        score = logits[0, pred_idx]
        score.backward()

        grads = x.grad.abs().squeeze().detach().cpu().numpy()
        input_np = input_vector.cpu().numpy()

        active = np.where(input_np > 0)[0]
        ranked = sorted(
            [(self.symptom_names[i], grads[i]) for i in active],
            key=lambda x: x[1],
            reverse=True
        )

        reasons = ", ".join(
            f"{s} ({v:.3f})" for s, v in ranked[:3]
        )

        explanation = (
            f"Diagnosed {pred_disease} "
            f"(Confidence: {conf.item():.2f}). "
            f"Key symptoms: {reasons}"
        )

        return pred_disease, explanation
