# core/student_model.py

from pathlib import Path
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# Paths
# -----------------------
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# -----------------------
# Device
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Load symptoms
# -----------------------
with open(ARTIFACTS_DIR / "symptom_list.pkl", "rb") as f:
    SYMPTOMS = pickle.load(f)

INPUT_DIM = len(SYMPTOMS)

# -----------------------
# Load labels (SOURCE OF TRUTH)
# -----------------------
df = pd.read_csv(ARTIFACTS_DIR / "training.csv")
LABELS = df.iloc[:, -1].astype(str).unique().tolist()

OUTPUT_DIM = len(LABELS)

assert OUTPUT_DIM == 470, f"Expected 470 classes, found {OUTPUT_DIM}"

# -----------------------
# Model
# -----------------------
class StudentNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc_out = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc_out(x)

# -----------------------
# Load trained weights
# -----------------------
model = StudentNet(INPUT_DIM, OUTPUT_DIM).to(DEVICE)

state = torch.load(
    ARTIFACTS_DIR / "student_demo.pth",
    map_location=DEVICE
)

model.load_state_dict(state, strict=True)
model.eval()

# -----------------------
# Prediction
# -----------------------
def predict(symptom_vector):
    x = torch.tensor(symptom_vector).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
        conf, idx = torch.max(probs, dim=1)

    return {
        "disease": LABELS[idx.item()],
        "confidence": float(conf.item())
    }

# -----------------------
# Test
# -----------------------
if __name__ == "__main__":
    dummy = np.zeros(INPUT_DIM)
    print(predict(dummy))
