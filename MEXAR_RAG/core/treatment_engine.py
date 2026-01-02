from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

meds = pd.read_csv(ARTIFACTS_DIR / "medications.csv").set_index("Disease").to_dict()
diets = pd.read_csv(ARTIFACTS_DIR / "diets.csv").set_index("Disease").to_dict()
prec = pd.read_csv(ARTIFACTS_DIR / "precautions.csv").set_index("Disease").T.to_dict("list")

def get_treatment(disease):
    return {
        "medications": meds.get("Meds", {}).get(disease, "Consult doctor"),
        "diet": diets.get("Diet", {}).get(disease, "Balanced diet"),
        "precautions": prec.get(disease, [])
    }
