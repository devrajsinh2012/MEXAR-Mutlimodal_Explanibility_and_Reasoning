RED_FLAGS = [
    "chest pain",
    "difficulty breathing",
    "severe bleeding",
    "unconscious",
    "seizure"
]

def check_safety(text, confidence):
    text = text.lower()

    for rf in RED_FLAGS:
        if rf in text:
            return "⚠️ Emergency symptoms detected. Seek immediate medical care."

    if confidence < 0.5:
        return "⚠️ Low confidence. This is not a medical diagnosis."

    return None
