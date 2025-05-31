from .load_model import get_classifier

classifier = get_classifier()

def predict_toxicity(text: str):
    result = classifier(text)[0]
    return {
        "toxic": result["label"].lower() == "toxic",
        "score": round(result["score"], 3)
    }