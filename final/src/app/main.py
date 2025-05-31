from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.model.predictor import predict_toxicity

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")
    return predict_toxicity(input.text)
