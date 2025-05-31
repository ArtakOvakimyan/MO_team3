from src.model.predictor import predict_toxicity

def test_predict_toxicity():
    result = predict_toxicity("You are stupid")
    assert isinstance(result, dict)
    assert "toxic" in result
    assert "score" in result
