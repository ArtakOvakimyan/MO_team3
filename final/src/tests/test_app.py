from fastapi.testclient import TestClient
from src.app.main import app

client = TestClient(app)

def test_api_predict():
    response = client.post("/predict", json={"text": "I hate you"})
    assert response.status_code == 200
    json_data = response.json()
    assert "toxic" in json_data
    assert "score" in json_data