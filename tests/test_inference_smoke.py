import json
import os
import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY", "devkey123")

def test_health():
    r = requests.get(f"{API_URL}/health", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True

def test_predict_smoke():
    with open("artifacts/sample_payload.json", "r", encoding="utf-8") as f:
        payload = json.load(f)

    headers = {"X-API-Key": API_KEY}
    r = requests.post(f"{API_URL}/predict", json=payload, headers=headers, timeout=10)
    assert r.status_code == 200
    data = r.json()
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability_class_1"] <= 1.0
