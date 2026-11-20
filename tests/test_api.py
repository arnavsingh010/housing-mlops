import pytest
from fastapi.testclient import TestClient

from housing_mlops.api import create_app
from housing_mlops.train import train

HOUSE = {
    "MedInc": 8.3252,
    "HouseAge": 41,
    "AveRooms": 6.98,
    "AveBedrms": 1.02,
    "Population": 322,
    "AveOccup": 2.56,
    "Latitude": 37.88,
    "Longitude": -122.23,
}


@pytest.fixture
def client(cfg):
    train(cfg)
    with TestClient(create_app(cfg["artifacts"]["model_path"])) as c:
        yield c


@pytest.fixture
def client_no_model(tmp_path):
    with TestClient(create_app(tmp_path / "missing.joblib")) as c:
        yield c


def test_home_page_is_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Predict price" in r.text


def test_health(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True}


def test_predict_returns_dollars(client):
    r = client.post("/predict", json=HOUSE)
    assert r.status_code == 200
    assert r.json()["predicted_value_usd"] > 1_000


def test_predict_batch(client):
    r = client.post("/predict/batch", json={"houses": [HOUSE, HOUSE]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_predict_rejects_out_of_range(client):
    assert client.post("/predict", json={**HOUSE, "Latitude": 99}).status_code == 422


def test_predict_rejects_missing_field(client):
    bad = {k: v for k, v in HOUSE.items() if k != "MedInc"}
    assert client.post("/predict", json=bad).status_code == 422


def test_batch_rejects_empty_list(client):
    assert client.post("/predict/batch", json={"houses": []}).status_code == 422


def test_model_info(client):
    body = client.get("/model-info").json()
    assert "model_type" in body["metadata"]


def test_no_model_gives_503_but_health_works(client_no_model):
    assert client_no_model.get("/health").json() == {"status": "ok", "model_loaded": False}
    assert client_no_model.post("/predict", json=HOUSE).status_code == 503
    assert client_no_model.get("/model-info").status_code == 503
