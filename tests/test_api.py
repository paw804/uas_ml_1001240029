from fastapi.testclient import TestClient
import pytest
import os
import joblib
import pandas as pd
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True

def test_predict_valid_input():
    payload = {
        "brand": "Maruti",
        "model": "Swift",
        "vehicle_age": 3,
        "km_driven": 35000,
        "fuel_type": "Petrol",
        "seller_type": "Individual",
        "transmission_type": "Manual",
        "mileage": 18.5,
        "engine": 1197,
        "max_power": 82.0,
        "seats": 5
    }
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "predicted_selling_price_inr" in data

def test_predict_missing_field():
    payload = {
        "brand": "Maruti",
        "model": "Swift",
        "vehicle_age": 3,
        # 'km_driven' dihilangkan untuk memicu error 422
        "fuel_type": "Petrol",
        "seller_type": "Individual",
        "transmission_type": "Manual",
        "mileage": 18.5,
        "engine": 1197,
        "max_power": 82.0,
        "seats": 5
    }
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 422

def test_predict_invalid_data_type():
    payload = {
        "brand": "Maruti",
        "model": "Swift",
        "vehicle_age": "tiga_tahun",
        "km_driven": 35000,
        "fuel_type": "Petrol",
        "seller_type": "Individual",
        "transmission_type": "Manual",
        "mileage": 18.5,
        "engine": 1197,
        "max_power": 82.0,
        "seats": 5
    }
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 422

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'models', 'model.joblib')

@pytest.fixture
def model_pipeline():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def test_behavior_older_car_cheaper(model_pipeline):
    if model_pipeline is None:
        pytest.skip("Model pipeline belum tersedia.")

    young_car = pd.DataFrame([{
        "brand": "Maruti", "model": "Swift", "vehicle_age": 2, "km_driven": 20000,
        "fuel_type": "Petrol", "seller_type": "Individual", "transmission_type": "Manual",
        "mileage": 19.0, "engine": 1197, "max_power": 82.0, "seats": 5
    }])
    old_car = pd.DataFrame([{
        "brand": "Maruti", "model": "Swift", "vehicle_age": 8, "km_driven": 20000,
        "fuel_type": "Petrol", "seller_type": "Individual", "transmission_type": "Manual",
        "mileage": 19.0, "engine": 1197, "max_power": 82.0, "seats": 5
    }])
    price_young = model_pipeline.predict(young_car)[0]
    price_old = model_pipeline.predict(old_car)[0]
    assert price_old < price_young

def test_behavior_higher_mileage_efficiency(model_pipeline):
    if model_pipeline is None:
        pytest.skip("Model pipeline belum tersedia.")

    efficient_car = pd.DataFrame([{
        "brand": "Hyundai", "model": "i20", "vehicle_age": 4, "km_driven": 40000,
        "fuel_type": "Diesel", "seller_type": "Individual", "transmission_type": "Manual",
        "mileage": 22.0, "engine": 1248, "max_power": 74.0, "seats": 5
    }])
    inefficient_car = pd.DataFrame([{
        "brand": "Hyundai", "model": "i20", "vehicle_age": 4, "km_driven": 40000,
        "fuel_type": "Diesel", "seller_type": "Individual", "transmission_type": "Manual",
        "mileage": 11.0, "engine": 1248, "max_power": 74.0, "seats": 5
    }])
    price_efficient = model_pipeline.predict(efficient_car)[0]
    price_inefficient = model_pipeline.predict(inefficient_car)[0]
    assert price_efficient != price_inefficient