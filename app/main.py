from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import joblib
import pandas as pd
import os

app = FastAPI(
    title="API Estimasi Harga Mobil Bekas (CarDekho)",
    description="REST API untuk memprediksi harga jual mobil bekas menggunakan model Machine Learning terbaik.",
    version="1.0.0"
)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'models', 'model.joblib')

model_pipeline = None
if os.path.exists(model_path):
    try:
        model_pipeline = joblib.load(model_path)
        print("✅ Model pipeline berhasil dimuat ke dalam memori API.")
    except Exception as e:
        print(f"❌ Gagal memuat model: {e}")

# ... existing code ...
class CarInputFeatures(BaseModel):
    brand: str = Field(..., description="Merek mobil", json_schema_extra={"example": "Maruti"})
    model: str = Field(..., description="Model mobil", json_schema_extra={"example": "Swift"})
    vehicle_age: int = Field(..., description="Umur kendaraan dalam tahun", json_schema_extra={"example": 3})
    km_driven: int = Field(..., description="Jarak tempuh total dalam kilometer", json_schema_extra={"example": 35000})
    fuel_type: str = Field(..., description="Tipe bahan bakar", json_schema_extra={"example": "Petrol"})
    seller_type: str = Field(..., description="Tipe penjual", json_schema_extra={"example": "Individual"})
    transmission_type: str = Field(..., description="Tipe transmisi", json_schema_extra={"example": "Manual"})
    mileage: float = Field(..., description="Konsumsi bahan bakar / efisiensi (kmpl)", json_schema_extra={"example": 18.5})
    engine: int = Field(..., description="Kapasitas mesin dalam CC", json_schema_extra={"example": 1197})
    max_power: float = Field(..., description="Tenaga maksimum dalam bhp", json_schema_extra={"example": 82.0})
    seats: int = Field(..., description="Jumlah tempat duduk", json_schema_extra={"example": 5})

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

@app.get("/", tags=["Info"])
def read_root():
    return {
        "status": "online",
        "service": "CarDekho Used Car Price Prediction API",
        "author": "Faula Dwi Susanti"
    }

@app.get("/health", tags=["Info"])
def health_check():
    is_loaded = model_pipeline is not None
    return {
        "status": "healthy" if is_loaded else "degraded",
        "model_loaded": is_loaded
    }

@app.post("/predict-harga", tags=["Prediction"])
def predict_price(features: CarInputFeatures):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model ML belum tersedia di server.")

    try:
        input_data = pd.DataFrame([features.model_dump()])
        predicted_price = model_pipeline.predict(input_data)[0]

        return {
            "status": "success",
            "predicted_selling_price_inr": round(float(predicted_price), 2),
            "currency": "INR"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))