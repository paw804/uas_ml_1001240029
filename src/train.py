import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def train_models():
    """
    Melatih 3 algoritma regresi (Linear Regression, Ridge, Random Forest)
    menggunakan Pipeline scikit-learn tanpa data leakage, melakukan 5-fold CV,
    dan menyimpan pipeline model terbaik beserta metadata.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'cardekho_dataset.csv')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print("❌ Dataset belum ditemukan! Jalankan python src/load_data.py terlebih dahulu.")
        return

    df = pd.read_csv(data_path, index_col=0)
    print("✅ Dataset berhasil dimuat untuk Training Tahap 3.")

    # Target adalah 'selling_price'
    X = df.drop(columns=['selling_price', 'car_name'], errors='ignore')
    y = df['selling_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"📊 Ukuran Data Train: {X_train.shape}, Ukuran Data Test: {X_test.shape}")

    # Mengidentifikasi fitur numerik dan kategorikal secara otomatis
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Menggabungkan transformer ke dalam ColumnTransformer (tanpa kebocoran data)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Daftar 3 algoritma regresi yang akan dibandingkan
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    print("\n" + "="*70)
    print("PERBANDINGAN 3 MODEL DENGAN 5-FOLD CROSS VALIDATION (R² Score)")
    print("="*70)

    best_score = -float('inf')
    best_model_name = None
    best_pipeline = None

    for name, model in models.items():
        # Pipeline utuh berisi preprocessor dan model regresi
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # 5-fold cross validation pada data train (R² score)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        print(f"🔹 {name:20} | R² Mean = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        if cv_scores.mean() > best_score:
            best_score = cv_scores.mean()
            best_model_name = name
            best_pipeline = pipeline

    print("="*70)
    print(f"🏆 Model Terbaik Berdasarkan CV: {best_model_name} dengan R² Mean: {best_score:.4f}")
    print("="*70)

    best_pipeline.fit(X_train, y_train)
    
    # Test set disentuh SEKALI di sini untuk evaluasi akhir
    y_pred = best_pipeline.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)

    print("\n" + "="*70)
    print("EVALUASI MODEL TERBAIK PADA TEST SET (Hanya disentuh sekali)")
    print("="*70)
    print(f"MAE  (Mean Absolute Error) : {test_mae:,.2f}")
    print(f"RMSE (Root Mean Squared Error): {test_rmse:,.2f}")
    print(f"R²   (Coefficient of Determination): {test_r2:.4f}")
    print("="*70)

    model_path = os.path.join(models_dir, 'model.joblib')
    joblib.dump(best_pipeline, model_path)
    print(f"💾 Pipeline utuh berhasil disimpan ke: {model_path}")

    metadata = {
        "best_model": best_model_name,
        "cv_r2_score": float(best_score),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_r2": float(test_r2),
        "features": {
            "numerical": numeric_features,
            "categorical": categorical_features
        }
    }

    metadata_path = os.path.join(models_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"📄 Metadata model berhasil disimpan ke: {metadata_path}")
    print("="*70)

if __name__ == "__main__":
    train_models()