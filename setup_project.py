import os

def create_project_structure():
    # Struktur folder wajib Tahap 1 sesuai instruksi soal
    folders = [
        "src",
        "app",
        "tests",
        "data",
        "models",
        "reports"
    ]

    files = {
        "src/__init__.py": "",
        "src/load_data.py": "",
        "src/eda.py": "# Skrip untuk menjawab 4 pertanyaan EDA Tahap 2\n",
        "src/train.py": "# Skrip training, Pipeline, dan evaluasi model Tahap 3\n",
        "src/evaluate.py": "# Skrip pengujian test set dan metrik Tahap 3\n",
        "app/__init__.py": "",
        "app/main.py": "# Skrip FastAPI Tahap 4\n",
        "tests/__init__.py": "",
        "tests/test_api.py": "# Test otomatis pytest Tahap 5\n",
        "requirements.txt": "pandas==2.1.4\nscikit-learn==1.3.2\nmatplotlib==3.8.2\nseaborn==0.13.0\n",
        "requirements-api.txt": "fastapi==0.104.1\nuvicorn==0.24.0.post1\npydantic==2.5.2\njoblib==1.3.2\nscikit-learn==1.3.2\npandas==2.1.4\n",
        ".gitignore": "data/\nmodels/\nreports/*.png\n__pycache__/\n*.pyc\n.pytest_cache/\n.venv/\nvenv/\n",
    }

    print("Membuat struktur direktori proyek ML End-to-End...")
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"📁 Created directory: {folder}/")

    for filepath, content in files.items():
        if not os.path.exists(filepath):
            with open(filepath, "w") as f:
                f.write(content)
            print(f"📄 Created file: {filepath}")
        else:
            print(f"⚠️ File already exists: {filepath}")

    print("\n✅ Setup Tahap 1 selesai!")

if __name__ == "__main__":
    create_project_structure()