UAS Machine Learning End-to-End: Estimasi Harga Kendaraan Bekas

Nama: Faula Dwi Susanti

NIM: 1001240029

Kasus Pilihan: Kasus B (Regresi: Estimasi Harga Kendaraan Bekas)

1. Deskripsi Masalah & Lingkungan Kerja

A. Deskripsi Masalah

Dalam industri marketplace otomotif, penentuan harga jual mobil bekas sering kali menjadi tantangan bagi penjual individu maupun dealer karena tingginya variasi faktor kondisi kendaraan. Sistem Machine Learning End-to-End ini dibangun untuk memprediksi atau mengestimasi harga jual wajar (numerik kontinu) dari mobil bekas berdasarkan karakteristik fisik dan spesifikasi teknisnya (seperti umur kendaraan, jarak tempuh, jenis bahan bakar, kapasitas mesin, dan tenaga maksimum).

Tantangan utama yang diselesaikan dalam proyek ini meliputi:

Menangani hubungan non-linear antara umur kendaraan dan penyusutan harga pasar.

Membangun alur pemrosesan data otomatis (Pipeline scikit-learn) untuk mencegah terjadinya kebocoran data (data leakage).

Menyediakan antarmuka REST API yang andal dan divalidasi ketat untuk konsumsi sistem eksternal.

B. Lingkungan Pengembangan & Versi Pustaka

Proyek ini dikembangkan di atas Python 3.10+. Pustaka utama yang digunakan tercantum di bawah ini:

Python: 3.10+ (Kompatibel dengan 3.13)

pandas: 3.0.3 (atau versi stabil terbaru)

scikit-learn: 1.9.0

FastAPI & Uvicorn: Untuk implementasi dan serving REST API

Pytest & TestClient: Untuk pengujian unit dan pengujian perilaku (behavioral testing)

2. Sumber dan Lisensi Data

Sumber Data: CarDekho Used Car Dataset

Tautan Sumber Publik: Kaggle - CarDekho Used Car Data

Lisensi: CC0: Public Domain / Open Database License.

Keterangan: Data diunduh secara otomatis melalui skrip src/load_data.py dari repositori publik yang aman untuk memastikan reproduksibilitas penuh oleh penguji.

3. Mengapa Folder Data dan Model Tidak Di-Commit?

Sesuai dengan praktik terbaik rekayasa perangkat lunak (Software Engineering Best Practices):

Alasan: Folder data/ (berisi dataset mentah berukuran besar) dan folder models/ (berisi file biner artefak model .joblib) dikecualikan dari pelacakan Git menggunakan .gitignore. Hal ini bertujuan agar repositori tetap ringan dan bersih dari file hasil komputasi yang dapat berubah.

Cara Reproduksi oleh Penguji: Penguji dapat dengan mudah mengkloning repositori ini, memasang dependensi, lalu menjalankan skrip src/load_data.py untuk mengunduh dataset secara mandiri serta src/train.py untuk melatih dan menghasilkan artefak model secara identik.

4. Cara Menjalankan Proyek dari Nol (Reproduksi Langkah demi Langkah)

Buka terminal (CMD / Terminal) pada direktori utama proyek, lalu jalankan perintah berikut secara berurutan:

A. Instalasi Dependensi

# Install pustaka untuk analisis, EDA, dan training model
pip install -r requirements.txt

# Install pustaka untuk REST API (FastAPI & Uvicorn)
pip install -r requirements-api.txt


B. Tahap 1: Memuat dan Memeriksa Data

Unduh dataset dan tampilkan informasi dasar (jumlah baris, kolom, tipe data, serta missing values):

python src/load_data.py


C. Tahap 2: Exploratory Data Analysis (EDA)

Jalankan analisis data eksploratif untuk menghasilkan 4 grafik wajib ke dalam folder reports/:

python src/eda.py


D. Tahap 3: Pelatihan dan Evaluasi Model

Latih tiga model regresi (Linear Regression, Ridge, dan Random Forest) menggunakan 5-fold Cross-Validation, lalu simpan pipeline terbaik beserta metadata ke folder models/:

python src/train.py


E. Tahap 4: Menjalankan REST API (FastAPI)

Jalankan server lokal menggunakan Uvicorn:

uvicorn app.main:app --reload


Akses dokumentasi interaktif Swagger UI di browser melalui: http://127.0.0.1:8000/docs

5. Contoh Pemanggilan API (cURL)

A. Contoh Request Valid (Berhasil - Status 200)

curl -X 'POST' \
  'http://127.0.0.1:8000/predict-harga' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
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
}'


B. Contoh Request Tidak Valid (Ditolak - Status 422 Validation Error)

Mengirimkan tipe data salah atau field wajib yang hilang:

curl -X 'POST' \
  'http://127.0.0.1:8000/predict-harga' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "brand": "Maruti",
  "model": "Swift",
  "vehicle_age": "tiga_tahun",
  "km_driven": 35000
}'


6. Pengujian Otomatis (Pytest)

Jalankan seluruh rangkaian unit test dan behavioral test untuk memverifikasi kesehatan API dan perilaku model:

python -m pytest tests/ -v
