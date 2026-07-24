import pandas as pd
import urllib.request
import os

def load_and_inspect_data():
    """
    Mengunduh dataset dari sumber publik, menyimpan ke folder data/,
    dan mencetak informasi sesuai instruksi Tahap 1.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, 'cardekho_dataset.csv')
    
    url = "https://raw.githubusercontent.com/manishkr1754/CarDekho_Used_Car_Price_Prediction/main/notebooks/data/cardekho_dataset.csv"
    
    if not os.path.exists(file_path):
        print("Mengunduh dataset dari internet. Mohon tunggu sebentar...")
        urllib.request.urlretrieve(url, file_path)
        print("✅ Dataset berhasil diunduh dan disimpan ke data/cardekho_dataset.csv")
    else:
        print("✅ Dataset sudah tersedia di folder data/.")
        
    df = pd.read_csv(file_path, index_col=0) 
    
    print("\n" + "="*50)
    print("INFORMASI DATASET (JAWABAN TAHAP 1)")
    print("="*50)
    
    print(f"Jumlah baris : {df.shape[0]}")
    print(f"Jumlah kolom : {df.shape[1]}")
    
    print("\n--- Tipe Tiap Kolom ---")
    print(df.dtypes)
    
    print("\n--- Jumlah Nilai Hilang (Missing Values) Per Kolom ---")
    print(df.isnull().sum())
    
    print("="*50)
    print("Tahap 1 selesai! Salin output di atas untuk Laporan PDF Anda.")

if __name__ == "__main__":
    load_and_inspect_data()