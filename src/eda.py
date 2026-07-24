# STREAMING_CHUNK:Importing required libraries for EDA and visualization...
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda():
    """
    Melakukan Exploratory Data Analysis (EDA) sesuai rubrik Tahap 2 Kasus B.
    Menghasilkan 4 grafik wajib ke folder reports/ dan mencetak informasi pemeriksaan wajib.
    """
    # STREAMING_CHUNK:Configuring file paths and directories...
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'cardekho_dataset.csv')
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print("❌ Dataset belum ditemukan! Jalankan python src/load_data.py terlebih dahulu.")
        return

    df = pd.read_csv(data_path, index_col=0)
    print("✅ Dataset berhasil dimuat untuk Analisis EDA Tahap 2.")

    # STREAMING_CHUNK:Executing mandatory dataset inspections...
    print("\n" + "="*70)
    print("PEMERIKSAAN WAJIB TAHAP 2 (Sesuai Ketentuan Soal UAS)")
    print("="*70)
    
    print("\n1. Pemeriksaan Nilai Hilang (Missing Values / df.isna().sum()):")
    missing_sum = df.isna().sum()
    print(missing_sum)

    print("\n2. Ringkasan Statistik Deskriptif (df.describe()):")
    print(df.describe())

    duplicates_count = df.duplicated().sum()
    print(f"\n3. Jumlah Baris Duplikat: {duplicates_count}")
    print("="*70)

    # STREAMING_CHUNK:Generating Chart 1 - Target distribution histogram...
    plt.figure(figsize=(9, 5))
    sns.histplot(df['selling_price'], kde=True, color='teal', bins=50)
    plt.title('Grafik 1: Distribusi Harga Jual Mobil (Selling Price - Regresi)', fontsize=12, fontweight='bold')
    plt.xlabel('Harga Jual (Selling Price dalam INR)', fontsize=10)
    plt.ylabel('Frekuensi', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt1_path = os.path.join(reports_dir, 'distribusi_target.png')
    plt.savefig(plt1_path, dpi=300)
    plt.close()
    print(f"📊 Grafik 1 tersimpan di: {plt1_path}")

    # STREAMING_CHUNK:Generating Chart 2 - Missing values bar plot...
    plt.figure(figsize=(9, 5))
    missing_data = df.isna().sum()
    missing_data.plot(kind='bar', color='salmon', edgecolor='black')
    plt.title('Grafik 2: Jumlah Nilai Hilang (Missing Values) per Kolom', fontsize=12, fontweight='bold')
    plt.xlabel('Kolom', fontsize=10)
    plt.ylabel('Jumlah Missing Value', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt2_path = os.path.join(reports_dir, 'missing_values.png')
    plt.savefig(plt2_path, dpi=300)
    plt.close()
    print(f"📊 Grafik 2 tersimpan di: {plt2_path}")

    # STREAMING_CHUNK:Generating Chart 3 - Vehicle age vs price relationship (Non-linear)...
    plt.figure(figsize=(9, 5))
    sns.scatterplot(x='vehicle_age', y='selling_price', data=df, alpha=0.4, color='forestgreen')
    plt.title('Grafik 3: Hubungan Umur Kendaraan dan Harga Jual (Non-Linear Depreciation)', fontsize=12, fontweight='bold')
    plt.xlabel('Umur Kendaraan (Tahun)', fontsize=10)
    plt.ylabel('Harga Jual (Selling Price)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt3_path = os.path.join(reports_dir, 'vehicle_age_vs_price.png')
    plt.savefig(plt3_path, dpi=300)
    plt.close()
    print(f"📊 Grafik 3 tersimpan di: {plt3_path}")

    # STREAMING_CHUNK:Generating Chart 4 - Numerical correlation heatmap...
    plt.figure(figsize=(10, 6))
    numeric_df = df.select_dtypes(include=['number'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar_kws={'label': 'Korelasi'})
    plt.title('Grafik 4: Heatmap Korelasi Fitur Numerik', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt4_path = os.path.join(reports_dir, 'correlation_heatmap.png')
    plt.savefig(plt4_path, dpi=300)
    plt.close()
    print(f"📊 Grafik 4 tersimpan di: {plt4_path}")

    print("="*70)
    print("✅ Tahap 2 EDA selesai! 4 grafik wajib telah dihasilkan di folder reports/")
    print("="*70)

if __name__ == "__main__":
    perform_eda()
