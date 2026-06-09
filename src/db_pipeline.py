import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables dari file .env
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")  
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "ab_testing_db")

def get_db_engine():
    """Membuat koneksi ke PostgreSQL."""
    connection_string = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    engine = create_engine(connection_string)
    return engine

def push_to_db(csv_path, table_name="experiment_logs"):
    """
    Fungsi Pipeline (Load): Mengirim data CSV ke tabel PostgreSQL.
    """
    print(f"[PROCESS] Membaca data dari {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Pastikan tipe data timestamp terformat dengan benar
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    engine = get_db_engine()
    print(f"[PROCESS] Mengunggah {len(df)} baris data ke tabel '{table_name}'...")
    
    # Push ke database (if_exists='replace' akan menimpa tabel jika sudah ada)
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print("[SUKSES] Data berhasil disimpan ke PostgreSQL!")

def fetch_from_db(query):
    """
    Fungsi Pipeline (Extract): Menarik data dari PostgreSQL menjadi DataFrame.
    """
    engine = get_db_engine()
    df = pd.read_sql(query, engine)
    return df

if __name__ == "__main__":
    # Mengunci path absolut berdasarkan lokasi script ini (folder src)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Mundur 1 folder (..), lalu masuk ke data/raw/ab_test_data.csv
    file_path = os.path.join(BASE_DIR, '..', 'data', 'raw', 'ab_test_data.csv')
    
    # Normalisasi path agar rapi (menghilangkan '..')
    file_path = os.path.normpath(file_path)
    
    try:
        # Jalankan pipeline untuk mendorong data
        push_to_db(file_path)
        
        # Tes menarik data kembali
        test_query = "SELECT * FROM experiment_logs LIMIT 5;"
        df_test = fetch_from_db(test_query)
        print("\n[TEST] Menampilkan 5 baris pertama dari Database:")
        print(df_test)
        
    except Exception as e:
        print(f"\n[GAGAL] Periksa kembali koneksi database atau password. Error: {e}")