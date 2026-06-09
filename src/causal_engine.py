import pandas as pd
import numpy as np
import warnings
from causalimpact import CausalImpact

# Menyembunyikan FutureWarning dari statsmodels
warnings.filterwarnings('ignore')

# 1. Patch untuk bug 'applymap' pada Pandas >= 2.1.0
if not hasattr(pd.DataFrame, 'applymap'):
    pd.DataFrame.applymap = pd.DataFrame.map

# 2. Patch untuk bug 'KeyError: 0' pada proses standarisasi pycausalimpact
def patched_standardize(self):
    from causalimpact.misc import standardize
    self.normed_pre_data, (mu, sig) = standardize(self.pre_data)
    self.normed_post_data = (self.post_data - mu) / sig
    # FIX: Mengganti pemanggilan [0] yang kadaluarsa menjadi .iloc[0]
    self.mu_sig = (mu.iloc[0], sig.iloc[0])

# Menyuntikkan fungsi yang sudah diperbaiki ke dalam library
CausalImpact._standardize_pre_post_data = patched_standardize

def run_causal_impact(df, pre_period, post_period):
    """
    Menjalankan analisis Causal Inference menggunakan library CausalImpact.
    
    Parameters:
    - df (pd.DataFrame): DataFrame dengan index Datetime. Kolom 'y' sebagai target, 'X' sebagai kontrol.
    - pre_period (list): [start_date, end_date] periode sebelum intervensi/fitur rilis.
    - post_period (list): [start_date, end_date] periode sesudah intervensi/fitur rilis.
    
    Returns:
    - ci (CausalImpact object): Objek utama yang menyimpan model dan hasil plotting.
    - summary_data (dict): Ringkasan metrik untuk divisualisasikan dengan mudah di dashboard.
    """
    # Menjalankan model CausalImpact
    ci = CausalImpact(df, pre_period, post_period)
    
    # Mengekstrak metrik utama agar mudah ditangkap oleh UI Dashboard (misal: Streamlit)
    summary_data = {
        "actual_cumulative": ci.summary_data.loc['actual', 'cumulative'],
        "predicted_cumulative": ci.summary_data.loc['predicted', 'cumulative'],
        "absolute_effect": ci.summary_data.loc['abs_effect', 'cumulative'],
        "relative_effect_pct": (ci.summary_data.loc['rel_effect', 'cumulative'] * 100),
        "p_value": ci.p_value,
        "is_significant": bool(ci.p_value < 0.05)
    }
    
    return ci, summary_data

if __name__ == "__main__":
    print("[STATUS] Memulai test Causal Engine...")
    
    # Generate data dummy sederhana untuk testing
    np.random.seed(42)
    dates = pd.date_range(start='2026-04-01', periods=60)
    X = np.random.normal(loc=1000000, scale=100000, size=60)
    y = X * 1.2 + np.random.normal(loc=0, scale=50000, size=60)
    y[40:] += 300000  # Dampak intervensi
    
    df_test = pd.DataFrame({'y': y, 'X': X}, index=dates)
    pre = ['2026-04-01', '2026-05-10']
    post = ['2026-05-11', '2026-05-30']
    
    try:
        model_ci, metrics = run_causal_impact(df_test, pre, post)
        print("[SUKSES] Engine berjalan tanpa error!")
        print(f"P-Value: {metrics['p_value']:.4f}")
        print(f"Signifikan: {metrics['is_significant']}")
        print(f"Kenaikan Relatif: {metrics['relative_effect_pct']:.2f}%")
    except Exception as e:
        print(f"[GAGAL] Terjadi error: {e}")