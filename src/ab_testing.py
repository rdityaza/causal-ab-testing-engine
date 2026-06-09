import pandas as pd
from scipy.stats import chisquare

def check_srm(df):
    # Menghitung user aktual di masing-masing grup
    observed_counts = df['group'].value_counts()
    print("Jumlah aktual per grup:")
    print(observed_counts)

    # Menghitung ekspektasi user (karena desainnya 50:50, berarti total dibagi 2)
    total_users = len(df)
    expected_counts = [total_users / 2, total_users / 2]

    # Hitung p-value dari Chi-Square
    chi2_stat, p_value = chisquare(f_obs=observed_counts.values, f_exp=expected_counts)

    print(f"\nChi-Square Statistic: {chi2_stat:.4f}")
    print(f"P-Value: {p_value:.4f}")

    # Threshold standar adalah 0.05 (atau 0.01 untuk lebih ketat)
    alpha = 0.05

    if p_value < alpha:
        return False
    else:
        return True
    
if __name__ == "__main__":
    # Load dummy data yang udah di-generate sebelumnya
    df_dummy = pd.read_csv('../data/raw/ab_test_data.csv')
    
    # Tes fungsinya
    is_safe = check_srm(df_dummy)
    
    if is_safe:
        print("[STATUS] Eksperimen berjalan baik (Tidak ada SRM).")
    else:
        print("[STATUS] Bahaya! Terjadi Sample Ratio Mismatch.")
    
