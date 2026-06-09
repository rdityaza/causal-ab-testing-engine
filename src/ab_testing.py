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
    
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

def calculate_significance(df, alpha=0.05):
    """
    Menghitung signifikansi statistik untuk Conversion Rate dan Revenue.
    
    Output: Dictionary berisi nilai aktual, p-value, dan status signifikansi.
    """
    control = df[df['group'] == 'Control']
    treatment = df[df['group'] == 'Treatment']
    
    # 1. Z-Test untuk Conversion Rate
    successes = [control['is_converted'].sum(), treatment['is_converted'].sum()]
    nobs = [len(control), len(treatment)]
    z_stat, p_val_cr = proportions_ztest(successes, nobs)
    
    # Hitung rate aktual
    ctrl_rate = successes[0] / nobs[0]
    treat_rate = successes[1] / nobs[1]
    
    # 2. Welch's T-Test untuk Revenue
    t_stat, p_val_rev = stats.ttest_ind(control['revenue'], treatment['revenue'], equal_var=False)
    
    # Hitung rata-rata revenue aktual
    ctrl_rev_mean = control['revenue'].mean()
    treat_rev_mean = treatment['revenue'].mean()
    
    # Masukkan SEMUA metrik ke dalam kamus balikan
    return {
        "conversion_rate": {
            "control_rate": ctrl_rate,
            "treatment_rate": treat_rate,
            "p_value": p_val_cr,
            "is_significant": bool(p_val_cr < alpha)
        },
        "revenue": {
            "control_mean": ctrl_rev_mean,
            "treatment_mean": treat_rev_mean,
            "p_value": p_val_rev,
            "is_significant": bool(p_val_rev < alpha)
        }
    }