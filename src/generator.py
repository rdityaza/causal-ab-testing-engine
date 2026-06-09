import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_ab_data(n_users=10000, days=30, seed=42):
    """
    Fungsi untuk men-generate data simulasi A/B Testing.
    Mengembalikan Pandas DataFrame.
    """
    np.random.seed(seed)
    start_date = datetime(2026, 5, 1)

    # Parameter Bisnis
    conversion_rate_control = 0.10
    conversion_rate_treatment = 0.12 
    avg_revenue_control = 150000 
    avg_revenue_treatment = 165000 

    # Generate User & Group
    user_ids = [f"USER_{i:05d}" for i in range(n_users)]
    groups = np.random.choice(['Control', 'Treatment'], size=n_users, p=[0.5, 0.5])

    # Generate Timestamps
    random_days = np.random.randint(0, days, size=n_users)
    timestamps = [start_date + timedelta(days=int(d)) for d in random_days]

    # Generate Metrics
    is_converted = []
    revenue = []

    for group in groups:
        if group == 'Control':
            converted = np.random.binomial(1, conversion_rate_control)
            rev = np.random.normal(avg_revenue_control, 20000) if converted else 0
        else:
            converted = np.random.binomial(1, conversion_rate_treatment)
            rev = np.random.normal(avg_revenue_treatment, 25000) if converted else 0
            
        is_converted.append(converted)
        revenue.append(round(rev, 2))

    # Compile into DataFrame
    df = pd.DataFrame({
        'user_id': user_ids,
        'timestamp': timestamps,
        'group': groups,
        'is_converted': is_converted,
        'revenue': revenue
    })

    return df.sort_values('timestamp').reset_index(drop=True)

if __name__ == "__main__":
    df_experiment = generate_ab_data()
    # Save ke folder data/raw/
    df_experiment.to_csv('../data/raw/ab_test_data.csv', index=False)
    print("Data berhasil di-generate dan disimpan ke data/raw/ab_test_data.csv")