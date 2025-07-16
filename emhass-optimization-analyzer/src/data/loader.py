import pandas as pd
import os

def load_optimization_results(csv_path: str) -> pd.DataFrame:
    """Load and validate optimization results from a CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The file {csv_path} does not exist.")
    
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Validate required columns
    required_cols = ['timestamp', 'P_PV', 'P_Load']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in the CSV: {missing_cols}")
    
    return df