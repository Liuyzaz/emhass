def load_data(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def clean_data(df):
    # Implement any necessary data cleaning steps here
    return df

def format_timestamp(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def extract_hours(df):
    df['hours'] = df['timestamp'].dt.hour
    return df

def create_deferrable_loads(df):
    hours = df['hours']
    
    # Create deferrable load masks
    mask_0 = (hours >= 19) & (hours < 20)
    df['P_deferrable0'] = 0
    df.loc[mask_0, 'P_deferrable0'] = 1000

    mask1 = (hours >= 6) & (hours < 7)
    mask2 = (hours >= 19) & (hours < 20)
    df['P_deferrable1'] = 0
    df.loc[mask1 | mask2, 'P_deferrable1'] = 2400

    mask3 = (hours >= 21) & (hours < 22)
    df['P_deferrable2'] = 0
    df.loc[mask3, 'P_deferrable2'] = 3000

    return df

def calculate_usage_time(df, column_name):
    interval_hours = 0.5  # 30 minutes per row
    usage_time = (df[column_name] > 0).sum() * interval_hours
    return usage_time

def calculate_grid_power(df):
    df['P_grid'] = df['P_Load'] + df['P_deferrable0'] + df['P_deferrable1'] + df['P_deferrable2'] - df['P_PV']
    df['P_grid_pos'] = df['P_grid'].apply(lambda x: x if x > 0 else 0)
    df['P_grid_neg'] = df['P_grid'].apply(lambda x: x if x < 0 else 0)
    return df

def calculate_costs(df):
    total_prod_revenue = (0.5 * df['unit_prod_price'] * 0.001 * (-df['P_grid_neg'])).sum()
    total_load_cost = 0.5 * (0.001 * df['P_grid_pos'] * df['unit_load_cost']).sum()
    optimal_cost = total_load_cost - total_prod_revenue
    return total_prod_revenue, total_load_cost, optimal_cost