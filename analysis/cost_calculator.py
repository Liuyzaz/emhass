def calculate_optimal_cost(df):
    if 'P_grid_neg' in df.columns:
        total_prod_revenue = (0.5 * df['unit_prod_price'] * 0.001 * (-df['P_grid_neg'])).sum()
    else:
        total_prod_revenue = 0
        print('P_grid_neg column not found in df')

    if 'P_grid_pos' not in df.columns:
        df['P_grid_pos'] = 0  # fallback if missing
    if 'unit_load_cost' not in df.columns:
        df['unit_load_cost'] = 0.1419  # fallback if not present

    total_load_cost = 0.5 * (0.001 * df['P_grid_pos'] * df['unit_load_cost']).sum()
    optimal_cost = total_load_cost - total_prod_revenue
    return optimal_cost


def calculate_original_cost(df):
    if 'P_grid_neg' in df.columns:
        origin_total_prod_revenue = (0.5 * df['unit_prod_price'] * 0.001 * (-df['P_grid_neg'])).sum()
    else:
        origin_total_prod_revenue = 0
        print('P_grid_neg column not found in df')

    if 'P_grid_pos' not in df.columns:
        df['P_grid_pos'] = 0  # fallback if missing
    if 'unit_load_cost' not in df.columns:
        df['unit_load_cost'] = 0.1419  # fallback if not present

    original_total_load_cost = 0.5 * (0.001 * df['P_grid_pos'] * df['unit_load_cost']).sum()
    original_cost = original_total_load_cost - origin_total_prod_revenue
    return original_cost


def calculate_savings(original_cost, optimal_cost):
    if original_cost == 0:
        return 0
    savings_percent = (original_cost - optimal_cost) / original_cost * 100
    return savings_percent


def load_and_process_data(file_path):
    import pandas as pd
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df