import pandas as pd
def load_data(file_path):
    import pandas as pd
    
    df = pd.read_csv(file_path)
    return df

def process_data(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    hours = df['timestamp'].dt.hour
    
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
    
    # Calculate P_grid
    df['P_grid'] = df['P_Load'] + df['P_deferrable0'] + df['P_deferrable1'] + df['P_deferrable2'] - df['P_PV']
    
    df['P_grid_pos'] = df['P_grid'].apply(lambda x: x if x > 0 else 0)
    df['P_grid_neg'] = df['P_grid'].apply(lambda x: x if x < 0 else 0)
    
    return df