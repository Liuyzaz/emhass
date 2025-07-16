import pandas as pd

def prepare_manual_df(optimal_df, deferrable_specs):
    columns_to_keep = ['timestamp','P_PV', 'P_Load', 'unit_load_cost', 'unit_prod_price']
    df = optimal_df[[col for col in columns_to_keep if col in optimal_df.columns]].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    hours = df['timestamp'].dt.hour

    for spec in deferrable_specs:
        col = spec['col']
        df[col] = 0
        if 'periods' in spec:
            for (start, end) in spec['periods']:
                mask = (hours >= start) & (hours < end)
                df.loc[mask, col] = spec['value']
        else:
            start = spec['start_hour']
            end = spec['end_hour']
            mask = (hours >= start) & (hours < end)
            df.loc[mask, col] = spec['value']
    return df


