import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go

# ---- Cost and revenue calculation ----
def calc_cost_revenue(df, interval_hours=0.5):
    # Fallbacks for missing columns
    for col in ['P_grid_neg', 'P_grid_pos', 'unit_load_cost']:
        if col not in df.columns:
            df[col] = 0 if col != 'unit_load_cost' else 0.1419
    # Revenue (feed-in)
    prod_revenue = (interval_hours * df['unit_prod_price'] * 0.001 * (-df['P_grid_neg'])).sum()
    # Cost (consumption)
    load_cost = interval_hours * (0.001 * df['P_grid_pos'] * df['unit_load_cost']).sum()
    # Net cost
    total_cost = load_cost - prod_revenue
    return {
        'prod_revenue': prod_revenue,
        'load_cost': load_cost,
        'total_cost': total_cost
    }

def calc_deferrable_usage(df, interval_hours=0.5, def_cols=None):
    if def_cols is None:
        def_cols = ['P_deferrable0', 'P_deferrable1', 'P_deferrable2']
    usage_hours = {}
    for col in def_cols:
        if col in df.columns:
            usage_hours[col] = (df[col] > 0).sum() * interval_hours
        else:
            usage_hours[col] = 0
    return usage_hours

def print_deferrable_operation_times(df, def_cols=None):
    if def_cols is None:
        def_cols = ['P_deferrable0', 'P_deferrable1', 'P_deferrable2']
    if 'timestamp' not in df.columns:
        print("No timestamp column found!")
        return
    for col in def_cols:
        if col in df.columns:
            active_times = df.loc[df[col] > 0, 'timestamp']
            print(f"{col} active at:")
            for t in active_times:
                print(f"  {pd.to_datetime(t)}")
        else:
            print(f"{col}: Not operated")

# ---- Plotting ----
def plot_deferrable_schedule(df, title='Deferrable Loads Operation'):
    import matplotlib.pyplot as plt
    import numpy as np
    time_axis = pd.to_datetime(df['timestamp'])
    x = np.arange(len(time_axis))
    bar_width = 0.25

    plt.figure(figsize=(16, 4))
    for i, col in enumerate(['P_deferrable0', 'P_deferrable1', 'P_deferrable2']):
        if col in df.columns:
            plt.bar(x + bar_width * (i-1), df[col], width=bar_width, label=col)
    # Highlight peak hours
    for start, end in [(7, 9), (17.5, 20.5)]:
        indices = [i for i, t in enumerate(time_axis)
                   if (t.hour + t.minute/60) >= start and (t.hour + t.minute/60) < end]
        if indices:
            plt.axvspan(indices[0] - 0.5, indices[-1] + 0.5, color='red', alpha=0.15, label='Peak Hours' if start==7 else None)
    plt.title(title)
    plt.ylabel('Power (W)')
    plt.xlabel('Time')
    plt.xticks(ticks=x[::2], labels=[t.strftime('%H:%M') for i, t in enumerate(time_axis) if i % 2 == 0], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cost_comparison(original_cost, optimal_cost):
    labels = ['Original', 'Optimized']
    colors = ['blue', 'orange']
    costs = [original_cost, optimal_cost]
    plt.figure(figsize=(6,5))
    plt.bar(labels, costs, color=colors)
    plt.ylabel('Total Cost (€)')
    plt.title('Total Cost Comparison')
    for i, v in enumerate(costs):
        plt.text(i, v + 0.01 * max(costs), f"{v:.2f} €", ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_savings_pie(original_cost, optimal_cost):
    import plotly.graph_objs as go
    labels = ['Savings', 'Remaining Cost']
    values = [original_cost - optimal_cost, optimal_cost]
    savings_percent = (original_cost - optimal_cost) / original_cost * 100 if original_cost else 0
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_traces(textinfo='label+percent', pull=[0.1, 0], marker=dict(colors=['green', 'orange']))
    fig.update_layout(title=f'Cost Savings Percentage: {savings_percent:.2f}%')
    fig.show()

def plot_energy_usage(df, interval_hours=0.5):
    import matplotlib.pyplot as plt
    import numpy as np
    time_axis = pd.to_datetime(df['timestamp'])
    x = np.arange(len(time_axis))
    bar_width = 0.4

    grid_import = df['P_grid_pos'] * interval_hours / 1000
    grid_export = df['P_grid_neg'] * interval_hours / 1000
    total_load = df['P_Load'] + sum([df.get(f'P_deferrable{i}', 0) for i in range(3)])
    solar_used = (total_load - df['P_grid_pos']).clip(lower=0)
    solar_used = np.minimum(solar_used, df['P_PV']) * interval_hours / 1000

    plt.figure(figsize=(16,5))
    plt.bar(x, grid_import, width=bar_width, label='Grid Import (kWh)', color='orange')
    plt.bar(x, solar_used, width=bar_width, bottom=grid_import, label='Solar Used (kWh)', color='green')
    plt.bar(x, grid_export, width=bar_width, label='Grid Export (kWh)', color='blue')
    # Highlight peak hours
    for start, end in [(7, 9), (17.5, 20.5)]:
        indices = [i for i, t in enumerate(time_axis)
                   if (t.hour + t.minute/60) >= start and (t.hour + t.minute/60) < end]
        if indices:
            plt.axvspan(indices[0] - 0.5, indices[-1] + 0.5, color='red', alpha=0.15, label='Peak Hours' if start==7 else None)
    plt.axhline(0, color='black', linewidth=1)
    plt.title('Energy Usage per Time Slot')
    plt.ylabel('Energy (kWh)')
    plt.xlabel('Time')
    plt.xticks(ticks=x[::2], labels=[t.strftime('%H:%M') for i, t in enumerate(time_axis) if i % 2 == 0], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_grid_saving_pie(df, optimal_df, interval_hours=0.5):
    total_grid_original = (df['P_grid_pos'] * interval_hours / 1000).sum()
    total_grid_optimal = (optimal_df['P_grid_pos'] * interval_hours / 1000).sum()
    energy_saving = total_grid_original - total_grid_optimal
    saving_percent = (energy_saving / total_grid_original) * 100 if total_grid_original else 0

    labels = ['Original Grid Consumption', 'Saved Energy']
    values = [total_grid_optimal, energy_saving]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
    fig.update_layout(
        title='Grid Energy Consumption and Savings',
        annotations=[dict(text='Grid', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    fig.show()
    # Print table as well
    print(pd.DataFrame({
        'Scenario': ['Original', 'Optimized', 'Saving'],
        'Grid Energy (kWh)': [total_grid_original, total_grid_optimal, energy_saving],
        'Saving (%)': [None, None, saving_percent]
    }))
