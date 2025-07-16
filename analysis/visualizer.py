import pandas as pd
def plot_deferrable_loads(df):
    import matplotlib.pyplot as plt
    import numpy as np

    time_axis = pd.to_datetime(df['timestamp'])
    time_labels = [t.strftime('%H:%M') for t in time_axis]
    x = np.arange(len(time_axis))
    bar_width = 0.25

    plt.figure(figsize=(16, 4))
    plt.bar(x - bar_width, df['P_deferrable0'], width=bar_width, label='Deferrable 0')
    plt.bar(x, df['P_deferrable1'], width=bar_width, label='Deferrable 1')
    plt.bar(x + bar_width, df['P_deferrable2'], width=bar_width, label='Deferrable 2')

    for start, end in [(7, 9), (17.5, 20.5)]:
        indices = [i for i, t in enumerate(time_axis)
                   if (t.hour + t.minute/60) >= start and (t.hour + t.minute/60) < end]
        if indices:
            plt.axvspan(indices[0] - 0.5, indices[-1] + 0.5, color='red', alpha=0.15, label='Peak Hours' if start == 7 else None)

    plt.title('Deferrable Loads Operation')
    plt.ylabel('Power (W)')
    plt.xlabel('Time')
    plt.xticks(ticks=x[::2], labels=[l for i, l in enumerate(time_labels) if i % 2 == 0], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cost_comparison(original_cost, optimal_cost):
    import matplotlib.pyplot as plt

    costs = [original_cost, optimal_cost]
    labels = ['Original', 'Optimized']
    colors = ['blue', 'orange']

    plt.figure(figsize=(6, 5))
    plt.bar(labels, costs, color=colors)
    plt.ylabel('Total Cost (€)')
    plt.title('Total Cost Comparison')
    for i, v in enumerate(costs):
        plt.text(i, v + 0.01 * max(costs), f"{v:.2f} €", ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    plt.show()