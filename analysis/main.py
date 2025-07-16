import pandas as pd
from data_processor import load_data
from cost_calculator import calculate_optimal_cost, calculate_original_cost
from visualizer import plot_deferrable_loads, plot_cost_comparison

def main(input_file):
    # Load data from the specified CSV file
    df = load_data(input_file)

    # Calculate costs
    optimal_cost = calculate_optimal_cost(df)
    original_cost = calculate_original_cost(df)

    # Visualize the results
    plot_deferrable_loads(df)
    plot_cost_comparison(original_cost, optimal_cost)

if __name__ == "__main__":
    input_file = 'analysis/data/input/Battery_opt_res_latest.csv'  # Update with the desired input file path
    main(input_file)