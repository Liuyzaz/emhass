from analyzer import OptimizationAnalyzer, OptimizationScenario
from data.loader import load_optimization_results
import pandas as pd

def main():
    # Path to the optimization results CSV file
    optimal_csv_path = 'data/sample_optimization_results.csv'
    
    # Create an instance of the OptimizationAnalyzer
    analyzer = OptimizationAnalyzer()
    
    # Analyze the PV optimization scenario
    results = analyzer.analyze_optimization(
        optimal_csv_path,
        scenario=OptimizationScenario.PV
    )
    
    # Print the analysis results
    print("Analysis Results:")
    print(f"Scenario: {results['scenario']}")
    print(f"Optimal Costs: {results['costs']['optimal_net_cost']:.2f} €")
    print(f"Original Costs: {results['costs']['original_net_cost']:.2f} €")
    print(f"Savings: {results['costs']['savings']:.2f} €")
    print(f"Energy Savings: {results['energy_metrics']['energy_savings']:.2f} kWh")

if __name__ == "__main__":
    main()