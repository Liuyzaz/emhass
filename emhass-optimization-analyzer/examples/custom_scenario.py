from analyzer import OptimizationAnalyzer, DeferrableConfig
import pandas as pd

def main():
    # Path to the optimization results CSV file
    csv_path = 'data/sample_optimization_results.csv'
    
    # Define custom deferrable loads
    custom_deferrables = [
        DeferrableConfig("EV_charging", 7000, 22, 6),  # Electric vehicle charging
        DeferrableConfig("Washing_machine", 1500, 18, 20)  # Washing machine
    ]
    
    # Create an instance of the OptimizationAnalyzer
    analyzer = OptimizationAnalyzer()
    
    # Analyze the optimization scenario with custom deferrables
    results = analyzer.analyze_optimization(
        optimal_csv_path=csv_path,
        scenario='custom',
        custom_deferrables=custom_deferrables,
        create_plots=True
    )
    
    # Print the results summary
    print("Analysis Results:")
    print(results)

if __name__ == "__main__":
    main()