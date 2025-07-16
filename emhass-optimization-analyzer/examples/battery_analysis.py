from analyzer import OptimizationAnalyzer, OptimizationScenario
from data.loader import load_optimization_results
import logging

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Path to the optimization results CSV file
    optimal_csv_path = 'data/sample_optimization_results.csv'

    # Create an instance of the OptimizationAnalyzer
    analyzer = OptimizationAnalyzer(logger)

    # Analyze the battery optimization scenario
    results = analyzer.analyze_optimization(
        optimal_csv_path,
        scenario=OptimizationScenario.BATTERY
    )

    # Print the analysis results
    logger.info("Analysis Results:")
    logger.info(results)

if __name__ == "__main__":
    main()