import logging
from analyzer.core import OptimizationAnalyzer
from analyzer.config import DeferrableConfig
from typing import List

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Path to the optimization results CSV
    optimal_csv_path = 'data/sample_optimization_results.csv'
    
    # Example of custom deferrables
    custom_deferrables: List[DeferrableConfig] = [
        DeferrableConfig("EV_charging", 7000, 22, 6)
    ]

    # Run the optimization analysis for the battery scenario
    analyzer = OptimizationAnalyzer(logger)
    results = analyzer.analyze_optimization(optimal_csv_path, create_plots=True)

    # Print the results summary
    logger.info("Analysis Results:")
    logger.info(results)

if __name__ == "__main__":
    main()