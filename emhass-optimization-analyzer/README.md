# EMHASS Optimization Analyzer

The EMHASS Optimization Analyzer is a Python-based tool designed to analyze optimization scenarios for energy management systems. It provides functionalities to load optimization results, calculate costs, and generate visualizations for various scenarios, including battery and photovoltaic (PV) systems.

## Features

- Analyze optimization results from CSV files.
- Support for multiple optimization scenarios (Battery, PV, Custom).
- Configurable deferrable load settings.
- Generate insightful plots for cost comparisons and energy usage.
- Unit tests to ensure functionality and reliability.

## Project Structure

```
emhass-optimization-analyzer
├── src
│   ├── analyzer
│   │   ├── __init__.py
│   │   ├── core.py
│   │   ├── config.py
│   │   └── plots.py
│   ├── data
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── utils
│   │   ├── __init__.py
│   │   └── helpers.py
│   └── main.py
├── tests
│   ├── __init__.py
│   ├── test_analyzer.py
│   └── test_data_loader.py
├── data
│   ├── sample_optimization_results.csv
│   └── sample_config.json
├── examples
│   ├── battery_analysis.py
│   ├── pv_analysis.py
│   └── custom_scenario.py
├── requirements.txt
├── setup.py
├── pyproject.toml
└── README.md
```

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

To run the optimization analysis, use the `main.py` script. You can specify the optimization scenario and the path to the optimization results CSV file.

Example command:

```
python src/main.py --scenario battery --csv_path data/sample_optimization_results.csv
```

## Examples

Check the `examples` directory for scripts demonstrating how to use the analyzer for different scenarios:

- `battery_analysis.py`: Analyze battery optimization.
- `pv_analysis.py`: Analyze photovoltaic optimization.
- `custom_scenario.py`: Analyze a custom scenario with user-defined deferrable loads.

## Testing

To run the unit tests, navigate to the `tests` directory and execute:

```
pytest
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.