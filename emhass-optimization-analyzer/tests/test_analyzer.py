import pytest
import pandas as pd
from src.analyzer.core import OptimizationAnalyzer
from src.analyzer.config import DeferrableConfig
from src.analyzer import OptimizationScenario

@pytest.fixture
def sample_data():
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=48, freq='30T'),
        'P_PV': [1000] * 48,
        'P_Load': [1500] * 48,
        'unit_load_cost': [0.1419] * 48,
        'unit_prod_price': [0.05] * 48,
    }
    return pd.DataFrame(data)

def test_load_optimization_results(sample_data):
    sample_data.to_csv('test_optimization_results.csv', index=False)
    analyzer = OptimizationAnalyzer()
    df = analyzer._load_optimization_results('test_optimization_results.csv')
    
    assert df.shape[0] == 48
    assert 'timestamp' in df.columns
    assert 'P_PV' in df.columns
    assert 'P_Load' in df.columns

def test_create_original_scenario(sample_data):
    analyzer = OptimizationAnalyzer()
    deferrables = [
        DeferrableConfig("P_deferrable0", 1000, 19, 20),
        DeferrableConfig("P_deferrable1", 2400, 6, 7),
        DeferrableConfig("P_deferrable2", 3000, 21, 22),
    ]
    original_df = analyzer._create_original_scenario(sample_data, deferrables)
    
    assert 'P_deferrable0' in original_df.columns
    assert 'P_deferrable1' in original_df.columns
    assert 'P_deferrable2' in original_df.columns
    assert original_df['P_deferrable0'].sum() > 0
    assert original_df['P_deferrable1'].sum() > 0
    assert original_df['P_deferrable2'].sum() > 0

def test_calculate_costs(sample_data):
    analyzer = OptimizationAnalyzer()
    original_df = analyzer._create_original_scenario(sample_data, [])
    costs = analyzer._calculate_costs(sample_data, original_df)
    
    assert 'original_net_cost' in costs
    assert 'optimal_net_cost' in costs
    assert 'savings' in costs

def test_analyze_optimization(sample_data):
    sample_data.to_csv('test_optimization_results.csv', index=False)
    analyzer = OptimizationAnalyzer()
    results = analyzer.analyze_optimization('test_optimization_results.csv', OptimizationScenario.BATTERY)
    
    assert results['scenario'] == 'battery'
    assert 'optimal_df' in results
    assert 'original_df' in results
    assert 'costs' in results
    assert 'energy_metrics' in results
    assert 'deferrable_analysis' in results
    assert 'deferrables_config' in results
    assert 'plots' in results  # If plots are generated