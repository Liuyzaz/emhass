import pandas as pd
import pytest
from src.data.loader import load_optimization_results

def test_load_optimization_results_valid():
    df = load_optimization_results('data/sample_optimization_results.csv')
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'timestamp' in df.columns
    assert 'P_PV' in df.columns
    assert 'P_Load' in df.columns

def test_load_optimization_results_missing_columns():
    with pytest.raises(ValueError, match="Missing required columns:"):
        load_optimization_results('data/invalid_sample.csv')

def test_load_optimization_results_invalid_file():
    with pytest.raises(FileNotFoundError):
        load_optimization_results('data/non_existent_file.csv')