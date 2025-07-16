from typing import Dict, List, Optional
import pandas as pd
import logging
from .config import OptimizationScenario, DeferrableConfig

class OptimizationAnalyzer:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.interval_hours = 0.5  # 30 minutes per slot
        self.default_deferrables = [
            DeferrableConfig("P_deferrable0", 1000, 19, 20),
            DeferrableConfig("P_deferrable1", 2400, 6, 7),
            DeferrableConfig("P_deferrable2", 3000, 21, 22),
        ]

    def analyze_optimization(self, 
                             optimal_csv_path: str,
                             scenario: OptimizationScenario = OptimizationScenario.BATTERY,
                             custom_deferrables: Optional[List[DeferrableConfig]] = None,
                             create_plots: bool = True) -> Dict:
        self.logger.info(f"Analyzing {scenario.value} optimization from {optimal_csv_path}")
        optimal_df = self._load_optimization_results(optimal_csv_path)
        deferrables = custom_deferrables or self.default_deferrables
        original_df = self._create_original_scenario(optimal_df, deferrables)
        costs = self._calculate_costs(optimal_df, original_df)
        energy_metrics = self._calculate_energy_metrics(optimal_df, original_df)
        deferrable_analysis = self._analyze_deferrable_operations(optimal_df, original_df, deferrables)
        
        results = {
            'scenario': scenario.value,
            'optimal_df': optimal_df,
            'original_df': original_df,
            'costs': costs,
            'energy_metrics': energy_metrics,
            'deferrable_analysis': deferrable_analysis,
            'deferrables_config': deferrables
        }
        
        if create_plots:
            results['plots'] = self._generate_all_plots(results)
        
        self._log_summary(results)
        return results

    def _load_optimization_results(self, csv_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            required_cols = ['timestamp', 'P_PV', 'P_Load']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.warning(f"Missing columns: {missing_cols}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading {csv_path}: {e}")
            raise

    def _create_original_scenario(self, optimal_df: pd.DataFrame, 
                                   deferrables: List[DeferrableConfig]) -> pd.DataFrame:
        base_cols = ['timestamp', 'P_PV', 'P_Load', 'unit_load_cost', 'unit_prod_price']
        df = optimal_df[[col for col in base_cols if col in optimal_df.columns]].copy()
        
        if 'unit_load_cost' not in df.columns:
            df['unit_load_cost'] = 0.1419
        if 'unit_prod_price' not in df.columns:
            df['unit_prod_price'] = 0.05
        
        hours = df['timestamp'].dt.hour
        
        for deferrable in deferrables:
            if deferrable.name == "P_deferrable1":
                mask1 = (hours >= 6) & (hours < 7)
                mask2 = (hours >= 19) & (hours < 20)
                mask = mask1 | mask2
            else:
                mask = (hours >= deferrable.start_hour) & (hours < deferrable.end_hour)
            
            df[deferrable.name] = 0
            df.loc[mask, deferrable.name] = deferrable.power_w
        
        deferrable_cols = [d.name for d in deferrables]
        total_deferrables = df[deferrable_cols].sum(axis=1)
        df['P_grid'] = df['P_Load'] + total_deferrables - df['P_PV']
        df['P_grid_pos'] = df['P_grid'].apply(lambda x: max(x, 0))
        df['P_grid_neg'] = df['P_grid'].apply(lambda x: min(x, 0))
        
        return df

    def _calculate_costs(self, optimal_df: pd.DataFrame, original_df: pd.DataFrame) -> Dict:
        costs = {}
        
        if 'P_grid_neg' in optimal_df.columns and 'P_grid_pos' in optimal_df.columns:
            optimal_revenue = (self.interval_hours * optimal_df['unit_prod_price'] * 0.001 * 
                               (-optimal_df['P_grid_neg'])).sum()
            optimal_cost_total = (self.interval_hours * 0.001 * optimal_df['P_grid_pos'] * 
                                  optimal_df['unit_load_cost']).sum()
            costs['optimal_net_cost'] = optimal_cost_total - optimal_revenue
            costs['optimal_revenue'] = optimal_revenue
            costs['optimal_total_cost'] = optimal_cost_total
        
        original_revenue = (self.interval_hours * original_df['unit_prod_price'] * 0.001 * 
                            (-original_df['P_grid_neg'])).sum()
        original_cost_total = (self.interval_hours * 0.001 * original_df['P_grid_pos'] * 
                               original_df['unit_load_cost']).sum()
        costs['original_net_cost'] = original_cost_total - original_revenue
        costs['original_revenue'] = original_revenue
        costs['original_total_cost'] = original_cost_total
        
        costs['savings'] = costs['original_net_cost'] - costs['optimal_net_cost']
        costs['savings_percent'] = (costs['savings'] / costs['original_net_cost']) * 100
        
        costs['original_cost_per_slot'] = (
            self.interval_hours * 0.001 * original_df['P_grid_pos'] * original_df['unit_load_cost'] -
            self.interval_hours * 0.001 * (-original_df['P_grid_neg']) * original_df['unit_prod_price']
        )
        
        if 'cost_fun_profit' in optimal_df.columns:
            costs['optimal_cost_per_slot'] = -optimal_df['cost_fun_profit']
        
        return costs

    def _calculate_energy_metrics(self, optimal_df: pd.DataFrame, original_df: pd.DataFrame) -> Dict:
        metrics = {}
        
        metrics['original_energy_per_slot'] = original_df['P_grid'] * self.interval_hours / 1000
        metrics['optimal_energy_per_slot'] = optimal_df['P_grid'] * self.interval_hours / 1000
        
        metrics['original_total_energy'] = metrics['original_energy_per_slot'].sum()
        metrics['optimal_total_energy'] = metrics['optimal_energy_per_slot'].sum()
        metrics['energy_savings'] = metrics['original_total_energy'] - metrics['optimal_total_energy']
        
        return metrics

    def _analyze_deferrable_operations(self, optimal_df: pd.DataFrame, original_df: pd.DataFrame,
                                       deferrables: List[DeferrableConfig]) -> Dict:
        analysis = {}
        
        for deferrable in deferrables:
            if deferrable.name in optimal_df.columns:
                original_active = original_df.loc[original_df[deferrable.name] > 0, 'timestamp']
                original_usage_time = (original_df[deferrable.name] > 0).sum() * self.interval_hours
                
                optimal_active = optimal_df.loc[optimal_df[deferrable.name] > 0, 'timestamp']
                optimal_usage_time = (optimal_df[deferrable.name] > 0).sum() * self.interval_hours
                
                analysis[deferrable.name] = {
                    'config': deferrable,
                    'original_active_times': original_active.tolist(),
                    'optimal_active_times': optimal_active.tolist(),
                    'original_usage_hours': original_usage_time,
                    'optimal_usage_hours': optimal_usage_time,
                    'schedule_changed': not original_active.equals(optimal_active)
                }
        
        return analysis

    def _generate_all_plots(self, results: Dict) -> Dict:
        plots = {}
        plots['deferrable_comparison'] = self._plot_deferrable_loads(results)
        plots['cost_comparison'] = self._plot_cost_comparison(results)
        plots['cost_per_slot'] = self._plot_cost_per_slot(results)
        plots['energy_comparison'] = self._plot_energy_comparison(results)
        plots['savings_pie'] = self._plot_savings_pie(results)
        return plots

    def _log_summary(self, results: Dict):
        costs = results['costs']
        energy_metrics = results['energy_metrics']
        
        self.logger.info(f"=== {results['scenario'].title()} Optimization Analysis ===")
        self.logger.info(f"Original cost: €{costs['original_net_cost']:.2f}")
        self.logger.info(f"Optimal cost: €{costs['optimal_net_cost']:.2f}")
        self.logger.info(f"Savings: €{costs['savings']:.2f} ({costs['savings_percent']:.2f}%)")
        self.logger.info(f"Energy savings: {energy_metrics['energy_savings']:.2f} kWh")