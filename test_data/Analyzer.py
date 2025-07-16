import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

class OptimizationScenario(Enum):
    """Supported optimization scenarios"""
    BATTERY = "battery"
    PV = "pv" 
    ALL = "all"
    NA = "na"
    CUSTOM = "custom"

@dataclass
class DeferrableConfig:
    """Configuration for deferrable loads"""
    name: str
    power_w: int
    start_hour: int
    end_hour: int
    duration_hours: float = 1.0

class OptimizationAnalyzer:
    """
    General analyzer for EMHASS optimization results.
    Works with any optimization scenario (Battery, PV, All, etc.)
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.interval_hours = 0.5  # 30 minutes per slot
        
        # Default deferrable load configurations
        self.default_deferrables = [
            DeferrableConfig("P_deferrable0", 1000, 19, 20),  # Appliance 1
            DeferrableConfig("P_deferrable1", 2400, 6, 7),    # Appliance 2 (morning)
            DeferrableConfig("P_deferrable2", 3000, 21, 22),  # Dryer
        ]
    
    def analyze_optimization(self, 
                           optimal_csv_path: str,
                           scenario: OptimizationScenario = OptimizationScenario.BATTERY,
                           custom_deferrables: Optional[List[DeferrableConfig]] = None,
                           create_plots: bool = True) -> Dict:
        """
        Main analysis method for any optimization scenario.
        
        :param optimal_csv_path: Path to the optimization results CSV
        :param scenario: Type of optimization scenario
        :param custom_deferrables: Custom deferrable load configuration
        :param create_plots: Whether to generate plots
        :return: Complete analysis results
        """
        self.logger.info(f"Analyzing {scenario.value} optimization from {optimal_csv_path}")
        
        # Load optimization results
        optimal_df = self._load_optimization_results(optimal_csv_path)
        
        # Use custom or default deferrable configuration
        deferrables = custom_deferrables or self.default_deferrables
        
        # Create original (naive) scenario
        original_df = self._create_original_scenario(optimal_df, deferrables)
        
        # Calculate costs
        costs = self._calculate_costs(optimal_df, original_df)
        
        # Calculate energy metrics
        energy_metrics = self._calculate_energy_metrics(optimal_df, original_df)
        
        # Analyze deferrable operations
        deferrable_analysis = self._analyze_deferrable_operations(optimal_df, original_df, deferrables)
        
        # Compile results
        results = {
            'scenario': scenario.value,
            'optimal_df': optimal_df,
            'original_df': original_df,
            'costs': costs,
            'energy_metrics': energy_metrics,
            'deferrable_analysis': deferrable_analysis,
            'deferrables_config': deferrables
        }
        
        # Generate plots if requested
        if create_plots:
            results['plots'] = self._generate_all_plots(results)
        
        self._log_summary(results)
        return results
    
    def _load_optimization_results(self, csv_path: str) -> pd.DataFrame:
        """Load and validate optimization results CSV"""
        try:
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Validate required columns
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
        """Create original (naive) deferrable load schedule"""
        # Start with base data
        base_cols = ['timestamp', 'P_PV', 'P_Load', 'unit_load_cost', 'unit_prod_price']
        df = optimal_df[[col for col in base_cols if col in optimal_df.columns]].copy()
        
        # Add fallback values if missing
        if 'unit_load_cost' not in df.columns:
            df['unit_load_cost'] = 0.1419
        if 'unit_prod_price' not in df.columns:
            df['unit_prod_price'] = 0.05
        
        # Create naive deferrable schedules
        hours = df['timestamp'].dt.hour
        
        for deferrable in deferrables:
            # Create time mask for this deferrable load
            if deferrable.name == "P_deferrable1":
                # Special case: operates both morning and evening
                mask1 = (hours >= 6) & (hours < 7)
                mask2 = (hours >= 19) & (hours < 20)
                mask = mask1 | mask2
            else:
                mask = (hours >= deferrable.start_hour) & (hours < deferrable.end_hour)
            
            df[deferrable.name] = 0
            df.loc[mask, deferrable.name] = deferrable.power_w
        
        # Calculate grid power
        deferrable_cols = [d.name for d in deferrables]
        total_deferrables = df[deferrable_cols].sum(axis=1)
        df['P_grid'] = df['P_Load'] + total_deferrables - df['P_PV']
        df['P_grid_pos'] = df['P_grid'].apply(lambda x: max(x, 0))
        df['P_grid_neg'] = df['P_grid'].apply(lambda x: min(x, 0))
        
        return df
    
    def _calculate_costs(self, optimal_df: pd.DataFrame, original_df: pd.DataFrame) -> Dict:
        """Calculate costs for both scenarios"""
        costs = {}
        
        # Optimal costs
        if 'P_grid_neg' in optimal_df.columns and 'P_grid_pos' in optimal_df.columns:
            optimal_revenue = (self.interval_hours * optimal_df['unit_prod_price'] * 0.001 * 
                             (-optimal_df['P_grid_neg'])).sum()
            optimal_cost_total = (self.interval_hours * 0.001 * optimal_df['P_grid_pos'] * 
                                optimal_df['unit_load_cost']).sum()
            costs['optimal_net_cost'] = optimal_cost_total - optimal_revenue
            costs['optimal_revenue'] = optimal_revenue
            costs['optimal_total_cost'] = optimal_cost_total
        
        # Original costs
        original_revenue = (self.interval_hours * original_df['unit_prod_price'] * 0.001 * 
                          (-original_df['P_grid_neg'])).sum()
        original_cost_total = (self.interval_hours * 0.001 * original_df['P_grid_pos'] * 
                             original_df['unit_load_cost']).sum()
        costs['original_net_cost'] = original_cost_total - original_revenue
        costs['original_revenue'] = original_revenue
        costs['original_total_cost'] = original_cost_total
        
        # Savings calculation
        costs['savings'] = costs['original_net_cost'] - costs['optimal_net_cost']
        costs['savings_percent'] = (costs['savings'] / costs['original_net_cost']) * 100
        
        # Cost per time slot
        costs['original_cost_per_slot'] = (
            self.interval_hours * 0.001 * original_df['P_grid_pos'] * original_df['unit_load_cost'] -
            self.interval_hours * 0.001 * (-original_df['P_grid_neg']) * original_df['unit_prod_price']
        )
        
        if 'cost_fun_profit' in optimal_df.columns:
            costs['optimal_cost_per_slot'] = -optimal_df['cost_fun_profit']
        
        return costs
    
    def _calculate_energy_metrics(self, optimal_df: pd.DataFrame, original_df: pd.DataFrame) -> Dict:
        """Calculate energy-related metrics"""
        metrics = {}
        
        # Energy per slot (kWh)
        metrics['original_energy_per_slot'] = original_df['P_grid'] * self.interval_hours / 1000
        metrics['optimal_energy_per_slot'] = optimal_df['P_grid'] * self.interval_hours / 1000
        
        # Total daily energy
        metrics['original_total_energy'] = metrics['original_energy_per_slot'].sum()
        metrics['optimal_total_energy'] = metrics['optimal_energy_per_slot'].sum()
        metrics['energy_savings'] = metrics['original_total_energy'] - metrics['optimal_total_energy']
        
        return metrics
    
    def _analyze_deferrable_operations(self, optimal_df: pd.DataFrame, original_df: pd.DataFrame,
                                     deferrables: List[DeferrableConfig]) -> Dict:
        """Analyze when deferrable loads operate in both scenarios"""
        analysis = {}
        
        for deferrable in deferrables:
            if deferrable.name in optimal_df.columns:
                # Original schedule
                original_active = original_df.loc[original_df[deferrable.name] > 0, 'timestamp']
                original_usage_time = (original_df[deferrable.name] > 0).sum() * self.interval_hours
                
                # Optimal schedule
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
        """Generate all visualization plots"""
        plots = {}
        
        # 1. Deferrable loads comparison
        plots['deferrable_comparison'] = self._plot_deferrable_loads(results)
        
        # 2. Cost comparison
        plots['cost_comparison'] = self._plot_cost_comparison(results)
        
        # 3. Cost per time slot
        plots['cost_per_slot'] = self._plot_cost_per_slot(results)
        
        # 4. Energy usage comparison
        plots['energy_comparison'] = self._plot_energy_comparison(results)
        
        # 5. Savings pie chart
        plots['savings_pie'] = self._plot_savings_pie(results)
        
        return plots
    
    def _plot_deferrable_loads(self, results: Dict):
        """Plot deferrable loads for original vs optimized"""
        optimal_df = results['optimal_df']
        original_df = results['original_df']
        deferrables = results['deferrables_config']
        
        # Original schedule
        time_axis = pd.to_datetime(original_df['timestamp'])
        time_labels = [t.strftime('%H:%M') for t in time_axis]
        x = np.arange(len(time_axis))
        bar_width = 0.25
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
        
        # Plot original
        for i, deferrable in enumerate(deferrables):
            if deferrable.name in original_df.columns:
                ax1.bar(x + i*bar_width, original_df[deferrable.name], 
                       width=bar_width, label=deferrable.name)
        
        self._add_peak_hours_highlight(ax1, time_axis)
        ax1.set_title('Original Deferrable Loads Operation')
        ax1.set_ylabel('Power (W)')
        ax1.legend()
        
        # Plot optimized
        if all(d.name in optimal_df.columns for d in deferrables):
            time_axis_opt = pd.to_datetime(optimal_df['timestamp'])
            x_opt = np.arange(len(time_axis_opt))
            
            for i, deferrable in enumerate(deferrables):
                ax2.bar(x_opt + i*bar_width, optimal_df[deferrable.name], 
                       width=bar_width, label=deferrable.name)
            
            self._add_peak_hours_highlight(ax2, time_axis_opt)
            ax2.set_title('Optimized Deferrable Loads Operation')
            ax2.set_ylabel('Power (W)')
            ax2.set_xlabel('Time')
            ax2.legend()
        
        plt.tight_layout()
        plt.show()
        return fig
    
    def _add_peak_hours_highlight(self, ax, time_axis):
        """Add peak hours highlighting to plot"""
        for start, end in [(7, 9), (17.5, 20.5)]:
            indices = [i for i, t in enumerate(time_axis)
                      if (t.hour + t.minute/60) >= start and (t.hour + t.minute/60) < end]
            if indices:
                ax.axvspan(indices[0] - 0.5, indices[-1] + 0.5, 
                          color='red', alpha=0.15, 
                          label='Peak Hours' if start == 7 else None)
    
    def _plot_cost_comparison(self, results: Dict):
        """Plot total cost comparison"""
        costs = results['costs']
        
        cost_values = [costs['original_net_cost'], costs['optimal_net_cost']]
        labels = ['Original', 'Optimized']
        colors = ['blue', 'orange']
        
        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(labels, cost_values, color=colors)
        ax.set_ylabel('Total Cost (€)')
        ax.set_title(f'Total Cost Comparison - {results["scenario"].title()} Scenario')
        
        for i, v in enumerate(cost_values):
            ax.text(i, v + 0.01 * max(cost_values), f"{v:.2f} €", 
                   ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        return fig
    
    def _plot_savings_pie(self, results: Dict):
        """Plot savings percentage as pie chart"""
        costs = results['costs']
        
        labels = ['Savings', 'Remaining Cost']
        values = [costs['savings'], costs['optimal_net_cost']]
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig.update_traces(textinfo='label+percent', pull=[0.1, 0], 
                         marker=dict(colors=['green', 'orange']))
        fig.update_layout(
            title=f'{results["scenario"].title()} Optimization: Cost Savings {costs["savings_percent"]:.2f}%'
        )
        fig.show()
        return fig
    
    def _plot_cost_per_slot(self, results: Dict):
        """Plot cost per time slot comparison"""
        if 'optimal_cost_per_slot' not in results['costs']:
            return None
            
        original_df = results['original_df']
        costs = results['costs']
        
        time_axis = pd.to_datetime(original_df['timestamp'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=time_axis,
            y=costs['original_cost_per_slot'],
            name='Original',
            marker_color='blue',
            hovertemplate='Time: %{x|%Y-%m-%d %H:%M}<br>Cost: %{y:.4f} €'
        ))
        
        fig.add_trace(go.Bar(
            x=time_axis,
            y=costs['optimal_cost_per_slot'],
            name='Optimized',
            marker_color='orange',
            hovertemplate='Time: %{x|%Y-%m-%d %H:%M}<br>Cost: %{y:.4f} €'
        ))
        
        fig.update_layout(
            barmode='group',
            title=f'Cost per Time Slot: {results["scenario"].title()} Scenario',
            xaxis_title='Time',
            yaxis_title='Cost (€/30min)',
            hovermode='x unified'
        )
        fig.show()
        return fig
    
    def _plot_energy_comparison(self, results: Dict):
        """Plot energy usage comparison"""
        original_df = results['original_df']
        optimal_df = results['optimal_df']
        energy_metrics = results['energy_metrics']
        
        time_axis = pd.to_datetime(original_df['timestamp'])
        time_axis_opt = pd.to_datetime(optimal_df['timestamp'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=energy_metrics['original_energy_per_slot'],
            mode='lines',
            name='Original Energy per Slot (kWh)',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=time_axis_opt,
            y=energy_metrics['optimal_energy_per_slot'],
            mode='lines',
            name='Optimized Energy per Slot (kWh)',
            line=dict(color='orange')
        ))
        
        fig.update_layout(
            title=f'Energy Usage: {results["scenario"].title()} Scenario',
            xaxis_title='Time',
            yaxis_title='Energy per Slot (kWh)',
            hovermode='x unified'
        )
        fig.show()
        return fig
    
    def _log_summary(self, results: Dict):
        """Log analysis summary"""
        costs = results['costs']
        energy_metrics = results['energy_metrics']
        
        self.logger.info(f"=== {results['scenario'].title()} Optimization Analysis ===")
        self.logger.info(f"Original cost: €{costs['original_net_cost']:.2f}")
        self.logger.info(f"Optimal cost: €{costs['optimal_net_cost']:.2f}")
        self.logger.info(f"Savings: €{costs['savings']:.2f} ({costs['savings_percent']:.2f}%)")
        self.logger.info(f"Energy savings: {energy_metrics['energy_savings']:.2f} kWh")

# Convenience function for easy usage
def analyze_optimization_scenario(csv_path: str, 
                                scenario: str = "battery",
                                custom_deferrables: Optional[List[DeferrableConfig]] = None) -> Dict:
    """
    Convenience function to analyze any optimization scenario.
    
    Usage:
        # Battery scenario
        results = analyze_optimization_scenario('battery_results.csv', 'battery')
        
        # PV scenario  
        results = analyze_optimization_scenario('pv_results.csv', 'pv')
        
        # Custom scenario
        custom_loads = [DeferrableConfig("EV_charging", 7000, 22, 6)]
        results = analyze_optimization_scenario('custom_results.csv', 'custom', custom_loads)
    """
    analyzer = OptimizationAnalyzer()
    scenario_enum = OptimizationScenario(scenario.lower())
    return analyzer.analyze_optimization(csv_path, scenario_enum, custom_deferrables)