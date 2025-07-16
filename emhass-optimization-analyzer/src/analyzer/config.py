from typing import List, Dict
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

# Default deferrable load configurations
DEFAULT_DEFERRABLES: List[DeferrableConfig] = [
    DeferrableConfig("P_deferrable0", 1000, 19, 20),  # Appliance 1
    DeferrableConfig("P_deferrable1", 2400, 6, 7),    # Appliance 2 (morning)
    DeferrableConfig("P_deferrable2", 3000, 21, 22),  # Dryer
]

# Default optimization scenarios
DEFAULT_SCENARIOS: Dict[str, OptimizationScenario] = {
    "battery": OptimizationScenario.BATTERY,
    "pv": OptimizationScenario.PV,
    "all": OptimizationScenario.ALL,
    "na": OptimizationScenario.NA,
    "custom": OptimizationScenario.CUSTOM,
}