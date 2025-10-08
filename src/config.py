"""
Configuration management for GridOps Agentic AI System.
Handles environment variables, system parameters, and constraint definitions.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from loguru import logger


class SystemConfig(BaseSettings):
    """System-wide configuration with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',
        protected_namespaces=()
    )
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(default=None, alias='OPENAI_API_KEY')
    anthropic_api_key: Optional[str] = Field(default=None, alias='ANTHROPIC_API_KEY')
    llm_provider: str = Field(default='openai', alias='LLM_PROVIDER')
    model_name: str = Field(default='gpt-4-turbo-preview', alias='MODEL_NAME')
    temperature: float = Field(default=0.1, alias='TEMPERATURE')
    
    # System Behavior
    log_level: str = Field(default='INFO', alias='LOG_LEVEL')
    max_iterations: int = Field(default=10, alias='MAX_ITERATIONS')
    timeout_seconds: int = Field(default=300, alias='TIMEOUT_SECONDS')
    
    # Network Configuration
    default_network: str = Field(default='ieee_123', alias='DEFAULT_NETWORK')
    voltage_tolerance_pu: float = Field(default=0.05, alias='VOLTAGE_TOLERANCE_PU')
    thermal_margin_percent: float = Field(default=20.0, alias='THERMAL_MARGIN_PERCENT')
    max_load_shed_percent: float = Field(default=30.0, alias='MAX_LOAD_SHED_PERCENT')
    
    # Report Configuration
    report_format: str = Field(default='markdown', alias='REPORT_FORMAT')
    generate_plots: bool = Field(default=True, alias='GENERATE_PLOTS')
    plot_dpi: int = Field(default=300, alias='PLOT_DPI')
    
    # State Management
    state_persistence: bool = Field(default=True, alias='STATE_PERSISTENCE')
    memory_retention_hours: int = Field(default=24, alias='MEMORY_RETENTION_HOURS')


class NetworkConstraints:
    """Power system operational constraints."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
        # Voltage constraints (per-unit)
        self.v_min_pu = 1.0 - config.voltage_tolerance_pu
        self.v_max_pu = 1.0 + config.voltage_tolerance_pu
        
        # Thermal constraints
        self.thermal_margin = config.thermal_margin_percent / 100.0
        
        # Load shedding constraints
        self.max_load_shed = config.max_load_shed_percent / 100.0
        
        # Protection coordination
        self.min_coordination_time_sec = 0.3  # Minimum time interval between devices
        self.max_fault_clearing_time_sec = 2.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Export constraints as dictionary."""
        return {
            "voltage": {
                "min_pu": self.v_min_pu,
                "max_pu": self.v_max_pu,
                "tolerance": self.config.voltage_tolerance_pu
            },
            "thermal": {
                "margin_percent": self.config.thermal_margin_percent,
                "margin_multiplier": self.thermal_margin
            },
            "load_shedding": {
                "max_percent": self.config.max_load_shed_percent,
                "max_multiplier": self.max_load_shed
            },
            "protection": {
                "min_coordination_time_sec": self.min_coordination_time_sec,
                "max_fault_clearing_time_sec": self.max_fault_clearing_time_sec
            }
        }


class PathConfig:
    """Project directory paths."""
    
    def __init__(self, base_path: Optional[Path] = None):
        if base_path is None:
            base_path = Path(__file__).parent.parent
        
        self.base = base_path
        self.src = base_path / "src"
        self.data = base_path / "data"
        self.networks = self.data / "networks"
        self.ieee_feeders = self.networks / "ieee_test_feeders"
        self.references = self.data / "references"
        self.models = base_path / "models"
        self.knowledge_base = self.models / "knowledge_base"
        self.reports = base_path / "reports"
        self.plots = base_path / "plots"
        self.logs = base_path / "logs"
        self.notebooks = base_path / "notebooks"
        self.tests = base_path / "tests"
        self.docs = base_path / "docs"
        
        # Create directories if they don't exist
        self._create_directories()
    
    def _create_directories(self):
        """Create all necessary directories."""
        for path in [
            self.data, self.networks, self.ieee_feeders, self.references,
            self.models, self.knowledge_base, self.reports, self.plots,
            self.logs, self.notebooks, self.tests, self.docs
        ]:
            path.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, str]:
        """Export paths as dictionary."""
        return {
            "base": str(self.base),
            "data": str(self.data),
            "networks": str(self.networks),
            "reports": str(self.reports),
            "plots": str(self.plots),
            "logs": str(self.logs)
        }


def setup_logging(config: SystemConfig, paths: PathConfig):
    """Configure logging with loguru."""
    logger.remove()  # Remove default handler
    
    # Console logging
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=config.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # File logging
    log_file = paths.logs / "gridops_{time:YYYY-MM-DD}.log"
    logger.add(
        sink=str(log_file),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="00:00",
        retention="30 days",
        compression="zip"
    )
    
    logger.info("Logging system initialized")
    logger.info(f"Log level: {config.log_level}")
    logger.info(f"Log directory: {paths.logs}")


def load_configuration() -> tuple[SystemConfig, NetworkConstraints, PathConfig]:
    """Load all configurations."""
    system_config = SystemConfig()
    paths = PathConfig()
    constraints = NetworkConstraints(system_config)
    
    setup_logging(system_config, paths)
    
    logger.info("Configuration loaded successfully")
    logger.info(f"LLM Provider: {system_config.llm_provider}")
    logger.info(f"Model: {system_config.model_name}")
    logger.info(f"Default Network: {system_config.default_network}")
    
    return system_config, constraints, paths


if __name__ == "__main__":
    # Test configuration loading
    config, constraints, paths = load_configuration()
    
    print("\n=== System Configuration ===")
    print(f"LLM Provider: {config.llm_provider}")
    print(f"Model: {config.model_name}")
    print(f"Temperature: {config.temperature}")
    
    print("\n=== Network Constraints ===")
    import json
    print(json.dumps(constraints.to_dict(), indent=2))
    
    print("\n=== Paths ===")
    print(json.dumps(paths.to_dict(), indent=2))
