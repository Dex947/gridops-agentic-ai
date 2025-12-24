# Changelog

All notable changes to the GridOps Agentic AI System will be documented in this file.

## [Unreleased]

## [1.0.4] - 2025-12-24

### Added
- **Protection Coordination Module** (`src/tools/protection_coordination.py`)
  - `ProtectionDevice` dataclass with IEC time-current curves
  - `ProtectionCoordination` class for coordination analysis
  - Support for fuse, recloser, relay, breaker, sectionalizer devices
  - `TripCharacteristic` enum: extremely_inverse, very_inverse, inverse, definite_time
  - Short-circuit analysis integration with pandapower
  - `create_default_protection_scheme()` for automatic device placement
  - Post-reconfiguration verification
  - Coordination time interval (CTI) checking
  - Device interrupting capacity validation

- **Reliability Indices Module** (`src/tools/reliability_indices.py`)
  - IEEE 1366 standard reliability indices:
    - SAIDI (System Average Interruption Duration Index)
    - SAIFI (System Average Interruption Frequency Index)
    - CAIDI (Customer Average Interruption Duration Index)
    - ASAI (Average Service Availability Index)
    - MAIFI (Momentary Average Interruption Frequency Index)
    - CEMI (Customers Experiencing Multiple Interruptions)
  - `CustomerData` dataclass for load point customer tracking
  - `OutageEvent` dataclass with sustained/momentary classification
  - `ReliabilityCalculator` class for index computation
  - Contingency impact estimation
  - Customer-minutes interrupted (CMI) tracking

- **Historical Case Database** (`src/core/case_database.py`)
  - `HistoricalCase` dataclass for contingency case storage
  - `CaseDatabase` class with JSON persistence
  - Similarity-based case search
  - Action success rate tracking
  - Recommended actions based on historical data
  - Operator feedback integration (rating 1-5)
  - ML training data export
  - Workflow state recording

- **New Test Suites**
  - `tests/unit/test_protection_coordination.py` (18 tests)
  - `tests/unit/test_reliability_indices.py` (25 tests)
  - `tests/unit/test_case_database.py` (24 tests)

- **Input Validation Functions** (`src/tools/powerflow_tools.py`)
  - `validate_line_ids()` - Validate line IDs exist in network
  - `validate_bus_ids()` - Validate bus IDs exist in network
  - `validate_load_ids()` - Validate load IDs exist in network
  - `validate_switch_ids()` - Validate switch IDs exist in network
  - `ValidationError` exception class for validation failures
- **Action Plan Schema Validation** (`src/core/state_manager.py`)
  - `validate_action_plan()` - Comprehensive schema validation for action plans
  - `validate_and_sanitize_action_plan()` - Validate and add default values
  - `ActionPlanValidationError` exception class
  - Validation for action types, target elements, priority, and costs
- **Wind Generation Profiles** (`src/tools/timeseries_analysis.py`)
  - `GenerationProfile.create_wind_profile()` - Wind profiles with patterns (low, moderate, high, variable)
  - `GenerationProfile.create_combined_renewable_profile()` - Combined solar+wind profiles
- **Battery Dispatch Optimization** (`src/tools/der_integration.py`)
  - `BatteryState` dataclass for battery state tracking
  - `BatteryDispatchResult` dataclass for dispatch results
  - `DERIntegration.optimize_battery_dispatch()` - Multi-mode battery optimization
  - Dispatch modes: peak_shaving, arbitrage, voltage_support, auto
  - SOC tracking and energy throughput calculation
- **Configurable Analysis Parameters** (`src/config.py`)
  - `max_reconfiguration_combinations` - Limit for topology optimization
  - `max_contingency_combinations` - Limit for N-k analysis
  - `parallel_workers` - Number of parallel workers
  - `der_curtailment_step` - DER curtailment step size
  - `hosting_capacity_step_kw` - Hosting capacity calculation step
  - `timeseries_default_timestep_minutes` - Default time-series timestep
  - `api_key_rotation_days` - Security: API key rotation period
  - `mask_api_keys_in_logs` - Security: Mask API keys in logs

### Changed
- **Updated Exports** (`src/tools/__init__.py`)
  - Added protection coordination exports
  - Added reliability indices exports
- **Updated Exports** (`src/core/__init__.py`)
  - Added case database exports
  - Added action plan validation exports

- **Enhanced Input Validation**
  - `apply_switching_action()` now validates line IDs before applying
  - `apply_load_shedding()` now validates load IDs before applying
  - `NetworkAnalyzer.find_alternative_paths()` validates bus IDs
  - `NetworkAnalyzer.get_buses_downstream()` validates bus ID
  - `NetworkReconfiguration.apply_switch_action()` returns success/failure
- **Configurable Reconfiguration** (`src/tools/reconfiguration.py`)
  - `NetworkReconfiguration.__init__()` accepts `max_combinations` and `max_iterations`
  - `exhaustive_search()` uses instance `max_combinations` as default
- **Updated Exports** (`src/tools/__init__.py`)
  - Added `BatteryState`, `BatteryDispatchResult` exports
  - Added validation function exports

## [1.0.3] - 2025-12-24

### Fixed
- **Line loading always 0%**: Fixed unrealistic line ratings in IEEE test networks (max_i_ka = 99999 kA)
  - Added `_fix_line_ratings()` method to NetworkLoader
  - Automatically calculates realistic ratings based on actual power flow currents
### Added
- **Optimal Power Flow (OPF)**: New `src/tools/opf_tools.py` module
  - AC OPF for optimal generator dispatch and loss minimization
  - DC OPF for fast linear approximations
  - Cost function optimization (min_loss, min_generation_cost, min_load_shedding)
  - Voltage setpoint optimization
  - Loss sensitivity analysis
- **N-k Contingency Analysis**: Enhanced `src/core/contingency_simulator.py`
  - N-2, N-3, and configurable N-k analysis
  - Parallel contingency evaluation with ThreadPoolExecutor
  - Contingency ranking by severity score
  - Smart combination limiting for large networks
- **Network Reconfiguration Optimization**: New `src/tools/reconfiguration.py` module
  - Branch exchange algorithm for loss minimization
  - Exhaustive search for small networks
  - Radiality constraint enforcement
  - Multiple objectives: min_loss, load_balancing, min_voltage_deviation
  - Tie switch and sectionalizing switch identification
- **DER Integration**: New `src/tools/der_integration.py` module
  - Solar PV, wind, battery storage, and other DER types
  - Hosting capacity calculation
  - DER dispatch optimization with curtailment
  - Impact analysis (voltage rise, reverse power flow)
- **Time-Series Analysis**: New `src/tools/timeseries_analysis.py` module
  - Daily load profiles (residential, industrial)
  - Solar generation profiles
  - 24-hour power flow simulation
  - Peak identification and energy calculations
  - Critical period detection
- **Advanced Features Guide**: `docs/user-guide/advanced-features.md`
- **Documentation restructure**: Created `docs/` folder with modular documentation
  - `docs/index.md` - Documentation home
  - `docs/getting-started/` - Installation, quickstart, configuration
  - `docs/user-guide/` - CLI reference, networks guide
  - `docs/architecture/` - System overview
  - `docs/case-studies/` - IEEE 33-bus analysis example
- **Pytest test suite**: 203 unit tests covering all modules (79% coverage)
  - `tests/conftest.py` - Shared fixtures
  - `tests/unit/test_network_loader.py` - Network loading tests
  - `tests/unit/test_powerflow_tools.py` - Power flow tests
  - `tests/unit/test_contingency_simulator.py` - Contingency tests
  - `tests/unit/test_state_manager.py` - State management tests
  - `tests/unit/test_network_analysis.py` - Network analysis tests
- **GitHub Actions CI**: `.github/workflows/ci.yml` with test, lint, and validate jobs
- **pyproject.toml**: Unified configuration for pytest, ruff, coverage, and mypy

### Changed
- **Code quality**: Fixed all ruff linting issues (881 auto-fixed, 2 manual fixes)
- **NetworkLoader**: Now automatically fixes unrealistic line ratings on load

## [1.0.2] - 2025-10-08

### Fixed
- **Network topology visualization**: Fixed `bus_geodata` attribute check to use `hasattr()` preventing AttributeError
- **Alternative path finding**: Fixed MultiGraph incompatibility by converting to simple Graph for shortest path algorithms
- **Model configuration**: Updated default model from `gpt-4` to `gpt-4o-mini` for broader API access

### Improved
- Graceful fallback to NetworkX spring layout when geographic coordinates unavailable
- Robust graph type handling in path-finding algorithms
- Better error handling for network topology plotting

### Validated
- All 4 visualizations now generate successfully (voltage, loading, topology, comparison)
- Network topology plot working with IEEE 33-bus network
- MultiGraph to Graph conversion working correctly
- No more "not implemented for multigraph type" errors

## [1.0.1] - 2025-10-08

### Fixed
- Pydantic protected namespace warning in SystemConfig
- LaTeX special character escaping in report generation
- API key propagation to LLM agents (Planner and Explainer)

### Added
- Comprehensive "Actual Results" section in README with real analysis data
- Case study: IEEE 33-bus network Line 5 outage analysis
- Results tables with baseline vs post-contingency metrics
- Embedded visualizations in documentation (voltage profile, line loading, comparison)
- Key findings and technical recommendations
- test_system.py validation script for core functionality

### Improved
- README documentation with actual execution results
- LaTeX report generation with proper character escaping
- Agent initialization with explicit API key passing
- Documentation clarity with real-world examples

### Validated
- Complete pipeline execution: network loading → analysis → reporting → PDF generation
- All 8 core system tests passing
- PDF compilation successful with pdflatex
- Visualization generation (4 plot types)
- Multi-agent workflow with LLM integration

## [1.0.0] - 2025-10-08

### Added

#### Core Infrastructure
- Complete project structure with modular architecture
- Configuration management system with environment variables and pydantic-settings
- State management with LangGraph TypedDict for workflow coordination
- Network loader supporting IEEE test feeders and custom networks
- Contingency simulator for N-1 and N-k scenario analysis
- Logging system with loguru (file and console output)
- Path configuration for organized project structure

#### Multi-Agent System
- **Planner Agent**: Generates 2-5 candidate reconfiguration strategies using LLM
- **PowerFlow User Agent**: Executes power flow simulations and applies network modifications
- **Constraint Checker Agent**: Validates voltage, thermal, and protection constraints with safety scoring
- **Explainer Agent**: Generates human-readable technical explanations with IEEE citations
- **Retrieval Agent**: Accesses standards database (IEEE, ANSI, NERC) and best practices

#### Power System Analysis
- Power flow tools with pandapower integration (BFSW, NR, GS algorithms)
- Network analysis tools with NetworkX for topology analysis
- Switching action application (line and explicit switches)
- Load shedding implementation with configurable limits
- Alternative path finding for contingency resolution
- Critical element identification (high loading, bridge elements)
- Bus centrality calculation for importance ranking

#### Workflow Orchestration
- LangGraph state graph implementation with 9 workflow nodes
- Sequential workflow: load → baseline → contingency → retrieve → plan → evaluate → select → explain → finalize
- Error handling and recovery mechanisms
- Session-based state persistence
- Workflow status tracking and iteration limits

#### Visualization
- Voltage profile plots (baseline vs post-action comparison)
- Line loading distribution histograms and bar charts
- Network topology graphs with switched element highlighting
- Performance comparison charts (4-panel metrics)
- High-resolution output (configurable DPI)
- Automatic plot generation pipeline

#### Reporting
- Comprehensive Markdown reports with embedded visualizations
- Professional LaTeX reports with structured sections
- Executive summary generation
- Power flow results tables
- Action evaluation comparisons
- Technical references and citations
- Timestamp-based file naming for version control

#### Command-Line Interface
- Argument parser with comprehensive options
- Network listing functionality
- Contingency type selection (line_outage, transformer_outage, load_increase, generator_outage)
- Multi-element contingency support
- Custom output directory specification
- Log level override capability
- Report format selection (Markdown/LaTeX/both)

#### Testing and Validation
- Module-level test code in `__main__` blocks
- Example scenarios and usage demonstrations
- Mock data for standalone module testing
- Integration test capability through main.py

### Dependencies
- pandapower ≥2.14.0 for power system analysis
- langgraph ≥0.2.0 for multi-agent orchestration
- langchain ≥0.2.0 for LLM integration
- langchain-openai ≥0.1.0 for OpenAI models
- langchain-anthropic ≥0.1.0 for Claude models
- networkx ≥3.1 for graph operations
- matplotlib ≥3.7.0 for visualization
- plotly ≥5.14.0 for interactive plots
- loguru ≥0.7.0 for logging
- pydantic ≥2.5.0 for configuration validation
- python-dotenv ≥1.0.0 for environment management

### Configuration
- Environment-based configuration (.env file)
- Network constraint definitions (voltage: 0.95-1.05 pu, thermal: 100%)
- LLM provider selection (OpenAI/Anthropic)
- Model selection and temperature control
- Maximum iteration limits
- Report format preferences
- Plot generation settings

### Documentation
- Comprehensive README.md with architecture diagrams
- Detailed installation and setup instructions
- Usage examples and quick start guide
- API documentation in docstrings
- CONTRIBUTING.md with development guidelines
- LICENSE file (MIT License)
- CHANGELOG.md with structured version history
- memory.json for project context and assumptions

### Standards and References
- IEEE Std 1547-2018: DER Interconnection
- ANSI C84.1-2020: Voltage Ratings
- IEEE Std 242-2001: Protection Coordination
- IEEE Std 1366-2012: Reliability Indices
- NERC TPL-001-4: Planning Requirements
- IEEE Std C37.117-2007: Load Shedding
- IEEE Std 738-2012: Conductor Thermal Ratings

### Performance Optimizations
- Network caching in loader
- Efficient power flow algorithms
- Selective N-1 testing (sample-based)
- State persistence for workflow resumption
- Lazy loading of visualization components
