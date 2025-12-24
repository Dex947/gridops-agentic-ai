# Configuration Reference

GridOps uses environment variables for configuration. All settings can be defined in a `.env` file in the project root.

## Environment Variables

### LLM Configuration

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic API key | - | `sk-ant-...` |
| `LLM_PROVIDER` | LLM provider to use | `openai` | `openai`, `anthropic` |
| `MODEL_NAME` | Model name | `gpt-4o-mini` | See model options |
| `TEMPERATURE` | Sampling temperature | `0.1` | `0.0` - `1.0` |

### System Behavior

| Variable | Description | Default | Range |
|----------|-------------|---------|-------|
| `LOG_LEVEL` | Logging verbosity | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `MAX_ITERATIONS` | Maximum workflow iterations | `10` | `1` - `100` |
| `TIMEOUT_SECONDS` | Operation timeout | `300` | `60` - `3600` |

### Network Constraints

| Variable | Description | Default | Unit |
|----------|-------------|---------|------|
| `VOLTAGE_TOLERANCE_PU` | Voltage tolerance | `0.05` | per-unit |
| `THERMAL_MARGIN_PERCENT` | Thermal safety margin | `20.0` | % |
| `MAX_LOAD_SHED_PERCENT` | Maximum load shedding | `30.0` | % |

### Report Configuration

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `REPORT_FORMAT` | Default report format | `markdown` | `markdown`, `latex`, `both` |
| `GENERATE_PLOTS` | Generate visualizations | `True` | `True`, `False` |
| `PLOT_DPI` | Plot resolution | `300` | `72` - `600` |

### State Management

| Variable | Description | Default |
|----------|-------------|---------|
| `STATE_PERSISTENCE` | Save workflow state | `True` |
| `MEMORY_RETENTION_HOURS` | State retention period | `24` |

## Example .env File

```ini
# =============================================================================
# GridOps Configuration
# =============================================================================

# LLM Configuration
OPENAI_API_KEY=sk-proj-your-key-here
LLM_PROVIDER=openai
MODEL_NAME=gpt-4o-mini
TEMPERATURE=0.1

# System Behavior
LOG_LEVEL=INFO
MAX_ITERATIONS=10
TIMEOUT_SECONDS=300

# Network Constraints (per ANSI C84.1-2020)
VOLTAGE_TOLERANCE_PU=0.05
THERMAL_MARGIN_PERCENT=20.0
MAX_LOAD_SHED_PERCENT=30.0

# Report Configuration
REPORT_FORMAT=markdown
GENERATE_PLOTS=True
PLOT_DPI=300

# State Management
STATE_PERSISTENCE=True
MEMORY_RETENTION_HOURS=24
```

## Constraint Configuration

### Voltage Limits

The voltage limits are calculated from `VOLTAGE_TOLERANCE_PU`:
- **Minimum:** `1.0 - VOLTAGE_TOLERANCE_PU` = 0.95 pu
- **Maximum:** `1.0 + VOLTAGE_TOLERANCE_PU` = 1.05 pu

Per ANSI C84.1-2020, Range A specifies ±5% for distribution systems.

### Thermal Limits

- **Normal Limit:** 100% of rated capacity
- **Emergency Limit:** 100% + `THERMAL_MARGIN_PERCENT` = 120%

Per IEEE Std 738-2012, emergency ratings allow temporary overloading.

### Protection Coordination

Hardcoded in `src/config.py`:
- **Minimum Coordination Time:** 0.3 seconds
- **Maximum Fault Clearing Time:** 2.0 seconds

## Programmatic Configuration

You can also configure GridOps programmatically:

```python
from src.config import SystemConfig, NetworkConstraints, PathConfig

# Create custom configuration
config = SystemConfig(
    llm_provider="openai",
    model_name="gpt-4o",
    temperature=0.2,
    voltage_tolerance_pu=0.05
)

# Create constraints
constraints = NetworkConstraints(config)

# Access constraint values
print(f"Voltage range: {constraints.v_min_pu} - {constraints.v_max_pu} pu")
print(f"Thermal margin: {constraints.thermal_margin * 100}%")
```

## Model Options

### OpenAI Models

| Model | Speed | Cost | Quality | Recommended For |
|-------|-------|------|---------|-----------------|
| `gpt-4o-mini` | Fast | $ | Good | Development, testing |
| `gpt-4o` | Medium | $$ | Excellent | Production |
| `gpt-4-turbo` | Medium | $$$ | Excellent | Complex analysis |
| `gpt-3.5-turbo` | Very Fast | ¢ | Basic | Quick prototyping |

### Anthropic Models

| Model | Speed | Cost | Quality | Recommended For |
|-------|-------|------|---------|-----------------|
| `claude-3-haiku-20240307` | Fast | $ | Good | Development |
| `claude-3-sonnet-20240229` | Medium | $$ | Excellent | Production |
| `claude-3-opus-20240229` | Slow | $$$ | Best | Complex analysis |

## Directory Structure

GridOps uses the following directory structure (auto-created):

```
gridops-agentic-ai/
├── data/
│   ├── networks/          # Custom network files
│   └── references/        # Technical references
├── models/
│   └── knowledge_base/    # Agent knowledge base
├── reports/               # Generated reports
├── plots/                 # Generated visualizations
├── logs/                  # System logs
├── state/                 # Workflow state files
└── tests/                 # Test files
```

## Next Steps

- [CLI Reference](../user-guide/cli-reference.md) - Command-line options
- [Networks Guide](../user-guide/networks.md) - Available networks
- [Architecture Overview](../architecture/overview.md) - System design
