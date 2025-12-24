# Quick Start Guide

Get up and running with GridOps in minutes.

## Basic Usage

### Analyze a Line Outage

```bash
python main.py \
  --network ieee_33 \
  --contingency "Line 5 outage causing downstream undervoltage" \
  --type line_outage \
  --elements 5
```

### What This Does

1. Loads the IEEE 33-bus distribution network
2. Runs baseline power flow analysis
3. Simulates Line 5 outage
4. Retrieves relevant IEEE standards
5. Generates corrective action proposals (if LLM available)
6. Evaluates and validates each proposal
7. Selects the best action
8. Generates explanation and reports

## Command-Line Options

```bash
python main.py --help
```

| Option | Description | Default |
|--------|-------------|---------|
| `--network` | Network name to analyze | `ieee_33` |
| `--contingency` | Contingency description | Required |
| `--type` | Contingency type | `line_outage` |
| `--elements` | Affected element indices | Required |
| `--no-plots` | Skip plot generation | False |
| `--no-latex` | Skip LaTeX report | False |
| `--output-dir` | Custom output directory | `./reports` |
| `--log-level` | Override log level | From config |
| `--list-networks` | List available networks | - |

## Contingency Types

| Type | Description | Example |
|------|-------------|---------|
| `line_outage` | Line/cable failure | `--type line_outage --elements 5` |
| `transformer_outage` | Transformer failure | `--type transformer_outage --elements 0` |
| `load_increase` | Sudden load increase | `--type load_increase --elements 15` |
| `generator_outage` | Generator trip | `--type generator_outage --elements 0` |

## Multiple Element Contingencies

Analyze N-k contingencies with multiple elements:

```bash
python main.py \
  --network ieee_33 \
  --contingency "Lines 5 and 8 simultaneous outage" \
  --type line_outage \
  --elements 5 8
```

## Output Files

Each execution generates:

### Reports (`reports/`)
- `run_<timestamp>.md` - Markdown report
- `run_<timestamp>.tex` - LaTeX report (compile with `pdflatex`)

### Visualizations (`plots/`)
- `voltage_profile.png` - Bus voltage comparison
- `line_loading.png` - Line loading distribution
- `network_topology.png` - Network graph
- `comparison.png` - Performance metrics

### State (`state/`)
- `state_<session_id>.json` - Complete workflow state

## Example Scenarios

### Scenario 1: Feeder Outage
```bash
python main.py \
  --network ieee_33 \
  --contingency "Main feeder line 0 outage" \
  --type line_outage \
  --elements 0
```

### Scenario 2: Load Growth
```bash
python main.py \
  --network ieee_33 \
  --contingency "50% load increase at bus 15" \
  --type load_increase \
  --elements 15
```

### Scenario 3: Multiple Failures
```bash
python main.py \
  --network ieee_33 \
  --contingency "Lines 3, 5, and 10 outage during storm" \
  --type line_outage \
  --elements 3 5 10
```

## Understanding the Output

### Execution Summary

```json
{
  "network": "ieee_33",
  "contingency": "Line 5 outage",
  "status": "complete",
  "proposed_actions": 3,
  "selected_action": "close_tie_switch_21",
  "violations": 5
}
```

### Key Metrics

| Metric | Healthy Range | Description |
|--------|---------------|-------------|
| Voltage | 0.95-1.05 pu | Per ANSI C84.1-2020 |
| Line Loading | < 100% | Thermal limit |
| Losses | Varies | System efficiency |

## Next Steps

- [Configuration Reference](configuration.md) - Customize settings
- [Networks Guide](../user-guide/networks.md) - Available networks
- [Architecture Overview](../architecture/overview.md) - How it works
