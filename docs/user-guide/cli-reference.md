# Command-Line Interface Reference

Complete reference for GridOps command-line options.

## Synopsis

```bash
python main.py [OPTIONS]
```

## Options

### Required Options

| Option | Description | Example |
|--------|-------------|---------|
| `--contingency` | Contingency description | `"Line 5 outage"` |
| `--elements` | Affected element indices | `5` or `5 8 10` |

### Network Options

| Option | Description | Default |
|--------|-------------|---------|
| `--network` | Network name to analyze | `ieee_33` |
| `--list-networks` | List available networks and exit | - |

### Contingency Options

| Option | Description | Default | Choices |
|--------|-------------|---------|---------|
| `--type` | Contingency type | `line_outage` | `line_outage`, `transformer_outage`, `load_increase`, `generator_outage` |

### Output Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output-dir` | Custom output directory | `./reports` |
| `--no-plots` | Skip plot generation | `False` |
| `--no-latex` | Skip LaTeX report | `False` |

### System Options

| Option | Description | Default | Choices |
|--------|-------------|---------|---------|
| `--log-level` | Override log level | From config | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

## Usage Examples

### Basic Analysis

```bash
python main.py \
  --network ieee_33 \
  --contingency "Line 5 outage" \
  --type line_outage \
  --elements 5
```

### Multiple Element Contingency

```bash
python main.py \
  --network ieee_33 \
  --contingency "Lines 5 and 8 simultaneous outage" \
  --type line_outage \
  --elements 5 8
```

### Custom Output Directory

```bash
python main.py \
  --network ieee_33 \
  --contingency "Line 10 outage" \
  --type line_outage \
  --elements 10 \
  --output-dir ./my_reports
```

### Skip Visualizations

```bash
python main.py \
  --network ieee_33 \
  --contingency "Line 5 outage" \
  --type line_outage \
  --elements 5 \
  --no-plots
```

### Debug Mode

```bash
python main.py \
  --network ieee_33 \
  --contingency "Line 5 outage" \
  --type line_outage \
  --elements 5 \
  --log-level DEBUG
```

### List Available Networks

```bash
python main.py --list-networks
```

Output:
```
============================================================
AVAILABLE NETWORKS
============================================================

ieee_33:
  Buses: 33
  Lines: 37
  Loads: 32
  Total Load: 3.715 MW

cigre_mv:
  Buses: 15
  Lines: 15
  Loads: 9
  Total Load: 13.890 MW
...
```

## Contingency Types

### Line Outage (`line_outage`)

Simulates line/cable failure by taking the specified line(s) out of service.

```bash
--type line_outage --elements 5
```

### Transformer Outage (`transformer_outage`)

Simulates transformer failure.

```bash
--type transformer_outage --elements 0
```

### Load Increase (`load_increase`)

Simulates sudden load increase at specified bus(es).

```bash
--type load_increase --elements 15
```

### Generator Outage (`generator_outage`)

Simulates generator trip.

```bash
--type generator_outage --elements 0
```

## Output Files

### Reports Directory (`reports/`)

| File | Description |
|------|-------------|
| `run_<timestamp>.md` | Markdown report |
| `run_<timestamp>.tex` | LaTeX report |

### Plots Directory (`plots/`)

| File | Description |
|------|-------------|
| `voltage_profile.png` | Bus voltage comparison |
| `line_loading.png` | Line loading distribution |
| `network_topology.png` | Network graph |
| `comparison.png` | Performance metrics |

### State Directory (`state/`)

| File | Description |
|------|-------------|
| `state_<session_id>.json` | Complete workflow state |

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Error (check logs) |
| `130` | Interrupted by user (Ctrl+C) |

## Environment Variables

The CLI respects environment variables from `.env`:

```bash
# Override model for this run
MODEL_NAME=gpt-4o python main.py --network ieee_33 ...

# Override log level
LOG_LEVEL=DEBUG python main.py --network ieee_33 ...
```

## See Also

- [Quick Start Guide](../getting-started/quickstart.md)
- [Configuration Reference](../getting-started/configuration.md)
- [Networks Guide](networks.md)
