# GridOps Quick Start Guide

Get started with GridOps in 5 minutes.

## Prerequisites

- Python 3.10+
- pip
- OpenAI or Anthropic API key

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API key
# Windows: notepad .env
# Linux/Mac: nano .env
```

Add one of:
```
OPENAI_API_KEY=sk-your-key-here
# OR
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 3. Verify Installation

```bash
python test_system.py
```

You should see: `✓ All tests passed! System is operational.`

## Your First Analysis

### Example 1: Simple Line Outage

```bash
python main.py \
  --network ieee_33 \
  --contingency "Line 5 outage" \
  --type line_outage \
  --elements 5
```

**What happens:**
1. Loads IEEE 33-bus network
2. Runs baseline power flow
3. Simulates line 5 outage
4. Agents generate 2-5 corrective action plans
5. Evaluates each plan with power flow
6. Selects safest action
7. Generates explanation and reports

**Outputs:**
- `reports/run_<timestamp>.md` - Markdown report
- `reports/run_<timestamp>.tex` - LaTeX report  
- `plots/*.png` - Visualizations
- `state/state_<session>.json` - Workflow state

### Example 2: View Available Networks

```bash
python main.py --list-networks
```

### Example 3: Multiple Elements

```bash
python main.py \
  --network ieee_33 \
  --contingency "Lines 5 and 10 outage" \
  --type line_outage \
  --elements 5 10
```

## Understanding the Output

### 1. Terminal Output

```
============================================================
GRIDOPS AGENTIC AI SYSTEM
============================================================
Session ID: 20251008_143020
Network: ieee_33
Contingency: Line 5 outage
...
============================================================
EXECUTION SUMMARY
============================================================
{
  "network": "ieee_33",
  "contingency": "Line 5 outage",
  "status": "complete",
  "proposed_actions": 3,
  "selected_action": "close_tie_switch_20"
}
```

### 2. Markdown Report

Open `reports/run_<timestamp>.md` for:
- Executive summary
- Network characteristics
- Baseline vs contingency analysis
- Proposed actions table
- Selected action details
- Technical explanation
- Visualizations

### 3. Visualizations

Check `plots/` directory for:
- **voltage_profile.png**: Bus voltages before/after
- **line_loading.png**: Line loading distribution
- **network_topology.png**: Network graph
- **comparison.png**: Performance metrics

## Common Use Cases

### Use Case 1: N-1 Contingency Planning

Test if network can handle any single line failure:

```bash
# Test critical line
python main.py --network ieee_33 \
  --contingency "Line 10 outage" \
  --type line_outage --elements 10
```

**Look for:** Violations resolved, safety score > 0.7

### Use Case 2: Load Growth Analysis

Analyze impact of load increase:

```bash
python main.py --network ieee_33 \
  --contingency "50% load increase at bus 15" \
  --type load_increase --elements 15
```

**Look for:** Thermal violations, proposed load shedding

### Use Case 3: Transformer Outage

```bash
python main.py --network ieee_33 \
  --contingency "Transformer 2 outage" \
  --type transformer_outage --elements 2
```

## Customization

### Change Network Constraints

Edit `.env`:
```
VOLTAGE_TOLERANCE_PU=0.06  # Allow ±6% instead of ±5%
THERMAL_MARGIN_PERCENT=30   # 30% margin instead of 20%
```

### Use Different LLM

Edit `.env`:
```
LLM_PROVIDER=anthropic
MODEL_NAME=claude-3-5-sonnet-20241022
TEMPERATURE=0.15
```

### Skip Plot Generation

```bash
python main.py --network ieee_33 \
  --contingency "Line 5 outage" \
  --type line_outage --elements 5 \
  --no-plots
```

## Troubleshooting

### Issue: "Configuration failed"

**Solution:** Check `.env` file exists and is properly formatted

### Issue: "Power flow did not converge"

**Solutions:**
- Check network has no isolated buses
- Verify load/generation balance
- Try different contingency element
- Reduce contingency severity

### Issue: "API rate limit exceeded"

**Solutions:**
- Wait a few minutes
- Reduce `TEMPERATURE` in `.env`
- Use different API key
- Switch to alternative provider

### Issue: "No suitable action found"

**Interpretation:** The contingency is severe and no safe solution exists within constraints. This is valid output indicating system limitations.

## Next Steps

1. **Read Full Documentation**: See [README.md](README.md)
2. **Explore Networks**: Try different IEEE test feeders
3. **Custom Networks**: Add your own in `data/networks/`
4. **Modify Agents**: Customize agent behavior in `src/agents/`
5. **Add Standards**: Extend reference database in `src/agents/retrieval.py`

## Getting Help

- **Documentation**: README.md, CONTRIBUTING.md
- **Examples**: Check `__main__` blocks in source files
- **Test System**: Run `python test_system.py`
- **Issues**: Open GitHub issue with logs

## Quick Reference

### Command Structure

```bash
python main.py \
  --network <name> \           # Network to analyze
  --contingency "<desc>" \     # Human description
  --type <type> \              # Contingency type
  --elements <ids> \           # Element indices (space-separated)
  [--no-plots] \               # Skip visualization
  [--no-latex] \               # Skip LaTeX report
  [--output-dir <path>] \      # Custom output directory
  [--log-level <level>]        # DEBUG, INFO, WARNING, ERROR
```

### Contingency Types

- `line_outage` - Line disconnection
- `transformer_outage` - Transformer failure
- `load_increase` - Load growth scenario
- `generator_outage` - Generator trip

### Output Locations

- Reports: `reports/run_<timestamp>.md|.tex`
- Plots: `plots/*.png`
- State: `state/state_<session>.json`
- Logs: `logs/gridops_<date>.log`

---

**Ready to analyze contingencies!** 🚀

Start with: `python main.py --network ieee_33 --contingency "Line 5 outage" --type line_outage --elements 5`
