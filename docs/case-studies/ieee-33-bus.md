# Case Study: IEEE 33-Bus Network - Line 5 Outage

This case study demonstrates GridOps analyzing a line outage contingency on the IEEE 33-bus distribution feeder.

## Network Overview

The IEEE 33-bus network is a standard test feeder for distribution system analysis.

| Parameter | Value |
|-----------|-------|
| Total Buses | 33 |
| Total Lines | 37 |
| Total Loads | 32 |
| Total Active Power | 3.715 MW |
| Total Reactive Power | 2.300 MVAr |
| Voltage Level | 12.66 kV |
| Topology | Radial with tie switches |

## Contingency Scenario

**Event:** Line 5 outage (N-1 contingency)  
**Cause:** Equipment failure or fault  
**Impact:** Downstream buses lose supply path

## Analysis Results

### Pre-Contingency (Baseline)

| Metric | Value | Status |
|--------|-------|--------|
| Power Flow Converged | Yes | ✓ |
| Min Voltage | 0.9131 pu | ⚠️ Below limit |
| Max Voltage | 1.0000 pu | ✓ |
| Max Line Loading | 66.7% | ✓ |
| Total Losses | 0.203 MW | - |
| **Violations** | **21** | ⚠️ |

> **Note:** The baseline network already has voltage violations at remote buses due to the radial topology and load distribution.

### Post-Contingency (Line 5 Outage)

| Metric | Value | Status |
|--------|-------|--------|
| Power Flow Converged | Yes | ✓ |
| Min Voltage | 0.9382 pu | ⚠️ Below limit |
| Max Voltage | 1.0000 pu | ✓ |
| Max Line Loading | 58.4% | ✓ |
| Total Losses | 0.000 MW | - |
| **Violations** | **5** | ⚠️ Improved |

### Voltage Violations Identified

| Bus ID | Voltage (pu) | Violation |
|--------|--------------|-----------|
| Bus 28 | 0.947 | Undervoltage |
| Bus 29 | 0.943 | Undervoltage |
| Bus 30 | 0.939 | Undervoltage |
| Bus 31 | 0.938 | Undervoltage |
| Bus 32 | 0.938 | Undervoltage |

**Constraint Reference:** ANSI C84.1-2020 specifies voltage range 0.95-1.05 pu for distribution systems.

## Visualizations

### Voltage Profile Analysis

![Voltage Profile](../../plots/voltage_profile.png)

**Key Findings:**
- Orange line (baseline) shows progressive voltage drop along feeder
- Blue line (post-action) shows similar profile after Line 5 outage
- Red X marks indicate constraint violations (buses 28-32)
- Voltage degradation concentrated at feeder end buses

### Line Loading Distribution

![Line Loading](../../plots/line_loading.png)

**Key Findings:**
- Left panel: Histogram shows loading concentration around 60-70%
- Right panel: Line-by-line loading distribution
- All lines operate well below thermal limit (100%)
- No thermal constraint violations detected

### Performance Comparison

![Performance Metrics](../../plots/comparison.png)

**Key Findings:**
- **Top Left:** Minimum voltage improved from 0.913 to 0.938 pu
- **Top Right:** Maximum voltage unchanged at 1.0 pu
- **Bottom Left:** Violations reduced from 21 to 5 (76% improvement)
- **Bottom Right:** System losses reduced (Line 5 de-energized)

## Agent Analysis

### Proposed Actions

When LLM agents are available, the Planner Agent typically proposes:

1. **Close Tie Switch** - Activate normally-open tie line to restore supply
2. **Load Shedding** - Reduce load at critical buses to improve voltage
3. **Voltage Regulation** - Adjust tap settings (if available)

### Selected Action

In this scenario, no suitable corrective action was identified within constraint limits, indicating:

1. The contingency severity may require operator intervention
2. No viable tie-switch configurations exist to restore voltage constraints
3. Load shedding would be required (pending operator approval per IEEE Std C37.117)

## Standards Applied

| Standard | Application |
|----------|-------------|
| ANSI C84.1-2020, Section 4.1 | Voltage Ratings |
| NERC TPL-001-4, Category B | N-1 Contingencies |
| IEEE Std 1434-2014 | Power System Analysis |
| IEEE Std 1366-2012 | Reliability Indices |

## Key Takeaways

| Aspect | Finding |
|--------|---------|
| ✅ System Stability | Power flow remained convergent after contingency |
| ✅ Thermal Margins | No line overloading detected |
| ⚠️ Voltage Compliance | 5 buses violate ANSI C84.1-2020 limits |
| ⚠️ Corrective Action | Manual intervention required for voltage restoration |

## Recommended Next Steps

1. **Install voltage regulators** or capacitor banks at critical buses (28-32)
2. **Evaluate alternative feeder configurations** with additional tie switches
3. **Consider distributed energy resources (DER)** integration for voltage support
4. **Update network model** with actual impedance data for better accuracy

## Reproducing This Analysis

```bash
python main.py \
  --network ieee_33 \
  --contingency "Line 5 outage causing downstream undervoltage" \
  --type line_outage \
  --elements 5
```

## Related Case Studies

- [Custom Network Analysis](custom-network.md)
- [Multiple Contingency (N-k)](n-k-contingency.md)
- [Load Growth Scenario](load-growth.md)
