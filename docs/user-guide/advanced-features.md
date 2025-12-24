# Advanced Features

This guide covers the advanced analysis capabilities in GridOps.

## Optimal Power Flow (OPF)

The OPF module optimizes generator dispatch and network operation.

### Basic Usage

```python
from src.tools.opf_tools import OPFTool, OPFObjective, run_opf_analysis
from src.core.network_loader import NetworkLoader

# Load network
loader = NetworkLoader()
net = loader.load_network("ieee_33")

# Run AC OPF for loss minimization
tool = OPFTool(voltage_limits=(0.95, 1.05))
result = tool.run_ac_opf(net, OPFObjective.MIN_LOSS)

print(f"Converged: {result.converged}")
print(f"Total Losses: {result.total_losses_mw:.3f} MW")
print(f"Min Voltage: {result.min_voltage_pu:.4f} pu")
```

### Optimization Objectives

| Objective | Description |
|-----------|-------------|
| `MIN_LOSS` | Minimize total network losses |
| `MIN_GENERATION_COST` | Minimize generation cost |
| `MIN_LOAD_SHEDDING` | Minimize load curtailment |
| `MAX_LOADABILITY` | Maximize network loadability |

### DC OPF

For faster analysis, use DC OPF (linear approximation):

```python
result = tool.run_dc_opf(net, OPFObjective.MIN_LOSS)
```

---

## N-k Contingency Analysis

Analyze multiple simultaneous failures beyond N-1.

### Basic Usage

```python
from src.core.contingency_simulator import ContingencySimulator

simulator = ContingencySimulator(
    voltage_limits=(0.95, 1.05),
    thermal_limit_percent=100.0
)

# Run N-2 analysis (two simultaneous failures)
results = simulator.run_n_k_analysis(
    net, 
    k=2, 
    max_combinations=100
)

# Get ranked results
ranked = simulator.rank_contingencies_by_severity(results)
for item in ranked[:5]:
    print(f"{item['contingency']}: severity={item['severity_score']:.2f}")
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `k` | Number of simultaneous failures | 2 |
| `max_combinations` | Maximum combinations to analyze | 500 |
| `parallel` | Use parallel processing | False |
| `max_workers` | Number of parallel workers | 4 |

### Severity Ranking

Contingencies are ranked by severity score:
- Non-convergence: 100 points
- Voltage violations: (deviation × 100) points
- Thermal violations: (overload / 10) points
- Each violation: 5 points

---

## Network Reconfiguration

Optimize network topology through switching operations.

### Basic Usage

```python
from src.tools.reconfiguration import NetworkReconfiguration, optimize_network_topology

# Quick optimization
result = optimize_network_topology(net, objective="min_loss")

print(f"Baseline Losses: {result.baseline_losses_mw:.3f} MW")
print(f"Optimized Losses: {result.optimized_losses_mw:.3f} MW")
print(f"Loss Reduction: {result.loss_reduction_percent:.2f}%")
```

### Optimization Objectives

| Objective | Description |
|-----------|-------------|
| `min_loss` | Minimize total power losses |
| `min_voltage_deviation` | Minimize voltage deviation from 1.0 pu |
| `load_balancing` | Balance loading across feeders |
| `max_loadability` | Maximize network loadability |

### Algorithms

**Branch Exchange** (default):
- Iteratively swaps tie and sectionalizing switches
- Maintains radial topology
- Fast for large networks

```python
reconfig = NetworkReconfiguration()
result = reconfig.branch_exchange(net, max_iterations=100)
```

**Exhaustive Search**:
- Tests all valid configurations
- Optimal for small networks
- Limited by `max_configs`

```python
result = reconfig.exhaustive_search(net, max_configs=1000)
```

### Radiality Constraint

The optimizer ensures the network remains radial:
- Connected graph
- No loops (edges = nodes - 1)

---

## Combining Features

### OPF-Based Corrective Actions

```python
# After contingency, find optimal dispatch
from src.tools.opf_tools import OPFTool

tool = OPFTool()
opf_result = tool.run_ac_opf(post_contingency_net, OPFObjective.MIN_LOSS)

if opf_result.converged:
    print("OPF found feasible operating point")
```

### Reconfiguration After Contingency

```python
# After line outage, find new optimal topology
from src.tools.reconfiguration import optimize_network_topology

result = optimize_network_topology(
    post_contingency_net,
    objective="min_loss"
)
```

---

## Performance Considerations

### N-k Analysis
- N-2 on 33-bus: ~500 combinations, ~30 seconds
- N-3 on 33-bus: ~5000 combinations, use `max_combinations` limit
- Enable `parallel=True` for large analyses

### OPF
- AC OPF: More accurate, slower
- DC OPF: Fast approximation, no voltage results

### Reconfiguration
- Branch exchange: O(n²) per iteration
- Exhaustive: O(n!) - limit with `max_configs`

---

## API Reference

### OPFTool

```python
class OPFTool:
    def __init__(self, voltage_limits, thermal_limit_percent)
    def run_ac_opf(net, objective, init) -> OPFResult
    def run_dc_opf(net, objective) -> OPFResult
    def optimize_voltage_setpoints(net, target_profile) -> Dict
    def calculate_loss_sensitivity(net) -> Dict
```

### ContingencySimulator (N-k methods)

```python
class ContingencySimulator:
    def generate_n_k_contingencies(net, k, max_combinations) -> List
    def simulate_n_k_contingency(net, line_indices) -> ContingencyResult
    def run_n_k_analysis(net, k, max_combinations, parallel) -> List
    def rank_contingencies_by_severity(results) -> List
```

### NetworkReconfiguration

```python
class NetworkReconfiguration:
    def get_switchable_elements(net) -> List[SwitchState]
    def get_tie_switches(net) -> List[int]
    def check_radiality(net) -> bool
    def branch_exchange(net, objective, max_iterations) -> ReconfigurationResult
    def exhaustive_search(net, objective, max_configs) -> ReconfigurationResult
    def find_optimal_topology(net, objective, method) -> ReconfigurationResult
```
