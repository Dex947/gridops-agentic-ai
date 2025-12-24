# Networks Guide

GridOps supports multiple distribution network models for contingency analysis.

## Built-in Networks

### IEEE 33-Bus (`ieee_33`)

The IEEE 33-bus radial distribution system is a standard test feeder.

| Parameter | Value |
|-----------|-------|
| Buses | 33 |
| Lines | 37 (32 sectionalizing + 5 tie) |
| Loads | 32 |
| Total Load | 3.715 MW, 2.300 MVAr |
| Voltage | 12.66 kV |
| Topology | Radial with tie switches |

```bash
python main.py --network ieee_33 --contingency "Line 5 outage" --type line_outage --elements 5
```

### CIGRE MV Benchmark (`cigre_mv`)

CIGRE medium voltage benchmark network for European distribution systems.

| Parameter | Value |
|-----------|-------|
| Buses | 15 |
| Lines | 15 |
| Loads | 9 |
| Total Load | ~14 MW |
| Voltage | 20 kV |

```bash
python main.py --network cigre_mv --contingency "Line 3 outage" --type line_outage --elements 3
```

### Kerber Networks

German distribution network models:

- **`kerber_landnetz`** - Rural network (Landnetz)
- **`kerber_dorfnetz`** - Village network (Dorfnetz)

## Listing Available Networks

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

kerber_landnetz:
  Buses: 15
  Lines: 14
  Loads: 13
  Total Load: 0.057 MW
...
```

## Custom Networks

### Creating a Custom Network

1. Create a pandapower network in Python:

```python
import pandapower as pp

# Create empty network
net = pp.create_empty_network(name="My Custom Network")

# Add buses
bus0 = pp.create_bus(net, vn_kv=11.0, name="Substation")
bus1 = pp.create_bus(net, vn_kv=11.0, name="Bus 1")
bus2 = pp.create_bus(net, vn_kv=11.0, name="Bus 2")

# Add external grid (slack bus)
pp.create_ext_grid(net, bus=bus0, vm_pu=1.02, name="Grid")

# Add lines
pp.create_line(net, from_bus=bus0, to_bus=bus1, length_km=1.5,
               std_type="NAYY 4x150 SE", name="Line 0-1")
pp.create_line(net, from_bus=bus1, to_bus=bus2, length_km=1.0,
               std_type="NAYY 4x150 SE", name="Line 1-2")

# Add loads
pp.create_load(net, bus=bus1, p_mw=0.5, q_mvar=0.2, name="Load 1")
pp.create_load(net, bus=bus2, p_mw=0.8, q_mvar=0.3, name="Load 2")

# Save to JSON
pp.to_json(net, "data/networks/my_network.json")
```

2. Use the custom network:

```bash
python main.py --network my_network --contingency "Line 0 outage" --type line_outage --elements 0
```

### Network File Location

Custom networks should be saved in:
```
data/networks/<network_name>.json
```

### Required Network Elements

A valid network must have:

| Element | Required | Description |
|---------|----------|-------------|
| `bus` | Yes | At least 2 buses |
| `ext_grid` | Yes | At least 1 external grid (slack) |
| `line` | Yes | At least 1 line |
| `load` | Recommended | Loads for realistic analysis |

### Optional Elements

| Element | Description |
|---------|-------------|
| `trafo` | Transformers |
| `gen` | Generators |
| `sgen` | Static generators (DER) |
| `switch` | Explicit switches |
| `shunt` | Shunt elements (capacitors) |

## Network Validation

GridOps automatically validates networks on load:

1. **Connectivity check** - Ensures all buses are reachable
2. **Power flow test** - Runs initial power flow
3. **Line rating fix** - Corrects unrealistic line ratings

### Common Issues

#### Unrealistic Line Ratings

Some test networks have `max_i_ka = 99999` which causes 0% loading. GridOps automatically corrects this.

#### Isolated Buses

If buses become isolated after contingency, power flow may not converge. Check network topology.

#### Missing Slack Bus

Every network needs at least one `ext_grid` element as the slack bus.

## Network Analysis Tools

### Get Network Summary

```python
from src.core.network_loader import NetworkLoader
from src.config import load_configuration

config, constraints, paths = load_configuration()
loader = NetworkLoader(networks_path=paths.networks)

net = loader.load_network("ieee_33")
summary = loader.get_network_summary(net)
print(summary)
```

### Find Switchable Elements

```python
switchable = loader.get_switchable_elements(net)
for elem in switchable[:5]:
    print(elem)
```

### Analyze Topology

```python
from src.tools.network_analysis import NetworkAnalyzer

analyzer = NetworkAnalyzer(net)

# Find alternative paths
paths = analyzer.find_alternative_paths(0, 10, max_paths=3)

# Identify critical lines
critical = analyzer.identify_critical_lines(threshold_loading=80.0)

# Calculate bus centrality
centrality = analyzer.calculate_centrality()
```

## See Also

- [Quick Start Guide](../getting-started/quickstart.md)
- [CLI Reference](cli-reference.md)
- [Architecture Overview](../architecture/overview.md)
