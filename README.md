# GridOps Agentic AI System

[![GitHub](https://img.shields.io/badge/GitHub-Dex947%2Fgridops--agentic--ai-blue?logo=github)](https://github.com/Dex947/gridops-agentic-ai)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange)](https://github.com/langchain-ai/langgraph)
[![pandapower](https://img.shields.io/badge/pandapower-Power%20Flow-red)](https://www.pandapower.org/)

> Multi-agent AI system for safe distribution feeder reconfiguration and load-shedding during contingencies.

## 🎯 Key Features

- **Multi-Agent Architecture** — Five specialized AI agents coordinated through LangGraph
- **Power System Analysis** — Three-phase unbalanced power flow using pandapower
- **Contingency Simulation** — Automated N-1 and N-k contingency analysis
- **Constraint Validation** — Voltage, thermal, and protection coordination checks
- **Explainable AI** — Human-readable explanations with IEEE standard citations
- **Automated Reporting** — Markdown, LaTeX, and PDF reports with visualizations

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Orchestrator                    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Planner  │→ │PowerFlow │→ │Constraint│→ │Explainer │   │
│  │  Agent   │  │   Agent  │  │ Checker  │  │  Agent   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│       ↓              ↓              ↓              ↑        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Retrieval Agent (Standards)            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  pandapower (Power Flow) • NetworkX (Topology) • matplotlib │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Dex947/gridops-agentic-ai.git
cd gridops-agentic-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your OpenAI or Anthropic API key
```

### Run Analysis

```bash
# Analyze line outage on IEEE 33-bus network
python main.py \
  --network ieee_33 \
  --contingency "Line 5 outage" \
  --type line_outage \
  --elements 5

# List available networks
python main.py --list-networks
```

### Example Output

```
================================================================================
EXECUTION SUMMARY
================================================================================
{
  "network": "ieee_33",
  "contingency": "Line 5 outage",
  "status": "complete",
  "proposed_actions": 3,
  "selected_action": "close_tie_switch_21",
  "violations": 5
}
```

## 📖 Documentation

| Guide | Description |
|-------|-------------|
| [Installation](docs/getting-started/installation.md) | Complete setup instructions |
| [Quick Start](docs/getting-started/quickstart.md) | Get running in minutes |
| [Configuration](docs/getting-started/configuration.md) | All configuration options |
| [CLI Reference](docs/user-guide/cli-reference.md) | Command-line options |
| [Networks](docs/user-guide/networks.md) | Available networks and custom networks |
| [Architecture](docs/architecture/overview.md) | System design and components |
| [Case Studies](docs/case-studies/ieee-33-bus.md) | Real analysis examples |

## 📊 Sample Visualizations

<table>
<tr>
<td><img src="plots/voltage_profile.png" alt="Voltage Profile" width="400"/></td>
<td><img src="plots/line_loading.png" alt="Line Loading" width="400"/></td>
</tr>
<tr>
<td align="center"><b>Voltage Profile</b></td>
<td align="center"><b>Line Loading</b></td>
</tr>
</table>

## 🔧 Project Structure

```
gridops-agentic-ai/
├── src/
│   ├── agents/           # AI agents (Planner, Explainer, etc.)
│   ├── core/             # Network loader, contingency simulator
│   ├── tools/            # Power flow, network analysis
│   ├── orchestrator.py   # LangGraph workflow
│   └── config.py         # Configuration
├── docs/                 # Documentation
├── data/networks/        # Custom network files
├── reports/              # Generated reports
├── plots/                # Generated visualizations
└── main.py               # CLI entry point
```

## 🧪 Testing

```bash
# Run system validation (no LLM required)
python test_system.py

# Run pytest suite
pytest tests/ -v
```

## 📋 Roadmap

See [ROADMAP.md](ROADMAP.md) for the development roadmap from cleanup to expansion.

**Current Focus:**
- ✅ Core multi-agent workflow
- ✅ Power flow analysis and visualization
- 🔄 Testing infrastructure
- 📅 Web interface
- 📅 Real-time data integration

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
pip install -e .
pip install pytest pytest-cov black ruff mypy

# Run checks
black src/ tests/
ruff check src/
pytest tests/ -v --cov=src
```

## 📚 References

- [IEEE Std 1547-2018](https://standards.ieee.org/standard/1547-2018.html) — DER Interconnection
- [ANSI C84.1-2020](https://www.nema.org/standards/view/american-national-standard-for-electric-power-systems-and-equipment-voltage-ratings-60-hertz) — Voltage Ratings
- [IEEE Std 242-2001](https://standards.ieee.org/standard/242-2001.html) — Protection Coordination
- [pandapower](https://www.pandapower.org/) — Power system analysis
- [LangGraph](https://github.com/langchain-ai/langgraph) — Agent orchestration

## 📄 Citation

```bibtex
@software{gridops2025,
  title={GridOps: Agentic AI System for Distribution Network Contingency Management},
  author={Dex947},
  year={2025},
  url={https://github.com/Dex947/gridops-agentic-ai}
}
```

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

**Built with:** Python • pandapower • LangGraph • OpenAI/Anthropic • NetworkX • Matplotlib
