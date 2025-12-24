# GridOps Agentic AI System Documentation

Welcome to the GridOps documentation. GridOps is a multi-agent AI system for safe distribution feeder reconfiguration and load-shedding during contingencies.

## Quick Links

- [Getting Started](getting-started/installation.md)
- [User Guide](user-guide/cli-reference.md)
- [Architecture](architecture/overview.md)
- [API Reference](api/core.md)
- [Case Studies](case-studies/ieee-33-bus.md)

## What is GridOps?

GridOps is an autonomous multi-agent system that analyzes distribution network contingencies and proposes safe reconfiguration strategies. The system uses five specialized AI agents coordinated through LangGraph to:

1. **Analyze** network contingencies (N-1, N-k scenarios)
2. **Generate** candidate switching and load-shedding plans
3. **Validate** thermal, voltage, and protection constraints
4. **Explain** decisions with technical rationale
5. **Report** findings with visualizations and references

## Key Features

- **Multi-Agent Architecture**: Five specialized agents working together
- **Power System Analysis**: Three-phase unbalanced power flow using pandapower
- **Contingency Simulation**: Automated N-1 and N-k contingency analysis
- **Constraint Validation**: Voltage, thermal, and protection coordination checks
- **Explainable AI**: Human-readable technical explanations with IEEE standard citations
- **Automated Reporting**: Markdown and LaTeX reports with visualizations

## System Requirements

- Python 3.10 or higher
- OpenAI or Anthropic API key (for LLM agents)
- Optional: LaTeX distribution for PDF report generation

## Getting Help

- [GitHub Issues](https://github.com/Dex947/gridops-agentic-ai/issues) - Bug reports and feature requests
- [GitHub Discussions](https://github.com/Dex947/gridops-agentic-ai/discussions) - Questions and community support
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute

## License

GridOps is released under the MIT License. See [LICENSE](../LICENSE) for details.
