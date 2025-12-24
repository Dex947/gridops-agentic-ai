# Installation Guide

This guide covers the complete installation process for GridOps Agentic AI System.

## Prerequisites

### Required
- **Python 3.10 or higher** - [Download Python](https://www.python.org/downloads/)
- **pip** - Python package manager (included with Python)
- **Git** - For cloning the repository

### Optional
- **LaTeX distribution** - For PDF report generation (TeX Live, MiKTeX, or MacTeX)
- **OpenAI API key** - For GPT-based agents
- **Anthropic API key** - For Claude-based agents

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Dex947/gridops-agentic-ai.git
cd gridops-agentic-ai
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```ini
# LLM Configuration
OPENAI_API_KEY=sk-your-key-here
# or
ANTHROPIC_API_KEY=your-anthropic-key-here

# Optional: Override defaults
LLM_PROVIDER=openai
MODEL_NAME=gpt-4o-mini
TEMPERATURE=0.1
LOG_LEVEL=INFO
```

### 5. Verify Installation

```bash
python main.py --list-networks
```

You should see a list of available networks:
```
============================================================
AVAILABLE NETWORKS
============================================================

ieee_33:
  Buses: 33
  Lines: 37
  Loads: 32
  Total Load: 3.715 MW
...
```

## API Key Setup

### OpenAI API

1. Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Create a new API key
3. Add credits at [Billing Dashboard](https://platform.openai.com/account/billing)
4. Add to `.env`: `OPENAI_API_KEY=sk-your-key`

### Anthropic API

1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Create an API key
3. Add to `.env`: `ANTHROPIC_API_KEY=your-key`

## What Works Without API Keys?

Even without LLM API keys, the system provides:

| Feature | Available |
|---------|-----------|
| Power flow analysis | ✅ Yes |
| Contingency simulation (N-1, N-k) | ✅ Yes |
| Constraint violation detection | ✅ Yes |
| All visualizations | ✅ Yes |
| Markdown/LaTeX/PDF reports | ✅ Yes |
| IEEE standards retrieval | ✅ Yes |
| AI-generated action proposals | ❌ Requires LLM |
| AI-generated explanations | ❌ Requires LLM |

## Recommended Models

| Model | Speed | Cost | Quality | Use Case |
|-------|-------|------|---------|----------|
| `gpt-4o-mini` | Fast | Low | Good | Testing, development |
| `gpt-4o` | Medium | Medium | Excellent | Production |
| `gpt-3.5-turbo` | Very Fast | Very Low | Basic | Quick analysis |
| `claude-3-sonnet` | Medium | Medium | Excellent | Alternative to GPT-4 |

**Default:** `gpt-4o-mini` (best balance of speed, cost, and quality)

## Troubleshooting

### Python Version Issues

```bash
# Check Python version
python --version

# If using multiple Python versions
python3.10 -m venv venv
```

### Dependency Conflicts

```bash
# Create fresh environment
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Import Errors

Ensure you're in the project root directory:
```bash
cd gridops-agentic-ai
python main.py --list-networks
```

### API Key Issues

If you see `proposed_actions: 0`:
1. Check API key is valid
2. Verify account has credits
3. Check network connectivity

## Next Steps

- [Quick Start Guide](quickstart.md) - Run your first analysis
- [Configuration Reference](configuration.md) - All configuration options
- [CLI Reference](../user-guide/cli-reference.md) - Command-line options
