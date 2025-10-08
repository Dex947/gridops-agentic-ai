# Contributing to GridOps Agentic AI System

Thank you for your interest in contributing to GridOps! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project follows a standard code of conduct. Be respectful, inclusive, and professional in all interactions.

## How to Contribute

### Reporting Bugs

1. Check existing [GitHub Issues](https://github.com/your-repo/gridops/issues)
2. Create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version)
   - Relevant logs and error messages

### Suggesting Enhancements

1. Open an issue with the `enhancement` label
2. Describe the feature and its benefits
3. Provide use cases and examples
4. Discuss implementation approach if possible

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the coding style (see below)
   - Add tests for new functionality
   - Update documentation
   - Keep commits atomic and well-described

4. **Test your changes**
   ```bash
   # Run existing tests
   python -m pytest tests/
   
   # Test specific modules
   python src/your_module.py
   ```

5. **Update CHANGELOG.md**
   - Add entry under `[Unreleased]` section
   - Follow existing format

6. **Commit your changes**
   ```bash
   git commit -m "Add feature: brief description"
   ```

7. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Open a Pull Request**
   - Provide clear title and description
   - Reference related issues
   - Explain what changed and why

## Coding Style

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use meaningful variable and function names

### Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings:
  ```python
  def example_function(param1: str, param2: int) -> bool:
      """
      Brief description of function.
      
      Args:
          param1: Description of param1
          param2: Description of param2
      
      Returns:
          Description of return value
      
      Raises:
          ValueError: Description of when this is raised
      """
      pass
  ```

### Code Structure

- Keep functions focused and modular
- Avoid deep nesting (max 3-4 levels)
- Use meaningful imports (avoid `from module import *`)
- Handle exceptions appropriately

### Logging

- Use `loguru` for logging
- Appropriate log levels:
  - `DEBUG`: Detailed diagnostic information
  - `INFO`: General informational messages
  - `WARNING`: Warning messages for recoverable issues
  - `ERROR`: Error messages for failures

### Testing

- Write unit tests for new functionality
- Use `pytest` framework
- Aim for >80% code coverage
- Test edge cases and error conditions

## Project Structure Guidelines

### Adding New Agents

1. Create agent file in `src/agents/`
2. Inherit from base patterns in existing agents
3. Implement required methods:
   - `__init__()`: Initialize with configuration
   - Main processing method
   - Helper methods as needed

4. Update `src/agents/__init__.py`
5. Add to orchestrator workflow if needed
6. Write tests in `tests/agents/`
7. Update documentation

### Adding New Tools

1. Create tool file in `src/tools/`
2. Implement as standalone functions or classes
3. Add comprehensive docstrings
4. Include `__main__` section with test/example
5. Update `src/tools/__init__.py`
6. Document in README.md

### Adding New Networks

1. Convert network to pandapower format
2. Save as JSON in `data/networks/`
3. Add to `NetworkLoader.AVAILABLE_NETWORKS` if built-in
4. Test with:
   ```bash
   python main.py --network your_network --list-networks
   ```

## Development Setup

### Environment Setup

```bash
# Clone repository
git clone https://github.com/your-repo/gridops.git
cd gridops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black ruff mypy
```

### Pre-commit Checks

Before committing:

```bash
# Format code
black src/ tests/

# Check style
ruff src/ tests/

# Type checking
mypy src/

# Run tests
pytest tests/ -v --cov=src
```

## Documentation Guidelines

### README Updates

- Update README.md for new features
- Add examples for new functionality
- Update feature list and architecture diagrams
- Keep installation instructions current

### Code Comments

- Comment complex logic
- Explain non-obvious decisions
- Reference standards/papers where applicable
- Keep comments up-to-date with code

### Changelog

Follow [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
## [Unreleased]

### Added
- New feature description

### Changed
- Changed behavior description

### Fixed
- Bug fix description

### Deprecated
- Deprecated feature description

### Removed
- Removed feature description
```

## Review Process

1. All PRs require at least one review
2. Address review comments promptly
3. Keep PR focused and reasonably sized
4. Ensure CI/CD checks pass
5. Maintainers will merge after approval

## Questions?

- Open a GitHub Discussion for general questions
- Open an Issue for specific problems
- Email maintainers for sensitive topics

## Recognition

Contributors will be acknowledged in:
- CHANGELOG.md for specific contributions
- README.md contributors section
- Release notes

Thank you for contributing to GridOps!
