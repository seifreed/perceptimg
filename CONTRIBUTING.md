# Contributing to perceptimg

Thank you for your interest in contributing to perceptimg! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your changes

## Development Setup

```bash
# Clone the repository
git clone https://github.com/seifreed/perceptimg.git
cd perceptimg

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pre-commit install
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=perceptimg --cov-report=term-missing

# Run specific test file
pytest tests/test_optimizer.py -v

# Run specific test
pytest tests/test_optimizer.py::TestOptimizer::test_optimize -v
```

## Code Style

This project uses the following tools for code quality:

| Tool | Purpose |
|------|---------|
| `ruff` | Linting |
| `black` | Code formatting |
| `mypy` | Type checking |
| `bandit` | Security analysis |

```bash
# Run all checks
ruff check perceptimg tests
black perceptimg tests --check
mypy perceptimg --ignore-missing-imports
bandit -r perceptimg -q --exclude perceptimg/tests
```

### Code Style Guidelines

1. **Follow PEP 8**: Use snake_case for functions/variables, PascalCase for classes
2. **Type hints**: All public functions must have type hints
3. **Docstrings**: Use Google-style docstrings for all public functions/classes
4. **Line length**: Maximum 100 characters
5. **Imports**: Use absolute imports from package root

### Example

```python
from __future__ import annotations

from typing import Sequence

from perceptimg.core.policy import Policy


def optimize_batch(
    images: Sequence[str],
    policy: Policy,
    max_workers: int | None = None,
) -> BatchResult:
    """Optimize multiple images in parallel.

    Args:
        images: List of image paths to optimize.
        policy: Optimization policy to apply.
        max_workers: Maximum number of parallel workers. Defaults to CPU count.

    Returns:
        BatchResult with successful and failed results.

    Raises:
        ValueError: If images list is empty.
    """
    if not images:
        raise ValueError("Images list cannot be empty")
    # Implementation...
```

## Submitting Changes

### Pull Request Process

1. **Create a branch**: `git checkout -b feature/amazing-feature`
2. **Make your changes**: Follow code style guidelines
3. **Add tests**: Ensure new code is tested
4. **Run tests**: All tests must pass
5. **Commit**: Use conventional commit messages
6. **Push**: `git push origin feature/amazing-feature`
7. **Open PR**: Create a Pull Request on GitHub

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:
```
feat(optimizer): add support for custom quality presets

- Add QualityPreset enum for predefined quality levels
- Support custom quality ranges in Policy
- Update documentation with examples

Closes #123
```

### Pull Request Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines (ruff, black, mypy, bandit)
- [ ] Documentation updated if needed
- [ ] CHANGELOG.md updated (if applicable)
- [ ] Commit messages follow convention

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Python version**: `python --version`
2. **perceptimg version**: `pip show perceptimg`
3. **OS**: Windows/Linux/macOS and version
4. **Steps to reproduce**: Minimal code example
5. **Expected behavior**: What you expected to happen
6. **Actual behavior**: What actually happened

### Feature Requests

For feature requests, please describe:

1. **Use case**: Why do you need this feature?
2. **Proposed solution**: How should it work?
3. **Alternatives**: Any alternative solutions you've considered?

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Questions? Feel free to [open an issue](https://github.com/seifreed/perceptimg/issues) or contact the maintainer.