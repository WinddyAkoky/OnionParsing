# Contributing to OnionParsing

Thank you for your interest in contributing! This guide covers the basics.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/<your-org>/OnionParsing.git
cd OnionParsing

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) with a line length of 100
- Use [ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Add type hints to all public methods
- Write docstrings for all public classes and functions

## Making Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run linting (`ruff check .`)
5. Commit with a descriptive message
6. Push and open a Pull Request

## Adding a New Processor

1. Create a new file in `onion_parsing/processors/`
2. Subclass `BaseProcessor` and use the `@register_processor("name")` decorator
3. Implement the `process(context, data)` method
4. Import the processor in `onion_parsing/processors/__init__.py`
5. Add configuration defaults in `onion_parsing/config/default.yaml`
6. Update `README.md` processor table

## Reporting Issues

- Use [GitHub Issues](https://github.com/<your-org>/OnionParsing/issues)
- Include Python version, OS, and steps to reproduce
- Attach relevant logs and configuration

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
