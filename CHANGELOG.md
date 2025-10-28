# Changelog

## [0.1.0] - 2024-10-28

### Added
- Modular Python package (`llm_training`) with core functionality
- Unified CLI (`llm-train`) for data prep, training, and evaluation workflows
- OmegaConf-based configuration system (JSON/YAML)
- Typer-based CLI with subcommands
- Support for tensorized model compression (Llama, Mixtral)
- Integrated WandB logging for experiment tracking
- Unit tests with pytest
- GitHub Actions CI workflow
- MIT License
- Docker support for reproducible environments
- Contribution guidelines (CONTRIBUTING.md)
- Environment variable template (.env.example)

### Improved
- Logging and error handling throughout
- Code organization with clear package structure
- Documentation with honest feature descriptions

### Fixed
- Completed model_loader.py implementation
- Fixed test suite to match actual API
- Corrected pyproject.toml package configuration
