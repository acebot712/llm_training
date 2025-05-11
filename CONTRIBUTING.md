# Contributing Guide

Thank you for considering contributing to LLM Training!

## Getting Started
1. Fork the repo and clone your fork.
2. Install dependencies:
   ```sh
   pip install -e .
   pip install pre-commit
   pre-commit install
   ```
3. Run tests:
   ```sh
   pytest
   ```

## Code Style
- Follow PEP8 and use type hints.
- Add Google/NumPy-style docstrings to all functions/classes.
- Run `pre-commit` before pushing (auto-formats and lints code).

## Submitting a PR
- Branch from `main`.
- Add/modify tests for your changes.
- Update documentation if needed.
- Ensure all tests pass and code is linted.
- Describe your changes clearly in the PR.

## Issues
- Please use GitHub Issues for bugs, feature requests, and questions.

## Thanks for contributing! 