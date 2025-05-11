# LLM Training (Refactored)

This repository provides a modular, extensible framework for data preparation, fine-tuning, evaluation, and compression of large language models (LLMs).

## Key Features
- Modular Python package: `llm_training`
- Unified CLI: `llm-train` for all workflows
- Hydra-based configuration system (YAML/CLI/env)
- Extensible: add new datasets, models, or tasks via plugins
- Robust logging, error handling, and progress bars
- Unit/integration tests and CI/CD
- Auto-generated documentation
- Docker support for reproducibility
- Community guidelines and changelog

## Quickstart

1. **Install dependencies:**
   ```sh
   pip install -e .
   ```
2. **Run data preparation:**
   ```sh
   llm-train data-prep --config configs/data_prep_config.yaml
   ```
3. **Fine-tune a model:**
   ```sh
   llm-train train --config configs/sft_config.yaml
   ```
4. **Evaluate a model:**
   ```sh
   llm-train eval --config configs/evaluate_config.yaml
   ```
5. **Model compression:**
   ```sh
   llm-train compress --config configs/compress_config.yaml
   ```

## Configuration
All workflows are configured via Hydra YAML files. See `configs/` for examples. Override any parameter via CLI:
```sh
llm-train train --config configs/sft_config.yaml model.learning_rate=1e-5
```

## Extending
- Add new datasets, models, or tasks by registering them in the `llm_training.registry`.
- See the developer docs in `docs/` for details.

## Testing
Run all tests with:
```sh
pytest
```

## Documentation
Auto-generated docs are in `docs/` and can be built with:
```sh
mkdocs build
```

## Docker
Build and run in a reproducible environment:
```sh
docker build -t llm_training .
docker run -it llm_training
```

## Contributing
See `CONTRIBUTING.md` for guidelines. All contributions and issues are welcome!

## Changelog
See `CHANGELOG.md` for release history.

## Citation

If you find this work useful, please cite it as follows:
```bibtex
@misc{your_repository,
  author = {Abhijoy Sarkar},
  title = {LLM Training},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/acebot712/llm_training}},
}
```

## Contact
[![Contact me on Codementor](https://www.codementor.io/m-badges/abhijoysarkar/find-me-on-cm-b.svg)](https://www.codementor.io/@abhijoysarkar?refer=badge)
