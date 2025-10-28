# LLM Training (Refactored)

This repository provides a modular, extensible framework for data preparation, fine-tuning, evaluation, and compression of large language models (LLMs).

## Key Features
- Modular Python package: `llm_training`
- Unified CLI: `llm-train` for data prep, training, and evaluation
- Configuration via JSON/YAML files with OmegaConf
- Support for tensorized model compression (Llama, Mixtral)
- Integrated WandB logging for experiment tracking
- Unit tests with pytest
- MIT licensed with community guidelines

## Quickstart

1. **Install dependencies:**
   ```sh
   pip install -e .
   ```
2. **Run data preparation:**
   ```sh
   llm-train data-prep --config configs/data_prep_config.json
   ```
3. **Fine-tune a model:**
   ```sh
   llm-train train --config configs/sft_config.json
   ```
4. **Evaluate a model:**
   ```sh
   llm-train eval --config configs/evaluate_config.json
   ```

For model compression, see the scripts in `scripts/compression/`.

## Configuration
All workflows are configured via JSON or YAML files using OmegaConf. See `configs/` for examples. The config files specify model paths, hyperparameters, dataset paths, and training arguments.

### Example Configs
- `data_prep_config.json` - Dataset preparation configuration
- `sft_config.json` - Supervised fine-tuning parameters
- `evaluate_config.json` - Evaluation benchmark settings
- `accelerate_config.yaml` - Multi-GPU training with Accelerate

## Project Structure
- `llm_training/` - Main Python package with core functionality
- `scripts/` - Standalone scripts for data prep and compression workflows
- `configs/` - Configuration files for different workflows
- `tests/` - Unit tests

## Testing
Run all tests with:
```sh
pytest
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
@misc{sarkar2024llmtraining,
  author = {Abhijoy Sarkar},
  title = {LLM Training: A Modular Framework for Fine-tuning Large Language Models},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/acebot712/llm_training}},
}
```

## Contact
[![Contact me on Codementor](https://www.codementor.io/m-badges/abhijoysarkar/find-me-on-cm-b.svg)](https://www.codementor.io/@abhijoysarkar?refer=badge)
