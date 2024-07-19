# LLM Training

This repository contains scripts for data preparation, fine-tuning (SFT) training, and evaluation of large language models (LLMs). Follow the instructions below to set up and run the scripts using either configuration files or command-line arguments.

## Table of Contents
- [Data Preparation](#data-preparation)
- [SFT Training](#sft-training)
- [Evaluation](#evaluation)

## Data Preparation

Prepare your dataset for training with the provided script.

### Usage

To run the script with a custom configuration file:
```sh
python scripts/data_prep.py --config_file config/data_prep_config.json
```

To override specific parameters:
```sh
python scripts/data_prep.py --config_file custom_config.json --sample_percentage 0.1 --output_dir "custom_output_dir"
```

For quick debugging, set `sample_percentage` to `0.1` and `output_dir` to `"data/debug"`.

## SFT Training

Fine-tune your model using the provided SFT script.

### Usage

To run the script with the configuration file:
```sh
accelerate launch --config_file "config/accelerate_config.yaml" scripts/sft.py --config_file config/sft_config.json
```

To override specific parameters via CLI arguments:
```sh
accelerate launch --config_file "config/accelerate_config.yaml" scripts/sft.py --config_file config/sft_config.json --num_train_epochs 3 --output_dir "custom_output_dir"
```

If no config file is provided, all required arguments must be provided via the CLI:
```sh
accelerate launch --config_file "config/accelerate_config.yaml" scripts/sft.py --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --dataset_path "data/sft" --output_dir "./outputs/sft_mmlu" --run_name "mmlu_finetune" --num_train_epochs 2 --logging_steps 5 --save_steps 0.25 --eval_steps 0.25 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 2 --learning_rate 2e-5 --max_seq_length 4096 --gradient_checkpointing true
```

For quick debugging, set `num_train_epochs` to `1`, `output_dir` to `"./outputs/debug"`, `dataset_path` to `"data/debug"`, and `run_name` to `"debug"`.

## Evaluation

Evaluate your fine-tuned model using the provided evaluation script.

### Usage

To run the script with the configuration file:
```sh
python script_name.py --config_file configs/evaluate_config.json
```

To override specific parameters via CLI arguments:
```sh
python script_name.py --config_file configs/evaluate_config.json --num_fewshot 5 --output_dir "custom_output_dir"
```

If no config file is provided, all required arguments must be provided via the CLI:
```sh
python script_name.py --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --model_args '{"revision": "main", "dtype": "float", "parallelize": true}' --datasets "mmlu,hellaswag,boolq" --num_fewshot 0 --batch_size "auto:4" --device "cuda:7" --output_dir "./evaluation_results" --limit 1.0
```

For quick debugging, set `num_fewshot` to `0`, `datasets` to `"mmlu"`, and `limit` to `0.05`.

---

### Configuration Files

Each script can be configured using JSON files. Below are examples of what these configuration files might look like:

#### data_prep_config.json
```json
{
    "dataset_name": "your_dataset_name",
    "sample_percentage": 1.0,
    "output_dir": "data/sft",
    "split_ratio": 0.2
}
```

#### sft_config.json
```json
{
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "dataset_path": "data/sft",
    "output_dir": "./outputs/sft_finetune",
    "run_name": "sft_finetune",
    "num_train_epochs": 2,
    "logging_steps": 5,
    "save_steps": 0.25,
    "eval_steps": 0.25,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-5,
    "max_seq_length": 4096,
    "gradient_checkpointing": true
}
```

#### evaluate_config.json
```json
{
    "model_name": "/home/ubuntu/abhijoy/model_compressor/outputs/sft_mmlu/checkpoint-3566",
    "model_args": {
        "revision": "main",
        "dtype": "float",
        "parallelize": false
    },
    "datasets": ["mmlu", "hellaswag", "boolq"],
    "num_fewshot": 0,
    "batch_size": "auto:4",
    "device": "cuda:7",
    "output_dir": "./evaluation_results",
    "limit": null
}
```

### Additional Notes

- Ensure you have all the required dependencies installed before running the scripts `pip install -r requirements.txt`.
- Adjust the configuration files according to your specific use case and environment.
- For detailed documentation on each parameter, refer to the script's inline comments and the respective libraries' documentation.

Feel free to open issues or contribute to the repository if you find any bugs or have suggestions for improvements.

### Contact
[![Contact me on Codementor](https://www.codementor.io/m-badges/abhijoysarkar/find-me-on-cm-b.svg)](https://www.codementor.io/@abhijoysarkar?refer=badge)