import json
import os
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class EvaluationConfig:
    config_file: Optional[str] = field(
        default=None, metadata={"help": "Path to the configuration file."}
    )
    model_name: str = field(
        default=None, metadata={"help": "Name of the pretrained model."}
    )
    model_args: Dict = field(
        default_factory=dict, metadata={"help": "Arguments for the model."}
    )
    datasets: List[str] = field(
        default_factory=list, metadata={"help": "List of datasets to evaluate on."}
    )
    num_fewshot: int = field(
        default=0, metadata={"help": "Number of few-shot examples."}
    )
    batch_size: str = field(
        default="auto:4", metadata={"help": "Batch size."}
    )
    device: str = field(
        default="cuda", metadata={"help": "Device to run evaluation on."}
    )
    output_dir: str = field(
        default="./evaluation_results", metadata={"help": "Directory to save evaluation results."}
    )
    limit: Optional[float] = field(
        default=None, metadata={"help": "Limit on the number of examples to evaluate."}
    )

def evaluate_model(config: EvaluationConfig):
    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize the model with the corrected arguments
    model_instance = HFLM(pretrained=config.model_name, device=config.device, **config.model_args)
    
    all_results = {}

    for dataset in config.datasets:
        print(f"Evaluating on {dataset}...")

        # Create a TaskManager for the task
        task_manager = TaskManager()
        
        # Evaluation configuration
        eval_config = {
            'model': model_instance,
            'model_args': config.model_args,
            'tasks': [dataset],
            'num_fewshot': config.num_fewshot,
            'batch_size': config.batch_size,
            'device': config.device,
            'bootstrap_iters': 1000,  # Number of bootstrap iterations for statistical significance
            'task_manager': task_manager,
            'limit': config.limit
        }
        
        # Run the evaluation
        dataset_results = simple_evaluate(**eval_config)
        
        # Store results in dictionary
        all_results[dataset] = dataset_results["results"].get(dataset)
    
    # Save the results to a single JSON file
    results_file = os.path.join(config.output_dir, f"{config.model_name.replace('/', '_')}_all_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on datasets.")
    parser.add_argument("--config_file", type=str, help="Path to the configuration file.")
    parser.add_argument("--model_name", type=str, help="Name of the pretrained model.")
    parser.add_argument("--model_args", type=str, help="JSON string of arguments for the model.")
    parser.add_argument("--datasets", type=str, help="Comma-separated list of datasets to evaluate on.")
    parser.add_argument("--num_fewshot", type=int, help="Number of few-shot examples.")
    parser.add_argument("--batch_size", type=str, help="Batch size.")
    parser.add_argument("--device", type=str, help="Device to run evaluation on.")
    parser.add_argument("--output_dir", type=str, help="Directory to save evaluation results.")
    parser.add_argument("--limit", type=float, help="Limit on the number of examples to evaluate.")
    return parser.parse_args()

def load_config_from_file(config_file):
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    return EvaluationConfig(**config_dict)

def main():
    args = parse_args()

    if args.config_file:
        config = load_config_from_file(args.config_file)
    else:
        config = EvaluationConfig(
            model_name=args.model_name,
            model_args=json.loads(args.model_args) if args.model_args else {},
            datasets=args.datasets.split(",") if args.datasets else [],
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.output_dir,
            limit=args.limit
        )

    # Ensure all required arguments are provided either via CLI or config file
    missing_args = []
    for field_name, field_value in config.__dict__.items():
        if field_value is None and field_name != 'config_file':
            missing_args.append(field_name)

    if missing_args:
        raise ValueError(f"Missing required arguments: {', '.join(missing_args)}")

    results = evaluate_model(config)

    # Print summary results
    print(results)

if __name__ == "__main__":
    main()
