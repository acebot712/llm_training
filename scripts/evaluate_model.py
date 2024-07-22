"""
Usage:
To run the script with the configuration file:
python script_name.py --config_file configs/evaluate_config.json

To override specific parameters via CLI arguments:
python script_name.py --config_file configs/evaluate_config.json --num_fewshot 5 --output_dir "custom_output_dir"

If no config file is provided, all required arguments must be provided via the CLI:
python script_name.py --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --model_args '{"revision": "main", "dtype": "float", "parallelize": true}' --datasets "mmlu,hellaswag,boolq" --num_fewshot 0 --batch_size "auto:4" --device "cuda:7" --output_dir "./evaluation_results" --limit 1.0

"""
import json
import os
import argparse
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from dotenv import load_dotenv
from datetime import datetime
from model_loader import LlamaCausalLMTensor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    config_file: Optional[str] = field(default=None, metadata={"help": "Path to the configuration file."})
    model_name: str = field(default=None, metadata={"help": "Name of the pretrained model."})
    model_args: Dict = field(default_factory=dict, metadata={"help": "Arguments for the model."})
    datasets: List[str] = field(default_factory=list, metadata={"help": "List of datasets to evaluate on."})
    num_fewshot: int = field(default=0, metadata={"help": "Number of few-shot examples."})
    batch_size: str = field(default="auto:4", metadata={"help": "Batch size."})
    device: str = field(default="cuda", metadata={"help": "Device to run evaluation on."})
    output_dir: str = field(default="./evaluation_results", metadata={"help": "Directory to save evaluation results."})
    limit: Optional[float] = field(default=None, metadata={"help": "Limit on the number of examples to evaluate."})
    is_tensorized: Optional[bool] = field(default=False, metadata={"help": "Whether the model is tensorized."})

    def validate(self):
        missing_args = [field_name for field_name, field_value in asdict(self).items() if field_value is None and field_name != 'config_file']
        if missing_args:
            raise ValueError(f"Missing required arguments: {', '.join(missing_args)}")

def evaluate_model(config: EvaluationConfig):
    logger.info("Starting evaluation with configuration: %s", config)
    
    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize the model with the corrected arguments
    if config.is_tensorized:
        model_instance = LlamaCausalLMTensor.from_pretrained(config.model_name)
    else:
        model_instance = HFLM(pretrained=config.model_name, device=config.device, **config.model_args)
    
    all_results = {}

    # Define the path for the final results file
    results_file = os.path.join(config.output_dir, f"{config.model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")

    for dataset in config.datasets:
        logger.info(f"Evaluating on {dataset}...")

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
        
        try:
            # Run the evaluation
            dataset_results = simple_evaluate(**eval_config)
            # Store results in dictionary
            all_results[dataset] = dataset_results["results"].get(dataset)
        except Exception as e:
            logger.error(f"Error during evaluation of {dataset}: {e}")
            all_results[dataset] = {"error": str(e)}

        # Save intermediate results to the final file
        with open(results_file, 'w') as f:
            json.dump({"results": all_results, "config": asdict(config)}, f, indent=2)
    
    logger.info("Evaluation completed. Results saved to %s", results_file)
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
    try:
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        return EvaluationConfig(**config_dict)
    except Exception as e:
        logger.error(f"Error loading configuration file {config_file}: {e}")
        raise

def main():
    args = parse_args()

    if args.config_file:
        config = load_config_from_file(args.config_file)
        
        # Override config values with CLI arguments if provided
        if args.model_name:
            config.model_name = args.model_name
        if args.model_args:
            config.model_args = json.loads(args.model_args)
        if args.datasets:
            config.datasets = args.datasets.split(",")
        if args.num_fewshot is not None:
            config.num_fewshot = args.num_fewshot
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.device:
            config.device = args.device
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.limit is not None:
            config.limit = args.limit
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

    try:
        # Validate configuration
        config.validate()

        results = evaluate_model(config)

        # Print summary results
        logger.info("Evaluation results: %s", results)
    except ValueError as e:
        logger.error(e)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()

