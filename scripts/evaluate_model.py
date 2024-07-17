import json
import os
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()   

def evaluate_model(model_name, model_args, datasets, num_fewshot, batch_size, device, output_dir, limit=1.0):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the model with the corrected arguments
    model_instance = HFLM(pretrained=model_name, device=device, **model_args)
    
    all_results = {}

    for dataset in datasets:
        print(f"Evaluating on {dataset}...")

        # Create a TaskManager for the task
        task_manager = TaskManager()
        
        # Evaluation configuration
        eval_config = {
            'model': model_instance,
            'model_args': model_args,
            'tasks': [dataset],
            'num_fewshot': num_fewshot,
            'batch_size': batch_size,
            'device': device,
            'bootstrap_iters': 1000,  # Number of bootstrap iterations for statistical significance
            'task_manager': task_manager,
            'limit': limit
        }
        
        # Run the evaluation
        dataset_results = simple_evaluate(**eval_config)
        
        # Store results in dictionary
        all_results[dataset] = dataset_results["results"].get(dataset)
    
    # Save the results to a single JSON file
    results_file = os.path.join(output_dir, f"{model_name.replace('/', '_')}_all_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

# Configuration parameters
model_name = "NousResearch/Llama-2-7b-chat-hf"
model_args = {"revision": "main", "dtype": "float", "parallelize": True}
datasets = ["mmlu", "hellaswag", "boolq"]  # Add your datasets here
num_fewshot = 0
batch_size = "auto:4"  # Set batch size to auto with recomputation
device = "cuda:5"
output_dir = "./evaluation_results"
limit = None

# Run evaluation
results = evaluate_model(model_name, model_args, datasets, num_fewshot, batch_size, device, output_dir, limit)

# Print summary results
print(results)
