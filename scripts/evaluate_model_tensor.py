import json
import os
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from dotenv import load_dotenv
from model_loader import LlamaCausalLMTensor, LlamaCausalLMTensor_train

# Load environment variables
load_dotenv()   

def load_model(model_location):
    print(model_location)
    if "train" in model_location:
        print("INSIDE IF")
        model = LlamaCausalLMTensor_train.from_pretrained(model_location)
    else:
        print("INSIDE ELSE")
        model = LlamaCausalLMTensor.from_pretrained(model_location)
    print(model)
    return model

def evaluate_model(model_name, model_args, datasets, num_fewshot, batch_size, device, output_dir, limit=1.0):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the model with the corrected arguments
    model=load_model(model_name)
    model_instance = HFLM(pretrained=model, device=device, **model_args)
    
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
model_name = "/home/ubuntu/profiler/compact/compact_Pintxo_1071_1000_supra3_no_gate_mpo_zero_98"
model_args = {"revision": "main", "dtype": "float", "parallelize": False}
datasets = ["mmlu", "hellaswag", "boolq"]  # Add your datasets here
num_fewshot = 0
batch_size = "auto:4"  # Set batch size to auto with recomputation
device = "cuda:7"
output_dir = "./evaluation_results"
limit = None

# Run evaluation
results = evaluate_model(model_name, model_args, datasets, num_fewshot, batch_size, device, output_dir, limit)

# Print summary results
print(results)
