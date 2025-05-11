"""
Evaluation utilities for LLMs.
"""
from typing import Dict, List, Optional
import logging
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from .model_loader import LlamaCausalLMTensor
import os
from datetime import datetime

class Evaluator:
    """
    Handles evaluation of LLMs on benchmark datasets.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        if self.config.get("is_tensorized"):
            return LlamaCausalLMTensor.from_pretrained(self.config["model_name"])
        else:
            return HFLM(pretrained=self.config["model_name"], device=self.config["device"], **self.config.get("model_args", {}))

    def run(self) -> Dict:
        os.makedirs(self.config["output_dir"], exist_ok=True)
        model_instance = self.load_model()
        all_results = {}
        results_file = os.path.join(
            self.config["output_dir"],
            f"{self.config['model_name'].replace('/', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        )
        for dataset in self.config["datasets"]:
            self.logger.info(f"Evaluating on {dataset}...")
            task_manager = TaskManager()
            eval_config = {
                'model': model_instance,
                'model_args': self.config.get("model_args", {}),
                'tasks': [dataset],
                'num_fewshot': self.config["num_fewshot"],
                'batch_size': self.config["batch_size"],
                'device': self.config["device"],
                'bootstrap_iters': 1000,
                'task_manager': task_manager,
                'limit': self.config.get("limit")
            }
            try:
                dataset_results = simple_evaluate(**eval_config)
                all_results[dataset] = dataset_results["results"].get(dataset)
            except Exception as e:
                self.logger.error(f"Error during evaluation of {dataset}: {e}")
                all_results[dataset] = {"error": str(e)}
            with open(results_file, 'w') as f:
                import json
                json.dump({"results": all_results, "config": self.config}, f, indent=2)
        self.logger.info(f"Evaluation completed. Results saved to {results_file}")
        return all_results 