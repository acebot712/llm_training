"""
Training for LLM fine-tuning.
"""
from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from trl import SFTConfig, SFTTrainer
from datasets import load_from_disk
import wandb
import logging

def train_model(config: Dict):
    """
    Fine-tunes an LLM using the provided config dict.
    """
    logger = logging.getLogger(__name__)
    if config.get("wandb_api_key"):
        wandb.login(key=config["wandb_api_key"])
        wandb.init(project=config.get("run_name", "llm_training"))
        logger.info("WandB login successful.")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], padding=True, truncation=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    logger.info("Model and tokenizer initialized.")
    dataset = load_from_disk(config["dataset_path"])
    logger.info("Dataset loaded from disk.")
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"] if "test" in dataset else None
    sft_config = SFTConfig(**config)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        data_collator=data_collator,
    )
    logger.info("SFTTrainer initialized.")
    trainer.train()
    trainer.save_model(config["output_dir"])
    logger.info(f"Model saved to {config['output_dir']}") 