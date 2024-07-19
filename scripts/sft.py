"""
Usage:
To run the script with the configuration file:
accelerate launch --config_file "config/accelerate_config.yaml" scripts/sft.py --config_file config/sft_config.json

To override specific parameters via CLI arguments:
accelerate launch --config_file "config/accelerate_config.yaml" scripts/sft.py --config_file config/sft_config.json --num_train_epochs 3 --output_dir "custom_output_dir"

If no config file is provided, all required arguments must be provided via the CLI:
accelerate launch --config_file "config/accelerate_config.yaml" scripts/sft.py --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --dataset_path "data/sft" --output_dir "./outputs/sft_mmlu" --run_name "mmlu_finetune" --num_train_epochs 2 --logging_steps 5 --save_steps 0.25 --eval_steps 0.25 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 2 --learning_rate 2e-5 --max_seq_length 4096 --gradient_checkpointing true
"""
import os
import logging
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, HfArgumentParser
from trl import SFTConfig, SFTTrainer
from datasets import load_from_disk
import wandb
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Optional

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ScriptArguments:
    config_file: Optional[str] = field(
        default=None, metadata={"help": "Path to the configuration file."}
    )
    model_name: Optional[str] = field(
        default=None, metadata={"help": "Name of the pre-trained model."}
    )
    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the dataset."}
    )
    output_dir: Optional[str] = field(
        default=None, metadata={"help": "Output directory for the trained model."}
    )
    run_name: Optional[str] = field(
        default=None, metadata={"help": "WandB run name."}
    )
    num_train_epochs: Optional[int] = field(
        default=None, metadata={"help": "Number of training epochs."}
    )
    logging_steps: Optional[int] = field(
        default=None, metadata={"help": "Logging steps."}
    )
    save_steps: Optional[float] = field(
        default=None, metadata={"help": "Save steps."}
    )
    eval_steps: Optional[float] = field(
        default=None, metadata={"help": "Evaluation steps."}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=None, metadata={"help": "Training batch size per device."}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=None, metadata={"help": "Evaluation batch size per device."}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=None, metadata={"help": "Number of gradient accumulation steps."}
    )
    learning_rate: Optional[float] = field(
        default=None, metadata={"help": "Learning rate."}
    )
    max_seq_length: Optional[int] = field(
        default=None, metadata={"help": "Maximum sequence length."}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=None, metadata={"help": "Enable gradient checkpointing."}
    )

def initialize_wandb(run_name):
    if os.getenv("LOCAL_RANK", "0") == "0":
        try:
            wandb.login(key=os.environ.get("WANDB_API_KEY"))
            wandb.init(project=run_name)
            logger.info("WandB login successful.")
        except Exception as e:
            logger.error(f"WandB login failed: {e}")
            raise

def initialize_model_and_tokenizer(model_name):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        logger.info("Model and tokenizer initialized.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error initializing model and tokenizer: {e}")
        raise

def load_dataset(dataset_path):
    try:
        dataset = load_from_disk(dataset_path)
        logger.info("Dataset loaded from disk.")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def prepare_trainer(model, tokenizer, train_dataset, eval_dataset, args):
    sft_config = SFTConfig(
        run_name=args.run_name,
        dataset_text_field="text",
        output_dir=args.output_dir,
        bf16=True,
        seed=42,
        num_train_epochs=args.num_train_epochs,
        log_level="info",
        logging_first_step=True,
        logging_strategy="steps",
        save_strategy="steps",
        eval_strategy="steps",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        report_to="wandb",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=0.1,
        warmup_ratio=0.0,
        max_grad_norm=1.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-5,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    try:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_config,
            data_collator=data_collator
        )
        logger.info("SFTTrainer initialized with custom data collator.")
        return trainer
    except Exception as e:
        logger.error(f"Error initializing SFTTrainer: {e}")
        raise

def main():
    # Parse CLI arguments
    parser = HfArgumentParser(ScriptArguments)
    args, = parser.parse_args_into_dataclasses()

    # Load configuration from JSON file if provided
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
            for key, value in config.items():
                if getattr(args, key) is None:
                    setattr(args, key, value)

    # Ensure all required arguments are provided either via CLI or config file
    missing_args = []
    for field in ScriptArguments.__dataclass_fields__:
        if field != 'config_file' and getattr(args, field) is None:
            missing_args.append(field)

    if missing_args:
        parser = HfArgumentParser(ScriptArguments)
        args, = parser.parse_args_into_dataclasses()
        for arg in missing_args:
            if getattr(args, arg) is None:
                raise ValueError(f"Missing required argument: {arg}")

    try:
        initialize_wandb(args.run_name)

        model, tokenizer = initialize_model_and_tokenizer(args.model_name)

        dataset = load_dataset(args.dataset_path)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        trainer = prepare_trainer(model, tokenizer, train_dataset, eval_dataset, args)

        logger.info(f"Parallel mode: {trainer.args.parallel_mode}")

        checkpoint = trainer.args.resume_from_checkpoint if trainer.args.resume_from_checkpoint else None
        trainer.train(resume_from_checkpoint=checkpoint)
        
        trainer.save_model()
        logger.info("Model training completed and saved.")
    except Exception as e:
        logger.error(f"Training failed: {e}")

if __name__ == "__main__":
    main()
