"""
Usage:
accelerate launch --config_file "config/accelerate_config.yaml" scripts/sft_tensor.py
"""
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from trl import SFTConfig, SFTTrainer
from datasets import load_from_disk
import wandb
from dotenv import load_dotenv
from model_loader import LlamaCausalLMTensor, LlamaCausalLMTensor_train

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_wandb():
    try:
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        logger.info("WandB login successful.")
    except Exception as e:
        logger.error(f"WandB login failed: {e}")
        raise

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

def initialize_model_and_tokenizer():
    try:
        model = load_model("/home/ubuntu/profiler/Downloaded_checkpoint_pintxo")
        # model = AutoModelForCausalLM.from_pretrained(
        #     "NousResearch/Llama-2-7b-chat-hf",
        #     torch_dtype="auto",
        #     trust_remote_code=True,
        #     attn_implementation="flash_attention_2",
        #     local_files_only=True
        # )
        tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf", padding=True, truncation=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        logger.info("Model and tokenizer initialized.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error initializing model and tokenizer: {e}")
        raise

def load_dataset():
    try:
        dataset = load_from_disk("data/sft_mmlu_pintxos")
        logger.info("Dataset loaded from disk.")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def prepare_trainer(model, tokenizer, train_dataset, eval_dataset):
    sft_config = SFTConfig(
        run_name="mmlu_finetune",
        dataset_text_field="text",
        output_dir="./outputs/sft_mmlu",
        bf16=True,
        seed=42,
        num_train_epochs=2,
        log_level="info",
        logging_first_step=True,
        logging_strategy="steps",
        save_strategy="steps",
        eval_strategy="steps",
        logging_steps=5,
        save_steps=0.25,
        eval_steps=0.25,
        gradient_checkpointing=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        report_to="wandb",
        gradient_accumulation_steps=2,
        max_seq_length=4096,
        learning_rate=2e-5,
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
    try:
        initialize_wandb()

        model, tokenizer = initialize_model_and_tokenizer()

        dataset = load_dataset()
        train_dataset = dataset["auxiliary_train"]
        eval_dataset = dataset["test"]

        trainer = prepare_trainer(model, tokenizer, train_dataset, eval_dataset)

        logger.info(f"Parallel mode: {trainer.args.parallel_mode}")

        checkpoint = trainer.args.resume_from_checkpoint if trainer.args.resume_from_checkpoint else None
        trainer.train(resume_from_checkpoint=checkpoint)
        
        trainer.save_model()
        logger.info("Model training completed and saved.")
    except Exception as e:
        logger.error(f"Training failed: {e}")

if __name__ == "__main__":
    main()
