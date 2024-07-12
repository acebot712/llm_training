import os
import pickle
import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, DatasetDict
from trl import SFTConfig, SFTTrainer
import multiprocessing

def load_and_split_dataset(dataset_name, split_ratio=0.2):
    try:
        dataset = load_dataset(dataset_name, split="train")
        return dataset.train_test_split(test_size=split_ratio)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def sample_dataset(dataset_dict, percentage=0.05):
    sampled_dataset = DatasetDict()
    try:
        for split in dataset_dict.keys():
            sample_size = int(len(dataset_dict[split]) * percentage)
            sampled_dataset[split] = dataset_dict[split].shuffle(seed=42).select(range(sample_size))
    except Exception as e:
        print(f"Error sampling dataset: {e}")
        raise
    return sampled_dataset

def apply_list_clean(example):
    try:
        for conversation in example["conversations"]:
            if conversation.get("from") == "human":
                conversation["role"] = "user"
            elif conversation.get("from") == "gpt":
                conversation["role"] = "assistant"
            if "from" in conversation:
                del conversation["from"]
            conversation["content"] = conversation.pop("value")
    except Exception as e:
        print(f"Error cleaning example: {e}")
        raise
    return example

def apply_template(example, tokenizer):
    try:
        example["text"] = tokenizer.apply_chat_template(
            example['conversations'], tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        print(f"Error applying template: {e}")
        raise
    return example

def prepare_dataset(dataset_name, tokenizer_name, sample_percentage=0.05, split_ratio=0.2):
    dataset_dict = load_and_split_dataset(dataset_name, split_ratio)
    sampled_dataset = sample_dataset(dataset_dict, sample_percentage)
    
    num_cpus = multiprocessing.cpu_count()
    
    try:
        sampled_dataset = sampled_dataset.map(apply_list_clean, num_proc=num_cpus)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        sampled_dataset = sampled_dataset.map(lambda x: apply_template(x, tokenizer), num_proc=num_cpus)
    except Exception as e:
        print(f"Error during dataset preparation: {e}")
        raise

    return sampled_dataset

if __name__ == "__main__":
    dataset_name = "teknium/OpenHermes-2.5"
    tokenizer_name = "microsoft/Phi-3-mini-4k-instruct"

    prepared_dataset = prepare_dataset(dataset_name, tokenizer_name)

    # Saving the prepared dataset
    prepared_dataset.save_to_disk("data/hermes_dataset")
