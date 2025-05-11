"""
Data preparation for LLM training.
"""
from typing import Optional
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer
import multiprocessing
import logging

def prepare_data(dataset_name: str, tokenizer_name: str, sample_percentage: float = 0.05, split_ratio: float = 0.2, output_dir: Optional[str] = None) -> DatasetDict:
    """
    Loads, samples, cleans, and saves a dataset for LLM training.
    """
    logger = logging.getLogger(__name__)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding=True, truncation=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset = load_dataset(dataset_name, split="train")
    dataset_dict = dataset.train_test_split(test_size=split_ratio)
    sampled = DatasetDict()
    for split in dataset_dict.keys():
        sample_size = int(len(dataset_dict[split]) * sample_percentage)
        sampled[split] = dataset_dict[split].shuffle(seed=42).select(range(sample_size))
    def apply_list_clean(example):
        for conversation in example["conversations"]:
            if conversation.get("from") == "human":
                conversation["role"] = "user"
            elif conversation.get("from") == "gpt":
                conversation["role"] = "assistant"
            if "from" in conversation:
                del conversation["from"]
            conversation["content"] = conversation.pop("value")
        return example
    def apply_template(example):
        example["text"] = tokenizer.apply_chat_template(
            example['conversations'], tokenize=False
        )
        return example
    num_cpus = multiprocessing.cpu_count()
    sampled = sampled.map(apply_list_clean, num_proc=num_cpus)
    sampled = sampled.map(apply_template, num_proc=num_cpus)
    if output_dir:
        sampled.save_to_disk(output_dir)
        logger.info(f"Dataset saved to disk at {output_dir}.")
    return sampled 