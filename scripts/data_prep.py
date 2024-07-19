"""
Usage: 

To run the script with a custom configuration file:
python scripts/data_prep.py --config_file config/data_prep_config.json

To override specific parameters:
python your_script.py --config_file custom_config.json --sample_percentage 0.1 --output_dir "custom_output_dir"
"""
import argparse
import json
import multiprocessing
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
            sampled_dataset[split] = (
                dataset_dict[split].shuffle(seed=42).select(range(sample_size))
            )
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
            example['conversations'], tokenize=False
        )
    except Exception as e:
        print(f"Error applying template: {e}")
        raise
    return example

def prepare_dataset(
    dataset_name, tokenizer, sample_percentage=0.05, split_ratio=0.2
):
    dataset_dict = load_and_split_dataset(dataset_name, split_ratio)
    sampled_dataset = sample_dataset(dataset_dict, sample_percentage)

    num_cpus = multiprocessing.cpu_count()

    try:
        sampled_dataset = sampled_dataset.map(apply_list_clean, num_proc=num_cpus)
        sampled_dataset = sampled_dataset.map(
            lambda x: apply_template(x, tokenizer), num_proc=num_cpus
        )
    except Exception as e:
        print(f"Error during dataset preparation: {e}")
        raise

    return sampled_dataset

def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Prepare dataset with optional overrides.")
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to the configuration file.")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset to use.")
    parser.add_argument("--tokenizer_name", type=str, help="Name of the tokenizer to use.")
    parser.add_argument("--sample_percentage", type=float, help="Percentage of the dataset to sample.")
    parser.add_argument("--split_ratio", type=float, help="Ratio for splitting the dataset into train/test.")
    parser.add_argument("--output_dir", type=str, help="Directory to save the processed dataset.")
    args = parser.parse_args()

    # Load default configuration
    with open(args.config_file, 'r') as f:
        config = json.load(f)

    # Override defaults with CLI arguments if provided
    dataset_name = args.dataset_name or config.get("dataset_name")
    tokenizer_name = args.tokenizer_name or config.get("tokenizer_name")
    sample_percentage = args.sample_percentage if args.sample_percentage is not None else config.get("sample_percentage")
    split_ratio = args.split_ratio if args.split_ratio is not None else config.get("split_ratio")
    output_dir = args.output_dir or config.get("output_dir")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding=True, truncation=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    prepared_dataset = prepare_dataset(dataset_name, tokenizer, sample_percentage, split_ratio)

    print(prepared_dataset)
    # Save the processed dataset to disk
    prepared_dataset.save_to_disk(output_dir)
    print(f"Dataset saved to disk at {output_dir}.")
    loaded_dataset = load_from_disk(output_dir)
    print("Dataset loaded from disk.")
    print(loaded_dataset)

if __name__ == "__main__":
    main()
