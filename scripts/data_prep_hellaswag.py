import multiprocessing
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_and_split_dataset(dataset_name):
    try:
        dataset = load_dataset(dataset_name, split=['train', 'validation', 'test'])
        return DatasetDict({
            "train": dataset[0],
            "validation": dataset[1],
            "test": dataset[2]
        })
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

def format_examples(example):
    try:
        example["text"] = f"### Context: {example['ctx']}\n### Options: {', '.join(example['endings'])}\n### Label: {example.get('label', 'N/A')}"
    except Exception as e:
        print(f"Error formatting example: {e}")
        raise
    return example

def prepare_dataset(dataset_name, sample_percentage=0.05):
    dataset_dict = load_and_split_dataset(dataset_name)
    sampled_dataset = sample_dataset(dataset_dict, sample_percentage)

    num_cpus = multiprocessing.cpu_count()

    try:
        formatted_dataset = sampled_dataset.map(format_examples, num_proc=num_cpus)
    except Exception as e:
        print(f"Error during dataset preparation: {e}")
        raise

    return formatted_dataset

if __name__ == "__main__":
    dataset_name = "Rowan/hellaswag"
    tokenizer_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding=True, truncation=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    prepared_dataset = prepare_dataset(dataset_name, sample_percentage=1)

    print(prepared_dataset)
    # Save the processed dataset to disk
    prepared_dataset.save_to_disk("data/sft_hellaswag")
    print("Dataset saved to disk.")
    loaded_dataset = load_from_disk("data/sft_hellaswag")
    print("Dataset loaded from disk.")
    print(loaded_dataset)
