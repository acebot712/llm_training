# accelerate launch --config_file "deepspeed_config.yaml" heal_llm.py
import multiprocessing
import os

import torch
from accelerate import Accelerator
from datasets import DatasetDict, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

os.environ["HF_HUB_CACHE"] = "/opt/dlami/nvme"

device_map = {"": Accelerator().local_process_index}
dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
tokenizer.pad_token = tokenizer.eos_token
ds = load_dataset("teknium/OpenHermes-2.5", split="train")
ds = ds.train_test_split(test_size=0.2)
print(ds)


def sample_dataset(dataset_dict, percentage=0.05):
    sampled_dataset = DatasetDict()
    for split in dataset_dict.keys():
        # Calculate the number of samples to select
        sample_size = int(len(dataset_dict[split]) * percentage)
        sampled_dataset[split] = (
            dataset_dict[split].shuffle(seed=42).select(range(sample_size))
        )
    return sampled_dataset


def apply_template(example):
    example["conversations"] = tokenizer.apply_chat_template(
        example['conversations'], tokenize=False, add_generation_prompt=True
    )
    return example


num_cpus = multiprocessing.cpu_count()


# Sample 5% of each split in the dataset
sampled_dataset = sample_dataset(ds, 0.05)
print(sampled_dataset)
print(sampled_dataset["train"][0])
sampled_dataset = sampled_dataset.map(apply_template, num_proc=num_cpus)
print(sampled_dataset)
print(sampled_dataset["train"][0])
sft_config = SFTConfig(
    dataset_text_field="input",
    output_dir="/home/sauron/davidmontero/llm_healing/sfttrainer-paperparams",
    bf16=True,
    seed=42,
    num_train_epochs=2,
    log_level="info",
    logging_steps=5,
    logging_first_step=True,
    logging_strategy="steps",
    # evaluation_strategy="epoch",
    # eval_steps=5,
    save_strategy="steps",
    # save_steps=5,
    gradient_checkpointing=True,
    per_device_eval_batch_size=16,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    # report_to="tensorboard",
    # training params
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    max_seq_length=4096,  # in the paper they use 4096
    learning_rate=2e-5,  # in the paper they use 2e-5
    lr_scheduler_type="cosine",  # used in the paper
    weight_decay=0.1,  # they use 0.1 in the paper (maybe too high?)
    warmup_ratio=0.0,  # no warmup specified in the paper, we are already using a small lr
    max_grad_norm=1.0,  # paper uses 1.0 and also recommended in other experiments
    adam_beta1=0.9,  # in the paper they use 0.9 for pretraining and rlhf
    adam_beta2=0.95,  # in the paper they use 0.95 for pretraining and rlhf
    adam_epsilon=1e-5,  # in the paper they use 1e-5 for pretraining and rlhf
)
trainer = SFTTrainer(
    model,
    tokenizer=tokenizer,
    train_dataset=sampled_dataset,
    # eval_dataset=test_dataset,
    args=sft_config,
)
trainer.train(
    resume_from_checkpoint="/home/sauron/davidmontero/llm_healing/sfttrainer/checkpoint-16500"
)
