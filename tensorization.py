import os
os.environ["HF_HUB_CACHE"] = "/opt/dlami/nvme"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin

# Set random seed for reproducibility
torch.random.manual_seed(0)

# Define the DeepSpeed configuration for inference
deepspeed_config = {
    "train_batch_size": 8,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 1.0,
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True
    }
}

# Initialize the Accelerator with DeepSpeedPlugin
accelerator = Accelerator(deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=deepspeed_config, zero_stage=3))

# Load the model and tokenizer without device_map and low_cpu_mem_usage
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    torch_dtype="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Prepare the model with the Accelerator
model = accelerator.prepare(model)

# Define the inference pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=accelerator.device  # Ensure the pipeline uses the correct device
)

# Define the messages
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]

# Define the generation arguments
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.1,
    "do_sample": True
}

# Run inference
output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
