from transformers import AutoModelForCausalLM, AutoTokenizer
from model_loader import LlamaCausalLMTensor

# Load the models
model_1 = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
model_2 = LlamaCausalLMTensor.from_pretrained("/home/ubuntu/profiler/Downloaded_checkpoint_pintxo")

# Function to print layer types
def print_layer_types(model):
    for name, module in model.named_modules():
        print(f"{name}: {type(module)}")

print("Model 1 Layers:")
print_layer_types(model_1)

print("\nModel 2 Layers:")
print_layer_types(model_2)
