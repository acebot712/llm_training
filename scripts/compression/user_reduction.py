import os
import torch
import logging
from transformers import AutoModelForCausalLM
from torch import nn
import tensorly as tl
from tensorly.decomposition import parafac
import inspect

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurations
MODEL_NAME = os.getenv("MODEL_NAME", "NousResearch/Llama-2-7b-chat-hf")
SAVE_PATH = os.getenv("SAVE_PATH", "decomposed_llama_model_state_dict.pt")
MODEL_CLASS_CODE_PATH = os.getenv("MODEL_CLASS_CODE_PATH", "decomposed_llama_model_class.py")
CUDA_DEVICE = os.getenv("CUDA_DEVICE", "cuda:1")

def load_model(model_name, device):
    """Load the original model."""
    logger.info(f"Loading the original model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    logger.info("Original model loaded successfully.")
    return model

def count_parameters(model):
    """Count the parameters of a model."""
    return sum(p.numel() for p in model.parameters())

def decompose_linear_layer(layer, rank, device):
    """Decompose a linear layer using CP decomposition."""
    logger.info(f"Decomposing layer with rank: {rank}")
    weight = layer.weight.data.cpu().numpy()  # Move to CPU to convert to numpy
    logger.info(f"Original weight shape: {weight.shape}")
    
    factors = parafac(weight, rank=rank)
    logger.info("Decomposition factors obtained.")
    
    decomposed_weight = tl.kruskal_to_tensor(factors)
    logger.info(f"Decomposed weight shape: {decomposed_weight.shape}")
    
    layer.weight.data = torch.tensor(decomposed_weight, dtype=layer.weight.dtype).to(device)
    logger.info("Layer weight updated with decomposed weight.")
    return layer

class DecomposedLlamaModel(nn.Module):
    def __init__(self, original_model, decomposed_layers):
        super(DecomposedLlamaModel, self).__init__()
        self.original_model = original_model
        self.decomposed_layers = decomposed_layers
        logger.info("DecomposedLlamaModel initialized with decomposed layers.")

    def forward(self, *args, **kwargs):
        for i, layer in enumerate(self.decomposed_layers):
            self.original_model.model.layers[i].mlp.gate_proj = layer
        return self.original_model(*args, **kwargs)

def decompose_model_to_target_reduction(original_model, original_params, target_reduction_percentage, device):
    """Decompose the model layers to achieve the target reduction percentage."""
    current_reduction_percentage = 0
    rank = original_model.model.layers[0].mlp.gate_proj.weight.shape[0]  # Start with the maximum rank

    while current_reduction_percentage < target_reduction_percentage and rank > 1:
        decomposed_layers = []
        for i in range(len(original_model.model.layers)):
            decomposed_layer = decompose_linear_layer(original_model.model.layers[i].mlp.gate_proj, rank=rank, device=device)
            decomposed_layers.append(decomposed_layer)

        decomposed_model = DecomposedLlamaModel(original_model, decomposed_layers)
        decomposed_params = count_parameters(decomposed_model)
        reduction = original_params - decomposed_params
        current_reduction_percentage = (reduction / original_params) * 100

        if current_reduction_percentage < target_reduction_percentage:
            rank -= 1

    return decomposed_model, current_reduction_percentage

def save_model_state(model, path):
    """Save the state dictionary of the model."""
    torch.save(model.state_dict(), path)
    logger.info(f"Decomposed model state dictionary saved to {path}")

def save_model_class_code(class_code, path):
    """Save the custom model class code."""
    with open(path, "w") as f:
        f.write(class_code)
    logger.info(f"DecomposedLlamaModel class code saved to {path}")

def main():
    device = torch.device(CUDA_DEVICE)
    
    # Load the original model
    original_model = load_model(MODEL_NAME, device)
    
    # Count parameters in the original model
    original_params = count_parameters(original_model)
    logger.info(f"Original model parameters: {original_params}")
    
    # Get user input for target reduction percentage
    target_reduction_percentage = float(input("Enter the desired reduction percentage: "))
    
    # Decompose the model to achieve the target reduction percentage
    decomposed_model, achieved_reduction_percentage = decompose_model_to_target_reduction(
        original_model, original_params, target_reduction_percentage, device
    )
    
    logger.info(f"Achieved reduction percentage: {achieved_reduction_percentage:.2f}%")
    
    # Count parameters in the decomposed model
    decomposed_params = count_parameters(decomposed_model)
    logger.info(f"Decomposed model parameters: {decomposed_params}")
    
    # Calculate reduction in parameters
    reduction = original_params - decomposed_params
    reduction_percentage = (reduction / original_params) * 100
    
    logger.info(f"Reduction in parameters: {reduction} parameters")
    logger.info(f"Reduction percentage: {reduction_percentage:.2f}%")
    
    # Save the decomposed model state dictionary
    save_model_state(decomposed_model, SAVE_PATH)
    
    # Save the custom model class code
    model_code = inspect.getsource(DecomposedLlamaModel)
    save_model_class_code(model_code, MODEL_CLASS_CODE_PATH)
    
    # Loading the decomposed model for testing
    logger.info("Loading the decomposed model for testing...")
    test_original_model = load_model(MODEL_NAME, device)
    
    with open(MODEL_CLASS_CODE_PATH, "r") as f:
        exec(f.read())
    
    test_decomposed_model = DecomposedLlamaModel(test_original_model, [])
    test_decomposed_model.load_state_dict(torch.load(SAVE_PATH))
    test_decomposed_model = test_decomposed_model.to(device)
    logger.info("Decomposed model loaded successfully.")

if __name__ == "__main__":
    main()
