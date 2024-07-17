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
RANK = int(os.getenv("DECOMPOSITION_RANK", 10))
SAVE_PATH = os.getenv("SAVE_PATH", "decomposed_llama_model_state_dict.pt")
MODEL_CLASS_CODE_PATH = os.getenv("MODEL_CLASS_CODE_PATH", "decomposed_llama_model_class.py")
CUDA_DEVICE = os.getenv("CUDA_DEVICE", "cuda:0")


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


def decompose_model_layers(model, rank, device):
    """Decompose all linear layers in the model."""
    logger.info("Decomposing all layers...")
    decomposed_layers = []
    for i, layer in enumerate(model.model.layers):
        logger.info(f"Processing layer {i+1}/{len(model.model.layers)}")
        decomposed_layer = decompose_linear_layer(layer.mlp.gate_proj, rank, device)
        decomposed_layers.append(decomposed_layer)
    logger.info("All layers decomposed.")
    return decomposed_layers


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
    
    # Decompose the model layers
    decomposed_layers = decompose_model_layers(original_model, RANK, device)
    
    # Wrap the original model with the decomposed layers
    original_model = nn.DataParallel(original_model)
    decomposed_model = DecomposedLlamaModel(original_model, decomposed_layers)
    
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
    
    # Load the decomposed model for testing
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
