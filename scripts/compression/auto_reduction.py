from transformers import AutoModelForCausalLM
import torch
import tensorly as tl
from tensorly.decomposition import parafac
from torch import nn

# Load the original model
model_name = "NousResearch/Llama-2-7b-chat-hf"
print(f"Loading the original model: {model_name}")
original_model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
print("Original model loaded successfully.")

# Function to count the parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Count parameters in the original model
original_params = count_parameters(original_model)
print(f"Original model parameters: {original_params}")

# Function to decompose a linear layer using CP decomposition with GPU
def decompose_linear_layer(layer, rank):
    print(f"Decomposing layer: {layer} with rank: {rank}")
    weight = layer.weight.data.cpu().numpy()  # Moving to CPU to convert to numpy
    print(f"Original weight shape: {weight.shape}")
    factors = parafac(weight, rank=rank)
    print("Decomposition factors obtained.")
    decomposed_weight = tl.kruskal_to_tensor(factors)
    print(f"Decomposed weight shape: {decomposed_weight.shape}")
    layer.weight.data = torch.tensor(decomposed_weight, dtype=layer.weight.dtype).to('cuda')
    print("Layer weight updated with decomposed weight.")
    return layer

# Apply decomposition to the first MLP layer in the model
original_layer = original_model.model.layers[0].mlp.gate_proj
print(f"Original layer: {original_layer}")
decomposed_layer = decompose_linear_layer(original_layer, rank=10)
print("First MLP layer decomposed.")

class DecomposedLlamaModel(nn.Module):
    def __init__(self, original_model, decomposed_layers):
        super(DecomposedLlamaModel, self).__init__()
        self.original_model = original_model
        self.decomposed_layers = decomposed_layers
        print("DecomposedLlamaModel initialized with decomposed layers.")

    def forward(self, *args, **kwargs):
        for i, layer in enumerate(self.decomposed_layers):
            print(f"Replacing layer {i} with decomposed layer.")
            self.original_model.model.layers[i].mlp.gate_proj = layer
        return self.original_model(*args, **kwargs)

# Decompose all layers and utilize multiple GPUs
print("Decomposing all layers...")
decomposed_layers = []
for i in range(len(original_model.model.layers)):
    print(f"Processing layer {i+1}/{len(original_model.model.layers)}")
    decomposed_layer = nn.DataParallel(decompose_linear_layer(original_model.model.layers[i].mlp.gate_proj, rank=10))
    decomposed_layers.append(decomposed_layer)
print("All layers decomposed.")

# Move the original model to DataParallel to utilize multiple GPUs
original_model = nn.DataParallel(original_model)

# Initialize decomposed model with decomposed layers
decomposed_model = DecomposedLlamaModel(original_model, decomposed_layers)

# Count parameters in the decomposed model
decomposed_params = count_parameters(decomposed_model)
print(f"Decomposed model parameters: {decomposed_params}")

# Calculate reduction in parameters
reduction = original_params - decomposed_params
reduction_percentage = (reduction / original_params) * 100

print(f"Reduction in parameters: {reduction} parameters")
print(f"Reduction percentage: {reduction_percentage:.2f}%")
