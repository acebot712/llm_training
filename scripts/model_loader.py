"""
Model loader mixtral
"""

import os
import torch.utils.checkpoint
from transformers import MixtralConfig, MixtralModel, MixtralForCausalLM
from transformers.modeling_utils import load_state_dict
from tqdm import tqdm
from typing import List, Tuple

import torch
import torch.nn as nn


class TensorizedLinearLayer(nn.Module):
    def __init__(
        self,
        tensorized_shapes: List[List[int]],
        bias: bool,
        contraction_map: str,
        full_contraction_map: str,
        reranked_input: List[int],
        reranked_output: List[int],
        input_dim: int,
        output_dim: int,
        use_full_contraction: bool = True,
        expected_batch_size: int = 64,
        path: List[Tuple[int]] = None,
        expr: List[str] = None,
        permutation: List[int] = None,
        full_path: List[Tuple[int]] = None,
        full_expr: List[str] = None,
        debug_info: bool = False,
        quant8=False,
    ):
        super(TensorizedLinearLayer, self).__init__()
        self.quant8 = quant8
        self.ttensors = []
        for i, ts in enumerate(tensorized_shapes):
            self.register_parameter(
                name=f"ttensor_{i}", param=torch.nn.Parameter(torch.randn(ts))
            )
            self.ttensors.append(self.get_parameter(f"ttensor_{i}"))
        if bias:
            self.register_parameter(
                name="bias", param=nn.Parameter(torch.rand([output_dim]))
            )
        else:
            self.bias = None
        self.contraction_map = contraction_map
        self.full_contraction_map = full_contraction_map
        self.reranked_input = reranked_input
        self.reranked_output = reranked_output
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_full_contraction = use_full_contraction
        self.batch_size = expected_batch_size
        self.path = path
        self.expr = expr
        self.permutation_ = permutation
        self.full_path = full_path
        self.full_expr = full_expr
        self.debug_info = debug_info
        # self.full_tn = None
        # self.tn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.use_full_contraction:
            output = self.full_contraction_forward(x)
        else:
            output = self.reconstructed_forward(x)
        if self.bias is not None:
            output += self.bias
        return output

    def full_contraction_forward(self, x: torch.Tensor) -> torch.Tensor:
        res_x = x.view([-1] + self.reranked_input)
        tensors = [res_x] + [t for t in self.ttensors]
        for i, step in enumerate(self.full_path):
            output = torch.einsum(self.full_expr[i], tensors[step[0]], tensors[step[1]])  # type: ignore[misc]
            tensors[step[0]] = output
            tensors.pop(step[1])  # type: ignore[misc]
        return output.view([s for s in x.shape[:-1]] + [self.output_dim])

    def reconstructed_forward(self, x: torch.Tensor) -> torch.Tensor:

        tensors = [t for t in self.ttensors]
        for i, step in enumerate(self.path):
            output = torch.einsum(self.expr[i], tensors[step[0]], tensors[step[1]])
            tensors[step[0]] = output
            tensors.pop(step[1])
        reranked_contracted = output.permute(self.permutation_)
        contracted_weight = reranked_contracted.reshape(
            [self.input_dim, self.output_dim]
        )
        output = torch.matmul(x, contracted_weight)
        return output


class TensorizedLinearLayer_t(nn.Module):
    def __init__(
        self,
        tensorized_shapes: List[List[int]],
        bias: bool,
        contraction_map: str,
        full_contraction_map: str,
        input_dim: int,
        output_dim: int,
        use_full_contraction: bool = True,
        expected_batch_size: int = 64,
        path: List[Tuple[int]] = None,
        expr: List[str] = None,
        full_path: List[Tuple[int]] = None,
        full_expr: List[str] = None,
        debug_info: bool = False,
        quant8=False,
    ):
        super(TensorizedLinearLayer_t, self).__init__()
        self.ttensors = []
        for i, ts in enumerate(tensorized_shapes):
            self.register_parameter(
                name=f"ttensor_{i}", param=torch.nn.Parameter(torch.randn(ts))
            )
            self.ttensors.append(self.get_parameter(f"ttensor_{i}"))
        if bias:
            self.register_parameter(
                name="bias", param=nn.Parameter(torch.rand([output_dim]))
            )
        else:
            self.bias = None
        self.contraction_map = contraction_map
        self.full_contraction_map = full_contraction_map
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_full_contraction = use_full_contraction
        self.batch_size = expected_batch_size
        self.path = path
        self.expr = expr
        self.full_path = full_path
        self.full_expr = full_expr
        self.debug_info = debug_info

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.use_full_contraction:
            output = self.full_contraction_forward(x)
        else:
            output = self.reconstructed_forward(x)
        if self.bias is not None:
            output += self.bias

        # self.ttensors[0]=self.ttensors[0].to(torch.int8)
        return output

    def full_contraction_forward(self, x: torch.Tensor) -> torch.Tensor:
        tensors = [x.view([-1, self.input_dim])] + [t for t in self.ttensors]
        for i, step in enumerate(self.full_path):
            # output = torch.einsum(self.full_expr[i], tensors_train_a1000[step[0]], tensors_train_a1000[step[1]])
            output = torch.matmul(tensors[step[0]], tensors[step[1]])  # type: ignore[misc]
            tensors[step[0]] = output
            tensors.pop(step[1])  # type: ignore[misc]
        return output.view([s for s in x.shape[:-1]] + [self.output_dim])

    def reconstructed_forward(self, x: torch.Tensor) -> torch.Tensor:
        tensors = [t for t in self.ttensors]
        for i, step in enumerate(self.path):
            # output = torch.einsum(self.expr[i], tensors_train_a1000[step[0]], tensors_train_a1000[step[1]])
            output = torch.matmul(tensors[step[0]], tensors[step[1]])  # type: ignore[misc]
            tensors[step[0]] = output
            tensors.pop(step[1])  # type: ignore[misc]
        output = torch.matmul(x, output)
        return output


def convert_name_to_attribute_path(name):
    """
    Convert dot-separated module name to Python attribute access syntax.
    """
    components = name.split(".")
    formatted_path = components[0]
    for i, component in enumerate(components[1:]):
        if component.isdigit():
            formatted_path += f"[{component}]"
        else:
            formatted_path += f".{component}"
    return formatted_path


class Mixtral8x7Tensor(MixtralForCausalLM):
    """
    Tensorized Mixtral8x7b model with support for custom tensorized layers.
    """

    config_class = MixtralConfig

    def __init__(self, config):
        super().__init__(config)
        self.tensorized_info = config.tensorized_layers_info
        self.model_path = config.location

        # Apply tensorized layers
        for name, arguments in tqdm(
            self.tensorized_info.items(), desc="Loading Tensorized Layers"
        ):
            tensorized_layer = TensorizedLinearLayer(**arguments)
            attribute_path = convert_name_to_attribute_path(name)
            print("Atribute_path------:", attribute_path)
            exec(
                f"self.{attribute_path} = tensorized_layer",
                {},
                {"self": self, "tensorized_layer": tensorized_layer},
            )

        # Update model weights
        state_dict = {}
        for file in os.listdir(self.model_path):
            if file.endswith(".safetensors") or file.endswith(".bin"):
                state_dict.update(load_state_dict(os.path.join(self.model_path, file)))
        self.load_state_dict(state_dict, strict=False)

    def forward(self, input_ids=None, **kwargs):
        return super().forward(input_ids=input_ids, **kwargs)


class Mixtral8x7Tensor_train(MixtralForCausalLM):
    """
    Tensorized Mixtral8x7b model with support for custom tensorized layers.
    """

    config_class = MixtralConfig

    def __init__(self, config):
        super().__init__(config)
        self.tensorized_info = config.tensorized_layers_info
        self.model_path = config.location

        # Apply tensorized layers
        for name, arguments in tqdm(
            self.tensorized_info.items(), desc="Loading Tensorized Layers"
        ):
            tensorized_layer = TensorizedLinearLayer_t(**arguments)
            attribute_path = convert_name_to_attribute_path(name)
            print("Atribute_path------:", attribute_path)
            exec(
                f"self.{attribute_path} = tensorized_layer",
                {},
                {"self": self, "tensorized_layer": tensorized_layer},
            )

        # Update model weights
        state_dict = {}
        for file in os.listdir(self.model_path):
            if file.endswith(".safetensors") or file.endswith(".bin"):
                state_dict.update(load_state_dict(os.path.join(self.model_path, file)))
        self.load_state_dict(state_dict, strict=False)

    def forward(self, input_ids=None, **kwargs):
        return super().forward(input_ids=input_ids, **kwargs)


import os
from tqdm import tqdm
import torch
import sys

from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    AutoTokenizer,
)

from transformers.modeling_utils import load_state_dict


def convert_name_to_attribute_path(name):
    """
    Convert dot-separated module name to Python attribute access syntax.
    """
    components = name.split(".")
    formatted_path = components[0]
    for i, component in enumerate(components[1:]):
        if component.isdigit():
            formatted_path += f"[{component}]"
        else:
            formatted_path += f".{component}"
    return formatted_path


class LlamaCausalLMTensor(LlamaForCausalLM):
    """
    Tensorized LlamaForCausalLM model with support for custom tensorized layers.
    """

    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.tensorized_info = config.tensorized_layers_info
        self.model_path = config.location
        self.model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        for name, arguments in tqdm(
            self.tensorized_info.items(), desc="Loading Tensorized Layers"
        ):
            tensorized_layer = TensorizedLinearLayer(**arguments)
            attribute_path = convert_name_to_attribute_path(name)
            exec(
                f"self.model.{attribute_path} = tensorized_layer",
                {},
                {"self": self, "tensorized_layer": tensorized_layer},
            )

        # Update model weights:
        state_dict = {}
        for file in os.listdir(self.model_path):
            if file.endswith(".safetensors") or file.endswith(".bin"):
                state_dict.update(load_state_dict(os.path.join(self.model_path, file)))
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.model


class LlamaCausalLMTensor_train(LlamaForCausalLM):
    """
    Tensorized LlamaForCausalLM model with support for custom tensorized layers.
    """

    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.tensorized_info = config.tensorized_layers_info
        self.model_path = config.location
        self.model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        for name, arguments in tqdm(
            self.tensorized_info.items(), desc="Loading Tensorized Layers"
        ):
            tensorized_layer = TensorizedLinearLayer_t(**arguments)
            attribute_path = convert_name_to_attribute_path(name)
            exec(
                f"self.model.{attribute_path} = tensorized_layer",
                {},
                {"self": self, "tensorized_layer": tensorized_layer},
            )

        # Update model weights:
        state_dict = {}
        for file in os.listdir(self.model_path):
            if file.endswith(".safetensors") or file.endswith(".bin"):
                state_dict.update(load_state_dict(os.path.join(self.model_path, file)))
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.model
