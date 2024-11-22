# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import config
import torch
import os

from olive.data.registry import Registry
from midas.model_loader import load_model

print("Loading user script")

# Helper latency-only dataloader that creates random tensors with no label
class RandomDataLoader:
    def __init__(self, create_inputs_func, batch_size, torch_dtype):
        self.create_input_func = create_inputs_func
        self.batch_size = batch_size
        self.torch_dtype = torch_dtype

    def __getitem__(self, idx):
        label = None
        return self.create_input_func(self.batch_size, self.torch_dtype), label

def midas_depth_inputs(batch_size, torch_dtype):
    inputs = {
        "input": torch.rand((batch_size, 3, config.image_height, config.image_width), dtype=torch_dtype),
    }
    return inputs


def midas_depth_load(model_name):
    model_type = os.path.splitext(os.path.basename(model_name))[0]
    model, _, net_w, net_h = load_model(torch.device("cpu"), model_name, model_type, optimize=True, height=config.image_height, square=False)
    print(f"Loaded model: {model_name} with width:{net_w} height:{net_h}")
    return model


def midas_depth_conversion_inputs(model=None):
    return tuple(midas_depth_inputs(1, torch.float32).values())


@Registry.register_dataloader()
def midas_depth_data_loader(dataset, batch_size, *args, **kwargs):
    return RandomDataLoader(midas_depth_inputs, batch_size, torch.float16)