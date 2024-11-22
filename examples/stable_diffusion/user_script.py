# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import config
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from huggingface_hub import model_info
from transformers.models.clip.modeling_clip import CLIPTextModel

from olive.data.registry import Registry

# model resolution:
# either 512x512
# or 640x360

# Helper latency-only dataloader that creates random tensors with no label
class RandomDataLoader:
    def __init__(self, create_inputs_func, batch_size, torch_dtype):
        self.create_input_func = create_inputs_func
        self.batch_size = batch_size
        self.torch_dtype = torch_dtype

    def __getitem__(self, idx):
        label = None
        return self.create_input_func(self.batch_size, self.torch_dtype), label


def get_base_model_name(model_name):
    return model_info(model_name).cardData.get("base_model") or model_name


# -----------------------------------------------------------------------------
# TEXT ENCODER
# -----------------------------------------------------------------------------


def text_encoder_inputs(batch_size, torch_dtype):
    return torch.zeros((batch_size, 77), dtype=torch_dtype)


def text_encoder_load(model_name):
    base_model_id = get_base_model_name(model_name)
    model = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder")
    return model


def text_encoder_conversion_inputs(model=None):
    return text_encoder_inputs(1, torch.int32)


@Registry.register_dataloader()
def text_encoder_data_loader(dataset, batch_size, *args, **kwargs):
    return RandomDataLoader(text_encoder_inputs, batch_size, torch.int32)


# -----------------------------------------------------------------------------
# Controlnet
# -----------------------------------------------------------------------------

def controlnet_inputs(batch_size, torch_dtype):
    inputs = {
        "sample": torch.rand((batch_size, 4, config.unet_sample_height, config.unet_sample_width), dtype=torch_dtype),
        "timestep": torch.rand((batch_size,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((batch_size, 77, config.cross_attention_dim), dtype=torch_dtype),
        "controlnet_cond": torch.rand((batch_size, 3, config.unet_sample_height * 8, config.unet_sample_width * 8), dtype=torch_dtype),
        "controlnet_scale": torch.rand((batch_size, 1), dtype=torch_dtype),
    }
    return inputs


def controlnet_load(model_name):
    model = ControlNetModel.from_pretrained(model_name)
    return model


def controlnet_conversion_inputs(model=None):
    return tuple(controlnet_inputs(1, torch.float32).values())


@Registry.register_dataloader()
def controlnet_data_loader(dataset, batch_size, *args, **kwargs):
    return RandomDataLoader(controlnet_inputs, batch_size, torch.float16)


# -----------------------------------------------------------------------------
# Unet-controlnet
# -----------------------------------------------------------------------------
def unet_controlnet_inputs(batch_size, torch_dtype, is_conversion_inputs=False):
    # TODO(jstoecker): Rename onnx::Concat_4 to text_embeds and onnx::Shape_5 to time_ids
    sample_width = config.unet_sample_width
    sample_height = config.unet_sample_height
    down_control_block =     [torch.rand((batch_size, 320,  sample_height, sample_width), dtype=torch_dtype)]
    down_control_block.append(torch.rand((batch_size, 320,  sample_height, sample_width), dtype=torch_dtype))
    down_control_block.append(torch.rand((batch_size, 320,  sample_height, sample_width), dtype=torch_dtype))
    down_control_block.append(torch.rand((batch_size, 320,  int(round(sample_height / 2)), int(round(sample_width / 2))), dtype=torch_dtype))
    down_control_block.append(torch.rand((batch_size, 640,  int(round(sample_height / 2)), int(round(sample_width / 2))), dtype=torch_dtype))
    down_control_block.append(torch.rand((batch_size, 640,  int(round(sample_height / 2)), int(round(sample_width / 2))), dtype=torch_dtype))
    down_control_block.append(torch.rand((batch_size, 640,  int(round(sample_height / 4)), int(round(sample_width / 4))), dtype=torch_dtype))
    down_control_block.append(torch.rand((batch_size, 1280, int(round(sample_height / 4)), int(round(sample_width / 4))), dtype=torch_dtype))
    down_control_block.append(torch.rand((batch_size, 1280, int(round(sample_height / 4)), int(round(sample_width / 4))), dtype=torch_dtype))
    down_control_block.append(torch.rand((batch_size, 1280, int(round(sample_height / 8)), int(round(sample_width / 8))), dtype=torch_dtype))
    down_control_block.append(torch.rand((batch_size, 1280, int(round(sample_height / 8)), int(round(sample_width / 8))), dtype=torch_dtype))
    down_control_block.append(torch.rand((batch_size, 1280, int(round(sample_height / 8)), int(round(sample_width / 8))), dtype=torch_dtype))

    mid_control_block = torch.rand((1, 1280, int(round(sample_height / 8)), int(round(sample_width / 8))), dtype=torch_dtype)
    inputs = {
        "sample": torch.rand((batch_size, 4, config.unet_sample_height, config.unet_sample_width), dtype=torch_dtype),
        "timestep": torch.rand((batch_size,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((batch_size, 77, config.cross_attention_dim), dtype=torch_dtype),
    }

    # use as kwargs since they won't be in the correct position if passed along with the tuple of inputs
    kwargs = {
        "return_dict": False,
    }
    if is_conversion_inputs:
        inputs["additional_inputs"] = {
            **kwargs,
            "down_block_additional_residuals": down_control_block,
            "mid_block_additional_residual": mid_control_block,
        }
    else:
        inputs.update(kwargs)
        inputs["controlnet_downblock1"] =  down_control_block[0]
        inputs["controlnet_downblock2"] =  down_control_block[1]
        inputs["controlnet_downblock3"] =  down_control_block[2]
        inputs["controlnet_downblock4"] =  down_control_block[3]
        inputs["controlnet_downblock5"] =  down_control_block[4]
        inputs["controlnet_downblock6"] =  down_control_block[5]
        inputs["controlnet_downblock7"] =  down_control_block[6]
        inputs["controlnet_downblock8"] =  down_control_block[7]
        inputs["controlnet_downblock9"] =  down_control_block[8]
        inputs["controlnet_downblock10"] = down_control_block[9]
        inputs["controlnet_downblock11"] = down_control_block[10]
        inputs["controlnet_downblock12"] = down_control_block[11]
        inputs["controlnet_midblock"] = mid_control_block

    return inputs


def unet_controlnet_load(model_name):
    #TODO: merge this with the other unet function
    base_model_id = get_base_model_name(model_name)
    model = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
    return model

def get_unet_ov_example_input():
    import numpy as np

    encoder_hidden_state = torch.ones((2, 77, 768))
    latents_shape = (2, 4, config.vae_sample_height // 8, config.vae_sample_width // 8)
    latents = torch.randn(latents_shape)
    t = torch.from_numpy(np.array(1, dtype=float))
    batch_size = 1
    sample_width = config.unet_sample_width
    sample_height = config.unet_sample_height
    torch_dtype = torch.float32
    down_control_block =     [torch.rand((batch_size, 320,  sample_height, sample_width), dtype=torch_dtype)]
    down_control_block.append(torch.rand((batch_size, 320,  sample_height, sample_width), dtype=torch_dtype))
    down_control_block.append(torch.rand((batch_size, 320,  sample_height, sample_width), dtype=torch_dtype))
    down_control_block.append(torch.rand((batch_size, 320,  int(round(sample_height / 2)), int(round(sample_width / 2))), dtype=torch_dtype))
    down_control_block.append(torch.rand((batch_size, 640,  int(round(sample_height / 2)), int(round(sample_width / 2))), dtype=torch_dtype))
    down_control_block.append(torch.rand((batch_size, 640,  int(round(sample_height / 2)), int(round(sample_width / 2))), dtype=torch_dtype))
    down_control_block.append(torch.rand((batch_size, 640,  int(round(sample_height / 4)), int(round(sample_width / 4))), dtype=torch_dtype))
    down_control_block.append(torch.rand((batch_size, 1280, int(round(sample_height / 4)), int(round(sample_width / 4))), dtype=torch_dtype))
    down_control_block.append(torch.rand((batch_size, 1280, int(round(sample_height / 4)), int(round(sample_width / 4))), dtype=torch_dtype))
    down_control_block.append(torch.rand((batch_size, 1280, int(round(sample_height / 8)), int(round(sample_width / 8))), dtype=torch_dtype))
    down_control_block.append(torch.rand((batch_size, 1280, int(round(sample_height / 8)), int(round(sample_width / 8))), dtype=torch_dtype))
    down_control_block.append(torch.rand((batch_size, 1280, int(round(sample_height / 8)), int(round(sample_width / 8))), dtype=torch_dtype))

    mid_control_block = torch.rand((1, 1280, int(round(sample_height / 8)), int(round(sample_width / 8))), dtype=torch_dtype)
    return (latents, t, encoder_hidden_state, None, None, None, None, None, down_control_block, mid_control_block)

def unet_controlnet_conversion_inputs(model=None):
    return tuple(unet_controlnet_inputs(1, torch.float32, True).values())


@Registry.register_dataloader()
def unet_controlnet_data_loader(dataset, batch_size, *args, **kwargs):
    return RandomDataLoader(unet_controlnet_inputs, batch_size, torch.float16)


# -----------------------------------------------------------------------------
# VAE ENCODER
# -----------------------------------------------------------------------------


def vae_encoder_inputs(batch_size, torch_dtype):
    return {"sample": torch.rand((batch_size, 3, config.vae_sample_height, config.vae_sample_width), dtype=torch_dtype)}


def vae_encoder_load(model_name):
    base_model_id = get_base_model_name(model_name)
    model = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")
    model.forward = lambda sample: model.encode(sample)[0].sample()
    return model


def vae_encoder_conversion_inputs(model=None):
    return tuple(vae_encoder_inputs(1, torch.float32).values())


@Registry.register_dataloader()
def vae_encoder_data_loader(dataset, batch_size, *args, **kwargs):
    return RandomDataLoader(vae_encoder_inputs, batch_size, torch.float16)


# -----------------------------------------------------------------------------
# VAE DECODER
# -----------------------------------------------------------------------------


def vae_decoder_inputs(batch_size, torch_dtype):
    return {
        "latent_sample": torch.rand(
            (batch_size, 4, config.unet_sample_height, config.unet_sample_width), dtype=torch_dtype
        )
    }


def vae_decoder_load(model_name):
    base_model_id = get_base_model_name(model_name)
    model = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")
    model.forward = model.decode
    return model


def vae_decoder_conversion_inputs(model=None):
    return tuple(vae_decoder_inputs(1, torch.float32).values())


@Registry.register_dataloader()
def vae_decoder_data_loader(dataset, batch_size, *args, **kwargs):
    return RandomDataLoader(vae_decoder_inputs, batch_size, torch.float16)


# -----------------------------------------------------------------------------
# SAFETY CHECKER
# -----------------------------------------------------------------------------


def safety_checker_inputs(batch_size, torch_dtype):
    return {
        "clip_input": torch.rand((batch_size, 3, 224, 224), dtype=torch_dtype),
        "images": torch.rand((batch_size, config.vae_sample_width, config.vae_sample_width, 3), dtype=torch_dtype),
    }


def safety_checker_load(model_name):
    base_model_id = get_base_model_name(model_name)
    model = StableDiffusionSafetyChecker.from_pretrained(base_model_id, subfolder="safety_checker")
    model.forward = model.forward_onnx
    return model


def safety_checker_conversion_inputs(model=None):
    return tuple(safety_checker_inputs(1, torch.float32).values())


@Registry.register_dataloader()
def safety_checker_data_loader(dataset, batch_size, *args, **kwargs):
    return RandomDataLoader(safety_checker_inputs, batch_size, torch.float16)
