# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import os
import json
import shutil
import sys
import warnings
from pathlib import Path
from typing import Dict

from packaging import version
import user_script

from olive.common.utils import set_tempdir
from olive.workflows import run as olive_run

# pylint: disable=redefined-outer-name
# ruff: noqa: TID252, T201

def update_config_with_provider(config: Dict, provider: str):
    if provider == "dml":
        # DirectML EP is the default, so no need to update config.
        return config
    elif provider == "cuda":
        from sd_utils.ort import update_cuda_config

        return update_cuda_config(config)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def optimize(
    model_id: str,
    model_type: str,
    provider: str,
):
    from google.protobuf import __version__ as protobuf_version

    # protobuf 4.x aborts with OOM when optimizing unet
    if version.parse(protobuf_version) > version.parse("3.20.3"):
        print("This script requires protobuf 3.20.3. Please ensure your package version matches requirements.txt.")
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent

    # Clean up previously optimized models, if any.
    shutil.rmtree(script_dir / "footprints", ignore_errors=True)

    # Load the entire PyTorch pipeline to ensure all models and their configurations are downloaded and cached.
    # This avoids an issue where the non-ONNX components (tokenizer, scheduler, and feature extractor) are not
    # automatically cached correctly if individual models are fetched one at a time.
    print(f"Download {model_type} PyTorch pipeline...")

    model_info = {}

    model_name = "midas_dpt_beit_large"
    model_path = "weights/dpt_beit_large_512.pt"
    print(f"\nOptimizing {model_name}")

    olive_config = None
    with (script_dir / f"config_{model_name}.json").open() as fin:
        olive_config = json.load(fin)
    olive_config = update_config_with_provider(olive_config, provider)

    olive_config["input_model"]["model_path"] = model_path

    run_res = olive_run(olive_config)

    from sd_utils.ort import save_optimized_onnx_submodel

    save_optimized_onnx_submodel(model_name, provider, model_info)
    return model_info


def parse_common_args(raw_args):
    parser = argparse.ArgumentParser("Common arguments")

    parser.add_argument("--model_type", choices=["controlnet_canny", "controlnet_depth", "sd"], default="sd", type=str)
    parser.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4", type=str)
    parser.add_argument(
        "--provider", default="dml", type=str, choices=["dml", "cuda"], help="Execution provider to use"
    )
    parser.add_argument("--optimize", action="store_true", help="Runs the optimization step")
    parser.add_argument("--clean_cache", action="store_true", help="Deletes the Olive cache")
    parser.add_argument("--test_unoptimized", action="store_true", help="Use unoptimized model for inference")
    parser.add_argument("--batch_size", default=1, type=int, help="Number of images to generate per batch")
    parser.add_argument(
        "--prompt",
        default=(
            "castle surrounded by water and nature, village, volumetric lighting, photorealistic, "
            "detailed and intricate, fantasy, epic cinematic shot, mountains, 8k ultra hd"
        ),
        type=str,
    )
    parser.add_argument(
        "--guidance_scale",
        default=7.5,
        type=float,
        help="Guidance scale as defined in Classifier-Free Diffusion Guidance",
    )
    parser.add_argument("--tempdir", default=None, type=str, help="Root directory for tempfile directories and files")

    return parser.parse_known_args(raw_args)


def parse_ort_args(raw_args):
    parser = argparse.ArgumentParser("ONNX Runtime arguments")

    parser.add_argument(
        "--static_dims",
        action="store_true",
        help="DEPRECATED (now enabled by default). Use --dynamic_dims to disable static_dims.",
    )
    parser.add_argument("--dynamic_dims", action="store_true", help="Disable static shape optimization")

    return parser.parse_known_args(raw_args)


def main(raw_args=None):
    common_args, extra_args = parse_common_args(raw_args)

    provider = common_args.provider

    script_dir = Path(__file__).resolve().parent

    if common_args.clean_cache:
        shutil.rmtree(script_dir / "cache", ignore_errors=True)

    ov_args, ort_args = None, None
    ort_args, extra_args = parse_ort_args(extra_args)

    if common_args.optimize:
        set_tempdir(common_args.tempdir)

        # TODO(jstoecker): clean up warning filter (mostly during conversion from torch to ONNX)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from sd_utils.ort import validate_args

            validate_args(ort_args, common_args.provider)
            optimize(common_args.model_id, common_args.model_type, common_args.provider)


if __name__ == "__main__":
    main()
