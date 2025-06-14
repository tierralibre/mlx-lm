# Copyright © 2023-2024 Apple Inc.

import argparse
from pathlib import Path
from typing import Callable, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

from .utils import (
    dequantize_model,
    fetch_from_hub,
    get_model_path,
    quantize_model,
    save,
    upload_to_hub,
)


def mixed_quant_predicate_builder(
    recipe: str, model: nn.Module
) -> Callable[[str, nn.Module, dict], Union[bool, dict]]:

    high_bits = 6
    group_size = 64

    if recipe == "mixed_2_6":
        low_bits = 2
    elif recipe == "mixed_3_4":
        low_bits = 3
        high_bits = 4
    elif recipe == "mixed_3_6":
        low_bits = 3
    elif recipe == "mixed_4_6":
        low_bits = 4
    else:
        raise ValueError("Invalid quant recipe {recipe}")

    down_keys = [k for k, _ in model.named_modules() if "down_proj" in k]
    if len(down_keys) == 0:
        raise ValueError("Model does not have expected keys for mixed quant.")

    # Look for the layer index location in the path:
    for layer_location, k in enumerate(down_keys[0].split(".")):
        if k.isdigit():
            break
    num_layers = len(model.layers)

    def mixed_quant_predicate(
        path: str,
        module: nn.Module,
        config: dict,
    ) -> Union[bool, dict]:
        """Implements mixed quantization predicates with similar choices to, for example, llama.cpp's Q4_K_M.
        Ref: https://github.com/ggerganov/llama.cpp/blob/917786f43d0f29b7c77a0c56767c0fa4df68b1c5/src/llama.cpp#L5265
        By Alex Barron: https://gist.github.com/barronalex/84addb8078be21969f1690c1454855f3
        """

        if not hasattr(module, "to_quantized"):
            return False

        index = (
            int(path.split(".")[layer_location])
            if len(path.split(".")) > layer_location
            else 0
        )
        use_more_bits = (
            index < num_layers // 8
            or index >= 7 * num_layers // 8
            or (index - num_layers // 8) % 3 == 2
        )
        if "v_proj" in path and use_more_bits:
            return {"group_size": group_size, "bits": high_bits}
        if "down_proj" in path and use_more_bits:
            return {"group_size": group_size, "bits": high_bits}
        if "lm_head" in path:
            return {"group_size": group_size, "bits": high_bits}

        return {"group_size": group_size, "bits": low_bits}

    return mixed_quant_predicate


QUANT_RECIPES = ["mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6"]

MODEL_CONVERSION_DTYPES = ["float16", "bfloat16", "float32"]


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_group_size: int = 64,
    q_bits: int = 4,
    dtype: Optional[str] = None,
    upload_repo: str = None,
    revision: Optional[str] = None,
    dequantize: bool = False,
    quant_predicate: Optional[
        Union[Callable[[str, nn.Module, dict], Union[bool, dict]], str]
    ] = None,
    task: str = "generate",
):

    # Check the save path is empty
    if isinstance(mlx_path, str):
        mlx_path = Path(mlx_path)

    if mlx_path.exists():
        raise ValueError(
            f"Cannot save to the path {mlx_path} as it already exists."
            " Please delete the file/directory or specify a new path to save to."
        )

    print("[INFO] Loading")
    model_path = get_model_path(hf_path, revision=revision)

    # Custom path for embedding task
    if task == "embed":
        import json, glob
        import safetensors.torch as sttorch
        from .tokenizer_utils import load_tokenizer
        from .utils import _get_classes

        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
        config["task"] = "embed"
        config["model_type"] = "qwen2_embed"
        # Filter HF config keys when building ArgsCls
        config = {k: v for k, v in config.items() if not k.startswith("hf_")}
        # Build model
        ModelCls, ArgsCls = _get_classes(config)
        allowed_keys = ArgsCls.__init__.__code__.co_varnames
        args = ArgsCls(**{k: v for k, v in config.items() if k in allowed_keys})
        model = ModelCls(args)
        # Load weights from safetensors
        weight_files = glob.glob(str(model_path / "*.safetensors"))
        weights = {}
        import torch
        for wf in weight_files:
            w = sttorch.load_file(wf, device="cpu")
            processed = {}
            for k, v in w.items():
                if v.dtype == torch.bfloat16:
                    v = v.to(torch.float16)
                processed[k] = mx.array(v.cpu().numpy())
            weights.update(processed)
        model.load_weights(list(weights.items()), strict=False)
        tokenizer = load_tokenizer(model_path, {}, eos_token_ids=config.get("eos_token_id", None))
    else:
        model, config, tokenizer = fetch_from_hub(model_path, lazy=True)

    if isinstance(quant_predicate, str):
        quant_predicate = mixed_quant_predicate_builder(quant_predicate, model)

    if dtype is None:
        dtype = config.get("torch_dtype", None)
    if dtype in MODEL_CONVERSION_DTYPES:
        print("[INFO] Using dtype:", dtype)
        dtype = getattr(mx, dtype)
        cast_predicate = getattr(model, "cast_predicate", lambda _: True)

        def set_dtype(k, v):
            if cast_predicate(k) and mx.issubdtype(v.dtype, mx.floating):
                return v.astype(dtype)
            else:
                return v

        model.update(tree_map_with_path(set_dtype, model.parameters()))

    if quantize and dequantize:
        raise ValueError("Choose either quantize or dequantize, not both.")

    if quantize:
        print("[INFO] Quantizing")
        model, config = quantize_model(
            model, config, q_group_size, q_bits, quant_predicate=quant_predicate
        )

    if dequantize:
        print("[INFO] Dequantizing")
        model = dequantize_model(model)

    # If hf_path is a local directory, do not attempt to create/update a model card
    hf_repo_for_card = None if Path(hf_path).exists() else hf_path
    save(
        mlx_path,
        model_path,
        model,
        tokenizer,
        config,
        hf_repo=hf_repo_for_card,
    )

    if upload_repo is not None:
        upload_to_hub(mlx_path, upload_repo)


def configure_parser() -> argparse.ArgumentParser:
    """
    Configures and returns the argument parser for the script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face model to MLX format"
    )

    parser.add_argument("--hf-path", type=str, help="Path to the Hugging Face model.")
    parser.add_argument(
        "--mlx-path", type=str, default="mlx_model", help="Path to save the MLX model."
    )
    parser.add_argument(
        "-q", "--quantize", help="Generate a quantized model.", action="store_true"
    )
    parser.add_argument(
        "--q-group-size", help="Group size for quantization.", type=int, default=64
    )
    parser.add_argument(
        "--q-bits", help="Bits per weight for quantization.", type=int, default=4
    )
    parser.add_argument(
        "--quant-predicate",
        help=f"Mixed-bit quantization recipe.",
        choices=QUANT_RECIPES,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--task",
        choices=["generate", "embed"],
        default="generate",
        help="Task type: generate (text generation) or embed (embeddings)",
    )

    parser.add_argument(
        "--dtype",
        help="Type to save the non-quantized parameters. Defaults to config.json's `torch_dtype` or the current model weights dtype.",
        type=str,
        choices=MODEL_CONVERSION_DTYPES,
        default=None,
    )
    parser.add_argument(
        "--upload-repo",
        help="The Hugging Face repo to upload the model to.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-d",
        "--dequantize",
        help="Dequantize a quantized model.",
        action="store_true",
        default=False,
    )
    return parser


def main():
    parser = configure_parser()
    args = parser.parse_args()
    convert(**vars(args))


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.convert ...` directly is deprecated."
        " Use `mlx_lm.convert ...` or `python -m mlx_lm convert ...` instead."
    )
    main()
