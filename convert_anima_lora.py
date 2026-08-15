from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LOKR_RANK = 32

COMFY_WEIGHT_RE = re.compile(
    r"^lora_unet_blocks_(?P<block>\d+)_(?P<layer>.+)\.(?P<direction>lora_down|lora_up)\.weight$"
)
COMFY_ALPHA_RE = re.compile(r"^lora_unet_blocks_(?P<block>\d+)_(?P<layer>.+)\.alpha$")
LOKR_PART_RE = re.compile(
    r"^(?P<base>.+)\.(?P<part>lokr_w1|lokr_w2|lokr_w2_a|lokr_w2_b|dora_scale|alpha)$"
)


def default_output_path(input_path: str | Path) -> Path:
    input_path = Path(input_path)
    return input_path.with_name(f"{input_path.stem}_diffusers{input_path.suffix}")


def convert_base_name(name: str) -> str:
    if name.startswith("diffusion_model."):
        return name

    match = re.match(r"^lora_unet_blocks_(?P<block>\d+)_(?P<layer>.+)$", name)
    if not match:
        raise ValueError(f"Unsupported Anima adapter key base: {name}")

    return f"diffusion_model.blocks.{match.group('block')}.{match.group('layer')}"


def convert_comfyui_entry(key: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor] | None:
    weight_match = COMFY_WEIGHT_RE.match(key)
    if weight_match:
        suffix = "lora_A.weight" if weight_match.group("direction") == "lora_down" else "lora_B.weight"
        target = (
            f"diffusion_model.blocks.{weight_match.group('block')}."
            f"{weight_match.group('layer')}.{suffix}"
        )
        return target, tensor

    alpha_match = COMFY_ALPHA_RE.match(key)
    if alpha_match:
        target = (
            f"diffusion_model.blocks.{alpha_match.group('block')}."
            f"{alpha_match.group('layer')}.lora_alpha"
        )
        return target, tensor

    return None


def _prepare_matrix(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D matrix for LoKr conversion, got shape {tuple(tensor.shape)}")
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.to(torch.float32)
    return tensor


def convert_lokr_group(
    base_name: str, group: dict[str, torch.Tensor], lokr_rank: int = LOKR_RANK
) -> dict[str, torch.Tensor]:
    if "lokr_w1" not in group:
        raise ValueError(f"Missing lokr_w1 for {base_name}")

    if "lokr_w2" in group:
        w2 = group["lokr_w2"]
    elif "lokr_w2_a" in group and "lokr_w2_b" in group:
        w2 = _prepare_matrix(group["lokr_w2_a"]) @ _prepare_matrix(group["lokr_w2_b"])
    else:
        raise ValueError(f"Missing lokr_w2 or lokr_w2_a/lokr_w2_b for {base_name}")

    w1 = _prepare_matrix(group["lokr_w1"])
    w2 = _prepare_matrix(w2)
    full_matrix = torch.kron(w1, w2)

    u, singular_values, vh = torch.linalg.svd(full_matrix, full_matrices=False)
    rank = min(lokr_rank, singular_values.shape[0], full_matrix.shape[0], full_matrix.shape[1])
    sqrt_s = singular_values[:rank].sqrt()

    lora_b = (u[:, :rank] * sqrt_s.unsqueeze(0)).to(group["lokr_w1"].dtype)
    lora_a = (sqrt_s.unsqueeze(1) * vh[:rank, :]).to(group["lokr_w1"].dtype)

    alpha = group.get("alpha")
    if alpha is None:
        alpha = torch.tensor(float(rank), dtype=group["lokr_w1"].dtype)

    target_base = convert_base_name(base_name)
    return {
        f"{target_base}.lora_A.weight": lora_a,
        f"{target_base}.lora_B.weight": lora_b,
        f"{target_base}.lora_alpha": alpha,
    }


def convert_state_dict(
    state_dict: dict[str, torch.Tensor], lokr_rank: int = LOKR_RANK
) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    lokr_groups: dict[str, dict[str, torch.Tensor]] = {}

    for key, tensor in state_dict.items():
        match = LOKR_PART_RE.match(key)
        if match:
            lokr_groups.setdefault(match.group("base"), {})[match.group("part")] = tensor

    for base_name, group in lokr_groups.items():
        if not any(part.startswith("lokr_") for part in group):
            continue
        converted.update(convert_lokr_group(base_name, group, lokr_rank=lokr_rank))
        consumed.update(f"{base_name}.{part}" for part in group)

    unsupported: list[str] = []
    for key, tensor in state_dict.items():
        if key in consumed:
            continue
        if key.startswith("diffusion_model."):
            converted[key] = tensor
            continue

        converted_entry = convert_comfyui_entry(key, tensor)
        if converted_entry is None:
            unsupported.append(key)
            continue

        target_key, target_tensor = converted_entry
        converted[target_key] = target_tensor

    if unsupported:
        raise ValueError(f"Unsupported adapter keys: {', '.join(sorted(unsupported))}")

    return converted


def convert_file(input_path: str | Path, output_path: str | Path | None = None) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path is not None else default_output_path(input_path)

    state_dict = load_file(str(input_path))
    converted = convert_state_dict(state_dict)
    save_file(converted, str(output_path))
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert CivitAI Anima LoRA safetensors into diffusers-compatible format."
    )
    parser.add_argument("input_path", help="Input safetensors path")
    parser.add_argument("output_path", nargs="?", help="Output safetensors path")
    args = parser.parse_args()

    output_path = convert_file(args.input_path, args.output_path)
    print(f"Saved converted LoRA to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
