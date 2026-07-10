#!/usr/bin/env python3
"""
ComfyUI形式のAnima LoRA → diffusers形式に変換するスクリプト

対応フォーマット:
  [標準LoRA]  lora_down.weight / lora_up.weight  → lora_A / lora_B
  [LoKr+DoRA] lokr_w1 / lokr_w2_a / lokr_w2_b   → Kronecker積展開 → SVD → lora_A / lora_B

キー変換規則:
  lora_unet_blocks_{N}_{sublayer}.* → diffusion_model.blocks.{N}.{sublayer}.*
"""
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LOKR_RANK = 32  # LoKr → LoRA 変換時のランク

KNOWN_SUFFIXES = {
    "lora_down.weight": "lora_down",
    "lora_up.weight":   "lora_up",
    "alpha":            "alpha",
    "lokr_w1":          "lokr_w1",
    "lokr_w2_a":        "lokr_w2_a",
    "lokr_w2_b":        "lokr_w2_b",
    "dora_scale":       "dora_scale",
}

SUBLAYER_MAP = {
    "cross_attn_k_proj":              "cross_attn.k_proj",
    "cross_attn_output_proj":         "cross_attn.output_proj",
    "cross_attn_q_proj":              "cross_attn.q_proj",
    "cross_attn_v_proj":              "cross_attn.v_proj",
    "mlp_layer1":                     "mlp.layer1",
    "mlp_layer2":                     "mlp.layer2",
    "self_attn_k_proj":               "self_attn.k_proj",
    "self_attn_output_proj":          "self_attn.output_proj",
    "self_attn_q_proj":               "self_attn.q_proj",
    "self_attn_v_proj":               "self_attn.v_proj",
    "adaln_modulation_cross_attn_1":  "adaln_modulation_cross_attn.1",
    "adaln_modulation_cross_attn_2":  "adaln_modulation_cross_attn.2",
    "adaln_modulation_mlp_1":         "adaln_modulation_mlp.1",
    "adaln_modulation_mlp_2":         "adaln_modulation_mlp.2",
    "adaln_modulation_self_attn_1":   "adaln_modulation_self_attn.1",
    "adaln_modulation_self_attn_2":   "adaln_modulation_self_attn.2",
}


def parse_base_key(comfy_base: str):
    """lora_unet_blocks_{N}_{sublayer} を (block_num, anima_sublayer) に分解。失敗時は None。"""
    if not comfy_base.startswith("lora_unet_blocks_"):
        return None
    rest = comfy_base[len("lora_unet_blocks_"):]
    idx = rest.find("_")
    if idx == -1:
        return None
    block_num = rest[:idx]
    sublayer_str = rest[idx + 1:]
    anima_sublayer = SUBLAYER_MAP.get(sublayer_str)
    if anima_sublayer is None:
        return None
    return block_num, anima_sublayer


def lokr_to_lora(w1: torch.Tensor, w2_a: torch.Tensor, w2_b: torch.Tensor, rank: int):
    """
    LoKr → 標準LoRA 変換
      W_delta = kron(w1, w2_a @ w2_b)
      W_delta ≈ lora_B @ lora_A  (rank-r SVD近似)
    戻り値: (lora_A, lora_B)  ※ lora_A: (r, in), lora_B: (out, r)
    """
    w2 = w2_a.float() @ w2_b.float()
    W = torch.kron(w1.float(), w2)          # (out, in)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    r = min(rank, S.shape[0])
    scale = S[:r].sqrt()
    lora_B = (U[:, :r] * scale).to(w1.dtype)   # (out, r)
    lora_A = (Vh[:r, :] * scale.unsqueeze(1)).to(w1.dtype)  # (r, in)
    return lora_A, lora_B


def convert_lora(input_path: str, output_path: str):
    print(f"入力: {input_path}")
    state = load_file(input_path)

    # diffusers形式か判定（キーが diffusion_model. で始まる場合はすでに変換済み）
    if any(k.startswith("diffusion_model.") for k in state.keys()):
        print("このLoRAはすでにdiffusers形式です。そのままコピーします。")
        save_file(dict(state), output_path)
        print(f"保存完了: {output_path}")
        return

    # キーをベースキーでグループ化（既知サフィックスで正確に分割）
    groups = defaultdict(dict)
    ungrouped = []
    for key, tensor in state.items():
        matched = False
        for full_suffix, short_name in KNOWN_SUFFIXES.items():
            if key.endswith("." + full_suffix):
                base = key[: -(len(full_suffix) + 1)]
                groups[base][short_name] = tensor
                matched = True
                break
        if not matched:
            ungrouped.append(key)

    converted = {}
    skipped_bases = []

    for base, tensors in groups.items():
        parsed = parse_base_key(base)
        if parsed is None:
            skipped_bases.append(base)
            continue
        block_num, anima_sublayer = parsed
        prefix = f"diffusion_model.blocks.{block_num}.{anima_sublayer}"

        # --- LoKr + DoRA 形式 ---
        if "lokr_w1" in tensors:
            w1   = tensors["lokr_w1"]
            w2_a = tensors.get("lokr_w2_a")
            w2_b = tensors.get("lokr_w2_b")
            if w2_a is None or w2_b is None:
                skipped_bases.append(base)
                continue
            lora_A, lora_B = lokr_to_lora(w1, w2_a, w2_b, LOKR_RANK)
            converted[f"{prefix}.lora_A.weight"] = lora_A
            converted[f"{prefix}.lora_B.weight"] = lora_B
            converted[f"{prefix}.lora_alpha"] = torch.tensor(float(LOKR_RANK))
            # dora_scale は近似として無視（ベースモデル重みが無いため厳密な適用不可）

        # --- 標準LoRA 形式 ---
        elif "lora_down" in tensors and "lora_up" in tensors:
            converted[f"{prefix}.lora_A.weight"] = tensors["lora_down"]
            converted[f"{prefix}.lora_B.weight"] = tensors["lora_up"]
            if "alpha" in tensors:
                converted[f"{prefix}.lora_alpha"] = tensors["alpha"]

        else:
            skipped_bases.append(base)

    total_in  = len(groups)
    total_out = len([k for k in converted if k.endswith(".lora_A.weight")])
    if ungrouped:
        print(f"未認識キー ({len(ungrouped)}): {ungrouped[:3]} ...")
    print(f"変換成功: {total_out} レイヤー / {total_in} レイヤー")
    if skipped_bases:
        print(f"スキップ ({len(skipped_bases)} レイヤー): {skipped_bases[:3]} ...")

    if not converted:
        print("ERROR: 変換できたキーが0件です。")
        sys.exit(1)

    save_file(converted, output_path)
    print(f"保存完了: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python convert_anima_lora.py <input.safetensors> [output.safetensors]")
        sys.exit(1)

    inp = sys.argv[1]
    stem = Path(inp).stem
    out = sys.argv[2] if len(sys.argv) >= 3 else f"{stem}_diffusers.safetensors"
    convert_lora(inp, out)
