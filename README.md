# convert_anima_lora.py

CivitAI の Anima 用 LoRA を AmuseAI (diffusers) で使用できる形式へ変換するツールです。

## 対応フォーマット

- **ComfyUI 標準 LoRA**
  - `lora_unet_blocks_*` のキー名を `diffusion_model.blocks.*` へ変換します。
  - `alpha` は diffusers が全キーに `lora` を要求するため `lora_alpha` に改名します。
- **LyCORIS (LoKr + DoRA)**
  - `lokr_w1`, `lokr_w2_a`, `lokr_w2_b` からクロネッカー積を復元し、SVD で標準 LoRA (`lora_A`, `lora_B`) に変換します。
  - `dora_scale` はベースモデル依存のため無視します。
- **diffusers 形式**
  - `diffusion_model.` で始まるキーはそのままコピーします。

## 必要ライブラリ

```bash
pip install safetensors torch numpy
```

## 使い方

```bash
python convert_anima_lora.py <input.safetensors> [output.safetensors]
```

出力ファイル名を省略した場合は `<元ファイル名>_diffusers.safetensors` になります。

## LoKr 変換について

LoKr は次の手順で標準 LoRA に近似変換します。

1. `w2 = w2_a @ w2_b`
2. `W = kron(w1, w2)`
3. `U, S, Vh = svd(W)`
4. `lora_B = U[:, :r] * sqrt(S[:r])`
5. `lora_A = sqrt(S[:r]) * Vh[:r, :]`

既定の rank は `32` です。必要なら `convert_anima_lora.py` 内の `LOKR_RANK` を調整してください。
