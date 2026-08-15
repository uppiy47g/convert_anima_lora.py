# convert_anima_lora.py

**Convert CivitAI Anima LoRA for use with AmuseAI (diffusers)**

CivitAI で配布されている Anima 向け LoRA を AmuseAI (diffusers) で使用可能な形式に変換するツールです。

---

## 背景

CivitAI で配布されている Anima 向け LoRA の多くは **ComfyUI 形式** または **LyCORIS (LoKr + DoRA) 形式** で保存されています。  
AmuseAI は内部で HuggingFace diffusers を使用しているため、これらのフォーマットをそのまま読み込むことができず、以下のエラーが発生します。

```
Adapter name(s) {'xxx'} not in the list of present adapters: set()
Invalid LoRA checkpoint. Make sure all LoRA param names contain 'lora' substring.
```

本ツールはこの問題を解決します。

---

## 対応フォーマット

| 入力形式 | 判定方法 | 変換方法 |
|---|---|---|
| ComfyUI 標準LoRA | `lora_unet_*` + `lora_down/up` | キー名変換 |
| LyCORIS (LoKr+DoRA) | `lora_unet_*` + `lokr_w1/w2(_a/_b)` | Kronecker積展開 → SVD → LoRA再構成 |
| diffusers形式（変換不要） | `diffusion_model.*` | そのままコピー |

---

## インストール

```bash
pip install safetensors torch numpy
```

---

## 使い方

```bash
python convert_anima_lora.py <input.safetensors> [output.safetensors]
```

出力ファイル名を省略した場合、`<元のファイル名>_diffusers.safetensors` として保存されます。

### 例

```bash
python convert_anima_lora.py mikami_chizu_anima.safetensors
python convert_anima_lora.py "@inkandwash-000014.safetensors"
python convert_anima_lora.py input.safetensors output_diffusers.safetensors
```

---

## 変換ルール詳細

### ComfyUI 形式 → diffusers 形式

| ComfyUI 形式 | diffusers 形式 |
|---|---|
| `lora_unet_blocks_{N}_{layer}.lora_down.weight` | `diffusion_model.blocks.{N}.{layer}.lora_A.weight` |
| `lora_unet_blocks_{N}_{layer}.lora_up.weight` | `diffusion_model.blocks.{N}.{layer}.lora_B.weight` |
| `lora_unet_blocks_{N}_{layer}.alpha` | `diffusion_model.blocks.{N}.{layer}.lora_alpha` |

`alpha` を `lora_alpha` に改名するのは、diffusers が全テンソルキーに `lora` 文字列を要求するためです。

### LyCORIS LoKr → 標準LoRA 変換

LoKr はクロネッカー積を用いて重み行列の変化量を表現します。

```
ΔW = W1 ⊗ (W2_a × W2_b)
```

diffusers は LoKr をネイティブサポートしていないため、以下の手順で標準 LoRA に変換します。

```
Step 1: w2 = w2_a @ w2_b
Step 2: W  = kron(w1, w2)
Step 3: U, S, Vh = SVD(W)
Step 4: lora_B = U[:, :r] × √S
        lora_A = √S × Vh[:r, :]
```

既定の rank は `32` です。必要なら `convert_anima_lora.py` 内の `LOKR_RANK` を調整してください。

#### DoRA について

`dora_scale` はベースモデルの重みを用いた正規化が必要なため、LoRA ファイル単体では厳密な適用ができません。  
本ツールでは `dora_scale` を無視します。実用上は AmuseAI のウェイトスライダーで強度を調整してください。

---

## 技術的背景

### なぜ ComfyUI 形式の Anima LoRA が `lora_unet_*` というキー名を持つのか

ComfyUI はモデルアーキテクチャに依らず `lora_unet_` プレフィックスを使用する命名規則を採用しています。  
Anima は U-Net ではなく DiT (Diffusion Transformer) アーキテクチャですが、ComfyUI で訓練された場合でも `lora_unet_blocks_*` というキー名になります。

### LoKr → 標準LoRA変換の品質について

SVD による低ランク近似のため、元の LoKr の表現能力を完全に保持するわけではありません。  
rank を大きくするほど元の品質に近づきますが、ファイルサイズも増加します。
