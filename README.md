# convert_anima_lora

**Convert CivitAI Anima LoRA for use with AmuseAI (diffusers)**

CivitAIで配布されているAnima向けLoRAをAmuseAI（diffusers）で使用可能な形式に変換するツールです。

---

## 背景

CivitAIで配布されているAnima向けLoRAの多くは **ComfyUI形式** または **LyCORIS (LoKr+DoRA) 形式** で保存されています。
AmuseAIは内部でHuggingFace diffusersを使用しているため、これらのフォーマットをそのまま読み込むことができず、以下のエラーが発生します。

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
| LyCORIS (LoKr+DoRA) | `lora_unet_*` + `lokr_w1/w2_a/w2_b` | Kronecker積展開 → SVD → LoRA再構成 |
| diffusers形式（変換不要） | `diffusion_model.*` | そのままコピー |

---

## インストール

```bash
pip install safetensors torch
```

---

## 使い方

```bash
python convert_anima_lora.py <input.safetensors> [output.safetensors]
```

出力ファイル名を省略した場合、`<元のファイル名>_diffusers.safetensors` として保存されます。

### 例

```bash
# ComfyUI 標準LoRA の変換
python convert_anima_lora.py mikami_chizu_anima.safetensors

# LoKr+DoRA LoRA の変換
python convert_anima_lora.py "@inkandwash-000014.safetensors"

# 出力ファイル名を指定
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

**注意:** `alpha` を `lora_alpha` に改名するのは、diffusersが全テンソルキーに `lora` 文字列を要求するためです。

### LyCORIS LoKr → 標準LoRA 変換

LoKrはクロネッカー積を用いて重み行列の変化量を表現します：

```
ΔW = W1 ⊗ (W2_a × W2_b)
```

diffusersはLoKrをネイティブサポートしていないため、以下の手順で標準LoRAに変換します：

```
Step 1: w2 = w2_a @ w2_b       # 右因子を復元
Step 2: W  = kron(w1, w2)       # Kronecker積でフル行列を再構成
Step 3: U, S, Vh = SVD(W)       # 特異値分解
Step 4: lora_B = U[:, :r] × √S  # (out, rank)
        lora_A = √S × Vh[:r, :] # (rank, in)
```

デフォルトのrank = 32（コード内の `LOKR_RANK` で変更可能）。

#### DoRAについて

DoRAの `dora_scale` はベースモデルの重みを用いた正規化が必要なため、LoRAファイル単体では厳密な適用が不可能です。本ツールでは `dora_scale` を無視します。
実用上はAmuseAIのウェイトスライダーで強度を調整してください。

---

## 動作確認済みLoRA

| LoRA | 形式 | 結果 |
|---|---|---|
| [Mikami Chizu (Taimanin RPGX)](https://civitai.com/models/2722200) | ComfyUI 標準LoRA | ✅ 動作確認済み |
| [Expressive Ink Anime Style](https://civitai.com/models/2742108) | LoKr + DoRA | ✅ 動作確認済み |
| [Artist Style muo](https://civitai.com/models/2741655) | diffusers形式 | ✅ 動作確認済み（パススルー） |

---

## 技術的背景

### なぜComfyUI形式のAnima LoRAが `lora_unet_*` というキー名を持つのか

ComfyUIはモデルアーキテクチャに依らず `lora_unet_` プレフィックスを使用する命名規則を採用しています。
AnimaはU-NetではなくDiT（Diffusion Transformer）アーキテクチャですが、ComfyUIで訓練された場合でも `lora_unet_blocks_*` というキー名になります。

### LoKr → 標準LoRA変換の品質について

SVDによる低ランク近似のため、元のLoKrの表現能力を完全に保持するわけではありません。
rankを大きくするほど元の品質に近づきますが、ファイルサイズも増加します。

```python
# LOKR_RANK を変更することでrank数を調整可能
LOKR_RANK = 32  # デフォルト値
```

---

## 環境

- Python 3.10+
- PyTorch 2.0+
- safetensors 0.4+
- AmuseAI v3.5.2 で動作確認

---

## License

MIT

---

## 謝辞

- [AmuseAI](https://github.com/amuse-ai) - AnimaモデルおよびAnimaPipeline
- [HuggingFace diffusers](https://github.com/huggingface/diffusers) - LoRAローダー実装
- [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) - LoKr/DoRAアルゴリズム
- [CivitAI](https://civitai.com) - LoRAモデルの配布
