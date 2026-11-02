---
title: 多模態微調
summary: 圖片微調的 multimodal_scope、必須遵守的載入順序，以及 Qwen3.5、gemma-4 等模型特有的四個坑。
updatedAt: 2026-10-29
---

# 多模態微調

流程全貌見 [微調總覽](index.md)；本頁只講圖片訓練特有的部分。
實測過的模型家族包括 SmolVLM、Gemma 4 與 Qwen3.5。

## `multimodal_scope`：允許訓練改動多少

| 值 | 視覺 encoder | bridge（projector） | 語言模型 |
|---|---|---|---|
| `text_only` | 凍結 | 凍結 | 訓練 |
| `text_and_bridge` | 凍結 | **訓練** | 訓練 |

`text_only` 讓圖片的解析完全保持預訓練的樣子，只調整「對這些特徵的解讀」。
`text_and_bridge` 適合圖片領域離預訓練分布很遠的情況（工業影像、文件掃描）。

> **先用 `text_only` 試。** 在一個人臉身分辨識的實測裡，encoder 本身就帶有身分訊號；
> 加入負例之後，held-out 的照片 3/3 答對（包含沒看過的照片），
> 而 `text_and_bridge` 在那個案例幫不上忙——bridge 已經滿秩。
> 「領域很遠就開 bridge」是假設，不是規律。

## 用 LoRA

~5B 的多模態模型做 full fine-tune，在 16 GB 顯卡上 **backward 時 OOM**。
DRAM 夠但 VRAM 不夠，**CPU offload 救不了**。多模態請用 LoRA。

## 載入順序

圖片訓練模式必須在載入模型**之前**開啟，否則會用 `AutoModelForCausalLM` 載入。
Qwen3.5、Gemma 3/4 這類架構**同時註冊了純文字的對應**，所以載入不會失敗——
只是得到一個**沒有 vision tower** 的模型。後端的共用流程保證這個順序。

## 四個模型特有的坑（後端都已處理）

### 1. Qwen3.5 在 ZeRO-3 下的權重 rename

某些多模態架構（Qwen3.5）註冊了一個給純文字子模型用的 checkpoint rename，
而 transformers 在 ZeRO-3 下會無條件套用到整個多模態模型上，**幾乎所有語言權重都綁不上**。
後端偵測到時會補一個 key mapping；上游修好的版本上這個 workaround 不會觸發。

### 2. TRL 的 `mm_token_type_ids` 長度不對

用 prompt／completion 格式的資料時，TRL 只從 prompt 算 `mm_token_type_ids`，
completion 的 token 補進了 `input_ids` 卻沒補進它，於是 Qwen3.5-VL 的 rope index 計算出錯。
後端從完整的 `input_ids` 重算。

### 3. ZeRO-3 沒有還原部分 buffer（gemma-4）

transformers 的 ZeRO-3 loader 會跳過「沒有直屬參數的 module」的 buffer。
gemma-4 的 42 個 `layer_scalar` 因此整場訓練都停在 1.0（真值 0.06–0.89），
**訓練出來的 adapter 無法重現訓練時的狀態**。後端已修；比對不到任何 key 時會發警告，而不是當作沒事。

### 4. mmproj 要從 merge 後的模型產生

轉 GGUF 時，視覺投影檔要從 merge 過的模型出，不是從 base。

## `max_image_pixels`

圖片進 processor 之前會縮到這個大小以內，直接影響每張圖的 token 數與 VRAM 峰值。

> 推論側的對應設定不同：gemma-4 轉 GGUF 後用 llama-server 上線時要加 `--image-min-tokens 280`。

## 相關

- [微調總覽](index.md)
- [DeepSpeed ZeRO-3 offload](deepspeed-offload.md)
- [GGUF 轉換](../models/gguf-conversion.md)
