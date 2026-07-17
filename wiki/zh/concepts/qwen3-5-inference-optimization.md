---
type: concept
title: Qwen3.5 推論最佳化
created: 2026-04-29
updated: 2026-04-29
tags: [qwen, optimization, model-specific, parameters]
related: [trusta-ast-inference-service, model-aware-parameter-adjustment]
sources: ["inference_manual.md"]
---
# Qwen3.5 推論最佳化

**Qwen3.5 推論最佳化** 是指 Trusta AST 推論服務中專為 **Qwen3.5** 模型系列所實作的一組硬編碼邏輯與參數覆寫。此最佳化之所以必要，是因為該模型對生成參數的敏感性及其 MoE（Mixture of Experts，專家混合）架構。

## 問題

以預設或非標準的生成參數執行 Qwen3.5，經常導致生成瑕疵，例如亂碼文字或多語言輸出錯誤。

## 解決方案：模型感知參數調整

當偵測到載入的模型名稱包含「qwen3.5」時，服務會自動強制套用特定的參數值，覆寫任何使用者提供的設定：

| 參數 | 強制值 | 原因 |
| :--- | :--- | :--- |
| `temperature` | `1.0` | 穩定輸出分佈。 |
| `top_p` | `0.95` | 確保最佳的權杖選擇。 |
| `top_k` | `20` | 限制詞彙取樣範圍。 |
| `repetition_penalty` | `1.0` | 避免重複問題。 |
| `enable_thinking` | `false` | 停用「思考」模式以避免瑕疵。 |

## 對使用者的影響

*   **透明性**：使用者無需手動為 Qwen3.5 調整這些參數；系統會自動處理。
*   **限制**：進階使用者無法透過 API 覆寫 Qwen3.5 模型的這些數值。這以犧牲細粒度控制為代價，確保了可靠性。
*   **適用性**：此邏輯專屬於 Qwen3.5 系列，除非加入類似的邏輯，否則不會自動套用至其他 MoE 模型。
