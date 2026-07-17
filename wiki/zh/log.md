# Wiki 操作日誌（繁體中文）

> 知識庫操作的時序記錄。追加模式。
> 格式：`## [YYYY-MM-DD] action | subject`

## [2026-04-29] ingest | Trusta AST 推理服務
- 匯入 `inference_manual.html`。
- 建立來源頁 `inference_manual.md`。
- 建立實體頁：`trusta-ast-inference-service`、`model-manager`、`worker-process`。
- 建立概念頁：`multi-process-isolation`、`qwen3-5-inference-optimization`、`inference-engine-comparison`、`openai-api-compatibility`。
- 建立比較頁 `legacy-vs-openai-api`。

## [2026-05-28] ingest | LLM Wiki 方法論 + TRUSTA SSD 產品線
- 匯入 LLM Wiki 格式轉換研究，以及 TRUSTA 企業級 SSD Q&A 資料集。
- 建立 `concepts/llm-wiki-pattern.md`、`concepts/file-format-conversion.md`、`research-evidence.md`。
- 建立 `synthesis/*` TRUSTA SSD 產品線頁面。

## [2026-07-16] update | 內容修正 + 中英雙語重構
- 移除捏造的成本／效能數字；競品對比數字移除（本專案未實測）。
- 將不存在的 `OffloadConfig` 範例改為真實機制（device_map / offload_folder / n_gpu_layers / DeepSpeed ZeRO-3）。
- 修正「自動重啟／自動恢復」（實為手動復原）與「同時多模型／負載平衡」（單一模型，載入第二個回 HTTP 409）。
- 修正 `F8` → `FP8`、`ModelManager` 方法簽章、過時的 `.html` 來源引用。
- 攤平巢狀的 `wiki/` 層，並將整個 wiki 切分為 `en/` 與 `zh/`。
