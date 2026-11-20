# 更新紀錄

新的在最上面。

## 2026-11-20 — Strata 在 Intel Arc 上

- [Strata](engines/strata.md)：安裝檔帶著套件並在第一次載入時安裝；Intel Arc（Windows 與 Linux）也有套件，但沒有 deep sleep；寫下支援的 GGUF 清單、第一次載入的流程與載入時的環境變數預設。
- 新增實測 [Strata 在 Intel Arc 上](benchmarks/strata-intel-arc.md)。
- wiki 的其餘部分仍是依後端 0.3.0 寫的。

## 2026-10-29 — 重建

整個 wiki 依後端 0.3.0 的程式碼重寫。

- 新增：服務架構、四個推論引擎（含 Strata）、四種 offload、記憶體估算、GGUF 轉換、微調、API（對話、錯誤、遠端存取）、維運，以及六份附條件的實測結果。
- TRUSTA SSD 產品的問答整理成一頁 [TRUSTA 企業級 SSD](products/trusta-ssd.md)。
- 移除：停在 2026-04 至 07 的舊頁（舊版架構、引擎比較、offload 說明、手冊的原始匯入、LLM Wiki 方法論、競品比較）。安裝與使用說明改由 [`docs/`](../../docs/installation.md) 負責。
