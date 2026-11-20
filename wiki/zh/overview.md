---
title: 全貌
summary: AI Scaler Toolkit 是在一台機器上執行與微調 LLM 的本地平台；本頁說明它的元件、硬體適配的三層手段與最重要的幾個限制。
updatedAt: 2026-10-29
---

# 全貌

## 一句話

**在自己的電腦上跑 LLM 的平台**：下載模型、對話、量化轉換、微調，全部在本機的硬體上完成，
資料不離開這台電腦。

**產品的核心是「在不夠大的顯卡上也跑得動」。** 這個目標決定了大部分設計：
四個推論引擎、多層 offload、GGUF 量化、DeepSpeed ZeRO-3。
NVMe SSD 在這裡被當成**記憶體階層的一部分**，不只是放檔案的地方。

## 元件

| 元件 | 是什麼 | 在 repo 的哪裡 |
|---|---|---|
| **後端** | FastAPI 服務：54 條路由、四個推論引擎、微調、RAG | `backend/` |
| **網頁介面** | 預先建置好的單頁應用，由後端直接提供 | `frontend/` |
| **console** | 無介面的 Python 用戶端，打的是同一組 API | `console/` |
| **桌面程式** | 啟動器、Windows 安裝檔、Linux AppImage | `deploy/` |

```
┌─ 網頁介面 ─┐   ┌─ console ─┐   ┌─ 其他應用程式（Hermes、程式助理…）─┐
└─────┬──────┘   └─────┬─────┘   └─────────────────┬──────────────────┘
      │ HTTP / SSE      │ HTTP                      │ OpenAI 相容 /v1/
      └─────────────────┴──────────────┬────────────┘
                                       ▼
                         ┌─ 後端（FastAPI）─┐
                         │  主行程 + worker  │
                         └────────┬─────────┘
                                  │ 管理子行程
              ┌───────────────────┼────────────────────┐
              ▼                   ▼                    ▼
        llama-server          vllm serve            Strata
          （GGUF）          （HF safetensors）      （GGUF，MoE）
```

`transformers` 引擎不在圖上——它**在後端自己的 worker 行程裡**執行，不是獨立的伺服器。
這個差異是 [服務架構](architecture.md) 的主題。

## 硬體適配的三層手段

| 層 | 手段 | 詳見 |
|---|---|---|
| **權重** | 量化（int4／int8／GGUF）、CPU offload、MoE 專家 offload | [Offload](memory/offload.md) |
| **KV cache** | vLLM 的 DRAM 層 → NVMe 層；llama.cpp 的 slot 快取 | [Offload](memory/offload.md)、[vLLM](engines/vllm.md) |
| **訓練** | DeepSpeed ZeRO-3 + CPU／NVMe offload | [DeepSpeed ZeRO-3 offload](training/deepspeed-offload.md) |

## 最容易被誤解的限制：一次一個

| 資源 | 同時幾個 | 超過時 |
|---|---|---|
| 推論模型 | **1** | `409`，要先卸載 |
| 訓練 job | **1** | 拒絕開始 |
| 生成請求 | 8（`MAX_CONCURRENT_GENERATIONS` 可調） | 排隊 |
| **訓練與推論同時用一張 GPU** | **不會擋** | 只在伺服器 log 留一行警告；16 GB 的卡上通常直接 OOM |

最後一列要特別注意：開始訓練前**先卸載推論模型**。

## 各平台能用的功能

| 功能 | Windows | Linux |
|---|---|---|
| transformers、llama.cpp | ✅ | ✅ |
| vLLM | ❌ | ✅（NVIDIA 預設安裝；Intel 要 `TRUSTA_SETUP_VLLM=1`） |
| Strata | NVIDIA 與 Intel Arc 套件（Intel 沒有 deep sleep） | NVIDIA 與 Intel Arc 套件（Intel 沒有 deep sleep） |
| 微調 | ❌ | ✅（WSL2 也算 Linux） |

後端用 `GET /health` 回報這台機器實際能用的引擎與微調；網頁介面依此決定要顯示哪些選項。
不支援的功能被呼叫時回 `501`，`detail` 是一句說明原因的話。

## 從這裡往哪裡讀

| 你想知道 | 去 |
|---|---|
| 後端的行程與狀態 | [服務架構](architecture.md) |
| 該用哪個推論引擎 | [推論引擎怎麼選](engines/index.md) |
| 在 API 上開發 | [API 總覽](api/index.md) |
| 讓別台電腦連進來 | [遠端存取與信任邊界](api/remote-access.md) |
| 術語 | [術語表](glossary.md) |
