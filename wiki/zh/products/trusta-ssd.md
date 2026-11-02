---
title: TRUSTA 企業級 SSD
summary: ADATA TRUSTA 企業級 SSD 的產品線、介面與規格、適用場景，以及它在 AI Scaler Toolkit 裡扮演的角色。
updatedAt: 2026-10-29
---

# TRUSTA 企業級 SSD

## 在 AST 裡的角色

AST 把 NVMe SSD 當成**記憶體階層的一部分**，不只是放檔案的地方：

| 用途 | 說明 | 詳見 |
|---|---|---|
| KV cache 的 NVMe 層 | 長對話與共用文件的 KV 存在 SSD，重訪時不用重算 | [vLLM KV offload 實測](../benchmarks/vllm-kv-offload.md) |
| 訓練的 NVMe offload | DeepSpeed ZeRO-3 把參數或 optimizer state 放到 SSD | [DeepSpeed ZeRO-3 offload](../training/deepspeed-offload.md) |
| 模型權重的讀取 | MoE 推論依 routing 分散讀入專家；切換時的 slot 存檔 | [Gen5 NVMe 實測](../benchmarks/gen5-ssd.md) |

本 wiki 的實測使用 **T7P5**（PCIe Gen5）。

## 產品線

| 型號 | 介面 | 外型 |
|---|---|---|
| **T7P5** | PCIe Gen5 NVMe | U.2、E1.S、E3.S |
| **TD7P40** | PCIe Gen4 NVMe | U.2 |
| **TB3P40** | PCIe Gen4 NVMe | M.2 |
| **TD5S30** | SATA III | 2.5 吋 |
| **TB3S30** | SATA III | M.2 |

T7P5 是旗艦系列，針對 AI 與資料中心的高吞吐、低延遲工作負載。

## 功能

- **NVMe**（PCIe Gen4／Gen5 型號）：較低的延遲、較高的 IOPS 與平行處理能力
- 選配 **FDP**（Flexible Data Placement）與 **SR-IOV**（Single Root I/O Virtualization）
- 可靠性：企業級元件、錯誤更正、斷電保護、溫度管理、wear-leveling

## 適用場景

| 場景 | 用途 |
|---|---|
| AI 與資料科學 | AI 訓練與推論、向量資料庫、HPC |
| 商用伺服器與虛擬化 | VM、容器、VDI、網站平台 |
| 物件儲存 | AI、HPC、資料庫、備份、雲端 |
| 資料庫與即時交易 | OLTP、OLAP、金融交易 |
| 資料中心開機碟 | 伺服器快速開機與穩定的 OS 運作 |

## 相關

- [全貌](../overview.md)
- [實測結果](../benchmarks/index.md)
