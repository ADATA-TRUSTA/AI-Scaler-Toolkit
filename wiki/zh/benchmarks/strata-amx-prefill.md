---
title: Strata 用 CPU 的 AMX 讀 prompt
summary: 讓 CPU 在 prefill 時分擔一部分專家的運算，4K prompt 讀取快約 90～110%（16 GB 卡 792 → 1,507 tok/s，48 GB 卡約 1,000 → 2,000）；32K 沒有增益。引擎預設值下，超過約 3K tokens 時 AMX 完全沒有作用。
updatedAt: 2026-11-20
measuredAt: 2026-11-18 to 2026-11-19
---

# Strata 用 CPU 的 AMX 讀 prompt

Strata 把大型 MoE 的專家放在 DRAM。讀長 prompt（prefill）會用到每一個專家，不在 GPU 上的專家要嘛經 PCIe 搬過去，
要嘛由 CPU 算。有 AMX（Intel 的矩陣運算單元）的 CPU 可以分擔一部分。這頁量這件事值多少。

## 條件

| 項目 | 機器 A | 機器 B |
|---|---|---|
| GPU | 16 GB VRAM，PCIe Gen5 x8（實測 host→device 27.8 GB/s） | 48 GB VRAM，PCIe Gen4 x16（實測 25.0 GB/s） |
| CPU | 伺服器等級 CPU，AMX | 20 核工作站等級 CPU，AMX（`amx_int8`） |
| RAM | 251 GB | 503 GiB |
| 作業系統 | Ubuntu 24.04 | Ubuntu 24.04 |
| 引擎 | Strata 0.1.57（v0.2.11 加 strata-sleep 的 patch），為該張卡在本機編的 | 同左 |
| 模型 | Qwen3.8-Flash-Next **Q8_0**（176 GB，6 個分片） | 同左 |
| 設定 | `--expert-cache auto --prefill auto --spec 4 --mtp`、128K context、int8 KV | `--spec 4 --spec-min-p 0.5 --mtp`、256K context，除非另外說明否則不開 vision |

一次一個人，`max_tokens` 1（所以時間就是讀 prompt 的時間），每個請求第一行都不同以避開前綴重用，先暖機一次，再 AMX 關、開交替。
速度是引擎自己印的 `read in … tok/s`。

## 機器 A：預設的上限讓 AMX 沒有作用

| CPU share 設定 | prompt | AMX 關 | AMX 開 |
|---|---|---|---|
| 預設（上限 3,072 tokens） | 約 4K | 786～807 tok/s | 795～807 tok/s：**AMX 沒有跑** |
| 上限調到 16,000 | 約 4K | 788～797（平均 792） | 1,404～1,587（平均 1,507）：**+91%** |
| 同上 | 約 8K | 1,474～1,494（平均 1,485） | 1,761～1,879（平均 1,820）：**+23%** |

CPU 的分擔只套用在小於某個上限的 chunk，預設 3,072 tokens；4K 的 prompt 超過它，走全部串流的路徑，CPU 不參與。
這個上限是引擎的 `STRATA_PREFILL_CPU_SHARE_MAX`，要搭配 `STRATA_PREFILL_CPU_SHARE=auto` 使用。

## 機器 B：看有多少專家在 RAM 裡有副本

讀取速度，tok/s。「budget」是 `--resident-budget-gib`：引擎替專家在 RAM 裡保留多少副本。

| 設定 | 4K AMX 關 | 4K AMX 開 | 32K AMX 關 | 32K AMX 開 |
|---|---|---|---|---|
| budget 80 GiB（部分專家要從模型檔讀） | 810～871 | 1,182～1,322 | 2,536 | 2,638 |
| **budget 125 GiB**（96 GiB 的 RAM 副本，不從檔案讀） | **985～1,013** | **1,815～2,156** | **3,642** | 3,616 |
| budget 80，prefill chunk 8,192（原 32,768） | 858～873 | 1,331～1,350 | 1,717 | 2,070 |
| budget 125，經 AST、開 vision | 1,006～1,031 | 1,862～2,205 | 3,595 | 3,519 |

- **4K 的上限是 PCIe。** AMX 關時，等專家搬過去佔 GPU 時間軸的 58～66%。AMX 開時降到 0.2～0.5 s，CPU 分走約 64～83% 的專家。
- **32K 不是。** 等搬運只佔 12～15%，GPU 自己的運算是大頭，所以 AMX 在 budget 小時只多約 4%，budget 大時沒有差別。
- **32K 時，把專家都留在 RAM 比 AMX 重要：** budget 125 讓 32K 的讀取在 AMX 關時快 44%，4K 快 17%；代價是多用約 16 GiB RAM。
- **prefill chunk 調小更差：** 每個 chunk 都要重新串流一輪專家，32K 從 2,536 掉到 1,717 tok/s。
- **CPU worker 數影響小：** 11 個 worker 的 AMX 約為 18 個（預設）的 97%，5 個約 90%。在 4K、budget 80 下量的。
- AMX 關、budget 80 時，48 GB 的卡在 4K 只比 16 GB 的卡快一點（810～871 對 786～807）：多出來的 VRAM 對 decode 與快取命中的幫助比對 prefill 大。

## 限制

1. 只有單人、一個模型、一種精度（Q8_0）。多人同時讀的數字在 [Strata 與 vLLM 在一張 48 GB 卡上](strata-vs-vllm.md)。
2. 引擎是在測試機上為各自的 GPU 架構編的，不是發佈的套件。
3. 沒有檢查輸出內容是否正確：只要求第一個 token。AMX 有跑是由速度差與上限設定推得，沒有用 CPU 的效能計數器確認。
4. 16K 以上的 prompt 只有機器 B 量過；機器 A 的 8K 只有一組比較。
5. 怎麼經 AST 用請求或設定打開 AMX，這頁沒有涵蓋。

## 相關

- [Strata](../engines/strata.md)
- [Strata 與 vLLM 在一張 48 GB 卡上](strata-vs-vllm.md)
