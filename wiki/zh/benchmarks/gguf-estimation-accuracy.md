---
title: GGUF 記憶體估算的準確度
summary: 對照 llama-fit-params 在 6 種架構上：權重與 KV 誤差都在 1 MiB 內且從不低估，compute buffer 一律高估 0–26%；recommend 給的參數實際用掉預算的 96.6–97.8%。
updatedAt: 2026-10-29
measuredAt: 2026-09-01
---

# GGUF 記憶體估算的準確度

估算器的設計見 [記憶體估算](../memory/estimation.md)。這裡只放核對結果。

## 對照 `llama-fit-params`

`llama-fit-params` 是 llama.cpp 自己的 dry-run 載入，所以是精確的參考值。
涵蓋 6 種架構：qwen2、qwen3、qwen35、qwen35moe、gemma4、gpt-oss（dense／MoE、tied／untied embedding、SWA、hybrid linear attention）。

| 項目 | 誤差 |
|---|---|
| GPU／host 權重 | ≤ 1 MiB（只有四捨五入） |
| GPU／host KV cache 與 recurrent state | ≤ 1 MiB |
| GPU compute buffer | **+0% ～ +26%，一律高估** |

權重與 KV **從未低估**。compute buffer 是唯一的近似項；要精確就加 `verify: true`。

## `recommend` 填滿預算的程度

Qwen2.5-0.5B，`n_ctx_max=262144`：

| 上限 | 建議的參數 | llama.cpp 實際 | 使用率 |
|---|---|---|---|
| 1000 MiB | `-ngl 25 -c 8960 -fa on` | 924 MiB | 97.3% |
| 1500 MiB | `-ngl 25 -c 47360 -fa on` | 1353 MiB | 96.6% |
| 3000 MiB | `-ngl 25 -c 158976 -fa on` | 2835 MiB | 97.8% |

## 量出來、容易誤解的行為

| 行為 | 實測 |
|---|---|
| 不開 flash attention 時 compute buffer 暴增 | 32B、32K ctx：344 → 5,256 MiB |
| `-ncmoe` 不增加 compute buffer，降 `-ngl` 會 | `-ngl 999` 與 `-ngl 999 -ncmoe 2` 都是 14 MiB；`-ngl 2` 變 18 MiB |
| tied embedding 多一份 `token_embd` | gemma-4-26B-A4B Q4_K_M：檔案 16,147 MiB，`-ngl 0` 時 host 16,895 MiB；`-ngl 1` 時 GPU 剛好多 748 MiB |
| hybrid linear attention 只有部分層有 KV | Qwen3.5-9B、c=4096：KV 178 MiB（8 層 attention 128 + 24 層 linear 50），不是 512 MiB |
| `-ngl 1` 不是「第一層」 | 只放 output 層 |

## 限制

估算的規則是照 llama.cpp 的原始碼寫的；llama.cpp 改了放置或 compute buffer 的計算，就需要重新核對。

## 相關

- [記憶體估算](../memory/estimation.md)
- [llama.cpp 引擎](../engines/llama-server.md)
