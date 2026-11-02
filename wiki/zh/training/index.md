---
title: 微調總覽
summary: 微調的流程、兩個策略、兩條不能違反的順序、phase 生命週期，以及即時事件流（SSE）怎麼讀。
updatedAt: 2026-10-29
---

# 微調總覽

> 微調**只支援 Linux**（WSL2 也算）。其他平台上 `POST /training/start` 回 `501` 與一句原因。

## 一分鐘版本

```
POST /training/start
   │
   ▼
spawn 訓練 worker 行程
   │
   ├─ num_gpus == 1 → 在行程內訓練
   └─ num_gpus  > 1 → 起一個 deepspeed 子行程
          │
          ▼
   共用的準備序列：tokenizer → 資料集（含切 eval）→ 模型
          ▼
   trainer → 訓練 → 存檔 → merge LoRA → 轉 GGUF（Q4_K_M）
```

**一次只能跑一個 job。** 而**訓練與推論不互斥**——開始訓練前請先卸載推論模型，
否則在 16 GB 的卡上通常會 OOM。

## 方法與策略

| 方法 | 說明 |
|---|---|
| full | 全參數 |
| LoRA | 只訓練 adapter |
| QLoRA | base 量化 + LoRA |

| 策略（`use_sft_trainer`） | 用什麼 | 資料誰預處理 |
|---|---|---|
| `true`（預設） | TRL 的 `SFTTrainer` | TRL（含圖片的 collate） |
| `false` | HF `Trainer` | 後端自己，**一律用 tokenizer** |

## 主要欄位

| 組 | 欄位 |
|---|---|
| 基本 | `model_name`、`method`、`dataset_path`、`output_dir` |
| LoRA | `lora_r`、`lora_alpha`、`lora_dropout`、`lora_target_modules` |
| 多模態 | `multimodal_scope`、`max_image_pixels` |
| 超參數 | `num_train_epochs`、`per_device_train_batch_size`、`gradient_accumulation_steps`、`learning_rate`、`warmup_steps` |
| 存檔 | `save_steps`、`save_total_limit`、`save_tokenizer` |
| 資料欄位名 | `text_field`、`prompt_field`、`completion_field`、`messages_field` |
| 評估 | `eval_split_ratio`、`eval_steps`、`eval_split_seed` |
| DeepSpeed | `use_deepspeed`、`deepspeed_profile`、`deepspeed_config`、`offload_folder` |
| 其他 | `packing`、`gradient_checkpointing`、`use_flash_attention`、`num_gpus`、`max_seq_length` |

完整範例：`GET /examples/training`。

### `save_steps` 要設小

不是為了存檔頻率，而是**長時間訓練被中斷時的損失**。API **沒有 resume**，只能從最後一個 checkpoint 重來。
實測：同時驅動螢幕的顯卡上，ZeRO-3 訓練約 61 分鐘被驅動的 watchdog（`cudaErrorLaunchTimeout`）中止。
見 [跑了才發現的坑](../operations/known-pitfalls.md)。

### eval 在任何處理之前切出來

`eval_split_ratio` 有值時，資料集**一載入就先切**，評估看到的才是訓練從沒碰過的樣本。

## 兩條不能違反的順序

兩條都是「順序錯了不會報錯」的類型，所以單 GPU 與多 GPU 共用同一份準備序列。

| 順序 | 違反的後果 |
|---|---|
| **DeepSpeed 的設定必須在模型載入之前建立** | ZeRO-3 在載入期的分片**靜默放棄**；16 GB 卡上就是 OOM 與訓得起來的差別 |
| **圖片訓練模式必須在模型載入之前開啟** | 多模態 checkpoint 被當成純文字模型載入，**整個 vision tower 不存在** |

## 存 LoRA 時繞過 DeepSpeed

DeepSpeed + LoRA 的存檔兩種方式都會壞：`gather=True` 會 OOM，`gather=False` 會 `FileExists`。
所以偵測到「DeepSpeed 且模型是 PEFT」時，只存 adapter。

## 進度：phase 與 status

| phase（細） | status（粗） |
|---|---|
| `pending`、`queued` | `starting` |
| `resolving_deepspeed`、`launching_workers`、`loading_tokenizer`、`loading_dataset`、`loading_model`、`preparing_trainer` | `initializing` |
| `training`、`saving`、`merging_lora`、`converting_gguf` | `running` |
| `completed` | `completed` |
| `stopped`、`cancelled` | `stopped` |
| `error` | `error` |

終止 phase：`completed`、`stopped`、`error`、`cancelled`。`pending`、`queued`、`cancelled` 保留給尚未實作的佇列功能。
多 GPU 時，子行程只跑到 `training`；存檔、merge、轉 GGUF 由父行程在子行程結束後做。

## 事件流

每個 job 一份結構化事件流，同時寫成兩個檔案：

```
{LOG_DIR}/training/{job_id}/events.jsonl   # 機器讀，一行一個 JSON
{LOG_DIR}/training/{job_id}/training.log   # 人讀
```

| 欄位 | 必有 | 說明 |
|---|---|---|
| `ts` | 是 | Unix 時間（float） |
| `seq` | 是 | 單調遞增，**續傳的依據** |
| `job_id` | 是 | |
| `type` | 是 | `stage`（階段轉換）／`resource`（GPU、DRAM、SSD 快照）／`metric`（loss、lr…）／`error`／`info` |
| `phase`、`msg`、`data` | 否 | |

### 兩種讀法

| 端點 | 用途 |
|---|---|
| `GET /training/{job_id}/log?since=<seq>` | 補抓歷史 |
| `GET /training/{job_id}/log/stream` | **SSE** 即時事件流 |

- 斷線重連時用 `Last-Event-ID` 標頭（或 `?since=`）帶上最後收到的 `seq`，從那裡接續；標頭優先。
- 收到終止 phase 的 `stage` 事件就是結束。
- 經過反向代理時**必須關掉代理的緩衝**，否則事件會積著一次湧入。見 [放在反向代理後面](../operations/reverse-proxy.md)。
- 事件檔在磁碟上，**服務重啟後歷史仍在**；不論哪個 uvicorn worker 啟動的 job 都讀得到。

## 端點

| 方法 | 路徑 | 說明 |
|---|---|---|
| POST | `/training/start` | 開始，回 `job_id` |
| GET | `/training/status` | 粗粒度狀態與進度 |
| POST | `/training/stop` | 停止 |
| GET | `/training/{job_id}/log`、`/log/stream` | 見上 |
| GET | `/training/error_details` | 錯誤詳情與完整 traceback |
| GET | `/training/status/{session_id}/history` | loss、learning rate 等歷史 |
| POST | `/training/force_cleanup_gpu` | 強制清理訓練用的 GPU 行程與記憶體 |

## 相關

- [多模態微調](multimodal.md)
- [DeepSpeed ZeRO-3 offload](deepspeed-offload.md)
- [GGUF 轉換](../models/gguf-conversion.md)
- [16 GB 顯卡上用 LoRA 訓練 35B MoE](../benchmarks/deepspeed-lora-16gb.md)
