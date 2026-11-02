---
title: 遠端存取與信任邊界
summary: 誰能呼叫這個服務：本機、別台機器、瀏覽器裡的網頁各有一套規則；預設只開推論 API，錯誤的 key 會被鎖定。
updatedAt: 2026-10-29
---

# 遠端存取與信任邊界

## 設計方向

服務預設只聽 `127.0.0.1`。要讓另一台電腦（例如另一台機器上的 Hermes）使用這個模型，
需要把服務開到網路上——而**那只需要推論 API，其他都不需要**。

管理 API 收的是檔案路徑與啟動參數，未驗證的模型伺服器正是從這裡被攻擊的。
所以設計不是「加上驗證」，而是**把開到網路的範圍縮到它真正的用途**。

## 三種來源，三套規則

判斷依據是 **TCP 連線的來源位址**，不是用戶端能自己設定的 header。

| 來源 | 要什麼 | 能用什麼 |
|---|---|---|
| **本機**（loopback） | 不用 key | 全部 |
| **別台機器** | `Authorization: Bearer <SERVICE_API_KEY>` | 依 `SERVICE_REMOTE_ACCESS`，預設**只有 `/v1/`** |
| **瀏覽器裡的網頁** | — | 只有本機自己提供的頁面；其他網站的網頁一律拒絕 |

### 別台機器

| 設定 | 說明 |
|---|---|
| `SERVICE_API_KEY` | 空的時候，別台機器一律被拒絕（遠端存取沒開） |
| `SERVICE_REMOTE_ACCESS=inference`（預設） | 只有 `/v1/` |
| `SERVICE_REMOTE_ACCESS=all` | 加上 `/inference/`、`/config/`、`/training/`、`/rag/`、`/system/`、`/examples/` |

- 這是**允許清單**：之後新增的路由對網路是關著的，直到有人決定要開。
- **網頁介面與 `/docs` 永遠只限本機**，帶 key 也不行。
- **桌面程式永遠用 `inference`。**

> ⚠️ `all` 只在你自己的保護之後使用：VPN（例如 Tailscale），或有 TLS 與驗證的反向代理。
> **絕不要直接放到網際網路上**——持有 key 的人能碰到的遠不只那個模型。

### `/health`

唯一對網路全開的路徑，讓別台機器在拿到 key 之前先確認服務連得上。
**沒有 key 時只回「它活著」**；本機或帶對 key 時才回完整內容（含 `supported_engines`）。

### 錯誤的 key 會被鎖定

**一分鐘內錯 10 次 → 該位址鎖 5 分鐘**（`429`）。

- 鎖定只存在記憶體，重啟就忘記。
- log 只記位址，不記送了什麼——錯的 key 可能是打錯的真 key。
- **路徑被拒不算錯 key**，打錯網址不會把自己鎖掉。

## 本機也包含瀏覽器裡的每一個網頁

這是最容易忽略的一點：這台機器上開著的任何網站，都能從瀏覽器呼叫 `127.0.0.1`。
所以來自本機的請求還要再過一道檢查：

| 情況 | 結果 |
|---|---|
| `Origin` 是本機提供的頁面（任何埠） | ✅ |
| `Origin` 是別的網站，或 `null`（sandboxed frame、`file://`） | ❌ `403` |
| 沒有 `Origin`，但 `Sec-Fetch-Site: cross-site` 且不是頁面導覽 | ❌ `403`（跨站的 `<img>` 之類，GET 也可能有副作用） |
| `Host` 不是 localhost 也不是 IP | ❌ `403`（防 DNS rebinding） |
| 不是瀏覽器（沒有 `Origin` 與 `Sec-Fetch-*`） | ✅ 腳本、agent、程式助理不受影響 |

CORS 只允許本機的 origin，而且不帶 credentials。

## 往下游的保護

| 對象 | 措施 |
|---|---|
| llama-server | 每次啟動帶一把內部 key（沒設就隨機產生）；CORS 只允許本服務；只綁 loopback |
| vLLM 的 admin 端點 | sleep／wake 等只給本機程式；`/collective_rpc`、`/update_weights` 等一律 404 |
| 載入請求的 extra args | `--host`、`--port`、`--api-key`、`--cors-*` 等由服務管理的旗標會被拒絕 |
| `llama_server_url` | 只接受 loopback |

## 放在反向代理後面

**這一層靠的是連線來源位址，而代理會改變它。** 兩種情況都不是你要的：

| 代理 | 後端看到的來源 | 結果 |
|---|---|---|
| 不送 `X-Forwarded-For` | 代理自己（127.0.0.1） | **每個外部請求都被當成本機，全開** |
| 送 `X-Forwarded-For` | 真正的用戶端 | **網頁介面與管理 API 全部 `403`** |

所以在代理後面，**驗證要由代理負責**。見 [放在反向代理後面](../operations/reverse-proxy.md)。

## 相關

- [錯誤模型](errors.md)
- [從症狀反查](../operations/troubleshooting.md)
- [`docs/usage.md`：連接其他應用程式](../../../docs/usage.md#連接其他應用程式)
