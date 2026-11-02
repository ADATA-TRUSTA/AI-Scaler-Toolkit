---
title: 放在反向代理後面
summary: 用 nginx 之類的反向代理對外提供 AST 時要注意的事：路徑前綴、SSE 緩衝、逾時，以及代理如何讓後端的遠端存取檢查失效。
updatedAt: 2026-10-29
---

# 放在反向代理後面

桌面程式不需要反向代理。本頁給要把服務放在 nginx 等代理後面、讓多人使用的部署。

## 最重要的一件事：驗證交給代理

後端判斷「請求從哪來」靠的是 **TCP 連線的來源位址**（見 [遠端存取](../api/remote-access.md)），
而代理會改變它：

| 代理的設定 | 後端看到的來源 | 結果 |
|---|---|---|
| 不送 `X-Forwarded-For` | 代理自己，也就是 127.0.0.1 | **每個外部請求都被當成本機：全開**。`SERVICE_API_KEY` 與 `SERVICE_REMOTE_ACCESS` 都不生效 |
| 送 `X-Forwarded-For` | 真正的用戶端 | **網頁介面與管理 API 一律 `403`**（*Access from other computers is not enabled*）。uvicorn 預設信任 127.0.0.1 送來的這個 header |

**兩種都不是你要的。** 實務做法：

- 代理**不送** `X-Forwarded-For`／`X-Real-IP`，讓後端把請求當成本機；
- **由代理負責驗證**：`auth_basic`、mTLS 或上游的 SSO，並加上 TLS；
- 或只把 `/v1/` 開到代理上，管理 API 不對外。

> 後端看不出前面有代理，所以**不會提醒你**。部署後請從另一台機器、不帶憑證打一次 `/config/models` 確認被擋住。

## 路徑前綴

後端**沒有 `/api` 前綴**。若要在 `/api/` 底下提供，用結尾的斜線讓代理剝掉前綴：

```nginx
location /api/ {
    proxy_pass http://127.0.0.1:8000/;   # 結尾的 / 會剝掉 /api
}
```

## SSE 必須關掉緩衝

代理預設會緩衝回應；對 SSE 來說事件會積著、最後一次湧入。

```nginx
proxy_http_version 1.1;
proxy_set_header Connection "";
proxy_buffering off;
proxy_cache off;
chunked_transfer_encoding on;
proxy_read_timeout 3600s;
```

| 設定 | 為什麼 |
|---|---|
| `proxy_buffering off` | **關鍵**，不關就不會即時送出 |
| `proxy_http_version 1.1` + `Connection ""` | 保持上游長連線 |
| `proxy_read_timeout` | 訓練與長 prefill 很久；預設 60 秒會切斷串流 |

後端的 SSE 回應也會帶 `X-Accel-Buffering: no`，但只有 nginx 認得，換別的代理就沒有效果。

受影響的端點：`POST /v1/chat/completions`（串流）與 `GET /training/{job_id}/log/stream`。

## 逾時

後端的生成逾時預設 1800 秒（`DEFAULT_GENERATION_TIMEOUT`），
而串流時**等第一個 byte 就是整段 prefill**——長 context 可能要好幾分鐘。
代理的 `proxy_read_timeout` 要跟上，否則症狀是「長 context 的第一個回應就斷線」。

## 相關

- [遠端存取與信任邊界](../api/remote-access.md)
- [從症狀反查](troubleshooting.md)
