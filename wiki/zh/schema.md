# Wiki Schema

## Domain
本知識庫涵蓋三個相關領域：
1. **Trusta AST LLM 推理/訓練服務**（`wiki/` 子目錄）— 架構、引擎、API、offload、fine-tuning。
2. **TRUSTA 企業級 SSD 產品線** — 與 offload 到 SSD/DRAM 的價值主張相呼應。
3. **LLM Wiki 方法論** — 由來源文件（HTML/Markdown）自動整理成結構化 wiki 的格式轉換技術。

## Conventions
- **File names**: lowercase, hyphens, no spaces (e.g., `llm-wiki-pattern.md`)
- **Every wiki page starts with YAML frontmatter** (see below)
- **Use `[[wikilinks]]` to link between pages** (minimum 2 outbound links per page)
- **When updating a page, always bump the `updated` date**
- **Every new page must be added to `index.md` under the correct section**
- **Every action must be appended to `log.md`**
- **Provenance markers**: 在頁尾標註來源文件，格式為 `^ [filename.md]`

## Frontmatter

所有 wiki 頁面必須包含以下 YAML frontmatter：

```yaml
---
title: 頁面標題
summary: 簡短摘要（一兩句話）
kind: concept|entity|comparison|overview
sources:
  - raw/articles/源文件名.md
createdAt: YYYY-MM-DD
updatedAt: YYYY-MM-DD
confidence: 0.0-1.0  # 可選，表示可信度
provenanceState: extracted|merged|inferred|ambiguous  # 可選
---
```

### frontmatter 字段說明

| 字段 | 必填 | 說明 |
|------|------|------|
| title | 是 | 頁面標題 |
| summary | 是 | 簡短摘要，一兩句話 |
| kind | 是 | 頁面類型：concept（概念）、entity（實體）、comparison（比較）、overview（概述） |
| sources | 是 | 來源文件列表 |
| createdAt | 是 | 創建日期 |
| updatedAt | 是 | 最後更新日期 |
| confidence | 否 | 可信度（0-1） |
| provenanceState | 否 | 來源狀態 |

## Tag Taxonomy

定義主要標籤分類：

### 技術類別
- `llm-wiki` - LLM Wiki 相關
- `format-conversion` - 格式轉換
- `html` - HTML 處理
- `pdf` - PDF 處理
- `image` - 圖片處理
- `ocr` - OCR 技術
- `vision` - 視覺能力

### 工具類別
- `tool` - 工具介紹
- `library` - 程式庫
- `api` - API 使用
- `npm` - npm 套件

### 概念類別
- `concept` - 概念說明
- `pattern` - 設計模式
- `workflow` - 工作流程
- `implementation` - 實作細節

### 知識類別
- `research` - 研究資料
- `tutorial` - 教學文件
- `reference` - 參考文檔
- `comparison` - 比較分析

**規則**：每個頁面的標籤必須來自此分類。如果需要新標籤，先在此添加，然後再使用。

## Page Thresholds

- **創建頁面**：當實體/概念出現在 2+ 來源中，或是一個來源的核心主題
- **更新現有頁面**：當來源提到已覆蓋的內容時
- **不創建頁面**：對於過場提及、細節或域外內容
- **分割頁面**：當頁面超過 200 行時，拆分成子主題並建立交叉連結
- **歸檔頁面**：當內容完全被取代時，移動到 `_archive/`，從 index 移除

## Entity Pages

每個重要實體的單獨頁面。包含：
- 概述 / 是什麼
- 關鍵事實和日期
- 與其他實體的關係（[[wikilinks]]）
- 來源引用

## Concept Pages

每個概念或主題的單獨頁面。包含：
- 定義 / 解釋
- 當前知識狀態
- 開放問題或爭議
- 相關概念（[[wikilinks]]）

## Comparison Pages

並排分析。包含：
- 比較什麼以及為什麼
- 比較維度（優先使用表格格式）
- 結論或綜合
- 來源

## Update Policy

當新資訊與現有內容衝突時：
1. 檢查日期 - 較新的來源通常取代較舊的
2. 如果確實有衝突，記錄兩個立場及其日期和來源
3. 在 frontmatter 標註衝突：`contradictedBy: [頁面名稱]`
4. 在檢查報告中標記供用戶審查

## Raw Sources Frontmatter

原始來源文件也需要 frontmatter：

```yaml
---
source_url: https://example.com/article
ingested: YYYY-MM-DD
sha256: <content 的 SHA256 hash>
---
```

`sha256` 用於在重新導入相同 URL 時檢測內容是否改變，避免重複處理。

## 目錄結構

```
wiki/
├── SCHEMA.md                 # 此文件
├── index.md                   # 頁面目錄
├── log.md                     # 操作日誌
├── raw/                       # 原始來源（不可修改）
│   └── articles/             # 網頁文章、文件
├── concepts/                  # 概念頁面
│   └── *.md
├── entities/                  # 實體頁面
│   └── *.md
├── comparisons/               # 比較頁面
│   └── *.md
└── queries/                   # 查詢結果
    └── *.md
```

## 相關資源

- [[LLM Wiki Pattern]] - Karpathy 提出的概念模式
- [[檔案格式轉換技術]] - 各種格式的轉換方法
- [[llm-wiki-compiler]] - Node.js 工具實現
