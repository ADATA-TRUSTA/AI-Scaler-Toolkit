# LLM Wiki 格式轉換研究佐證

## 研究概述

本研究查詢 LLM Wiki 如何將不同格式的資料（HTML、PDF、圖片、程式碼等）轉換為 Markdown 檔案。

## 研究日期

2026-05-28

## 查詢方法

1. 網頁瀏覽器查詢 GitHub 原始碼
2. 直接讀取源碼文件
3. 分析實現細節

## 佐證來源

### 主要來源

#### 1. llm-wiki-compiler 原始碼庫
**網址：** https://github.com/atomicstrata/llm-wiki-compiler

**關鍵檔案：**
- [src/ingest/web.ts](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/src/ingest/web.ts) - HTML 轉換
- [src/ingest/pdf.ts](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/src/ingest/pdf.ts) - PDF 轉換
- [src/ingest/image.ts](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/src/ingest/image.ts) - 圖片轉換
- [src/ingest/file.ts](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/src/ingest/file.ts) - 文字/程式碼轉換
- [src/ingest/transcript.ts](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/src/ingest/transcript.ts) - 影片字幕
- [README.md](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/README.md) - 完整文件

**關鍵發現：**
- LLM Wiki 不是一個單一工具，而是一個概念模式
- llm-wiki-compiler 是具體實現的 Node.js 工具
- 不同格式使用不同的專用庫進行轉換

#### 2. Karpathy 的 LLM Wiki 原始概念
**網址：** https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f

**關鍵發現：**
- 這是一個概念設計，而非具體軟體
- 描述了三層架構：Raw Sources、The Wiki、The Schema
- 強調 LLM 自動化整理知識的模式

#### 3. Hermes Agent llm-wiki Skill
**位置：** /home/test/.hermes/skills/research/llm-wiki/SKILL.md

**關鍵發現：**
- Hermes Agent 使用 web_extract 工具處理資料轉換
- web_extract 使用 Firecrawl 服務
- 支援 URL、PDF 的 markdown 提取

## 技術細節驗證

### HTML 轉換
**驗證方式：** 直接查看 [src/ingest/web.ts](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/src/ingest/web.ts) 原始碼

**工具確認：**
- `@mozilla/readability` - 提取可讀內容
- `turndown` - HTML 轉 Markdown

### PDF 轉換
**驗證方式：** 直接查看 [src/ingest/pdf.ts](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/src/ingest/pdf.ts) 原始碼

**工具確認：**
- `pdf-parse` - 基於 pdfjs-dist

### 圖片轉換
**驗證方式：** 直接查看 [src/ingest/image.ts](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/src/ingest/image.ts) 原始碼

**工具確認：**
- `Anthropic Claude Vision API`
- 僅支援 Anthropic 提供商
- 支援 .jpg, .png, .gif, .webp

### 文字/程式碼轉換
**驗證方式：** 直接查看 [src/ingest/file.ts](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/src/ingest/file.ts) 原始碼

**工具確認：**
- 內建處理
- .md 直接讀取
- .txt 包裝成 code block

## 儲存格式驗證

**來源：** [README.md - What it produces](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/README.md)

**驗證 frontmatter 格式：**
```yaml
---
title: Knowledge Compilation
summary: Techniques for converting knowledge representations into forms that support efficient reasoning.
kind: concept
sources:
  - knowledge-compilation.md
createdAt: "2026-04-05T12:00:00Z"
updatedAt: "2026-04-05T12:00:00Z"
---
```

**驗證來源引用格式：**
```markdown
This paragraph is grounded in the source. ^[source.md]
```

## 研究結論

1. **LLM Wiki 是概念模式**，需要具體工具實現
2. **格式轉換依賴專用庫**，每個格式使用最合適的工具
3. **圖片轉換需要 LLM Vision**，因為需要理解視覺內容
4. **所有內容最終轉為 Markdown**，加上 YAML frontmatter 和來源標記
5. **結構化存儲**，支持交叉連結和可追溯性

## 相關查詢記錄

本研究的完整內容已儲存至：
- `raw/articles/llm-wiki-format-conversion-2026-05-28.md` - 完整研究報告
- `concepts/llm-wiki-pattern.md` - LLM Wiki Pattern 概念說明
- `concepts/file-format-conversion.md` - 格式轉換技術詳解
- `SCHEMA.md` - 知識庫結構規範
- `index.md` - 頁面目錄
- `log.md` - 操作日誌

## 研究工具

- Chrome DevTools Protocol (CDP)
- GitHub API
- 網頁瀏覽器自動化
- 原始碼直接讀取

## 研究限制

1. **僅查詢公開資源** - 未訪問私有倉庫
2. **未測試實際轉換** - 僅分析原始碼，未執行轉換
3. **聚焦於 llm-wiki-compiler** - 其他實現未深入查詢
4. **未驗證性能指標** - 僅關注功能實現

## 附註

本研究的佐證完全基於公開可得的原始碼和文檔，所有引用來源均可通過提供的網址直接驗證。

---

**研究完成時間：** 2026-05-28
**研究人員：** Hermes Agent
**佐證級別：** 高（基於原始碼直接分析）
