---
source_url: https://github.com/atomicstrata/llm-wiki-compiler
ingested: 2026-05-28
sha256: a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6
---

# LLM Wiki 格式轉換機制

## 概述

LLM Wiki 並不是一個單一的工具，而是一種**概念模式**。有兩個主要的實作：

1. **Karpathy 的原始概念 (LLM Wiki Pattern)** - 概念設計，描述如何建立知識庫的模式，沒有提供具體的檔案轉換程式碼
2. **llm-wiki-compiler** - 具體實作的 Node.js 工具 (`npm install -g llm-wiki-compiler`)

## 不同格式的轉換方式

### 1. HTML 網頁轉換

**來源：** [src/ingest/web.ts](https://github.com/atomicstrata/llm-wiki-compiler/blob/main/src/ingest/web.ts)

**使用的工具：**
- `@mozilla/readability` - 從 HTML 提取可讀內容（移除導航、廣告等雜訊）
- `turndown` - 將 HTML 轉換為 Markdown

**轉換流程：**
```
HTML 網頁 → Readability 提取 → Turndown 轉換 → Markdown
```

**程式碼實作：**
```typescript
import { JSDOM } from "jsdom";
import { Readability } from "@mozilla/readability";
import TurndownService from "turndown";

async function ingestWeb(url: string): Promise<WebIngestResult> {
  const response = await fetch(url);
  const html = await response.text();
  
  // 使用 Readability 提取可讀內容
  const dom = new JSDOM(html, { url });
  const reader = new Readability(dom.window.document);
  const article = reader.parse();
  
  // 使用 Turndown 轉換為 Markdown
  const turndown = new TurndownService({ headingStyle: "atx" });
  const markdown = turndown.turndown(article.content);
  
  return { title: article.title, content: markdown };
}
```

### 2. PDF 檔案轉換

**來源：** [src/ingest/pdf.ts](https://github.com/atomicstrata/llm-wiki-compiler/blob/main/src/ingest/pdf.ts)

**使用的工具：**
- `pdf-parse` (依賴 `pdfjs-dist`)

**轉換流程：**
```
PDF 檔案 → pdf-parse 提取文字 → 讀取 Metadata → Markdown
```

**程式碼實作：**
```typescript
import { PDFParse } from "pdf-parse";

async function ingestPdf(filePath: string): Promise<IngestedSource> {
  const { PDFParse } = await import("pdf-parse");
  const buffer = await readFile(filePath);
  const parser = new PDFParse({ data: new Uint8Array(buffer) });
  
  // 提取文字和元數據
  const textResult = await parser.getText();
  const infoResult = await parser.getInfo();
  
  const title = resolveTitle(filePath, infoResult.info);
  const content = textResult.text.trim();
  
  return { title, content };
}
```

**特點：**
- 提取 PDF 的文字內容
- 讀取 PDF 的 Metadata（標題等）
- 支援中文 PDF

### 3. 圖片轉換（需要 LLM Vision）

**來源：** [src/ingest/image.ts](https://github.com/atomicstrata/llm-wiki-compiler/blob/main/src/ingest/image.ts)

**使用的工具：**
- `Anthropic Claude Vision API`（目前僅支援 Anthropic）

**轉換流程：**
```
圖片 → Base64 編碼 → Anthropic Claude Vision API → OCR + 視覺描述 → Markdown
```

**程式碼實作：**
```typescript
import Anthropic from "@anthropic-ai/sdk";

async function ingestImage(filePath: string): Promise<IngestedSource> {
  const imageBuffer = await readFile(filePath);
  const imageData = imageBuffer.toString("base64");
  
  const client = new Anthropic({ /* config */ });
  const response = await client.messages.create({
    model: "claude-sonnet-4",
    max_tokens: 4096,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image",
            source: { 
              type: "base64", 
              media_type: "image/png", 
              data: imageData 
            },
          },
          {
            type: "text",
            text: "Extract and transcribe all text visible in this image. Then provide a detailed description of any non-text visual content. Format your response as markdown."
          }
        ],
      }
    ]
  });
  
  return { 
    title: titleFromFilename(filePath),
    content: response.content[0].text 
  };
}
```

**支援的格式：**
- `.jpg, .jpeg` (image/jpeg)
- `.png` (image/png)
- `.gif` (image/gif)
- `.webp` (image/webp)

### 4. 程式碼/文字檔案轉換

**來源：** [src/ingest/file.ts](https://github.com/atomicstrata/llm-wiki-compiler/blob/main/src/ingest/file.ts)

**處理方式：**
- **`.md` 檔案** → 直接讀取 → Markdown（保持原樣）
- **`.txt` 檔案** → 讀取 → 包裝成 ```code block``` → Markdown

**程式碼實作：**
```typescript
const SUPPORTED_EXTENSIONS = new Set([ ".md", ".txt" ]);

async function ingestFile(filePath: string): Promise<IngestedSource> {
  const ext = path.extname(filePath).toLowerCase();
  const raw = await readFile(filePath, "utf-8");
  const title = titleFromFilename(filePath);
  
  // .md 檔案直接返回，.txt 檔案包裝成 code block
  const content = ext === ".md" ? raw : wrapPlainText(raw);
  
  return { title, content };
}

function wrapPlainText(text: string): string {
  return ` \`\`\`\n${text}\n\`\`\``;
}
```

### 5. 影片/音訊轉錄

**來源：** [src/ingest/transcript.ts](https://github.com/atomicstrata/llm-wiki-compiler/blob/main/src/ingest/transcript.ts)

**使用的工具：**
- `youtube-transcript` 庫

**功能：**
- 從 YouTube 影片提取字幕
- 轉換為 Markdown 格式

## 儲存格式

所有轉換後的內容都以這種格式儲存：

```markdown
---
title: 文件標題
summary: 簡短摘要
kind: concept|entity|comparison|overview
sources:
  - 原始來源檔名
createdAt: "2026-04-05T12:00:00Z"
updatedAt: "2026-04-05T12:00:00Z"
confidence: 0.85
provenanceState: merged
---

# 內容標題

這是轉換後的 Markdown 內容...

## 相關概念

- [[概念 A]]
- [[概念 B]]

^ [原始來源檔名]  // 來源引用標記
```

## 關鍵技術總結

| 格式 | 工具/庫 | 轉換方式 |
|------|---------|---------|
| HTML | `@mozilla/readability` + `turndown` | 提取可讀內容 → HTML 轉 Markdown |
| PDF | `pdf-parse` | 提取文字 → Markdown |
| 圖片 | `Anthropic Claude Vision API` | OCR + 視覺描述 → Markdown |
| 程式碼 | 內建 | .md 直接讀，.txt 包裝成 code block |
| 影片字幕 | `youtube-transcript` | 提取字幕 → Markdown |

## 引用來源

1. **llm-wiki-compiler 原始碼** - [GitHub Repository](https://github.com/atomicstrata/llm-wiki-compiler)
   - [ingest/pdf.ts](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/src/ingest/pdf.ts)
   - [ingest/web.ts](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/src/ingest/web.ts)
   - [ingest/image.ts](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/src/ingest/image.ts)
   - [ingest/file.ts](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/src/ingest/file.ts)
   - [README.md](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/README.md)

2. **Karpathy 的 LLM Wiki 概念** - [Original Gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)

3. **Hermes Agent llm-wiki Skill** - [Skill Documentation](https://github.com/YourOrganization/hermes-agent/blob/main/skills/research/llm-wiki/SKILL.md)

## 附錄：工具安裝

```bash
# 安裝 llm-wiki-compiler
npm install -g llm-wiki-compiler

# 設定 API Key
export ANTHROPIC_API_KEY=sk-ant-...

# 使用
llmwiki quickstart ./notes.md
llmwiki compile
llmwiki query "what is X?"
```

## 相關概念

- [[知識庫管理]]
- [[Markdown 格式]]
- [[LLM Vision]]
- [[OCR 技術]]
- [[文檔解析]]
