---
title: File Format Conversion for LLM Wiki
summary: 各種檔案格式轉換成 Markdown 的方法和技术，包括 HTML、PDF、圖片、程式碼等。
kind: concept
sources:
  - raw/articles/llm-wiki-format-conversion-2026-05-28.md
createdAt: "2026-05-28"
updatedAt: "2026-05-28"
confidence: high
---

# 檔案格式轉換技術

LLM Wiki 需要將各種來源格式轉換為統一的 Markdown 格式。不同格式使用不同的工具和方法。

## HTML 網頁轉換

### 工具組合
- **@mozilla/readability**：從 HTML 提取可讀內容
- **TurndownService**：將 HTML 轉換為 Markdown

### 工作流程
```
HTML 網頁 → Readability 處理 → Turndown 轉換 → Markdown
```

### Readability 的作用
- 提取文章主要內容
- 移除導航欄、側邊欄、廣告等雜訊
- 保留標題、段落、列表、表格等結構
- 處理 HTML 階層結構

### Turndown 的作用
- 將 HTML 標籤轉換為 Markdown 語法
- 支援標題、粗體、斜體、代碼塊、列表、引用等
- 可自訂轉換規則
- 保持 Markdown 的可讀性

### 實作示例
```typescript
import { JSDOM } from "jsdom";
import { Readability } from "@mozilla/readability";
import TurndownService from "turndown";

async function convertHtmlToMarkdown(html: string, url: string): Promise<string> {
  const dom = new JSDOM(html, { url });
  const reader = new Readability(dom.window.document);
  const article = reader.parse();
  
  if (!article) throw new Error("Could not extract content");
  
  const turndown = new TurndownService({ headingStyle: "atx" });
  return turndown.turndown(article.content);
}
```

## PDF 檔案轉換

### 工具：pdf-parse
基於 pdfjs-dist，提取 PDF 的文字內容和元數據。

### 工作流程
```
PDF 檔案 → pdf-parse 讀取 → 提取文字和元數據 → Markdown
```

### 功能特點
- **文字提取**：從 PDF 頁面中提取所有文字內容
- **元數據讀取**：獲取標題、作者、建立日期等資訊
- **支援多語言**：包括中文、日文、韓文等
- **保持結構**：盡量保留段落和列表結構

### 實作示例
```typescript
import { PDFParse } from "pdf-parse";

async function convertPdfToMarkdown(filePath: string): Promise<{title: string, content: string}> {
  const buffer = await readFile(filePath);
  const parser = new PDFParse({ data: new Uint8Array(buffer) });
  
  // 提取文字
  const textResult = await parser.getText();
  // 讀取元數據
  const infoResult = await parser.getInfo();
  
  const title = infoResult.info.Title || extractTitleFromFilename(filePath);
  const content = textResult.text.trim();
  
  return { title, content };
}
```

## 圖片轉換（需要 LLM Vision）

### 原理
圖片無法直接轉為文字，需要通過 LLM 的視覺能力進行 OCR 和描述。

### 工作流程
```
圖片 → Base64 編碼 → LLM Vision API → OCR + 視覺描述 → Markdown
```

### 支援的工具
- **Anthropic Claude Vision**：目前最成熟的方案
- 支援：`.jpg`, `.png`, `.gif`, `.webp`
- 需要 API Key 和支援 Vision 的模型

### 提示詞設計
```
"Extract and transcribe all text visible in this image. 
Then provide a detailed description of any non-text visual content. 
Format your response as markdown."
```

### 實作示例
```typescript
import Anthropic from "@anthropic-ai/sdk";

async function convertImageToMarkdown(imagePath: string): Promise<string> {
  const buffer = await readFile(imagePath);
  const base64Image = buffer.toString("base64");
  
  const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
  
  const response = await client.messages.create({
    model: "claude-sonnet-4",
    max_tokens: 4096,
    messages: [{
      role: "user",
      content: [
        {
          type: "image",
          source: {
            type: "base64",
            media_type: "image/png",
            data: base64Image
          }
        },
        {
          type: "text",
          text: "Extract and transcribe all text visible in this image. Then provide a detailed description of any non-text visual content. Format your response as markdown."
        }
      ]
    }]
  });
  
  return response.content[0].text;
}
```

## 程式碼和文字檔案

### Markdown 檔案（.md）
- 直接讀取內容
- 保持原樣作為 Markdown

### 文字檔案（.txt）
- 讀取純文字內容
- 包裝成 Markdown 代碼塊

### 實作示例
```typescript
async function convertFileToMarkdown(filePath: string): Promise<{title: string, content: string}> {
  const ext = path.extname(filePath).toLowerCase();
  const content = await readFile(filePath, "utf-8");
  
  let markdownContent;
  if (ext === ".md") {
    markdownContent = content;  // 直接返回
  } else if (ext === ".txt") {
    markdownContent = `\`\`\`\n${content}\n\`\`\``;  // 包裝成代碼塊
  } else {
    throw new Error(`Unsupported file type: ${ext}`);
  }
  
  return {
    title: path.basename(filePath, ext).replace(/[-_]/g, " "),
    content: markdownContent
  };
}
```

## 影片字幕提取

### 工具：youtube-transcript
從 YouTube 影片提取字幕內容。

### 工作流程
```
YouTube 影片 URL → youtube-transcript API → 提取字幕 → Markdown 格式
```

### 輸出格式
```markdown
## [時間戳]

[字幕內容]
```

## 格式比較

| 格式 | 工具 | 需要 LLM | 複雜度 | 準確度 |
|------|------|----------|--------|--------|
| HTML | Readability + Turndown | 否 | 低 | 高 |
| PDF | pdf-parse | 否 | 低 | 中 |
| 圖片 | Claude Vision | 是 | 中 | 高 |
| .md | 內建 | 否 | 無 | 完美 |
| .txt | 內建 | 否 | 無 | 完美 |
| 字幕 | youtube-transcript | 否 | 低 | 高 |

## 最佳實踐

1. **優先使用原生 Markdown**：來源如果是 .md 檔案，直接使用
2. **HTML 清理**：使用 Readability 移除雜訊
3. **PDF 處理**：檢查文字選擇性，掃描 PDF 可能需要 OCR
4. **圖片轉換**：確保使用高品質的 Vision 模型
5. **來源標記**：每個段落都應該標記來源
6. **錯誤處理**：對無法轉換的檔案提供清晰的錯誤訊息

## 相關概念

- [[HTML 解析]]
- [[PDF 處理]]
- [[OCR 技術]]
- [[LLM Vision]]
- [[Markdown 格式]]
