---
title: File Format Conversion for LLM Wiki
summary: Methods and techniques for converting various file formats into Markdown, including HTML, PDF, images, and code.
kind: concept
sources:
  - raw/articles/llm-wiki-format-conversion-2026-05-28.md
createdAt: "2026-05-28"
updatedAt: "2026-05-28"
confidence: high
---

# File Format Conversion Techniques

An LLM Wiki needs to convert various source formats into a unified Markdown format. Different formats use different tools and methods.

## HTML Web Page Conversion

### Tool Combination
- **@mozilla/readability**: Extracts readable content from HTML
- **TurndownService**: Converts HTML into Markdown

### Workflow
```
HTML web page → Readability processing → Turndown conversion → Markdown
```

### Role of Readability
- Extracts the main content of an article
- Removes noise such as navigation bars, sidebars, and ads
- Preserves structures such as headings, paragraphs, lists, and tables
- Handles the HTML hierarchical structure

### Role of Turndown
- Converts HTML tags into Markdown syntax
- Supports headings, bold, italics, code blocks, lists, quotes, and more
- Allows custom conversion rules
- Maintains the readability of the Markdown

### Implementation Example
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

## PDF File Conversion

### Tool: pdf-parse
Based on pdfjs-dist, it extracts the text content and metadata of a PDF.

### Workflow
```
PDF file → pdf-parse reads → extract text and metadata → Markdown
```

### Feature Highlights
- **Text extraction**: Extracts all text content from PDF pages
- **Metadata reading**: Retrieves information such as title, author, and creation date
- **Multi-language support**: Including Chinese, Japanese, Korean, and more
- **Structure preservation**: Retains paragraph and list structure as much as possible

### Implementation Example
```typescript
import { PDFParse } from "pdf-parse";

async function convertPdfToMarkdown(filePath: string): Promise<{title: string, content: string}> {
  const buffer = await readFile(filePath);
  const parser = new PDFParse({ data: new Uint8Array(buffer) });
  
  // Extract text
  const textResult = await parser.getText();
  // Read metadata
  const infoResult = await parser.getInfo();
  
  const title = infoResult.info.Title || extractTitleFromFilename(filePath);
  const content = textResult.text.trim();
  
  return { title, content };
}
```

## Image Conversion (Requires LLM Vision)

### Principle
Images cannot be converted directly into text; they require the visual capabilities of an LLM for OCR and description.

### Workflow
```
Image → Base64 encoding → LLM Vision API → OCR + visual description → Markdown
```

### Supported Tools
- **Anthropic Claude Vision**: Currently the most mature solution
- Supports: `.jpg`, `.png`, `.gif`, `.webp`
- Requires an API key and a Vision-capable model

### Prompt Design
```
"Extract and transcribe all text visible in this image. 
Then provide a detailed description of any non-text visual content. 
Format your response as markdown."
```

### Implementation Example
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

## Code and Text Files

### Markdown Files (.md)
- Read the content directly
- Keep it as-is as Markdown

### Text Files (.txt)
- Read the plain text content
- Wrap it into a Markdown code block

### Implementation Example
```typescript
async function convertFileToMarkdown(filePath: string): Promise<{title: string, content: string}> {
  const ext = path.extname(filePath).toLowerCase();
  const content = await readFile(filePath, "utf-8");
  
  let markdownContent;
  if (ext === ".md") {
    markdownContent = content;  // Return directly
  } else if (ext === ".txt") {
    markdownContent = `\`\`\`\n${content}\n\`\`\``;  // Wrap into a code block
  } else {
    throw new Error(`Unsupported file type: ${ext}`);
  }
  
  return {
    title: path.basename(filePath, ext).replace(/[-_]/g, " "),
    content: markdownContent
  };
}
```

## Video Subtitle Extraction

### Tool: youtube-transcript
Extracts subtitle content from YouTube videos.

### Workflow
```
YouTube video URL → youtube-transcript API → extract subtitles → Markdown format
```

### Output Format
```markdown
## [timestamp]

[subtitle content]
```

## Format Comparison

| Format | Tool | Requires LLM | Complexity | Accuracy |
|------|------|----------|--------|--------|
| HTML | Readability + Turndown | No | Low | High |
| PDF | pdf-parse | No | Low | Medium |
| Image | Claude Vision | Yes | Medium | High |
| .md | Built-in | No | None | Perfect |
| .txt | Built-in | No | None | Perfect |
| Subtitles | youtube-transcript | No | Low | High |

## Best Practices

1. **Prefer native Markdown**: If the source is a .md file, use it directly
2. **HTML cleanup**: Use Readability to remove noise
3. **PDF processing**: Check text selectability; scanned PDFs may require OCR
4. **Image conversion**: Ensure the use of a high-quality Vision model
5. **Source tagging**: Every paragraph should be tagged with its source
6. **Error handling**: Provide clear error messages for files that cannot be converted

## Related Concepts

- [[HTML Parsing]]
- [[PDF Processing]]
- [[OCR Techniques]]
- [[LLM Vision]]
- [[Markdown Format]]
