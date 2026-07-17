# Supporting Evidence for LLM Wiki Format Conversion Research

## Research Overview

This research investigates how an LLM Wiki converts data in different formats (HTML, PDF, images, code, etc.) into Markdown files.

## Research Date

2026-05-28

## Query Methods

1. Web browser queries of GitHub source code
2. Directly reading source files
3. Analyzing implementation details

## Supporting Sources

### Primary Sources

#### 1. llm-wiki-compiler Source Repository
**URL:** https://github.com/atomicstrata/llm-wiki-compiler

**Key files:**
- [src/ingest/web.ts](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/src/ingest/web.ts) - HTML conversion
- [src/ingest/pdf.ts](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/src/ingest/pdf.ts) - PDF conversion
- [src/ingest/image.ts](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/src/ingest/image.ts) - Image conversion
- [src/ingest/file.ts](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/src/ingest/file.ts) - Text/code conversion
- [src/ingest/transcript.ts](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/src/ingest/transcript.ts) - Video subtitles
- [README.md](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/README.md) - Complete documentation

**Key findings:**
- LLM Wiki is not a single tool, but a conceptual pattern
- llm-wiki-compiler is a concrete Node.js tool implementation
- Different formats use different specialized libraries for conversion

#### 2. Karpathy's Original LLM Wiki Concept
**URL:** https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f

**Key findings:**
- This is a conceptual design, not a concrete piece of software
- It describes a three-layer architecture: Raw Sources, The Wiki, The Schema
- It emphasizes the pattern of an LLM automatically organizing knowledge

#### 3. Hermes Agent llm-wiki Skill
**Location:** /home/test/.hermes/skills/research/llm-wiki/SKILL.md

**Key findings:**
- The Hermes Agent uses the web_extract tool to handle data conversion
- web_extract uses the Firecrawl service
- It supports markdown extraction from URLs and PDFs

## Technical Detail Verification

### HTML Conversion
**Verification method:** Directly viewing the [src/ingest/web.ts](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/src/ingest/web.ts) source code

**Tools confirmed:**
- `@mozilla/readability` - Extracts readable content
- `turndown` - HTML to Markdown

### PDF Conversion
**Verification method:** Directly viewing the [src/ingest/pdf.ts](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/src/ingest/pdf.ts) source code

**Tools confirmed:**
- `pdf-parse` - Based on pdfjs-dist

### Image Conversion
**Verification method:** Directly viewing the [src/ingest/image.ts](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/src/ingest/image.ts) source code

**Tools confirmed:**
- `Anthropic Claude Vision API`
- Only the Anthropic provider is supported
- Supports .jpg, .png, .gif, .webp

### Text/Code Conversion
**Verification method:** Directly viewing the [src/ingest/file.ts](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/src/ingest/file.ts) source code

**Tools confirmed:**
- Built-in processing
- .md is read directly
- .txt is wrapped into a code block

## Storage Format Verification

**Source:** [README.md - What it produces](https://raw.githubusercontent.com/atomicstrata/llm-wiki-compiler/main/README.md)

**Verified frontmatter format:**
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

**Verified source citation format:**
```markdown
This paragraph is grounded in the source. ^[source.md]
```

## Research Conclusions

1. **LLM Wiki is a conceptual pattern**, requiring a concrete tool to implement it
2. **Format conversion relies on specialized libraries**, with each format using the most appropriate tool
3. **Image conversion requires LLM Vision**, because it needs to understand visual content
4. **All content is ultimately converted to Markdown**, with YAML frontmatter and source markers added
5. **Structured storage**, supporting cross-linking and traceability

## Related Query Records

The complete contents of this research have been saved to:
- `raw/articles/llm-wiki-format-conversion-2026-05-28.md` - Complete research report
- `concepts/llm-wiki-pattern.md` - LLM Wiki Pattern concept explanation
- `concepts/file-format-conversion.md` - Detailed explanation of format conversion techniques
- `SCHEMA.md` - Knowledge base structure specification
- `index.md` - Page directory
- `log.md` - Operation log

## Research Tools

- Chrome DevTools Protocol (CDP)
- GitHub API
- Web browser automation
- Direct source code reading

## Research Limitations

1. **Only public resources were queried** - Private repositories were not accessed
2. **Actual conversion was not tested** - Only the source code was analyzed; no conversion was executed
3. **Focused on llm-wiki-compiler** - Other implementations were not investigated in depth
4. **Performance metrics were not verified** - Only functional implementation was of concern

## Additional Notes

The supporting evidence for this research is based entirely on publicly available source code and documentation; all cited sources can be verified directly through the URLs provided.

---

**Research completion time:** 2026-05-28
**Researcher:** Hermes Agent
**Evidence level:** High (based on direct source code analysis)
