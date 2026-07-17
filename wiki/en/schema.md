# Wiki Schema

## Domain
This knowledge base covers three related areas:
1. **Trusta AST LLM inference/training service** (the `wiki/` subdirectory) — architecture, engines, API, offload, fine-tuning.
2. **TRUSTA enterprise-grade SSD product line** — echoing the value proposition of offloading to SSD/DRAM.
3. **LLM Wiki methodology** — the format conversion technique for automatically organizing source documents (HTML/Markdown) into a structured wiki.

## Conventions
- **File names**: lowercase, hyphens, no spaces (e.g., `llm-wiki-pattern.md`)
- **Every wiki page starts with YAML frontmatter** (see below)
- **Use `[[wikilinks]]` to link between pages** (minimum 2 outbound links per page)
- **When updating a page, always bump the `updated` date**
- **Every new page must be added to `index.md` under the correct section**
- **Every action must be appended to `log.md`**
- **Provenance markers**: Mark the source document at the bottom of the page, in the format `^ [filename.md]`

## Frontmatter

All wiki pages must include the following YAML frontmatter:

```yaml
---
title: Page title
summary: Short summary (one or two sentences)
kind: concept|entity|comparison|overview
sources:
  - raw/articles/source-file-name.md
createdAt: YYYY-MM-DD
updatedAt: YYYY-MM-DD
confidence: 0.0-1.0  # Optional, indicates confidence level
provenanceState: extracted|merged|inferred|ambiguous  # Optional
---
```

### Frontmatter Field Descriptions

| Field | Required | Description |
|------|------|------|
| title | Yes | Page title |
| summary | Yes | Short summary, one or two sentences |
| kind | Yes | Page type: concept, entity, comparison, overview |
| sources | Yes | List of source files |
| createdAt | Yes | Creation date |
| updatedAt | Yes | Last updated date |
| confidence | No | Confidence level (0-1) |
| provenanceState | No | Provenance state |

## Tag Taxonomy

Defines the main tag categories:

### Technical Categories
- `llm-wiki` - LLM Wiki related
- `format-conversion` - Format conversion
- `html` - HTML processing
- `pdf` - PDF processing
- `image` - Image processing
- `ocr` - OCR techniques
- `vision` - Vision capabilities

### Tool Categories
- `tool` - Tool introduction
- `library` - Libraries
- `api` - API usage
- `npm` - npm packages

### Concept Categories
- `concept` - Concept explanation
- `pattern` - Design patterns
- `workflow` - Workflows
- `implementation` - Implementation details

### Knowledge Categories
- `research` - Research material
- `tutorial` - Tutorial documents
- `reference` - Reference documentation
- `comparison` - Comparative analysis

**Rule**: The tags on each page must come from this taxonomy. If a new tag is needed, add it here first, then use it.

## Page Thresholds

- **Create a page**: When an entity/concept appears in 2+ sources, or is the core topic of a source
- **Update an existing page**: When a source mentions already-covered content
- **Do not create a page**: For passing mentions, minor details, or out-of-domain content
- **Split a page**: When a page exceeds 200 lines, split it into subtopics and establish cross-links
- **Archive a page**: When content is completely superseded, move it to `_archive/` and remove it from the index

## Entity Pages

A separate page for each important entity. Includes:
- Overview / what it is
- Key facts and dates
- Relationships to other entities ([[wikilinks]])
- Source citations

## Concept Pages

A separate page for each concept or topic. Includes:
- Definition / explanation
- Current state of knowledge
- Open questions or controversies
- Related concepts ([[wikilinks]])

## Comparison Pages

Side-by-side analysis. Includes:
- What is being compared and why
- Comparison dimensions (prefer table format)
- Conclusion or synthesis
- Sources

## Update Policy

When new information conflicts with existing content:
1. Check the dates - newer sources usually supersede older ones
2. If there is a genuine conflict, record both positions along with their dates and sources
3. Note the conflict in the frontmatter: `contradictedBy: [page name]`
4. Flag it in the lint report for user review

## Raw Sources Frontmatter

Raw source files also need frontmatter:

```yaml
---
source_url: https://example.com/article
ingested: YYYY-MM-DD
sha256: <SHA256 hash of the content>
---
```

`sha256` is used to detect whether the content has changed when re-ingesting the same URL, avoiding duplicate processing.

## Directory Structure

```
wiki/
├── SCHEMA.md                 # This file
├── index.md                   # Page index
├── log.md                     # Operation log
├── raw/                       # Raw sources (immutable)
│   └── articles/             # Web articles, documents
├── concepts/                  # Concept pages
│   └── *.md
├── entities/                  # Entity pages
│   └── *.md
├── comparisons/               # Comparison pages
│   └── *.md
└── queries/                   # Query results
    └── *.md
```

## Related Resources

- [[LLM Wiki Pattern]] - The conceptual pattern proposed by Karpathy
- [[File Format Conversion Techniques]] - Conversion methods for various formats
- [[llm-wiki-compiler]] - The Node.js tool implementation
