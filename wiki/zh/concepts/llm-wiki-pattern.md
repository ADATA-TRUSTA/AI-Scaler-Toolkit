---
title: LLM Wiki Pattern
summary: Karpathy 提出的知識庫構建概念模式，通過 LLM 自動化整理來源資料並建立結構化的 wiki。
kind: concept
sources:
  - raw/articles/llm-wiki-format-conversion-2026-05-28.md
createdAt: "2026-05-28"
updatedAt: "2026-05-28"
confidence: high
---

# LLM Wiki Pattern

LLM Wiki Pattern 是由 Andrej Karpathy 提出的一種知識庫構建方法論，核心思想是通過 LLM 自動化地將來源資料轉換為結構化的、相互關聯的 Markdown 知識庫。

## 核心概念

### 與傳統 RAG 的區別

**傳統 RAG（檢索增強生成）：**
```
query → search chunks → answer → forget
```
- 每次查詢都重新檢索和處理
- 知識不會累積
- 沒有持久化的知識結構

**LLM Wiki Pattern：**
```
sources → compile → wiki → query → save → richer wiki → better answers
```
- 將知識編譯成持久的 wiki
- 概念頁面相互連結
- 查詢結果可以保存並豐富知識庫

## 架構層級

### Layer 1: Raw Sources（原始來源）
- 不可變的來源文件集合
- 包括文章、論文、圖片、數據文件等
- LLM 只讀取，不修改

### Layer 2: The Wiki（知識庫）
- LLM 生成的 Markdown 文件目錄
- 包含摘要、實體頁面、概念頁面、比較分析等
- LLM 完全擁有此層級：創建頁面、更新內容、維護交叉連結

### Layer 3: The Schema（模式層）
- 定義文件結構和命名規範
- 規範標籤體系
- 確保知識庫的一致性

## 工作流程

1. ** ingest（導入）**：獲取來源資料並保存到 raw/目錄
2. **Compile（編譯）**：LLM 分析來源，提取概念，創建/更新 wiki 頁面
3. **Query（查詢）**：用戶查詢知識庫，LLM 基於編譯後的知識回答
4. **Save（保存）**：將有价值的查詢結果保存為新的 wiki 頁面
5. **Lint（檢查）**：定期檢查知識庫質量，發現問題並修正

## 實現工具

### llm-wiki-compiler
Node.js 工具，實現了完整的 LLM Wiki Pattern：
- 支援多種格式：HTML、PDF、圖片、文字檔案
- 自動提取概念並創建 wiki 頁面
- 建立頁面間的交叉連結
- 提供查詢和編譯功能
- 支援 Obsidian 兼容格式

```bash
npm install -g llm-wiki-compiler
export ANTHROPIC_API_KEY=sk-ant-...
llmwiki quickstart ./notes.md
llmwiki compile
```

## 關鍵特點

- **自動化**：LLM 負責整理、歸檔、維護
- **累積性**：知識庫隨著時間增長而變得更有價值
- **可查詢**：結構化的知識支持高效查詢
- **可追溯**：每個段落都有來源引用標記
- **相互連結**：概念頁面之間建立豐富的交叉連結

## 相關概念

- [[知識庫管理]]
- [[RAG vs LLM Wiki]]
- [[Markdown 格式]]
- [[文檔解析]]
