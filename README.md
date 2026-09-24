# xdoc-qa

**Generating questions that can only be answered by reading several documents together.**

Most QA datasets ask about one passage of one document. Real questions often need more:
*"How would Kant and Mill judge a lie told to save a friend?"* has no single source.
This project builds such cross-document questions automatically from long PDFs.

> Work in progress: this README grows one step at a time.

## The corpus

Four classics of moral philosophy, each answering *what makes an action good?* differently:

| Document | Author | Answer in one word |
|---|---|---|
| Nicomachean Ethics | Aristotle | virtue |
| Groundwork of the Metaphysics of Morals | Kant | duty |
| Utilitarianism | J. S. Mill | consequences |
| On the Genealogy of Morality | Nietzsche | critique of morality itself |

Same topics, different positions, different words for the same idea
(*happiness*, *eudaimonia*, *utility*): a natural testbed for cross-document questions.

## Methodology

### 1. PDF conversion

**Goal:** get reliable text out of any PDF, page by page.

A PDF page is either **written text** (you can select it with the mouse) or a **photo of a page**
(a scan: for the computer it is just an image). Many PDFs mix both, and old scans often carry
a hidden, low-quality text layer. Every later step is built on this text, so it has to be right.

**How it works.** For every page:

1. **Read** the text already in the page, if any.
2. **Score** it: the share of words that are real words of the language (via `wordfreq`).
3. **Decide**:
   - good text → keep the page as it is;
   - no text and the page is one big image → it is a scan → **OCR**;
   - text with a low score → **OCR**, but keep the original if OCR does not score better.
4. **Rebuild** OCR pages as *page image + invisible text*: they look the same, but their text is now readable.
5. **Re-read and re-score** everything, and write a report.

**Tools:** `PyMuPDF` to read, render and write PDFs; `RapidOCR` (PaddleOCR models) for OCR.
Both install with `pip` alone, no system dependencies.

**Output:** for each document, a converted PDF with the same pages, and one line per page with
its text, its origin (`native` / `ocr`) and its score.

**Design choices**

- *Decide per page, not per document*: real PDFs are often mixed.
- *Deterministic OCR, not an LLM*: a vision LLM may silently "fix" or invent words;
  answers must be grounded in the exact text, so a visible error beats an invisible one.
- *Never make a page worse*: if OCR does not beat the existing text, the original is kept.

**Results**

| Document | Pages | Kept as is | OCR | Low score, original kept | Mean score |
|---|---|---|---|---|---|
| Aristotle | 274 | 267 | 1 | 6 | 0.98 |
| Nietzsche | 229 | 228 | – | 1 | 0.99 |
| Mill | 121 | 120 | 1 (scanned page) | – | 1.00 |
| Kant | 53 | 53 | – | – | 1.00 |

All four PDFs are born-digital: OCR was almost never needed, and the pipeline recognised it on its own.

**Known limitation:** the score measures how *English* a page is, not how *correct* it is.
A page of Nietzsche quoting Tertullian in Latin scores 0.78 while being perfectly fine.
OCR reads the same Latin and scores the same, so the original is kept:
OCR replaces existing text only if it scores clearly better (+0.05).

### 2. Document structure

**Goal:** split each book along the author's own structure (books, chapters, sections)
instead of cutting it every N words, so that every piece is a complete unit of meaning.

**How it works.** [PageIndex](https://github.com/VectifyAI/PageIndex) builds the table of
contents of each PDF as a tree: every section has a title, a first and last page, a short
LLM summary and its sub-sections. The smallest sections (the *leaves*) will become our chunks.
Documents are processed one at a time.

1. **Flash mode first:** the tree is built from the PDF bookmarks and the page layout
   (font sizes, bold, numbering), with no LLM; then the LLM writes the summaries.
2. **Standard mode as fallback:** if flash finds no real structure, the LLM reads the
   pages and reconstructs the headings.
3. **Summary check:** small models sometimes answer in JSON, cut the answer half-way or refuse.
   Every summary is cleaned into plain text; unusable ones are redone once, for that section only.

**LLM:** the model is set in `.env` (`LLM_MODEL`), never in the code. Two options:

- **Local (default):** an open-source model (Gemma 3) running with [Ollama](https://ollama.com):
  no API key, no cost, nothing leaves the machine.
- **Cloud:** any provider supported by [LiteLLM](https://docs.litellm.ai/docs/providers)
  (Gemini, OpenAI, Anthropic…): set `LLM_MODEL` and the provider's API key in `.env`.

**Design choices**

- *Keep the author's structure untouched*: PageIndex can merge and split sections to optimise
  its own search; we switch that off and size the chunks ourselves in the next step, with measured criteria.
- *Limited parallelism*: `LLM_CONCURRENCY` in `.env` caps simultaneous LLM calls
  (low for a local model, higher for a cloud one).
- *Built once, cached*: trees are saved and reused.

## Getting started

Requirements: Python 3.11+ and an LLM, either

- local: [Ollama](https://ollama.com) with the model pulled (`ollama pull gemma3:4b`), or
- cloud: an API key of any LiteLLM provider, e.g. Gemini.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env              # choose the LLM here: LLM_MODEL (+ API key if cloud)

# put the PDFs in data/raw/, then
python -m xdocqa.pdf_conversion   # step 1
python -m xdocqa.structure        # step 2
```