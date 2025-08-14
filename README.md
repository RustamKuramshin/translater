# Translater (LangChain + OpenAI)

## Translate PDFs to Markdown in your target language with AI

A Python-based document and text translator powered by LangChain and OpenAI.
This tool automatically extracts text from PDFs, intelligently splits it into chunks, translates it into your chosen target language while preserving structure, and returns clean, well-formatted Markdown.
It supports session resumption, page range selection, customizable LLM parameters, logging, and dry-run mode.

* **Quick start** – works out of the box (just set your `OPENAI_API_KEY`)
* **Accurate translation** – GPT-4o / GPT-4o-mini via `langchain-openai`
* **Output format** – clean Markdown with page headers
* **Scalable** – chunked translation with size control
* **Reliable** – progress stored in SQLite for safe resumption

---

## Table of Contents

* What This Project Does
* Features
* Architecture & How It Works
* Requirements
* Installation
* Quick Start
* Configuration (`config.yaml`)
* Usage Examples
* Environment Variables & Security
* Troubleshooting
* FAQ
* Roadmap
* Contributing

---

## What This Project Does

This tool translates English PDF documents into your target language while preserving the original structure — including headings, lists, code blocks, tables, and formatting.
The result is delivered in Markdown, making it ideal for publications, documentation, and technical articles.

Under the hood, it uses **LangChain** for loading and splitting text, and **OpenAI** (via `langchain-openai`) for high-quality machine translation. Large files are split into overlapping chunks to reduce sentence-boundary translation errors.

---

## Features

* Translate **PDF → Markdown** in a configurable target language — no summaries, no omissions
* Preserve formatting:

  * headings, lists, numbered lists
  * inline formatting (bold, italic, `inline code`)
  * links and code blocks
  * simple tables (Markdown format)
* Customizable LLM parameters: model, temperature, top\_p, penalties, max\_tokens, timeout
* Manual control over chunk size and overlap
* Selectable page range: entire file, first N pages, or P–Q range
* Session resumption – progress stored in SQLite (`.translator_state.sqlite3`)
* **Dry-run mode** – test splitting without calling the LLM
* Detailed logging and LLM call timing

---

## Architecture & How It Works

1. **Load & split PDF** – `PyPDFLoader` (`langchain-community`) + `RecursiveCharacterTextSplitter`.
2. **Normalize page ranges** – user-friendly CLI syntax with safe filtering.
3. **Chunk-by-chunk translation** – generates system & user prompts for each chunk, then calls `ChatOpenAI`.
4. **Retries** – exponential backoff with a configurable number of attempts.
5. **Reassembly** – merges translated chunks into a single Markdown file with page separators.
6. **Reliability** – stores the status of each chunk in SQLite, allowing safe interruption and resumption.

**Main files:**

* `translater.py` – CLI and logic for loading/splitting/translating/assembling
* `config.yaml` – example configuration
* `run.sh` – simple runner with `OPENAI_API_KEY` check
* `requirements.txt` – dependencies
* `install.sh` – dependency installer

---

## Requirements

* Python **3.9+** (recommended: 3.10–3.12)
* OpenAI account and API key (`OPENAI_API_KEY`)
* Access to GPT-4o / GPT-4o-mini (or another compatible OpenAI model)

**Dependencies** (see `requirements.txt`):

* `langchain`, `langchain-openai`, `langchain-community`
* `pypdf`, `PyYAML`

---

## Installation

**Option 1 – Script:**

```bash
./install.sh
```

**Option 2 – Manual:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Start

1. **Export your OpenAI API key**

```bash
export OPENAI_API_KEY=sk-...
```

2. **Run translation (example)**

```bash
./run.sh --input hands-high-performance-spring-5.pdf --output out.md --config config.yaml --page-end 10
```

Or directly via Python:

```bash
python3 translater.py --input input.pdf --output output.ru.md --config config.yaml --target-lang ru
```

Note: `run.sh` tries to use the installed CLI command `pdf-translater` (if in PATH), otherwise runs `translater.py`. The main method in this repo is direct execution of `translater.py`.

---

## Configuration (`config.yaml`)

Example in repo root. Structure:

```yaml
llm:
  model: gpt-4o-2024-11-20
  temperature: 0.5
  max_tokens: null
  top_p: null
  frequency_penalty: null
  presence_penalty: null
  request_timeout: 120
split:
  chunk_size: 2000
  chunk_overlap: 200
runtime:
  target_lang: ru           # target language (code or name)
  output: null              # default output: <input>.<lang>.md
  db_path: ./.translator_state.sqlite3
  resume: true
  max_retries: 3
  retry_backoff: 2.0
  dry_run: false
  page_start: null          # 1-based inclusive
  page_end: null            # if only page_end set, translates first N pages
```

Any CLI parameter overrides the configuration file.

---

## Usage Examples

* Translate entire document:

  ```bash
  python3 translater.py --input doc.pdf
  ```
* Translate to Spanish with default-named output file:

  ```bash
  python3 translater.py --input doc.pdf --target-lang es
  # output defaults to doc.es.md
  ```
* Specify output file explicitly:

  ```bash
  python3 translater.py --input doc.pdf --output doc.ru.md
  ```
* First 5 pages only:

  ```bash
  python3 translater.py --input doc.pdf --page-end 5
  ```
* Pages 10–20:

  ```bash
  python3 translater.py --input doc.pdf --page-start 10 --page-end 20
  ```
* Start fresh (ignore saved progress):

  ```bash
  python3 translater.py --input doc.pdf --no-resume
  ```
* Dry-run (no LLM calls):

  ```bash
  python3 translater.py --input doc.pdf --dry-run
  ```
* Fine-tune LLM parameters and language:

  ```bash
  python3 translater.py --input doc.pdf --model gpt-4o-mini --temperature 0.2 --request-timeout 180 --target-lang fr
  ```

---

## Environment Variables & Security

* `OPENAI_API_KEY` – **Required**. Must be set before running.
* Store your API key securely (e.g., in `.bashrc`, `.zshrc`, or a secret manager). **Never** commit it to the repository.
* Translating large documents can incur costs (token-based billing). Control costs by lowering `chunk_size`, limiting `page_end`, using `temperature` \~0.2–0.5, or choosing cheaper models.

---

## Troubleshooting

* **Error:** `OPENAI_API_KEY is not set` → export your key.
* **Missing dependency:** install via `./install.sh` or `pip install -r requirements.txt`.
* **PDF unreadable / empty output:** ensure the PDF contains selectable text (no OCR is performed). For scanned PDFs, run OCR first.
* **Interrupted translation:** safe to resume – just restart with the same parameters (default `--resume` enabled).
* **Slow / timeouts:** increase `--request-timeout`, reduce `chunk_size`, check network and OpenAI usage limits.

---

## FAQ

* **Other formats?** Currently focused on PDFs. Non-PDF files may not work correctly.
* **Change model?** Yes – set `--model` or edit `llm.model` in `config.yaml`. Compatible OpenAI models supported.
* **Original layout preserved?** Semantic structure (headings, lists, code) is kept in Markdown, but exact visual layout is not reproduced.
* **File size limits?** Yes – depends on model and chunking settings. For large PDFs, use `page_end` and/or smaller `chunk_size`.

---

## Roadmap

* Support for DOCX/HTML
* Configurable system prompt
* Cloud storage integration for results
* PyPI packaging with CLI command (`pdf-translater`)

---

## Contributing

* Suggestions and issues welcome.
* PRs with improved translation quality, new features, and tests are encouraged.
