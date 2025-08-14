#!/usr/bin/env python3
"""
AI PDF Translator CLI

Translates English PDF documents to Russian using OpenAI Chat Completions via LangChain.

Key features:
- CLI with config file support (YAML or JSON). CLI args override config.
- Reads OpenAI API key from OPENAI_API_KEY environment variable.
- Splits PDF into logical chunks and translates iteratively.
- Produces Markdown output, attempting to preserve structure.
- High-quality translation system prompt avoiding summarization or omissions.
- Logging of progress to terminal.
- Token-efficient chunking parameters.
- Resumable processing via local SQLite DB (portable single file).

Usage:
  python translater.py --input path/to/file.pdf --output out.md [--config config.yaml]

Dependencies (install as needed):
  - langchain
  - langchain-openai
  - langchain-community
  - pypdf (for PyPDFLoader)
  - PyYAML (optional for YAML config)

"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import sqlite3
import sys
import textwrap
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Try importing YAML support if available
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore

# LangChain imports
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
except Exception as e:
    # Lazy error if user tries to run without deps
    PyPDFLoader = None  # type: ignore
    RecursiveCharacterTextSplitter = None  # type: ignore
    HumanMessage = None  # type: ignore
    SystemMessage = None  # type: ignore
    ChatOpenAI = None  # type: ignore

APP_NAME = "pdf_translator"
DEFAULT_DB_FILE = ".translator_state.sqlite3"

# ---------------------- Config ----------------------
@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: Optional[int] = None  # None lets OpenAI decide
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    request_timeout: int = 120
    # You can extend with more OpenAI params if needed


@dataclass
class SplitConfig:
    chunk_size: int = 2000  # characters, aim for token efficiency
    chunk_overlap: int = 200


@dataclass
class RuntimeConfig:
    input: str = ""
    output: Optional[str] = None
    config: Optional[str] = None
    db_path: str = DEFAULT_DB_FILE
    resume: bool = True
    max_retries: int = 3
    retry_backoff: float = 2.0
    dry_run: bool = False
    # Page range (1-based inclusive). If only page_end is set, translates first N pages.
    page_start: Optional[int] = None
    page_end: Optional[int] = None


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @staticmethod
    def from_files_and_args(config_path: Optional[str], args: argparse.Namespace) -> "AppConfig":
        cfg = AppConfig()
        # Load file config if provided
        if config_path:
            file_cfg = load_config_file(Path(config_path))
            cfg = merge_config(cfg, file_cfg)
        # Override with CLI args
        cfg.runtime.input = args.input or cfg.runtime.input
        if args.output:
            cfg.runtime.output = args.output
        if args.db_path:
            cfg.runtime.db_path = args.db_path
        if args.no_resume:
            cfg.runtime.resume = False
        if args.resume:
            cfg.runtime.resume = True
        if args.max_retries is not None:
            cfg.runtime.max_retries = args.max_retries
        if args.retry_backoff is not None:
            cfg.runtime.retry_backoff = args.retry_backoff
        if args.dry_run:
            cfg.runtime.dry_run = True
        # Page range overrides
        if getattr(args, 'page_start', None) is not None:
            cfg.runtime.page_start = args.page_start
        if getattr(args, 'page_end', None) is not None:
            cfg.runtime.page_end = args.page_end
        # LLM overrides
        if args.model:
            cfg.llm.model = args.model
        if args.temperature is not None:
            cfg.llm.temperature = args.temperature
        if args.max_tokens is not None:
            cfg.llm.max_tokens = args.max_tokens
        if args.top_p is not None:
            cfg.llm.top_p = args.top_p
        if args.frequency_penalty is not None:
            cfg.llm.frequency_penalty = args.frequency_penalty
        if args.presence_penalty is not None:
            cfg.llm.presence_penalty = args.presence_penalty
        if args.request_timeout is not None:
            cfg.llm.request_timeout = args.request_timeout
        # Split overrides
        if args.chunk_size is not None:
            cfg.split.chunk_size = args.chunk_size
        if args.chunk_overlap is not None:
            cfg.split.chunk_overlap = args.chunk_overlap
        return cfg


# ---------------------- Logging ----------------------
logger = logging.getLogger(APP_NAME)


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)


# ---------------------- Utilities ----------------------

def _parse_minimal_yaml(text: str) -> Dict[str, Any]:
    """
    Minimal YAML parser supporting a subset sufficient for our config files:
    - Nested mappings by indentation (spaces only)
    - key: value pairs with scalars (str, int, float, bool, null)
    - key: on a line starts a nested mapping
    - Comments starting with # and blank lines are ignored
    - No sequences/lists support
    This is a fallback used only when PyYAML is not installed.
    """
    def parse_scalar(val: str):
        v = val.strip()
        if v == "" or v.lower() == "null":
            return None
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        # Try int/float
        try:
            if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
                return int(v)
            return float(v)
        except Exception:
            pass
        # Strip quotes if present
        if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
            return v[1:-1]
        return v

    lines = text.splitlines()
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(0, root)]

    for raw in lines:
        # Remove comments
        line = raw.split('#', 1)[0].rstrip('\n')
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(' '))
        # Ensure spaces only
        if '\t' in line:
            raise ValueError("Tabs are not supported in YAML indentation")
        # Adjust stack according to indentation
        while stack and indent < stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError("Invalid indentation structure in YAML")
        current = stack[-1][1]
        stripped = line.strip()
        if stripped.endswith(":"):
            key = stripped[:-1].strip()
            new_map: Dict[str, Any] = {}
            current[key] = new_map
            stack.append((indent + 2, new_map))
        else:
            if ":" not in stripped:
                raise ValueError(f"Invalid line in YAML: {raw}")
            key, val = stripped.split(":", 1)
            key = key.strip()
            current[key] = parse_scalar(val)
    return root


def load_config_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            with path.open("r", encoding="utf-8") as f:
                text = f.read()
            if yaml is not None:
                return yaml.safe_load(text) or {}
            # Fallback minimal parser
            return _parse_minimal_yaml(text)
        elif path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        else:
            raise ValueError("Unsupported config format. Use .yaml/.yml or .json")
    except Exception as e:
        raise RuntimeError(f"Failed to parse config file {path}: {e}")


def deep_update_dict(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = deep_update_dict(d[k], v)
        else:
            d[k] = v
    return d


def merge_config(base: AppConfig, override: Dict[str, Any]) -> AppConfig:
    # Convert dataclass to nested dict, update, then back
    d = {
        "llm": asdict(base.llm),
        "split": asdict(base.split),
        "runtime": asdict(base.runtime),
    }
    deep_update_dict(d, override)
    # Build new AppConfig
    return AppConfig(
        llm=LLMConfig(**d.get("llm", {})),
        split=SplitConfig(**d.get("split", {})),
        runtime=RuntimeConfig(**d.get("runtime", {})),
    )


def file_content_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _format_eta(seconds: float) -> str:
    # Format seconds into H:MM:SS
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def print_progress(done: int, total: int, start_time: float, prefix: str = "Progress") -> None:
    """
    Render a simple, dependency-free progress bar to stdout on a single line.
    Shows percentage, bar, counts, and ETA.
    """
    total = max(total, 1)
    done = min(max(done, 0), total)
    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = total - done
    eta = (remaining / rate) if rate > 0 else 0.0
    width = 30
    pct = done / total
    filled = int(width * pct)
    bar = "█" * filled + "-" * (width - filled)
    msg = f"\r{prefix} |{bar}| {pct*100:5.1f}% ({done}/{total}) ETA { _format_eta(eta) }"
    try:
        sys.stdout.write(msg)
        sys.stdout.flush()
    except Exception:
        # Fallback to logging if stdout write fails for some reason
        logger.info(f"{prefix}: {done}/{total} ({pct*100:.1f}%)")


# ---------------------- DB (Resumable) ----------------------
class StateDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS translations (
                file_hash TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                total_chunks INTEGER NOT NULL,
                page INTEGER,
                status TEXT NOT NULL,
                translated_md TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (file_hash, chunk_id)
            )
            """
        )
        self.conn.commit()

    def get_completed_chunk_ids(self, file_hash: str) -> List[int]:
        cur = self.conn.execute(
            "SELECT chunk_id FROM translations WHERE file_hash=? AND status='done' ORDER BY chunk_id",
            (file_hash,),
        )
        return [r[0] for r in cur.fetchall()]

    def upsert_chunk(self, file_hash: str, chunk_id: int, total_chunks: int, page: Optional[int], status: str, translated_md: Optional[str]) -> None:
        now = dt.datetime.utcnow().isoformat()
        self.conn.execute(
            """
            INSERT INTO translations(file_hash, chunk_id, total_chunks, page, status, translated_md, created_at, updated_at)
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(file_hash, chunk_id) DO UPDATE SET
                total_chunks=excluded.total_chunks,
                page=excluded.page,
                status=excluded.status,
                translated_md=excluded.translated_md,
                updated_at=excluded.updated_at
            """,
            (file_hash, chunk_id, total_chunks, page, status, translated_md, now, now),
        )
        self.conn.commit()

    def fetch_all_translations(self, file_hash: str) -> List[Tuple[int, Optional[int], str]]:
        cur = self.conn.execute(
            "SELECT chunk_id, page, translated_md FROM translations WHERE file_hash=? AND status='done' ORDER BY chunk_id",
            (file_hash,),
        )
        return [(r[0], r[1], r[2]) for r in cur.fetchall()]

    def close(self):
        self.conn.close()


# ---------------------- PDF Loading & Chunking ----------------------
@dataclass
class Chunk:
    chunk_id: int
    page: Optional[int]
    text: str


def load_and_chunk_pdf(pdf_path: Path, split_cfg: SplitConfig, page_start: Optional[int] = None, page_end: Optional[int] = None) -> List[Chunk]:
    if PyPDFLoader is None or RecursiveCharacterTextSplitter is None:
        raise RuntimeError("Required dependencies missing. Please install langchain, langchain-community, and pypdf.")

    loader = PyPDFLoader(str(pdf_path))

    # Normalize page range (1-based inclusive to 0-based inclusive bounds)
    start0: Optional[int] = None
    end0: Optional[int] = None
    if page_start is not None and page_start <= 0:
        raise ValueError("--page-start must be >= 1")
    if page_end is not None and page_end <= 0:
        raise ValueError("--page-end must be >= 1")

    if page_start is None and page_end is not None:
        # First N pages: 1..page_end
        start0, end0 = 0, page_end - 1
    elif page_start is not None and page_end is None:
        # From start to the end
        start0, end0 = page_start - 1, None
    elif page_start is not None and page_end is not None:
        # Inclusive range
        s0, e0 = page_start - 1, page_end - 1
        if s0 > e0:
            logger.warning(f"Page range start ({page_start}) > end ({page_end}), swapping.")
            s0, e0 = e0, s0
        start0, end0 = s0, e0
    else:
        start0, end0 = None, None

    # Heuristic separators to better preserve structure
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=split_cfg.chunk_size,
        chunk_overlap=split_cfg.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: List[Chunk] = []
    chunk_id = 0

    # Prefer lazy loading to avoid parsing the entire PDF when only a range is needed
    lazy = getattr(loader, "lazy_load", None)
    if callable(lazy):
        docs_iter = lazy()
        for d in docs_iter:
            page = d.metadata.get("page") if isinstance(d.metadata, dict) else None
            # Filter by page range if bounds specified
            if page is not None:
                if start0 is not None and page < start0:
                    continue
                if end0 is not None and page > end0:
                    # We've gone past the end of the requested range; stop iterating further pages
                    break
            parts = splitter.split_text(d.page_content)
            for p in parts:
                text = p.strip()
                if not text:
                    continue
                chunks.append(Chunk(chunk_id=chunk_id, page=page, text=text))
                chunk_id += 1
    else:
        # Fallback: load all pages (may be slower for large PDFs)
        docs = loader.load()
        for d in docs:
            page = d.metadata.get("page") if isinstance(d.metadata, dict) else None
            # Filter by page range if bounds specified
            if page is not None:
                if start0 is not None and page < start0:
                    continue
                if end0 is not None and page > end0:
                    continue
            parts = splitter.split_text(d.page_content)
            for p in parts:
                text = p.strip()
                if not text:
                    continue
                chunks.append(Chunk(chunk_id=chunk_id, page=page, text=text))
                chunk_id += 1

    return chunks


# ---------------------- LLM Prompt & Translation ----------------------
SYSTEM_PROMPT = (
    "You are a professional English-to-Russian translator. Your goal is to translate technical and general texts with high fidelity, preserving the meaning, tone, and structure.\n"
    "Rules:\n"
    "- Translate from English to Russian. Do not summarize or omit any content. Do not add extra explanations.\n"
    "- Preserve lists, numbered steps, headings, and code blocks. Convert structure to clean Markdown.\n"
    "- Keep tables readable in Markdown if possible.\n"
    "- Preserve inline formatting: bold (**), italics (*), inline code (`), links [text](url).\n"
    "- Keep numbers, units, dates consistent; localize where appropriate but do not distort meaning.\n"
    "- If a sentence is incomplete due to chunking, translate it naturally and continue; do not repeat previous content.\n"
    "Output: Return only the translated text in Russian as valid Markdown, without any preface or notes."
)


def ensure_openai_env():
    if os.getenv("OPENAI_API_KEY") is None:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")


def build_llm(cfg: LLMConfig) -> ChatOpenAI:
    ensure_openai_env()
    if ChatOpenAI is None:
        raise RuntimeError("Missing dependency: langchain-openai. Install it to proceed.")
    # Build ChatOpenAI with provided parameters
    # Note: Some parameters may be None and will be ignored by client
    client = ChatOpenAI(
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        top_p=cfg.top_p,
        frequency_penalty=cfg.frequency_penalty,
        presence_penalty=cfg.presence_penalty,
        timeout=cfg.request_timeout,
    )
    return client


def translate_chunk(llm: ChatOpenAI, text: str, max_retries: int, backoff: float) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Translate the following content to Russian. Return only Markdown.\n\n{text}"),
            ]
            logger.info(f"LLM request start (attempt {attempt}/{max_retries}) | input chars: {len(text)}")
            t0 = time.perf_counter()
            resp = llm.invoke(messages)
            dt_s = time.perf_counter() - t0
            # Try to log token usage if available
            usage = None
            try:
                # LangChain AIMessage may have usage in response_metadata or .usage_metadata
                meta = getattr(resp, 'response_metadata', None) or {}
                usage = meta.get('token_usage') or getattr(resp, 'usage_metadata', None)
            except Exception:
                usage = None
            if usage:
                logger.info(f"LLM request success in {dt_s:.2f}s | usage: {usage}")
            else:
                logger.info(f"LLM request success in {dt_s:.2f}s")
            # resp is an AIMessage with .content
            return str(resp.content).strip()
        except Exception as e:
            dt_s = time.perf_counter() - t0 if 't0' in locals() else 0.0
            last_err = e
            wait = backoff * (2 ** (attempt - 1))
            logger.warning(f"LLM request error after {dt_s:.2f}s (attempt {attempt}/{max_retries}): {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    assert last_err is not None
    # Log final failure
    logger.error(f"LLM request failed after {max_retries} attempts: {last_err}")
    raise last_err


# ---------------------- Assembly ----------------------

def assemble_markdown(chunks_data: List[Tuple[int, Optional[int], str]]) -> str:
    # chunks_data: list of (chunk_id, page, translated_md) sorted by chunk_id
    out_lines: List[str] = []
    current_page: Optional[int] = None
    for _, page, md in chunks_data:
        # Insert page header when page changes (approximate original formatting)
        if page is not None and page != current_page:
            if out_lines:
                out_lines.append("")
            out_lines.append(f"\n\n---\n\n# Страница {page + 1}\n")
            current_page = page
        out_lines.append(md.rstrip())
        out_lines.append("")
    return "\n".join(out_lines).strip() + "\n"


# ---------------------- CLI ----------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdf-translator",
        description="Translate English PDF to Russian Markdown using OpenAI via LangChain.",
    )
    p.add_argument("--input", required=True, help="Path to input PDF file")
    p.add_argument("--output", help="Path to output Markdown file. Default: <input>.ru.md")
    p.add_argument("--config", help="Path to config file (YAML or JSON)")

    # LLM params
    p.add_argument("--model", help="OpenAI model name (e.g., gpt-4o-mini)")
    p.add_argument("--temperature", type=float, help="Sampling temperature")
    p.add_argument("--max-tokens", dest="max_tokens", type=int, help="Max tokens for response")
    p.add_argument("--top-p", dest="top_p", type=float, help="Top-p nucleus sampling")
    p.add_argument("--frequency-penalty", dest="frequency_penalty", type=float, help="Frequency penalty")
    p.add_argument("--presence-penalty", dest="presence_penalty", type=float, help="Presence penalty")
    p.add_argument("--request-timeout", dest="request_timeout", type=int, help="Request timeout (s)")

    # Splitting
    p.add_argument("--chunk-size", dest="chunk_size", type=int, help="Chunk size in characters")
    p.add_argument("--chunk-overlap", dest="chunk_overlap", type=int, help="Chunk overlap in characters")

    # Page range (1-based inclusive)
    p.add_argument("--page-start", dest="page_start", type=int, help="Start page number (1-based). If omitted and --page-end is set, translates the first N pages.")
    p.add_argument("--page-end", dest="page_end", type=int, help="End page number (1-based, inclusive). If only this is set, it means translate the first N pages.")

    # Runtime & resume
    p.add_argument("--db-path", help=f"Path to SQLite state DB (default: {DEFAULT_DB_FILE})")
    res_group = p.add_mutually_exclusive_group()
    res_group.add_argument("--resume", action="store_true", help="Resume from previous progress (default)")
    res_group.add_argument("--no-resume", action="store_true", help="Do not resume; re-translate all chunks")
    p.add_argument("--max-retries", type=int, help="Max retries for LLM calls")
    p.add_argument("--retry-backoff", type=float, help="Initial backoff (s) for retries")
    p.add_argument("--dry-run", action="store_true", help="Load and split PDF, but do not call LLM")

    return p


# ---------------------- Main Flow ----------------------

def main(argv: Optional[List[str]] = None) -> int:
    setup_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Build config
    cfg = AppConfig.from_files_and_args(args.config, args)

    input_path = Path(cfg.runtime.input).expanduser().resolve()
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 2
    if input_path.suffix.lower() != ".pdf":
        logger.warning("Input is not a .pdf file. Proceeding anyway, but PDF loader expects PDF.")

    output_path = Path(cfg.runtime.output) if cfg.runtime.output else input_path.with_suffix(".ru.md")
    output_path = output_path.expanduser().resolve()

    db_path = Path(cfg.runtime.db_path).expanduser().resolve()
    db = StateDB(db_path)

    # Load & split
    logger.info("Loading and chunking PDF...")
    try:
        chunks = load_and_chunk_pdf(input_path, cfg.split, cfg.runtime.page_start, cfg.runtime.page_end)
    except Exception as e:
        logger.exception(f"Failed to load/split PDF: {e}")
        return 3

    total_chunks = len(chunks)
    if cfg.runtime.page_start is not None or cfg.runtime.page_end is not None:
        logger.info(f"Prepared {total_chunks} chunks for translation for selected pages (start={cfg.runtime.page_start}, end={cfg.runtime.page_end}).")
    else:
        logger.info(f"Prepared {total_chunks} chunks for translation.")

    # Compute job id (file hash + page range) for state separation
    base_hash = file_content_sha256(input_path)
    if cfg.runtime.page_start is not None or cfg.runtime.page_end is not None:
        page_spec = f"pages={cfg.runtime.page_start if cfg.runtime.page_start is not None else ''}:{cfg.runtime.page_end if cfg.runtime.page_end is not None else ''}"
        file_hash = hashlib.sha256((base_hash + '|' + page_spec).encode('utf-8')).hexdigest()
    else:
        file_hash = base_hash

    # Prepare LLM
    if not cfg.runtime.dry_run:
        try:
            llm = build_llm(cfg.llm)
        except Exception as e:
            logger.exception(f"Failed to initialize LLM: {e}")
            return 4
    else:
        llm = None  # type: ignore

    # Determine which chunks to process
    completed_ids = set(db.get_completed_chunk_ids(file_hash)) if cfg.runtime.resume else set()
    if completed_ids:
        logger.info(f"Resuming: {len(completed_ids)} chunks already completed. Will skip them.")

    # Initialize progress tracking
    processed_done = len(completed_ids)
    progress_start_time = time.time()
    print_progress(processed_done, total_chunks, progress_start_time)

    # Process chunks
    start_time = time.time()
    for ch in chunks:
        if ch.chunk_id in completed_ids and cfg.runtime.resume:
            continue

        logger.info(f"Translating chunk {ch.chunk_id + 1}/{total_chunks} (page {'' if ch.page is None else ch.page + 1})...")
        db.upsert_chunk(file_hash, ch.chunk_id, total_chunks, ch.page, "in_progress", None)

        try:
            if cfg.runtime.dry_run:
                translated = f"[DRY-RUN] {ch.text[:100]}..."
            else:
                translated = translate_chunk(llm, ch.text, cfg.runtime.max_retries, cfg.runtime.retry_backoff)
            db.upsert_chunk(file_hash, ch.chunk_id, total_chunks, ch.page, "done", translated)
            # Update progress after successful translation
            processed_done += 1
            print_progress(processed_done, total_chunks, progress_start_time)
        except Exception as e:
            logger.exception(f"Failed translating chunk {ch.chunk_id}: {e}")
            db.upsert_chunk(file_hash, ch.chunk_id, total_chunks, ch.page, "error", None)
            # On error, stop processing further to allow resume
            break

        # After each chunk, assemble current output for safety
        try:
            translated_chunks = db.fetch_all_translations(file_hash)
            md = assemble_markdown(translated_chunks)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                f.write(md)
        except Exception as e:
            logger.warning(f"Failed to write interim output: {e}")

    # Finish progress bar line
    try:
        sys.stdout.write("\n")
        sys.stdout.flush()
    except Exception:
        pass
    elapsed = time.time() - start_time

    # Final assembly
    translated_chunks = db.fetch_all_translations(file_hash)
    if translated_chunks:
        md = assemble_markdown(translated_chunks)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                f.write(md)
        except Exception as e:
            logger.exception(f"Failed to write final output: {e}")
            return 5

    db.close()

    done_count = len(translated_chunks)
    if done_count == total_chunks:
        logger.info(f"Done. Translated {done_count}/{total_chunks} chunks in {elapsed:.1f}s. Output: {output_path}")
        return 0
    else:
        logger.warning(
            f"Stopped early. Translated {done_count}/{total_chunks} chunks in {elapsed:.1f}s. "
            f"You can resume with --resume (default)."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
