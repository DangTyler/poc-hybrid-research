import os
import time
import hashlib
import re
from typing import List, Tuple
from io import BytesIO

import requests
from dotenv import load_dotenv
from xml.etree import ElementTree as ET

# Import ingestion utilities with a safe fallback
try:
    from .papers_ingest import load_to_neo4j, build_graph_with_langchain, load_to_pinecone
except Exception:  # pragma: no cover
    from ingest.papers_ingest import load_to_neo4j, build_graph_with_langchain, load_to_pinecone


def _read_all_bytes(uploaded_file) -> bytes:
    """Read all bytes from Streamlit's UploadedFile safely.
    Prefers getvalue(); falls back to read(). Returns empty bytes if unavailable.
    """
    if uploaded_file is None:
        return b""
    try:
        data = uploaded_file.getvalue()
        if data:
            return data
    except Exception:
        pass
    try:
        return uploaded_file.read()
    except Exception:
        return b""


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pypdf, then fall back to pdfminer.six.
    Returns an empty string if nothing can be extracted.
    """
    if not pdf_bytes:
        return ""

    # First try: pypdf
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(BytesIO(pdf_bytes))
        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")
        text = "\n".join(parts).strip()
        if text:
            return text
    except Exception:
        pass

    # Fallback: pdfminer.six
    try:
        from pdfminer.high_level import extract_text  # type: ignore
        text = extract_text(BytesIO(pdf_bytes)) or ""
        return text.strip()
    except Exception:
        return ""


def _strip_ext(name: str) -> str:
    base = os.path.basename(name or "document")
    if "." in base:
        return ".".join(base.split(".")[:-1])
    return base


_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)


def _arxiv_title_from_filename(name: str) -> str:
    m = _ARXIV_ID_RE.search(name or "")
    if not m:
        return ""
    arxiv_id = m.group(1)
    try:
        url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200 and resp.text:
            root = ET.fromstring(resp.text)
            # The first entry/title
            ns = {"a": "http://www.w3.org/2005/Atom"}
            entry = root.find("a:entry", ns)
            if entry is not None:
                title_el = entry.find("a:title", ns)
                if title_el is not None and title_el.text:
                    return title_el.text.strip()
    except Exception:
        return ""
    return ""


_SKIP_PREFIXES = (
    "arxiv", "journal", "proceedings", "ieee", "acm", "springer", "Elsevier",
)
_SKIP_TOKENS = ("abstract", "introduction", "figure", "table", "doi:")


def _guess_title_from_text(text: str) -> str:
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln and ln.strip()]
    # Consider first 40 lines; choose plausible one
    for ln in lines[:40]:
        low = ln.lower()
        if any(low.startswith(p) for p in _SKIP_PREFIXES):
            continue
        if any(tok in low for tok in _SKIP_TOKENS):
            continue
        if len(ln) < 5 or len(ln) > 180:
            continue
        # Skip numeric-only / page numbers
        if re.fullmatch(r"[0-9ivx\.\-]+", low):
            continue
        return ln
    return lines[0] if lines else ""


def _extract_title_from_pdf(pdf_bytes: bytes, fallback_name: str) -> str:
    # 1) Try arXiv from filename
    arxiv_title = _arxiv_title_from_filename(fallback_name)
    if arxiv_title:
        return arxiv_title
    # 2) Try PDF metadata title
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(BytesIO(pdf_bytes))
        meta_title = getattr(reader, "metadata", None)
        if meta_title and getattr(meta_title, "title", None):
            t = str(meta_title.title).strip()
            if t:
                return t
    except Exception:
        pass
    # 3) Guess from text
    text = _extract_text_from_pdf(pdf_bytes)
    guess = _guess_title_from_text(text)
    return guess or _strip_ext(fallback_name)


def _read_file(uploaded_file) -> Tuple[str, str, bytes]:
    """Return (title, content, raw_bytes) for an UploadedFile.
    - PDFs: title from arXiv or metadata/first heading; content from text extraction
    - Text files: title from filename stem; content is UTF-8 text
    - Others: best-effort UTF-8 decode
    """
    raw = _read_all_bytes(uploaded_file)
    name = getattr(uploaded_file, "name", "document")
    lower = (name or "document").lower()

    if lower.endswith((".txt", ".md", ".csv")):
        try:
            content = raw.decode("utf-8", errors="ignore")
        except Exception:
            content = ""
        return _strip_ext(name), content, raw

    if lower.endswith(".pdf"):
        title = _extract_title_from_pdf(raw, name)
        content = _extract_text_from_pdf(raw)
        return title or _strip_ext(name), content, raw

    # Fallback
    try:
        content = raw.decode("utf-8", errors="ignore")
    except Exception:
        content = ""
    return _strip_ext(name), content, raw


def ingest_uploaded_docs(files: List, dataset: str, run_builder: bool = True, run_pinecone: bool = True):
    """
    Ingest user-uploaded documents as generic Papers with inferred relationships.
    - Stable IDs computed from SHA1(file_bytes) to avoid duplicates on rerun
    - Realistic titles for PDFs via arXiv/metadata/heading
    - Optionally builds graph edges and embeddings
    """
    load_dotenv()

    if not files:
        return {"ingested": 0}

    papers = []
    for f in files:
        title, content, raw = _read_file(f)
        if not content or len(content.strip()) < 50:
            continue
        sha = hashlib.sha1(raw or b"\0").hexdigest()[:16]
        paper_id = f"{dataset}-{sha}"
        papers.append({
            "paperId": paper_id,
            "title": title,
            "abstract": content,
            "citationCount": 0,
            "references": [],
            "url": "",
        })

    if not papers:
        return {"ingested": 0}

    load_to_neo4j(papers, dataset=dataset)
    if run_builder:
        build_graph_with_langchain(papers, dataset=dataset)
    if run_pinecone:
        load_to_pinecone(papers)

    return {"ingested": len(papers)}

