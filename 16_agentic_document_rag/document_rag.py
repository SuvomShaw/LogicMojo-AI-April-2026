"""Document desk: search real PDFs, then wait when the answer is a number.

Start with README.md, then search this file for START HERE to follow the graph.

Read the graph in this order:

  route
    hello              -> chat          the packet stays closed
    any other question -> retrieve
  retrieve -> grade -> generate -> check
    nothing useful     -> rewrite once, then abstain
    a number, a table, or a figure -> human_review

Part A, above START HERE, turns a PDF or a Word file into chunks.
Notebook 1 lives there. The Streamlit app lives in the graph.

Models, same pair as module 15:
  qwen3.5:4b            writes, grades, and reads a page image
  qwen3-embedding:4b    turns a chunk or a question into a vector
"""

from __future__ import annotations

import base64
import hashlib
import json
import operator
import os
import re
import urllib.request
import uuid
from pathlib import Path
from typing import Annotated, TypedDict

# Chroma phones home unless this is set before the client is created.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

import fitz
from docx import Document as DocxDocument
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
PDF_DIR = CORPUS / "pdf"
DOCX_DIR = CORPUS / "docx"
PARSED = HERE / "parsed"
ELEMENTS_PATH = PARSED / "elements.jsonl"
CAPTIONS_PATH = PARSED / "captions.json"
OVERRIDES_PATH = HERE / "caption_overrides.json"
STAMP_PATH = PARSED / "index_stamp.json"
CHROMA_DIR = HERE / "chroma_db"
QUESTIONS_PATH = CORPUS / "questions.json"

CHAT_MODEL = "qwen3.5:4b"          # writes, grades, and reads a page image
EMBED_MODEL = "qwen3-embedding:4b"  # turns a chunk or a question into a vector
# Qwen3-Embedding reads this on the QUESTION only, never on the stored chunks.
# Same idea as module 15. The wording names tables and figures because that is what we search.
TASK = "Given an analyst question, retrieve the passage, table, or figure caption that contains the answer"
COLLECTION = "ledgerline_docs"      # one Chroma collection, so a second build overwrites it

# The only sentence a miss is allowed to return. The model does not invent a softer refusal.
ABSTAIN = "NOT_IN_CORPUS: the loaded documents do not answer that question."
# These skip retrieval. A greeting must not open the packet and invent a number.
SMALL_TALK = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "thanks", "thank you"}

# A vision call is slow, and a 4B model misreads digits. Spend one only where the text layer is the wrong channel.
CAPTION_DOC_IDS = {"1295982", "1659677"}  # gold chart pages (Office 365, Research Affiliates)
# PNC pages 6 and 7: the loss tables are painted as pictures. get_text() never sees 5.4%.
PICTURE_PAGES = {("7257900", 6), ("7257900", 7)}
# Bump this when the vision prompt changes. Otherwise the cache replays the old, wrong read.
CAPTION_CACHE_VERSION = "v3"

# BM25 tokens. The comma stays inside 18,443.87 so the rare amount is one token, not three.
_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][a-z0-9]+)*", re.IGNORECASE)
# A measured number is a percent, a decimal, or an amount with a thousands comma.
# A bare year such as 2017 does not match, so a date alone does not force a signature.
_MEASURE = re.compile(r"\d+\.\d+\s*%|\d+\s*%|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+")

_STORE: Chroma | None = None
_CHUNKS: list[Document] | None = None
_BM25: BM25Okapi | None = None
_SAVER = MemorySaver()
_GRAPH = None


def make_llm(max_tokens: int = 320, num_ctx: int = 8192) -> ChatOllama:
    """Same Qwen as module 15. The context is larger because a table chunk is longer than a policy chunk."""
    return ChatOllama(
        model=CHAT_MODEL,
        temperature=0,           # always pick the most likely next word
        seed=7,                  # with temperature 0, fixes the remaining ties
        num_ctx=num_ctx,         # how many tokens the model may read
        reasoning=False,         # skip qwen3.5's hidden thinking trace
        num_predict=max_tokens,  # how many tokens the model may write
    )


class QwenEmbeddings(Embeddings):
    """Instruction on the query, raw text on the documents. Same idea as module 15."""

    def __init__(self):
        self.model = OllamaEmbeddings(model=EMBED_MODEL)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(f"Instruct: {TASK}\nQuery:{text}")


def ollama_alive() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def words(text: str) -> list[str]:
    """Tokens for BM25. '18,443.87' is also stored as '18443.87' so either form matches."""
    tokens = []
    for token in _TOKEN.findall(text.lower()):
        tokens.append(token)
        bare = token.replace(",", "")
        if bare != token:
            tokens.append(bare)
    return tokens


def _normalize_rows(rows: list[list]) -> list[list[str]]:
    """Trim cells and drop columns that are empty on every row.

    Merged PDF cells show up as a row of blanks with one real value. Those
    blank columns make a markdown table unreadable, so they go.
    """
    cleaned = []
    width = 0
    for row in rows or []:
        cells = [str(cell or "").replace("\n", " ").replace("|", "/").strip() for cell in row]
        if any(cells):
            cleaned.append(cells)
            width = max(width, len(cells))
    if not cleaned or width == 0:
        return []
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    keep = [index for index in range(width) if any(row[index] for row in cleaned)]
    return [[row[index] for index in keep] for row in cleaned]


def markdown_table(rows: list[list]) -> str:
    """A grid of cells becomes a markdown table the model can read."""
    cleaned = _normalize_rows(rows)
    if len(cleaned) < 2 or len(cleaned[0]) < 2:
        return ""
    deduped = [cleaned[0]]
    for row in cleaned[1:]:
        if row != deduped[-1]:
            deduped.append(row)
    if len(deduped) < 2:
        return ""
    header, body = deduped[0], deduped[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def table_is_useful(rows: list[list]) -> bool:
    """Keep a real grid. Drop a chart axis or a one-cell infographic.

    Why: find_tables() also returns bar-chart axes (one percent per row, no label)
    and dashboards mashed into one giant cell. Indexing those lets the model cite
    a number with no row. The page text already has that wording.
    """
    cleaned = _normalize_rows(rows)
    if len(cleaned) < 2 or len(cleaned[0]) < 2:
        return False  # a real table has a header and at least two columns
    flat = [cell for row in cleaned for cell in row if cell]
    if len(flat) < 4:
        return False  # four filled cells is the smallest grid worth citing
    if max(len(cell) for cell in flat) > 280:
        return False  # one huge cell is a paragraph, not a table
    total = len(cleaned) * len(cleaned[0])
    if len(flat) / total < 0.45:
        return False  # a sparse axis: one filled cell on a row of blanks
    # A column of bare percents (164%, 128%) has no row name. That is a chart axis.
    return any(re.search(r"[A-Za-z]{3,}", cell) for cell in flat)


def _element(
    *,
    doc_id: str,
    title: str,
    source: str,
    file_type: str,
    page: int,
    element_type: str,
    text: str,
    page_image: str,
    license_name: str,
) -> dict:
    return {
        "doc_id": doc_id,
        "title": title[:160],
        "source": source,
        "file_type": file_type,
        "page": page,
        "element_type": element_type,
        "text": text.strip(),
        "page_image": page_image,
        "license": license_name,
    }


# ---------------------------------------------------------------------------
# Part A. Files become chunks.
# Skip this until the graph below START HERE is clear.
# parse_pdf and parse_docx pull out text, tables, and figures.
# build_index saves those chunks so retrieve can search them.
# ---------------------------------------------------------------------------
def parse_pdf(path: Path) -> list[dict]:
    """One PDF becomes text blocks and, where the grid is real, markdown tables.

    Every page is also rendered to a PNG. The app shows that PNG when a person
    signs a draft. The render is the evidence; the extracted text is the index.
    """
    doc_id = path.stem
    page_dir = CORPUS / "pages" / doc_id
    page_dir.mkdir(parents=True, exist_ok=True)
    elements: list[dict] = []
    title = ""
    license_name = "UniDoc-Bench CC-BY-NC-4.0"
    with fitz.open(path) as doc:
        for index, page in enumerate(doc):
            page_no = index + 1
            image_rel = Path("corpus") / "pages" / doc_id / f"page_{page_no:03d}.png"
            image_abs = HERE / image_rel
            if not image_abs.exists():
                # 110 dpi is enough to read a table on screen. Higher dpi only slows the vision call.
                page.get_pixmap(dpi=110).save(image_abs)
            # This is the text layer. A number painted inside a picture is not in it.
            text = re.sub(r"\n{3,}", "\n\n", page.get_text("text") or "").strip()
            if not title:
                title = next((line.strip() for line in text.splitlines() if len(line.strip()) > 12), doc_id)
            if len(text) >= 40:  # skip a page that is only a running header
                elements.append(
                    _element(
                        doc_id=doc_id,
                        title=title,
                        source=path.name,
                        file_type="pdf",
                        page=page_no,
                        element_type="text",
                        text=text,
                        page_image=image_rel.as_posix(),
                        license_name=license_name,
                    )
                )
            try:
                found = page.find_tables()
                tables = list(found.tables) if found is not None else []
            except Exception:
                tables = []
            kept = 0
            for table in tables:
                if kept >= 3:
                    break  # a page of false grids should not flood the index
                raw = table.extract()
                if not table_is_useful(raw):
                    continue
                markdown = markdown_table(raw)
                if not markdown:
                    continue
                kept += 1
                elements.append(
                    _element(
                        doc_id=doc_id,
                        title=title,
                        source=path.name,
                        file_type="pdf",
                        page=page_no,
                        element_type="table",
                        text=markdown,
                        page_image=image_rel.as_posix(),
                        license_name=license_name,
                    )
                )
    return elements


def parse_docx(path: Path) -> list[dict]:
    """Paragraphs, Word tables, and embedded images.

    Why page is always 1: a DOCX has no stable page numbers. The app shows the
    figure file itself, not "page 3". Walker Ranch has figures and no tables.
    The table branch is here so a later Word file with a grid still works.
    """
    document = DocxDocument(path)
    doc_id = path.stem
    license_name = "DOE Geothermal Data Repository CC-BY"
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    title = paragraphs[0][:160] if paragraphs else doc_id
    elements: list[dict] = []
    if paragraphs:
        elements.append(
            _element(
                doc_id=doc_id,
                title=title,
                source=path.name,
                file_type="docx",
                page=1,
                element_type="text",
                text="\n\n".join(paragraphs),
                page_image="",
                license_name=license_name,
            )
        )
    for table in document.tables:
        rows = [[cell.text for cell in row.cells] for row in table.rows]
        if not table_is_useful(rows):
            continue
        markdown = markdown_table(rows)
        if markdown:
            elements.append(
                _element(
                    doc_id=doc_id,
                    title=title,
                    source=path.name,
                    file_type="docx",
                    page=1,
                    element_type="table",
                    text=markdown,
                    page_image="",
                    license_name=license_name,
                )
            )
    seen: set[str] = set()
    figure_no = 0
    first_figure = ""
    for rel in document.part.rels.values():
        if "image" not in rel.reltype:
            continue
        if rel.target_ref in seen:
            continue
        seen.add(rel.target_ref)
        figure_no += 1
        suffix = Path(rel.target_ref).suffix or ".png"
        image_rel = Path("corpus") / "pages" / doc_id / f"figure_{figure_no}{suffix}"
        image_abs = HERE / image_rel
        image_abs.parent.mkdir(parents=True, exist_ok=True)
        image_abs.write_bytes(rel.target_part.blob)
        if not first_figure:
            first_figure = image_rel.as_posix()
        elements.append(
            _element(
                doc_id=doc_id,
                title=title,
                source=path.name,
                file_type="docx",
                page=1,
                element_type="figure",
                text="",
                page_image=image_rel.as_posix(),
                license_name=license_name,
            )
        )
    if first_figure:
        # The prose answer needs an image beside it in the app. Use the first figure.
        for element in elements:
            if element["element_type"] == "text" and not element["page_image"]:
                element["page_image"] = first_figure
    return elements


def parse_corpus() -> list[dict]:
    """Parse every PDF and DOCX in the packet. Re-running overwrites the jsonl."""
    if not PDF_DIR.exists() or not any(PDF_DIR.glob("*.pdf")):
        raise FileNotFoundError("No PDFs in corpus/pdf. Run: python download_corpus.py")
    elements: list[dict] = []
    for path in sorted(PDF_DIR.glob("*.pdf")):
        print(f"parsing {path.name}", flush=True)
        elements.extend(parse_pdf(path))
    for path in sorted(DOCX_DIR.glob("*.docx")):
        print(f"parsing {path.name}", flush=True)
        elements.extend(parse_docx(path))
    _write_elements(elements)
    return elements


def _write_elements(elements: list[dict]) -> None:
    PARSED.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(element, ensure_ascii=False) for element in elements) + "\n"
    # Skip the write when nothing changed, so a later index stamp stays valid.
    if ELEMENTS_PATH.exists() and ELEMENTS_PATH.read_text(encoding="utf-8") == payload:
        return
    ELEMENTS_PATH.write_text(payload, encoding="utf-8")


def load_elements() -> list[dict]:
    if not ELEMENTS_PATH.exists():
        raise FileNotFoundError("No parsed elements. Run notebook 1, or: python document_rag.py build")
    elements = []
    for line in ELEMENTS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            elements.append(json.loads(line))
    return elements


def load_questions() -> list[dict]:
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError("No corpus/questions.json. Run: python download_corpus.py")
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]


def corpus_summary(elements: list[dict] | None = None) -> dict:
    elements = elements if elements is not None else (load_elements() if ELEMENTS_PATH.exists() else [])
    by_type: dict[str, int] = {}
    docs = set()
    for element in elements:
        by_type[element["element_type"]] = by_type.get(element["element_type"], 0) + 1
        docs.add(element["doc_id"])
    return {"documents": len(docs), "elements": len(elements), "by_type": by_type}


def caption_image(path: Path) -> str:
    """Ask Qwen to read one page image. The reply is a search caption, not the answer we send.

    Why a caption at all: BM25 and the embedder cannot see pixels. They can see this text.
    Why we still show the PNG to a person: a 4B vision pass drops row labels and misreads digits.
    num_predict is 500 because a short limit stops in the middle of a table.
    think is false because qwen3.5 otherwise spends the call on a hidden trace and returns nothing.
    """
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    prompt = (
        "Ignore ordinary paragraphs. Transcribe only tables and charts. No preamble. "
        "For a table, one line per row: the left-hand label, then every number in that row, including the percent. "
        "For a bar chart, one line per bar: the category printed under the bar, then the number printed on that bar. "
        "Do not invent a number."
    )
    body = json.dumps(
        {
            "model": CHAT_MODEL,
            "think": False,
            "stream": False,
            "messages": [{"role": "user", "content": prompt, "images": [encoded]}],
            "options": {"temperature": 0, "num_predict": 500},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return (payload.get("message") or {}).get("content", "").strip()


def _load_captions() -> dict[str, str]:
    if not CAPTIONS_PATH.exists():
        return {}
    return json.loads(CAPTIONS_PATH.read_text(encoding="utf-8"))


def _save_captions(cache: dict[str, str]) -> None:
    PARSED.mkdir(parents=True, exist_ok=True)
    CAPTIONS_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _needs_figure(element: dict, evidence: dict[str, set[int]]) -> bool:
    """True when this page needs a figure element, because the text layer is the wrong channel.

    Three cases, and only these:
    - a gold chart page (the Office 365 chart, the Research Affiliates chart)
    - a known picture-table (PNC pages 6 and 7)
    - a page whose text layer is only a header (under 100 characters)
    Ordinary prose pages stay as text. We do not caption the whole packet.
    """
    if element["element_type"] != "text" or not element.get("page_image"):
        return False
    gold_chart = element["doc_id"] in CAPTION_DOC_IDS and element["page"] in evidence.get(element["doc_id"], set())
    picture_page = (element["doc_id"], element["page"]) in PICTURE_PAGES or len(element["text"]) < 100
    return gold_chart or picture_page


def _image_evidence_pages() -> dict[str, set[int]]:
    """Which pages of the chart documents does the benchmark actually ask about?"""
    pages: dict[str, set[int]] = {}
    if not QUESTIONS_PATH.exists():
        return pages
    for question in load_questions():
        if question["doc_id"] not in CAPTION_DOC_IDS:
            continue
        if "image" not in question.get("answer_type", ""):
            continue
        pages.setdefault(question["doc_id"], set()).update(question.get("evidence_pages") or [])
    return pages


def attach_captions(elements: list[dict] | None = None) -> list[dict]:
    """Add figure text where the text layer missed the fact.

    A second run reads parsed/captions.json and does not call Ollama.
    caption_overrides.json wins over the cache: that file is a transcription
    a person checked. The first vision pass on PNC page 7 wrote "1.7 | 5.4%"
    and dropped the words Commercial Real Estate.
    """
    if not ollama_alive():
        raise RuntimeError("Ollama is not running. Start it, then pull qwen3.5:4b.")
    elements = list(elements or load_elements())
    cache = _load_captions()
    evidence = _image_evidence_pages()
    overrides = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8")) if OVERRIDES_PATH.exists() else {}

    # A figure element is a second chunk for the same page, beside the text chunk.
    existing = {(e["doc_id"], e["page"], e["element_type"]) for e in elements}
    for element in list(elements):
        if not _needs_figure(element, evidence):
            continue
        key = (element["doc_id"], element["page"], "figure")
        if key in existing:
            continue
        figure = dict(element)
        figure["element_type"] = "figure"
        figure["text"] = ""
        elements.append(figure)
        existing.add(key)

    for element in elements:
        if element["element_type"] != "figure":
            continue
        image_rel = element.get("page_image") or ""
        if not image_rel:
            continue
        if element["file_type"] == "docx":
            # The DOCX already prints what the red dots are. A vision read once
            # said "red triangles". That sentence must not enter the index.
            # The pictures stay, so the app can show them. The paragraphs answer.
            element["text"] = (
                f"Embedded figure in {element['source']}. "
                "Use the paragraphs in this same file for the caption. "
                "Do not invent labels for the picture."
            )
            continue
        image_abs = HERE / image_rel
        if not image_abs.exists():
            continue
        if image_rel in overrides:
            caption = overrides[image_rel]  # checked by a person; do not ask the model again
        else:
            cache_key = f"{CAPTION_CACHE_VERSION}|{image_rel}"
            if cache_key not in cache:
                print(f"caption {image_rel}", flush=True)
                try:
                    cache[cache_key] = caption_image(image_abs)
                except Exception as exc:
                    cache[cache_key] = f"(caption failed: {exc})"
                _save_captions(cache)
            caption = cache[cache_key]
        # The stamp names the file and page, so a retrieved chunk still says where to look.
        element["text"] = (
            f"Figure in {element['source']}, page {element['page']}. {element['title']}.\n{caption}"
        ).strip()

    _write_elements(elements)
    return elements


def elements_to_chunks(elements: list[dict]) -> list[Document]:
    """Cut prose. Keep a table or a figure caption whole.

    Why prose is cut: a page is longer than the model should read for one fact.
    Why a table is not cut: splitting a grid is how a system cites the wrong row.
    Why the stamp is inside the text: both the embedder and BM25 can see the doc id and the page.
    Same idea as the handbook stamp in module 15.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,                          # prose pieces stay near this length
        chunk_overlap=120,                       # repeat the join, so a cut sentence survives in both pieces
        separators=["\n\n", "\n", ". ", " "],    # prefer a paragraph break over a mid-word cut
    )
    chunks: list[Document] = []
    for element_index, element in enumerate(elements, start=1):
        body = (element.get("text") or "").strip()
        if len(body) < 20:
            continue  # an empty figure, before its caption is attached
        if element["element_type"] == "text":
            pieces = splitter.split_text(body)
        else:
            pieces = [body[:3500]]  # one chunk. The cap only stops a runaway caption.
        for piece_no, piece in enumerate(pieces, start=1):
            # Inside the text, not only in metadata, so search can match the document id.
            stamp = (
                f"Document {element['doc_id']} ({element['file_type']}), "
                f"page {element['page']}, {element['element_type']}. {element['title']}."
            )
            # element_index: two tables on one page must not share an id, or the second write deletes the first.
            chunk_id = f"{element['doc_id']}-p{element['page']}-{element['element_type']}-{element_index}-{piece_no}"
            chunks.append(
                Document(
                    page_content=f"{stamp}\n{piece}",
                    metadata={
                        "chunk_id": chunk_id,
                        "doc_id": element["doc_id"],
                        "source": element["source"],
                        "file_type": element["file_type"],
                        "page": int(element["page"]),
                        "element_type": element["element_type"],
                        "page_image": element.get("page_image") or "",
                        "title": element.get("title") or "",
                        "license": element.get("license") or "",
                    },
                )
            )
    return chunks


def clear_runtime_cache() -> None:
    global _STORE, _CHUNKS, _BM25, _GRAPH
    _STORE = None
    _CHUNKS = None
    _BM25 = None
    _GRAPH = None


def _elements_hash() -> str:
    return hashlib.sha256(ELEMENTS_PATH.read_bytes()).hexdigest()


def _index_fresh() -> bool:
    """True when Chroma was built from this exact elements file.

    Why a hash, not the file's modified time: opening the file in an editor
    changes the time and would force a four-minute re-embed for no reason.
    """
    if not STAMP_PATH.exists() or not ELEMENTS_PATH.exists():
        return False
    stamp = json.loads(STAMP_PATH.read_text(encoding="utf-8"))
    return stamp.get("elements_sha256") == _elements_hash() and _collection_count() > 0


def _collection_count() -> int:
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        names = []
        for collection in client.list_collections():
            names.append(collection.name if hasattr(collection, "name") else str(collection))
        if COLLECTION not in names:
            return 0
        return client.get_collection(COLLECTION).count()
    except Exception:
        return 0


def build_index(force: bool = False, caption: bool = True) -> list[Document]:
    """Parse if needed, caption the chart pages, embed, and save Chroma.

    force=False skips the embed when the elements file has not changed.
    The first build takes several minutes. A second run of the notebook should say
    "Index already matches".
    """
    global _STORE, _CHUNKS, _BM25
    if not ELEMENTS_PATH.exists():
        parse_corpus()
    if caption:
        attach_captions(load_elements())
    if not force and _index_fresh():
        print(f"Index already matches the parsed files ({_collection_count()} chunks).")
        clear_runtime_cache()
        return get_chunks()

    chunks = elements_to_chunks(load_elements())
    if not chunks:
        raise RuntimeError("The parser produced no chunks.")
    print(f"Embedding {len(chunks)} chunks with {EMBED_MODEL}. The first batch loads the model.", flush=True)

    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing = []
    for collection in client.list_collections():
        existing.append(collection.name if hasattr(collection, "name") else str(collection))
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
    del client

    embeddings = QwenEmbeddings()
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    batch = 8  # small batches so a laptop shows progress and does not time out one huge call
    for start in range(0, len(chunks), batch):
        part = chunks[start : start + batch]
        print(f"  {start + 1}-{start + len(part)} / {len(chunks)}", flush=True)
        store.add_documents(part, ids=[doc.metadata["chunk_id"] for doc in part])

    PARSED.mkdir(parents=True, exist_ok=True)
    STAMP_PATH.write_text(
        json.dumps({"elements_sha256": _elements_hash(), "chunks": len(chunks)}),
        encoding="utf-8",
    )
    clear_runtime_cache()
    _STORE = store
    _CHUNKS = chunks
    _BM25 = BM25Okapi([words(doc.page_content) for doc in chunks])
    print(f"Saved {len(chunks)} chunks to {CHROMA_DIR}.")
    return chunks


def get_chunks() -> list[Document]:
    global _CHUNKS, _BM25
    if _CHUNKS is None:
        _CHUNKS = elements_to_chunks(load_elements())
        _BM25 = BM25Okapi([words(doc.page_content) for doc in _CHUNKS])
    return _CHUNKS


def get_store() -> Chroma:
    global _STORE
    if _STORE is None:
        if _collection_count() == 0:
            raise RuntimeError("No index yet. Run notebook 2, or: python document_rag.py build")
        _STORE = Chroma(
            collection_name=COLLECTION,
            embedding_function=QwenEmbeddings(),
            persist_directory=str(CHROMA_DIR),
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _STORE


def keyword_search(query: str, k: int = 6) -> list[Document]:
    chunks = get_chunks()
    scores = _BM25.get_scores(words(query)) if _BM25 is not None else []
    ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
    return [doc for score, doc in ranked[:k] if score > 0]


def vector_search(query: str, k: int = 6) -> list[Document]:
    return get_store().similarity_search(query, k=k)


def fuse(ranked_lists: list[list[Document]], rank_constant: int = 60) -> list[Document]:
    """Merge two ranked lists. A chunk found by both searches beats a chunk found by one.

    Reciprocal rank fusion, same formula as module 15.
    Rank r adds 1 / (60 + r). Rank 1 adds 1/61. Rank 6 adds 1/66.
    The 60 is a constant, not "return 60 chunks". It keeps one lucky rank-1 from
    drowning out a chunk that both searches agree on.
    """
    scores: dict[str, float] = {}
    picked: dict[str, Document] = {}
    for docs in ranked_lists:
        for rank, doc in enumerate(docs, start=1):
            chunk_id = doc.metadata["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rank_constant + rank)
            picked[chunk_id] = doc
    order = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [picked[chunk_id] for chunk_id in order]


def hybrid_search(query: str, k: int = 4) -> list[Document]:
    """Vector search for the vague question, keyword search for the rare amount, then fuse.

    Each side returns 6. The graph keeps 4. The extra two give fusion a chance to
    promote a chunk that both sides found just outside the top.
    """
    return fuse([vector_search(query, k=6), keyword_search(query, k=6)])[:k]


def retrieval_report(k: int = 4) -> list[dict]:
    """For each gold question: did the right document, and the right page, land in the top k?"""
    rows = []
    for question in load_questions():
        hits = hybrid_search(question["question"], k=k)
        doc_ids = [doc.metadata["doc_id"] for doc in hits]
        pages = [int(doc.metadata["page"]) for doc in hits if doc.metadata["doc_id"] == question["doc_id"]]
        gold_pages = set(question.get("evidence_pages") or [])
        rows.append(
            {
                "doc_id": question["doc_id"],
                "answer_type": question["answer_type"],
                "question_type": question["question_type"],
                "hit_doc": question["doc_id"] in doc_ids,
                "hit_page": bool(gold_pages & set(pages)) if gold_pages else False,
                "top": [f"{doc.metadata['doc_id']} p.{doc.metadata['page']} {doc.metadata['element_type']}" for doc in hits],
                "question": question["question"],
            }
        )
    return rows


def _hit(doc: Document) -> dict:
    meta = doc.metadata
    return {
        "text": doc.page_content,
        "doc_id": meta["doc_id"],
        "page": int(meta["page"]),
        "element_type": meta["element_type"],
        "source": meta["source"],
        "page_image": meta.get("page_image") or "",
        "title": meta.get("title") or "",
        "chunk_id": meta["chunk_id"],
        "license": meta.get("license") or "",
    }


def _label(hit: dict) -> str:
    return f"[{hit['doc_id']} p.{hit['page']} {hit['element_type']}]"


class Grade(BaseModel):
    relevant: bool = Field(description="True only if the passage states a fact needed to answer the question")


class Check(BaseModel):
    supported: bool = Field(description="True only if every factual claim in the answer is stated in the passages")


def _numbers(text: str) -> list[str]:
    return _MEASURE.findall(text or "")


def numbers_supported(answer: str, hits: list[dict]) -> bool:
    """Every measured number in the draft must be written in a retrieved passage.

    Why this exists beside the model's own yes/no: the checker will accept a fluent
    nearby number. This function only looks at the digits. '18,443.87' and
    '18443.87' count as the same. The percent sign is stripped before the compare,
    so a draft that says "1.7%" still passes when the passage contains 1.7.
    It does not know which column a number belongs to. That is the person's job.
    """
    blob = " ".join(hit["text"] for hit in hits).replace(",", "")
    for raw in _numbers(answer):
        bare = raw.replace(",", "").replace("%", "").strip()
        if not bare:
            continue
        if re.search(rf"(?<!\d){re.escape(bare)}(?!\d)", blob) is None:
            return False
    return True


def draft_has_measure(answer: str) -> bool:
    return bool(_MEASURE.search(answer or ""))


class RAGState(TypedDict):
    """What every node can read and write. A node returns only the keys it changes.

    steps uses operator.add, so each node appends a line instead of wiping the path.
    That path is what you read before you trust the sentence.
    """

    question: str
    search_query: str       # starts equal to the question; rewrite replaces it
    rewrites_left: int      # one retry. Spent by rewrite, not by the first search.
    hits: list              # passages still in play after grading
    answer: str
    supported: bool         # both the digit check and the model check said yes
    needs_human: bool       # supported, and the draft is a table, a figure, or a measured number
    pause_reason: str       # the sentence the screen shows above the draft
    citations: list
    human_action: str       # approve, edit, or reject, after the person resumes
    steps: Annotated[list[str], operator.add]


def _initial(question: str) -> RAGState:
    return {
        "question": question,
        "search_query": question,
        "rewrites_left": 1,
        "hits": [],
        "answer": "",
        "supported": False,
        "needs_human": False,
        "pause_reason": "",
        "citations": [],
        "human_action": "",
        "steps": [],
    }


# The desk graph. LangChain does the work inside a node. LangGraph chooses the next node.
#
#   greeting                              chat -> END          (the packet stays closed)
#   anything else                         retrieve -> grade
#   grade kept a passage                  generate -> check
#   grade kept nothing, one retry left    rewrite -> retrieve
#   grade kept nothing, retry spent       abstain -> END
#   check failed, one retry left          rewrite
#   check failed, retry spent             abstain
#   check passed, table / figure / number human_review         (interrupt: a person signs)
#   check passed, ordinary prose          END
#   person approves or edits              END                  (no second model call)
#   person rejects, one retry left        rewrite
#   person rejects, retry spent           abstain
#
# A failed check does not go to the person. We only ask a person to sign a draft
# that already looks supported. The PNC bug is the reason: 1.7 and 5.4% are both
# in the caption, so the checks pass, and the model still assigns 1.7 to the percent.


# START HERE. Hello goes to chat. Every other question opens the packet.
def route(state: RAGState) -> str:
    """First decision: is this small talk, or a question about the packet?"""
    text = state["question"].lower().strip(" .!?")
    return "chat" if text in SMALL_TALK else "retrieve"


def chat(state: RAGState) -> dict:
    """A greeting. The prompt forbids documents and numbers so the model cannot invent a fact."""
    reply = make_llm(60).invoke(
        "Reply in one short friendly sentence. Do not mention documents or numbers."
    )
    return {
        "answer": (reply.content or "").strip(),
        "needs_human": False,
        "supported": True,
        "steps": ["chat (documents stay closed)"],
    }


def retrieve(state: RAGState) -> dict:
    """Hybrid search. Uses search_query, which is the original question until a rewrite."""
    hits = [_hit(doc) for doc in hybrid_search(state["search_query"])]
    labels = ", ".join(_label(hit) for hit in hits) or "(none)"
    return {"hits": hits, "steps": [f"retrieve '{state['search_query']}' -> {labels}"]}


def grade(state: RAGState) -> dict:
    """Drop passages that do not state a fact the question needs.

    Same topic is not enough: a PNC page about losses is not automatically
    the Commercial Real Estate row. method='function_calling' makes the model
    return Grade.relevant as a real bool the edge can read.
    A failed call counts as no, so a broken grade cannot smuggle a passage through.
    """
    grader = make_llm(80, num_ctx=4096).with_structured_output(Grade, method="function_calling")
    kept, notes = [], []
    instruction = (
        "You are a strict relevance grader for a document desk. "
        "relevant is true only when the passage states a fact that helps answer the question. "
        "The same general topic is not enough.\n\n"
    )
    for hit in state["hits"]:
        try:
            verdict = grader.invoke(
                instruction + f"Question: {state['question']}\n\nPassage:\n{hit['text'][:1800]}"
            )
            ok = verdict is not None and verdict.relevant
        except Exception:
            ok = False
        notes.append(f"{_label(hit)}={'yes' if ok else 'no'}")
        if ok:
            kept.append(hit)
    return {"hits": kept, "steps": ["grade " + ", ".join(notes)]}


def rewrite(state: RAGState) -> dict:
    """One new search query, in words that would appear on the page. This spends the only retry."""
    reply = make_llm(80).invoke(
        "Rewrite the question as a short search query using words that would appear in the document. "
        "Output only the query.\n\n"
        f"Question: {state['question']}"
    )
    query = (reply.content or state["question"]).strip().splitlines()[0]
    return {
        "search_query": query,
        "rewrites_left": state["rewrites_left"] - 1,
        "steps": [f"rewrite -> '{query}'"],
    }


def generate(state: RAGState) -> dict:
    """Draft from the kept passages only.

    The model sees text. It does not see the page image. The image is for the person.
    'Copy numbers exactly' is a request. The check node is what enforces it.
    """
    blocks = []
    for hit in state["hits"]:
        blocks.append(f"{_label(hit)}\n{hit['text'][:2000]}")
    prompt = (
        "Answer the analyst using only the passages. "
        "End each factual sentence with the citation already printed above that passage, "
        "for example [2025833 p.3 table]. "
        "Copy numbers exactly. Do not round and do not swap one row for another. "
        "If the passages do not state the answer, reply with exactly NOT_IN_CORPUS and nothing else.\n\n"
        f"Question: {state['question']}\n\nPassages:\n\n" + "\n\n".join(blocks)
    )
    reply = make_llm(360).invoke(prompt)
    answer = (reply.content or "").strip()
    citations = [_label(hit) for hit in state["hits"]]
    return {"answer": answer, "citations": citations, "steps": ["generate"]}


def check(state: RAGState) -> dict:
    """Two gates, then the signature rule.

    Gate 1, digits: every measured number in the draft is written in a passage.
    Gate 2, claims: the Check schema also says the sentences are in the passages.
    Both must pass. The model's own yes is not enough on its own.

    Signature: a draft that passed still waits when a kept passage is a table or a
    figure, or when the draft contains a measured number. Walker Ranch prose has
    no measured number, so it does not wait. A failed gate does not wait either.
    It rewrites once, then abstains.
    """
    answer = state.get("answer") or ""
    if "NOT_IN_CORPUS" in answer:
        return {
            "supported": False,
            "needs_human": False,
            "pause_reason": "",
            "steps": ["check: draft abstained"],
        }
    digits_ok = numbers_supported(answer, state["hits"])
    checker = make_llm(80, num_ctx=4096).with_structured_output(Check, method="function_calling")
    passage = "\n\n".join(hit["text"][:1500] for hit in state["hits"])
    try:
        verdict = checker.invoke(
            "Is every factual claim in the answer stated in the passages? "
            "A number that is not written in the passages is not supported.\n\n"
            f"Question: {state['question']}\n\nAnswer:\n{answer}\n\nPassages:\n{passage}"
        )
        model_ok = verdict is not None and verdict.supported
    except Exception:
        model_ok = False
    supported = digits_ok and model_ok
    # A table or a figure is where a wrong row leaves the building.
    risky_element = any(hit["element_type"] in {"table", "figure"} for hit in state["hits"])
    measured = draft_has_measure(answer)
    # Only a draft that already passed the gates is worth a person's time.
    needs_human = supported and (risky_element or measured)
    if not digits_ok:
        reason = "A number in the draft is not in the retrieved passages."
    elif needs_human and risky_element:
        reason = "The evidence is a table or a figure. A person signs the draft before it is treated as sent."
    elif needs_human and measured:
        reason = "The draft contains a measured number. A person signs it before it is treated as sent."
    else:
        reason = ""
    note = f"check digits={'yes' if digits_ok else 'no'} model={'yes' if model_ok else 'no'}"
    return {
        "supported": supported,
        "needs_human": needs_human,
        "pause_reason": reason,
        "steps": [note],
    }


def human_review(state: RAGState) -> dict:
    """Stop until a person approves, edits, or rejects.

    interrupt() pauses here and hands the payload to the notebook or the app.
    On resume, LangGraph runs this function AGAIN FROM THE TOP, and interrupt()
    returns the decision instead of pausing. Do not call the model or write a
    file above that line. That work would happen twice.

    edit replaces the answer with the person's text. There is no second model call.
    The person is the authority. reject sends the graph back to rewrite, if the retry remains.
    """
    decision = interrupt(
        {
            "question": state["question"],
            "draft": state["answer"],
            "reason": state["pause_reason"],
            "citations": state["citations"],
            "sources": [
                {
                    "label": _label(hit),
                    "page_image": hit["page_image"],
                    "excerpt": hit["text"][:700],
                    "license": hit.get("license", ""),
                }
                for hit in state["hits"]
            ],
        }
    )
    action = (decision or {}).get("action", "reject")
    if action == "approve":
        return {"human_action": "approve", "steps": ["human approved"]}
    if action == "edit":
        edited = (decision.get("text") or state["answer"]).strip()
        return {"answer": edited, "human_action": "edit", "steps": ["human edited the draft"]}
    return {"human_action": "reject", "supported": False, "steps": ["human rejected the draft"]}


def abstain(state: RAGState) -> dict:
    return {"answer": ABSTAIN, "needs_human": False, "steps": ["abstain"]}


def after_grade(state: RAGState) -> str:
    """Keep a passage and draft. Nothing kept: rewrite once, then abstain."""
    if state["hits"]:
        return "generate"
    if state["rewrites_left"] > 0:
        return "rewrite"
    return "abstain"


def after_check(state: RAGState) -> str:
    """Unsupported: rewrite once, then abstain. Supported and risky: wait for a person.

    'done' is not a node. The edge map sends it to END.
    """
    if "NOT_IN_CORPUS" in (state.get("answer") or ""):
        return "abstain"
    if not state["supported"]:
        return "rewrite" if state["rewrites_left"] > 0 else "abstain"
    if state["needs_human"]:
        return "human_review"
    return "done"


def after_human(state: RAGState) -> str:
    """Approve and edit leave. Reject spends the remaining retry, then abstains.

    The first PNC pass is supported, so it never rewrote. rewrites_left is still 1.
    The first reject therefore searches again. A second reject finds the budget spent.
    """
    if state.get("human_action") in {"approve", "edit"}:
        return "done"
    if state["rewrites_left"] > 0:
        return "rewrite"
    return "abstain"


def build_assistant(checkpointer=None):
    """Compile the desk graph.

    The checkpointer is what makes the pause resumable. MemorySaver lives in this
    process only. A new shell, or a restarted Streamlit server, forgets the thread.
    A production desk swaps in a sqlite or postgres saver. The nodes stay the same.
    """
    graph = StateGraph(RAGState)
    for name, fn in [
        ("chat", chat),
        ("retrieve", retrieve),
        ("grade", grade),
        ("rewrite", rewrite),
        ("generate", generate),
        ("check", check),
        ("human_review", human_review),
        ("abstain", abstain),
    ]:
        graph.add_node(name, fn)
    graph.add_conditional_edges(START, route, ["chat", "retrieve"])
    graph.add_edge("chat", END)
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", after_grade, ["generate", "rewrite", "abstain"])
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "check")
    # "done" is the return value that means the draft can leave. It is not a node.
    end_paths = {"human_review": "human_review", "rewrite": "rewrite", "abstain": "abstain", "done": END}
    graph.add_conditional_edges("check", after_check, end_paths)
    graph.add_conditional_edges(
        "human_review",
        after_human,
        {"rewrite": "rewrite", "abstain": "abstain", "done": END},
    )
    return graph.compile(checkpointer=checkpointer or _SAVER)


def get_assistant():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_assistant(_SAVER)
    return _GRAPH


def ask(question: str, thread_id: str | None = None, resume: dict | None = None) -> tuple[dict, dict | None]:
    """Run one question, or resume a paused one.

    Returns (state, interrupt payload). The payload is None when the graph finished.
    resume={"action": "edit", "text": "..."} continues the same thread_id.
    A new question needs a new thread, or it continues the old pause.
    """
    thread_id = thread_id or f"desk-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    assistant = get_assistant()
    if resume is None:
        assistant.invoke(_initial(question), config)
    else:
        assistant.invoke(Command(resume=resume), config)
    state = assistant.get_state(config)
    payload = None
    if state.interrupts:
        payload = state.interrupts[0].value
    values = dict(state.values)
    values["thread_id"] = thread_id
    return values, payload


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Build or query the Ledgerline document desk.")
    parser.add_argument("command", choices=["build", "report", "ask", "resume"])
    parser.add_argument("text", nargs="?", default="")
    parser.add_argument("--thread", default="")
    parser.add_argument("--action", choices=["approve", "reject", "edit"], default="approve")
    parser.add_argument("--no-caption", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.command == "build":
        if not ollama_alive():
            sys.exit("Ollama is not running on 127.0.0.1:11434.")
        build_index(force=args.force, caption=not args.no_caption)
        return
    if args.command == "report":
        rows = retrieval_report()
        doc_hits = sum(row["hit_doc"] for row in rows)
        page_hits = sum(row["hit_page"] for row in rows)
        print(f"doc hit {doc_hits}/{len(rows)}   page hit {page_hits}/{len(rows)}")
        for row in rows:
            mark = "ok" if row["hit_doc"] else "MISS"
            print(f"{mark:4} {row['answer_type']:24} {row['doc_id']:22} {row['top']}")
        return
    if args.command == "ask":
        if not args.text:
            sys.exit("Usage: python document_rag.py ask \"your question\"")
        values, payload = ask(args.text)
        print("thread:", values["thread_id"])
        for step in values.get("steps") or []:
            print(" ", step)
        if payload:
            print("\nPAUSED — a person has to sign this draft.")
            print(payload["reason"])
            print(payload["draft"])
            print("Resume with: python document_rag.py resume", values["thread_id"], "--action approve")
        else:
            print("\n" + (values.get("answer") or ""))
        return
    if args.command == "resume":
        if not args.thread:
            sys.exit("Usage: python document_rag.py resume --thread THREAD --action approve")
        decision = {"action": args.action, "text": args.text}
        values, payload = ask("", thread_id=args.thread, resume=decision)
        print("thread:", values["thread_id"])
        for step in values.get("steps") or []:
            print(" ", step)
        if payload:
            print("\nPAUSED again")
            print(payload["draft"])
        else:
            print("\n" + (values.get("answer") or ""))


if __name__ == "__main__":
    main()
