"""Download the document packet: a few real finance PDFs, their questions, and one DOCX.

The PDFs come from Salesforce UniDoc-Bench (CC-BY-NC 4.0). The full finance
archive is about 360 MB and stays in the Hugging Face cache. This script copies
seven short documents into corpus/pdf/.

The DOCX is the Walker Ranch seismic note from the DOE Geothermal Data
Repository (CC-BY). It is here so the parser meets a Word file with figures.
It is a different subject on purpose: a finance question must not be answered
from it.

Run from this folder:

    python download_corpus.py
"""

from __future__ import annotations

import json
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

HERE = Path(__file__).parent
CORPUS = HERE / "corpus"
PDF_DIR = CORPUS / "pdf"
DOCX_DIR = CORPUS / "docx"

# Seven short notes, not the whole finance archive. Ids are the UniDoc file stems.
# Chosen so the packet has prose, a real grid, and a chart, without a 360 MB folder.
DOC_IDS = [
    "0082969",  # prose, euro amounts, direct-lending funds
    "1295982",  # chart, Office 365 vs perpetual Office
    "1659677",  # chart plus a small percent table, Research Affiliates
    "2025833",  # town ledger, exact dollar amounts
    "6002966",  # WAM shareholder letter, NTA per share
    "6838482",  # call-center contract, price table
    "7257900",  # PNC loss scenario, a percent in a table
]

WALKER_URL = "https://gdr.openei.org/files/833/Walker_Ranch_3D_seismic_images.docx"
WALKER_NAME = "walker_ranch_seismic.docx"

WALKER_QUESTION = {
    "doc_id": "walker_ranch_seismic",
    "question": "In the Walker Ranch seismic figures, what do the red dots show?",
    "answer": "Micro-earthquakes, from the LBL Raft River catalog.",
    "question_type": "factual_retrieval",
    "answer_type": "figure_caption",
    "evidence_pages": [1],
    "origin": "Caption printed in the DOE Walker Ranch DOCX.",
}


def _pages(paths) -> list[int]:
    pages = []
    for path in paths:
        match = re.search(r"page_(\d+)", str(path))
        if match:
            pages.append(int(match.group(1)))
    return pages


def _extract_pdfs(archive: Path) -> list[str]:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    wanted = set(DOC_IDS)
    saved = []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = Path(member.name).name
            # macOS packs AppleDouble sidecars (._0082969.pdf) next to the real PDFs. Skip them.
            if name.startswith("._") or not name.lower().endswith(".pdf"):
                continue
            if Path(name).stem not in wanted:
                continue
            target = PDF_DIR / name
            source = tar.extractfile(member)
            if source is None:
                continue
            with target.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            saved.append(name)
    missing = wanted - {Path(name).stem for name in saved}
    if missing:
        raise SystemExit(f"These PDFs were not in the archive: {sorted(missing)}")
    return sorted(saved)


def _download_docx() -> Path:
    DOCX_DIR.mkdir(parents=True, exist_ok=True)
    target = DOCX_DIR / WALKER_NAME
    if target.exists() and target.stat().st_size > 10_000:
        return target
    request = urllib.request.Request(WALKER_URL, headers={"User-Agent": "LogicMojo-teaching/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        target.write_bytes(response.read())
    return target


def _write_questions() -> int:
    parquet = hf_hub_download(
        "Salesforce/UniDoc-Bench",
        "data/finance-00000-of-00001.parquet",
        repo_type="dataset",
    )
    frame = pd.read_parquet(parquet)
    questions = []
    for row in frame.itertuples(index=False):
        paths = list(row.gt_image_paths)
        doc_id = str(paths[0]).split("/")[2]
        if doc_id not in DOC_IDS:
            continue
        questions.append(
            {
                "doc_id": doc_id,
                "question": row.question,
                "answer": row.answer,
                "question_type": row.question_type,
                "answer_type": row.answer_type,
                "evidence_pages": _pages(paths),
                "origin": "Salesforce/UniDoc-Bench finance split",
            }
        )
    questions.append(WALKER_QUESTION)
    payload = {
        "pdf_license": "CC-BY-NC-4.0",
        "pdf_source": "https://huggingface.co/datasets/Salesforce/UniDoc-Bench",
        "pdf_paper": "https://arxiv.org/abs/2510.03663",
        "docx_license": "CC-BY",
        "docx_source": WALKER_URL,
        "docx_note": (
            "Walker Ranch 3D seismic images, DOE Geothermal Data Repository. "
            "Figure captions are part of the published note. The underlying survey "
            "data remains with its owners; these notes use the published DOCX only."
        ),
        "questions": questions,
    }
    CORPUS.mkdir(parents=True, exist_ok=True)
    (CORPUS / "questions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return len(questions)


def main() -> None:
    print("Fetching the UniDoc-Bench finance archive (cached by Hugging Face after the first run)...")
    archive = hf_hub_download("Salesforce/UniDoc-Bench", "finance_pdfs.tar.gz", repo_type="dataset")
    saved = _extract_pdfs(Path(archive))
    print(f"PDFs in {PDF_DIR}:")
    for name in saved:
        print(f"  {name}")
    docx = _download_docx()
    print(f"DOCX: {docx.name} ({docx.stat().st_size // 1024} KB)")
    count = _write_questions()
    print(f"Wrote corpus/questions.json with {count} questions.")
    print("Next: open 01_parse_real_documents.ipynb, or run: python document_rag.py build")


if __name__ == "__main__":
    main()
