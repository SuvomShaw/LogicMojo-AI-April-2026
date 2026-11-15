"""Copy the Berka bank SQLite out of the FINCH dataset and check the saved SQL.

FINCH (domyn/FINCH, CC-BY-NC-4.0) packs 33 finance databases, about 2 GB.
This demo uses one of them: the Czech bank file BIRD calls `financial`.
It is 71 MB, 8 tables, and 1,056,320 transactions. The other files stay
in the Hugging Face cache.

The download also runs every checked query in questions.json. If a result
changed, the script stops. The published lowercase queries are not the
answer key. This file still stores POPLATEK MESICNE, OWNER, and east Bohemia.

    python download_corpus.py
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

from huggingface_hub import hf_hub_download

HERE = Path(__file__).parent
DATA = HERE / "data"
DB_PATH = DATA / "financial.sqlite"
QUESTIONS_PATH = HERE / "questions.json"
HF_FILE = "text2sql-db/text2sql/bird/financial.sqlite"


def _same(got, expected) -> bool:
    def norm(value):
        if isinstance(value, float):
            return round(value, 2)
        return value

    return [[norm(cell) for cell in row] for row in got] == expected


def main() -> None:
    print("Fetching the Berka bank SQLite from domyn/FINCH (cached after the first run)...")
    cached = Path(hf_hub_download("domyn/FINCH", HF_FILE, repo_type="dataset"))
    DATA.mkdir(parents=True, exist_ok=True)
    if not DB_PATH.exists() or DB_PATH.stat().st_size != cached.stat().st_size:
        shutil.copyfile(cached, DB_PATH)
    print(f"Database: {DB_PATH} ({DB_PATH.stat().st_size // (1024 * 1024)} MB)")

    payload = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        failed = 0
        for question in payload["questions"]:
            rows = connection.execute(question["checked_sql"]).fetchall()
            got = [list(row) for row in rows]
            mark = "ok  " if _same(got, question["result"]) else "MISS"
            if mark == "MISS":
                failed += 1
            print(f"{mark} {question['id']:32} {got}")
        if failed:
            raise SystemExit(f"{failed} checked queries no longer match questions.json")
    finally:
        connection.close()
    print(f"Checked {len(payload['questions'])} queries against the live file.")
    print("Next: open 01_the_bank_database.ipynb")


if __name__ == "__main__":
    main()
