"""Bank desk: a question becomes one SELECT, then a person signs a count.

Students: open README.md in this folder first, then search this file for START HERE.

  hello                         -> chat
  delete, update, insert, drop  -> refuse       the SQL is never written
  a bank question               -> link tables -> write SQL -> guard -> execute
  a count, sum, or amount       -> a person signs
  a label, such as a card type  -> send it
  no matching table             -> abstain

semantic_layer.json says what names like A2 mean.
data/financial.sqlite is the database. download_corpus.py fetches it.

Model: qwen3.5:4b writes the SQL and the one-sentence answer.
"""

from __future__ import annotations

import json
import operator
import re
import sqlite3
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DB_PATH = DATA / "financial.sqlite"
LAYER_PATH = HERE / "semantic_layer.json"
QUESTIONS_PATH = HERE / "questions.json"
CATALOG_PATH = DATA / "value_catalog.json"

CHAT_MODEL = "qwen3.5:4b"
ABSTAIN = "NOT_IN_DATABASE: the Berka bank tables do not answer that question."
REFUSE = "REFUSED: this desk can only read the database. It will not insert, update, or delete."
SMALL_TALK = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "thanks", "thank you"}

# A write in the question never reaches the model. The SQL guard is the second lock.
_WRITE = re.compile(r"\b(delete|drop|truncate|insert|alter|update)\b", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|reindex)\b",
    re.IGNORECASE,
)
# Comments are removed before the guard runs, so a comment cannot hide a second statement.
_LINE_COMMENT = re.compile(r"--.*?$", re.MULTILINE)
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_MEASURE = re.compile(r"\d+\.\d+|\d{1,3}(?:,\d{3})+|\d+")
_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][a-z0-9]+)*", re.IGNORECASE)
# These words sit in almost every question and almost every column note.
# Leaving them in makes "how many" look like a match for every table.
_STOP = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "how", "many",
    "what", "which", "who", "whom", "does", "do", "did", "is", "are", "was",
    "were", "be", "there", "with", "by", "from", "that", "this", "than", "had",
    "has", "have", "into", "over", "per", "its", "his", "her", "their", "not",
    "each", "after", "id",
    # "average" sits on A11 (average salary) and in "average loan". It must not choose the table.
    "average",
}

# These columns are short code lists or names. Their real spellings go into the prompt.
# A million-row amount column does not: the model does not need every balance.
_PROBE = {
    "account": ["frequency"],
    "client": ["gender"],
    "disp": ["type"],
    "card": ["type"],
    "loan": ["status"],
    "district": ["A2", "A3"],
    "order": ["k_symbol"],
    "trans": ["type", "operation", "k_symbol"],
}

_LAYER: dict | None = None
_CARDS: list[dict] | None = None
_BM25: BM25Okapi | None = None
_SAVER = MemorySaver()
_GRAPH = None


def make_llm(max_tokens: int = 220) -> ChatOllama:
    """Same Qwen settings as module 15. Temperature 0 keeps a re-run on one path."""
    return ChatOllama(
        model=CHAT_MODEL,
        temperature=0,
        seed=7,
        num_ctx=8192,            # the district-name list plus four tables fits here
        reasoning=False,         # skip qwen3.5's hidden thinking trace
        num_predict=max_tokens,
    )


def ollama_alive() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def words(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text or "")]


def query_words(text: str) -> list[str]:
    """Content words only. 'accounts' is also tried as 'account', so the plural still finds the table."""
    tokens = []
    for token in words(text):
        if token in _STOP or len(token) < 2:
            continue
        tokens.append(token)
        if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
            tokens.append(token[:-1])
    return tokens


def load_layer() -> dict:
    global _LAYER
    if _LAYER is None:
        _LAYER = json.loads(LAYER_PATH.read_text(encoding="utf-8"))
    return _LAYER


def load_questions() -> list[dict]:
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError("questions.json is missing.")
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]


def require_db() -> Path:
    if not DB_PATH.exists():
        raise FileNotFoundError("No data/financial.sqlite. Run: python download_corpus.py")
    return DB_PATH


def connect() -> sqlite3.Connection:
    """A read-only connection. query_only is the database lock. The guard is the one you can read."""
    require_db()
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def qtable(name: str) -> str:
    """Quote a table whose name is a SQL word. `order` is one of those."""
    if name.lower() in {"order", "group", "index", "select"}:
        return f'"{name}"'
    return name


def value_catalog(force: bool = False) -> dict[str, list[str]]:
    """Distinct stored values for the short columns. Cached, because `trans` has a million rows.

    Why this exists: the model will happily filter on 'monthly' or 'east bohemia'.
    Neither string is in the file. The list below is what is in the file.
    """
    if CATALOG_PATH.exists() and not force:
        return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    layer = load_layer()
    catalog: dict[str, list[str]] = {}
    connection = connect()
    try:
        for table, columns in _PROBE.items():
            codes = layer["tables"][table].get("codes") or {}
            for column in columns:
                rows = connection.execute(
                    f"SELECT {column} AS value, COUNT(*) AS n FROM {qtable(table)} GROUP BY 1 ORDER BY n DESC"
                ).fetchall()
                gloss = codes.get(column) or {}
                shown = []
                for row in rows:
                    raw = row["value"]
                    label = "(blank)" if raw is None or str(raw).strip() == "" else str(raw)
                    meaning = gloss.get(label)
                    shown.append(f"{label} = {meaning}" if meaning else label)
                catalog[f"{table}.{column}"] = shown
    finally:
        connection.close()
    DATA.mkdir(parents=True, exist_ok=True)
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    return catalog


def _cards() -> tuple[list[dict], BM25Okapi]:
    """One short card per column, plus one card per district or region name.

    A warehouse with 185 tables (BookSQL, also inside FINCH) cannot paste every
    column into the prompt. This bank has eight tables, and we still link,
    so the transaction table stays out of a question about cards.
    """
    global _CARDS, _BM25
    if _CARDS is not None and _BM25 is not None:
        return _CARDS, _BM25
    layer = load_layer()
    catalog = value_catalog()
    cards = []
    for table, spec in layer["tables"].items():
        codes = spec.get("codes") or {}
        for column, meaning in spec["columns"].items():
            # Short on purpose. A card that lists all 77 districts ranks worse
            # for the one name the question actually used.
            text = f"{table} {column}: {meaning}"
            gloss = codes.get(column) or {}
            if gloss:
                text += ". " + "; ".join(f"{code} means {label}" for code, label in gloss.items())
            cards.append({"table": table, "column": column, "text": text})
            if gloss:
                continue
            for value in catalog.get(f"{table}.{column}") or []:
                if value == "(blank)":
                    continue
                cards.append(
                    {"table": table, "column": column, "text": f"{table} {column} value {value}"}
                )
    _CARDS = cards
    _BM25 = BM25Okapi([words(card["text"]) for card in cards])
    return cards, _BM25


def link_tables(question: str) -> list[str]:
    """Tables for this question, then the bridge table that joins them.

    A word that lives on one or two cards is specific: Pisek, male, gold, unemployment.
    That card's table is in. A word that lives on many cards, such as account or amount,
    is not, unless it is the table's own name.
    Children are not pulled in. Matching account does not add a million transactions.
    If two chosen tables meet only through disp, disp is added, because that is the join.
    """
    cards, _index = _cards()
    tokens = list(dict.fromkeys(query_words(question)))
    if not tokens:
        return []
    layer = load_layer()
    table_names = set(layer["tables"])
    card_words = [(card, set(words(card["text"]))) for card in cards]
    ordered: list[str] = []

    def add(table: str) -> None:
        if table not in ordered:
            ordered.append(table)

    for token in tokens:
        hits = [card for card, tokens_in_card in card_words if token in tokens_in_card]
        if token in table_names:
            add(token)
            continue
        if hits and len(hits) <= 2:
            for card in hits:
                add(card["table"])
    if not ordered:
        return []
    # Add the middle of the only shortest join. card + client brings disp.
    # client + account has two middles (disp and district), so neither is guessed.
    edges: dict[str, set[str]] = {name: set() for name in layer["tables"]}
    for table, spec in layer["tables"].items():
        for fk in spec.get("foreign_keys") or []:
            edges[table].add(fk["ref_table"])
            edges[fk["ref_table"]].add(table)

    def shortest(start: str, goal: str) -> list[list[str]]:
        paths = []
        queue = [[start]]
        while queue:
            path = queue.pop(0)
            if paths and len(path) > len(paths[0]):
                break
            node = path[-1]
            if node == goal and len(path) > 1:
                paths.append(path)
                continue
            for nxt in edges[node]:
                if nxt not in path:
                    queue.append(path + [nxt])
        return paths

    for left in list(ordered):
        for right in list(ordered):
            if left >= right:
                continue
            found = shortest(left, right)
            if len(found) != 1:
                continue
            for table in found[0][1:-1]:
                add(table)
    return ordered[:6]


def schema_text(tables: list[str]) -> str:
    """The semantic layer for the linked tables, including the spellings that are really stored."""
    layer = load_layer()
    catalog = value_catalog()
    blocks = []
    for table in tables:
        spec = layer["tables"][table]
        lines = [f"TABLE {qtable(table)}", spec["purpose"]]
        codes = spec.get("codes") or {}
        for column, meaning in spec["columns"].items():
            lines.append(f"  {column}: {meaning}")
            stored = catalog.get(f"{table}.{column}")
            gloss = codes.get(column) or {}
            if stored:
                lines.append("    stored values: " + "; ".join(stored[:80]))
            elif gloss:
                lines.append("    codes: " + "; ".join(f"{code} = {label}" for code, label in gloss.items()))
        for fk in spec.get("foreign_keys") or []:
            lines.append(f"  foreign key {fk['column']} -> {fk['ref_table']}.{fk['ref_column']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


class SqlDraft(BaseModel):
    sql: str = Field(description="One SQLite SELECT statement. No semicolon. No explanation.")


def extract_sql(text: str) -> str:
    """Keep the statement and drop the prose a model often adds around it."""
    raw = (text or "").strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    match = re.search(r"\b(with|select)\b", raw, flags=re.IGNORECASE)
    if match:
        raw = raw[match.start():]
    return raw.strip().rstrip(";").strip()


def guard_sql(sql: str) -> str:
    """Return an empty string when the statement is one read. Otherwise return the reason it is blocked.

    Why two checks: the read-only connection would also reject a write, but the
    error would look like a database failure. You should see a rule.
    """
    if not sql:
        return "The model returned no SQL."
    stripped = _BLOCK_COMMENT.sub(" ", _LINE_COMMENT.sub(" ", sql))
    if ";" in stripped:
        return "More than one statement. The desk runs a single SELECT."
    if _FORBIDDEN.search(stripped):
        return "This is not a read. Insert, update, delete, and drop are blocked."
    if not re.match(r"(?is)(select|with)\b", stripped):
        return "The statement must start with SELECT or WITH."
    return ""


def execute_sql(sql: str, max_rows: int = 20, seconds: float = 4) -> dict:
    """Run one guarded statement. Stop a scan that runs too long, and do not pull a million rows into memory."""
    connection = connect()
    started = time.time()

    def _too_long() -> int:
        # sqlite calls this every few thousand steps. Non-zero aborts the query.
        return 1 if time.time() - started > seconds else 0

    connection.set_progress_handler(_too_long, 10000)
    try:
        cursor = connection.execute(sql)
        columns = [item[0] for item in cursor.description] if cursor.description else []
        fetched = cursor.fetchmany(max_rows + 1)
        truncated = len(fetched) > max_rows
        rows = [list(row) for row in fetched[:max_rows]]
        return {"columns": columns, "rows": rows, "truncated": truncated, "error": ""}
    except sqlite3.Error as exc:
        return {"columns": [], "rows": [], "truncated": False, "error": str(exc)}
    finally:
        connection.set_progress_handler(None, 0)
        connection.close()


def diagnose(sql: str, result: dict) -> str:
    """A database error, zero rows, or a count of 0. All three usually mean the filter value is not stored that way."""
    if result.get("error"):
        return result["error"]
    if not result.get("rows"):
        return (
            "The query returned zero rows. Copy filter values from the stored-values list, "
            "including capital letters."
        )
    aggregate = re.search(r"(?i)\b(count|sum|avg|min|max)\s*\(", sql or "")
    only = result["rows"][0] if len(result["rows"]) == 1 and len(result["rows"][0]) == 1 else None
    if aggregate and only is not None and only[0] in (0, 0.0):
        return (
            "The query returned 0. A zero here usually means the filter spelling does not match "
            "the stored-values list."
        )
    return ""


def write_sql(question: str, tables: list[str], problem: str = "", previous: str = "") -> str:
    """Ask Qwen for one SELECT. On a repair, the previous statement and the database message go in the prompt."""
    prompt = (
        "Write one SQLite SELECT for the Berka bank.\n"
        "Use only the tables below, and only the tables the question needs.\n"
        "Copy filter values exactly from the stored-values list, including case.\n"
        "Dates are YYYY-MM-DD. A year filter is strftime('%Y', column) compared with the four-digit year.\n"
        "One statement. No semicolon. No INSERT, UPDATE, DELETE, or DROP.\n"
        "Use COUNT(*) for a how-many question.\n\n"
    )
    if "order" in tables:
        # Said only when that table is linked. Said on every question, the model writes ORDER BY order.
        prompt += "The table order is a reserved word. Write it as \"order\".\n"
    prompt += f"{schema_text(tables)}\n\nQuestion: {question}\n"
    if problem:
        prompt += f"\nThe previous SQL failed.\nSQL: {previous}\nProblem: {problem}\nWrite a corrected SELECT.\n"
    try:
        draft = make_llm(240).with_structured_output(SqlDraft, method="function_calling").invoke(prompt)
        raw = draft.sql if draft is not None else ""
    except Exception:
        reply = make_llm(240).invoke(prompt + "\nOutput only the SQL.")
        raw = reply.content or ""
    return extract_sql(raw)


def preview(question: str) -> dict:
    """Link, write, guard, execute, and repair once. No sentence and no person.

    Notebook 2 uses this to score the query. The graph in notebook 3 decides
    whether that result is allowed to leave.
    """
    tables = link_tables(question)
    steps = [f"link -> {', '.join(tables) or '(none)'}"]
    empty = {"columns": [], "rows": [], "truncated": False, "error": ""}
    if not tables:
        return {"tables": [], "sql": "", "result": empty, "problem": "no table", "steps": steps}
    problem = ""
    previous = ""
    sql = ""
    result = empty
    for attempt in range(2):
        sql = write_sql(question, tables, problem=problem, previous=previous)
        steps.append(("repair " if attempt else "generate ") + sql)
        blocked = guard_sql(sql)
        if blocked:
            problem = blocked
            previous = sql
            steps.append("guard " + blocked)
            continue
        result = execute_sql(sql)
        problem = diagnose(sql, result)
        if problem:
            steps.append("execute " + problem)
            previous = sql
            continue
        steps.append(f"execute {len(result['rows'])} row(s)")
        break
    return {"tables": tables, "sql": sql, "result": result, "problem": problem, "steps": steps}


def _norm(value):
    if isinstance(value, float):
        return round(value, 2)
    return value


def same_result(got: list, expected: list) -> bool:
    return [[_norm(cell) for cell in row] for row in got] == [[_norm(cell) for cell in row] for row in expected]


def sql_report() -> list[dict]:
    """For each saved question: did the executed result match the checked result?"""
    rows = []
    for question in load_questions():
        got = preview(question["question"])
        matched = bool(got["result"]["rows"]) and not got["problem"] and same_result(got["result"]["rows"], question["result"])
        rows.append(
            {
                "id": question["id"],
                "match": matched,
                "sql": got["sql"],
                "rows": got["result"]["rows"],
                "expected": question["result"],
                "steps": got["steps"],
            }
        )
    return rows


def is_measured(sql: str, result: dict) -> bool:
    """A count, sum, average, or any numeric cell waits for a person. A card type does not."""
    if re.search(r"(?is)\b(count|sum|avg|min|max)\s*\(", sql or ""):
        return True
    for row in result.get("rows") or []:
        for cell in row:
            if isinstance(cell, (int, float)) and not isinstance(cell, bool):
                return True
    return False


def cells_sentence(result: dict) -> str:
    """A sentence built from the cells, used when the model's sentence adds a number."""
    if len(result["rows"]) == 1 and len(result["rows"][0]) == 1:
        column = result["columns"][0] if result["columns"] else "result"
        return f"{column}: {result['rows'][0][0]}"
    parts = []
    for row in result["rows"][:5]:
        parts.append(", ".join(f"{column}={value}" for column, value in zip(result["columns"], row)))
    return "Result rows: " + "; ".join(parts)


def sentence_grounded(answer: str, result: dict, question: str) -> bool:
    """Every result cell appears in the sentence, and every number in the sentence appears in the cells or the question."""
    flat = answer.replace(",", "").lower()
    allowed = (question + " " + " ".join(str(cell) for row in result["rows"] for cell in row if cell is not None)).replace(",", "").lower()
    for row in result["rows"]:
        for cell in row:
            if cell is None:
                continue
            if str(cell).replace(",", "").lower() not in flat:
                return False
    for raw in _MEASURE.findall(answer):
        if raw.replace(",", "") not in allowed:
            return False
    return True


def write_sentence(question: str, sql: str, result: dict) -> str:
    shown = [dict(zip(result["columns"], row)) for row in result["rows"][:8]]
    prompt = (
        "Write one sentence that answers the question using only the rows. "
        "Copy numbers and labels exactly. Do not invent a value. "
        "If the rows do not answer the question, reply with exactly NOT_IN_DATABASE.\n\n"
        f"Question: {question}\nSQL: {sql}\nRows: {json.dumps(shown, default=str)}"
    )
    reply = make_llm(120).invoke(prompt)
    return (reply.content or "").strip()


class SQLState(TypedDict):
    """What every node can read. A node returns only the keys it changes.

    steps uses operator.add, so each node appends a line. Read the path before the sentence.
    """

    question: str
    tables: list
    sql: str
    result: dict
    problem: str
    answer: str
    repairs_left: int          # one repair. Spent by the repair node, not by the first attempt.
    needs_human: bool
    pause_reason: str
    human_action: str
    steps: Annotated[list[str], operator.add]


def _initial(question: str) -> SQLState:
    return {
        "question": question,
        "tables": [],
        "sql": "",
        "result": {"columns": [], "rows": [], "truncated": False, "error": ""},
        "problem": "",
        "answer": "",
        "repairs_left": 1,
        "needs_human": False,
        "pause_reason": "",
        "human_action": "",
        "steps": [],
    }


# The branch graph. LangChain does the work inside a node. LangGraph chooses the next node.
#
#   greeting                         chat -> END
#   delete / update / insert / drop  refuse -> END     (the SQL is never written)
#   no table matches the question    link -> abstain
#   tables found                     generate -> guard -> execute
#   guard or execute failed, 1 left  repair -> guard
#   still failed                     abstain
#   result is a count, sum, or amount  draft -> human   (interrupt)
#   result is a label, such as a card type  draft -> END
#   person approves or edits         END               (no second model call on the sentence)
#   person rejects, 1 repair left    repair
#   person rejects, repair spent     abstain


# START HERE. Hello, a write, or a bank question.
def route(state: SQLState) -> str:
    """First decision: small talk, a write, or a question about the bank."""
    text = state["question"].lower().strip(" .!?")
    if text in SMALL_TALK:
        return "chat"
    if _WRITE.search(state["question"]):
        return "refuse"
    return "link"


def chat(state: SQLState) -> dict:
    """A greeting. The prompt forbids the database so a hello cannot invent a balance."""
    reply = make_llm(60).invoke(
        "The user greeted you. Reply with one short hello. "
        "Do not mention the database, SQL, or any number.\n\n"
        f"User: {state['question']}"
    )
    return {"answer": (reply.content or "").strip(), "steps": ["chat (database stays closed)"]}


def refuse(state: SQLState) -> dict:
    """A write request ends here. The model is not asked to draft the DELETE."""
    return {"answer": REFUSE, "steps": ["refuse write"]}


def link(state: SQLState) -> dict:
    tables = link_tables(state["question"])
    label = ", ".join(tables) if tables else "(none)"
    return {"tables": tables, "steps": [f"link -> {label}"]}


def after_link(state: SQLState) -> str:
    """No matching table means the question is not about this bank. Do not invent SQL."""
    return "generate" if state["tables"] else "abstain"


def generate(state: SQLState) -> dict:
    sql = write_sql(state["question"], state["tables"])
    return {"sql": sql, "problem": "", "steps": [f"generate {sql}"]}


def guard(state: SQLState) -> dict:
    blocked = guard_sql(state["sql"])
    return {"problem": blocked, "steps": ["guard ok" if not blocked else "guard " + blocked]}


def after_guard(state: SQLState) -> str:
    if not state["problem"]:
        return "execute"
    if state["repairs_left"] > 0:
        return "repair"
    return "abstain"


def execute(state: SQLState) -> dict:
    result = execute_sql(state["sql"])
    problem = diagnose(state["sql"], result)
    if result.get("error"):
        note = "execute error " + result["error"]
    elif problem:
        note = "execute " + problem
    else:
        note = f"execute {len(result['rows'])} row(s)"
    return {"result": result, "problem": problem, "steps": [note]}


def after_execute(state: SQLState) -> str:
    """One repair for a bad filter. After that, a count of zero is shown to a person; a crash abstains."""
    if state["problem"] and state["repairs_left"] > 0:
        return "repair"
    if state["result"].get("error") or not state["result"].get("rows"):
        return "abstain"
    return "draft"


def repair(state: SQLState) -> dict:
    """One new SELECT. This spends the only repair. The person rejecting a draft also lands here."""
    hint = state["problem"] or "The previous SQL was rejected."
    sql = write_sql(state["question"], state["tables"], problem=hint, previous=state.get("sql") or "")
    return {
        "sql": sql,
        "problem": "",
        "repairs_left": state["repairs_left"] - 1,
        "steps": [f"repair {sql}"],
    }


def draft(state: SQLState) -> dict:
    """Turn the grid into one sentence. A number that is not in the grid is replaced, not sent."""
    result = state["result"]
    answer = write_sentence(state["question"], state["sql"], result)
    if "NOT_IN_DATABASE" in answer or not sentence_grounded(answer, result, state["question"]):
        answer = cells_sentence(result)
        note = "sentence rebuilt from the result cells"
    else:
        note = "sentence"
    needs = is_measured(state["sql"], result)
    reason = "The result is a measured number. A person signs the sentence before it is sent to the client." if needs else ""
    return {"answer": answer, "needs_human": needs, "pause_reason": reason, "steps": [note]}


def after_draft(state: SQLState) -> str:
    return "human" if state["needs_human"] else "done"


def human_review(state: SQLState) -> dict:
    """Stop until a person approves, edits, or rejects.

    interrupt() pauses here. On resume, LangGraph runs this function again from
    the top, and interrupt() returns the decision. Do not call the model above
    this line. That work would run twice.

    edit replaces the sentence. There is no second model call. The person is the authority.
    """
    decision = interrupt(
        {
            "question": state["question"],
            "draft": state["answer"],
            "sql": state["sql"],
            "reason": state["pause_reason"],
            "columns": state["result"].get("columns") or [],
            "rows": state["result"].get("rows") or [],
            "tables": state["tables"],
        }
    )
    action = (decision or {}).get("action", "reject")
    if action == "approve":
        return {"human_action": "approve", "steps": ["human approved"]}
    if action == "edit":
        edited = (decision.get("text") or state["answer"]).strip()
        return {"answer": edited, "human_action": "edit", "steps": ["human edited the sentence"]}
    note = (decision or {}).get("text") or ""
    problem = "The person rejected this result."
    if note:
        problem = f"{problem} {note}"
    return {"human_action": "reject", "problem": problem, "steps": ["human rejected the result"]}


def after_human(state: SQLState) -> str:
    """Approve and edit leave. Reject spends the remaining repair, then abstains."""
    if state.get("human_action") in {"approve", "edit"}:
        return "done"
    if state["repairs_left"] > 0:
        return "repair"
    return "abstain"


def abstain(state: SQLState) -> dict:
    return {"answer": ABSTAIN, "needs_human": False, "steps": ["abstain"]}


def build_assistant(checkpointer=None):
    """Compile the branch graph.

    MemorySaver lives in this process. Restarting Streamlit forgets a draft
    that was waiting. A production desk swaps in sqlite or postgres. The nodes stay the same.
    """
    graph = StateGraph(SQLState)
    for name, fn in [
        ("chat", chat),
        ("refuse", refuse),
        ("link", link),
        ("generate", generate),
        ("guard", guard),
        ("execute", execute),
        ("repair", repair),
        ("draft", draft),
        ("human_review", human_review),
        ("abstain", abstain),
    ]:
        graph.add_node(name, fn)
    graph.add_conditional_edges(START, route, ["chat", "refuse", "link"])
    graph.add_edge("chat", END)
    graph.add_edge("refuse", END)
    graph.add_conditional_edges("link", after_link, ["generate", "abstain"])
    graph.add_edge("generate", "guard")
    graph.add_conditional_edges("guard", after_guard, ["execute", "repair", "abstain"])
    graph.add_conditional_edges("execute", after_execute, ["repair", "draft", "abstain"])
    graph.add_edge("repair", "guard")
    graph.add_conditional_edges("draft", after_draft, {"human": "human_review", "done": END})
    graph.add_conditional_edges(
        "human_review",
        after_human,
        {"repair": "repair", "abstain": "abstain", "done": END},
    )
    graph.add_edge("abstain", END)
    return graph.compile(checkpointer=checkpointer or _SAVER)


def get_assistant():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_assistant(_SAVER)
    return _GRAPH


def ask(question: str, thread_id: str | None = None, resume: dict | None = None) -> tuple[dict, dict | None]:
    """Run one question, or resume a paused one.

    Returns (state, interrupt payload). The payload is None when the graph finished.
    A new question needs a new thread, or it continues the old pause.
    """
    thread_id = thread_id or f"branch-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    assistant = get_assistant()
    if resume is None:
        assistant.invoke(_initial(question), config)
    else:
        assistant.invoke(Command(resume=resume), config)
    state = assistant.get_state(config)
    payload = state.interrupts[0].value if state.interrupts else None
    values = dict(state.values)
    values["thread_id"] = thread_id
    return values, payload


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Ask the Berka branch desk, or score the saved questions.")
    parser.add_argument("command", choices=["report", "ask", "resume"])
    parser.add_argument("text", nargs="?", default="")
    parser.add_argument("--thread", default="")
    parser.add_argument("--action", choices=["approve", "reject", "edit"], default="approve")
    args = parser.parse_args()

    if args.command == "report":
        if not ollama_alive():
            sys.exit("Ollama is not running on 127.0.0.1:11434.")
        rows = sql_report()
        matched = sum(row["match"] for row in rows)
        print(f"result match {matched}/{len(rows)}")
        for row in rows:
            mark = "ok  " if row["match"] else "MISS"
            print(f"{mark} {row['id']:32} got={row['rows']} expected={row['expected']}")
            print(f"     {row['sql']}")
        return
    if args.command == "ask":
        if not args.text:
            sys.exit('Usage: python text2sql.py ask "your question"')
        values, payload = ask(args.text)
        print("thread:", values["thread_id"])
        for step in values.get("steps") or []:
            print(" ", step)
        if payload:
            print("\nPAUSED — a person has to sign this sentence.")
            print(payload["reason"])
            print(payload["sql"])
            print(payload["draft"])
            print("Resume with: python text2sql.py resume --thread", values["thread_id"], "--action approve")
        else:
            print("\n" + (values.get("answer") or ""))
        return
    if not args.thread:
        sys.exit("Usage: python text2sql.py resume --thread THREAD --action approve")
    values, payload = ask("", thread_id=args.thread, resume={"action": args.action, "text": args.text})
    print("thread:", values["thread_id"])
    for step in values.get("steps") or []:
        print(" ", step)
    if payload:
        print("\nPAUSED again")
        print(payload.get("draft"))
    else:
        print("\n" + (values.get("answer") or ""))


if __name__ == "__main__":
    main()
