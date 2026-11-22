"""FastAPI endpoints for the sales and reporting agents."""

import asyncio
import logging
import os
import re
from calendar import month_name
from csv import DictReader
from io import StringIO
from time import perf_counter

import httpx
from fastapi import FastAPI, HTTPException
from groq import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from ollama import ResponseError
from pydantic import BaseModel

AGENT_TIMEOUT_SECONDS = 30
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama").strip().lower()
logger = logging.getLogger("uvicorn.error")


# The tools share the same report. Each row is a single sale.
SALES_CSV = """month,region,sales
March,North,700
March,South,900
March,East,600
March,North,500
February,South,1800
"""
sales_rows = list(DictReader(StringIO(SALES_CSV)))
months = {name.lower() for name in list(month_name)[1:]}


def totals_by_region(month: str) -> dict[str, int]:
    # Both tools that need sales totals use this same Python calculation.
    totals = {}
    requested_month = month.strip().lower()

    for row in sales_rows:
        if row["month"].lower() == requested_month:
            region = row["region"]
            # Start at zero the first time we see a region.
            totals[region] = totals.get(region, 0) + int(row["sales"])

    # An empty dictionary means the report has no rows for that month.
    return totals


@tool
def sales_by_region(month: str) -> dict[str, int]:
    """Return sales totals by region for a full month name, such as March."""
    return totals_by_region(month)


@tool
def list_report_months() -> list[str]:
    """List the months that have rows in the sales report."""
    # The set gives us each month once, even if it has several sales rows.
    return sorted({row["month"] for row in sales_rows})


@tool
def compare_months(first_month: str, second_month: str) -> dict:
    """Compare total sales across all regions for two different full month names."""
    first_month = first_month.strip().title()
    second_month = second_month.strip().title()
    if first_month == second_month:
        return {"error": "Provide two different months."}

    first_totals = totals_by_region(first_month)
    second_totals = totals_by_region(second_month)

    # None means there were no rows for that month. 0 would be a real total.
    return {
        first_month: sum(first_totals.values()) if first_totals else None,
        second_month: sum(second_totals.values()) if second_totals else None,
    }


def build_model():
    """Choose the model client when the API process starts."""
    if MODEL_PROVIDER == "ollama":
        return ChatOllama(model="qwen3.5:4b", temperature=0, reasoning=False)

    if MODEL_PROVIDER == "groq":
        if not os.getenv("GROQ_API_KEY", "").strip():
            raise RuntimeError("Set GROQ_API_KEY in the server terminal before starting Groq.")

        # ChatGroq reads GROQ_API_KEY from this process's environment.
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            temperature=0,
            timeout=AGENT_TIMEOUT_SECONDS,
            max_retries=0,
        )

    raise RuntimeError("MODEL_PROVIDER must be 'ollama' or 'groq'.")


# Both agents use the same model client. Each has its own prompt and tools.
model = build_model()
logger.info("Sales API provider=%s", MODEL_PROVIDER)
sales_agent = create_agent(
    model=model,
    tools=[sales_by_region],
    system_prompt=(
        "Answer questions about the top sales region in a single month. "
        "Call sales_by_region once with the requested full month name. "
        "Use the tool result to report the top region and total. "
        "If there is a tie, name all the winners and their shared total. "
        "If the result is empty, say 'No sales data is available for [month].'"
    ),
)

# Same model, different job and tool list. The report agent can choose a tool
# based on whether Priya wants to see available months or compare totals.
report_agent = create_agent(
    model=model,
    tools=[list_report_months, compare_months],
    system_prompt=(
        "You answer questions about the sales report. "
        "To list available months, call list_report_months. "
        "To compare two months, call compare_months once with both full month names. "
        "If a comparison question gives fewer than two different months, ask for them. "
        "Use the tool result to report both months and their totals. "
        "A null value means no data. Say which month has no data and that a "
        "comparison is unavailable."
    ),
)


# ChatRequest and ChatResponse describe the JSON fields the client uses.
app = FastAPI(title="Sales Agent API")


class ChatRequest(BaseModel):
    # FastAPI reads the JSON body. Pydantic checks that question is present and text.
    question: str


class ChatResponse(BaseModel):
    # The client can always read the answer field on a successful chat response.
    answer: str


@app.get("/health")
def health():
    # This checks the API process without making a model call.
    return {"status": "ok"}


def read_question(request: ChatRequest) -> str:
    # Both routes receive the same JSON, so they use the same blank-input check.
    question = request.question.strip()
    if not question:
        # Pydantic accepts spaces as text. After strip(), we can see it's blank.
        # 422 tells the caller to fix the input. FastAPI sends the detail as JSON.
        raise HTTPException(status_code=422, detail="Please enter a question.")
    return question


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # POST sends a question in the JSON body for the server to process.
    question = read_question(request)

    # Ask for a month before spending a model call on an incomplete question.
    words = set(re.findall(r"[a-z]+", question.lower()))
    if words.isdisjoint(months):
        return {"answer": "Which month should I check?"}

    return await run_agent(sales_agent, question)


@app.post("/report", response_model=ChatResponse)
async def report(request: ChatRequest):
    # Listing available reports does not require a month in the question.
    return await run_agent(report_agent, read_question(request))


async def run_agent(agent, question: str) -> dict[str, str]:
    # Measure the full run, including time spent waiting for model responses.
    started = perf_counter()
    agent_call = agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": 10},  # Stop if the agent keeps looping.
    )
    try:
        # The step limit above and the timeout here cover different failures.
        run = await asyncio.wait_for(agent_call, timeout=AGENT_TIMEOUT_SECONDS)
    except (TimeoutError, httpx.TimeoutException, APITimeoutError) as error:
        logger.warning("Sales agent timed out | provider=%s", MODEL_PROVIDER)
        raise HTTPException(status_code=504, detail="The sales agent took too long.") from error
    except RateLimitError as error:
        logger.warning("Groq rate limit reached")
        raise HTTPException(status_code=503, detail="Groq is rate limited. Try again later.") from error
    except APIConnectionError as error:
        logger.warning("Could not connect to Groq")
        raise HTTPException(status_code=503, detail="Cannot connect to Groq.") from error
    except APIStatusError as error:
        logger.warning("Groq returned HTTP %s", error.status_code)
        raise HTTPException(status_code=502, detail="Groq could not run the model.") from error
    except (ConnectionError, httpx.ConnectError) as error:
        logger.warning("Could not connect to Ollama")
        raise HTTPException(status_code=503, detail="Cannot connect to Ollama.") from error
    except ResponseError as error:
        logger.warning("Ollama returned HTTP %s", error.status_code)
        raise HTTPException(status_code=502, detail="Ollama could not run the model.") from error

    tool_names = [message.name for message in run["messages"] if isinstance(message, ToolMessage)]
    logger.info(
        "Provider: %s | Agent took %.2f seconds | Tools: %s",
        MODEL_PROVIDER,
        perf_counter() - started,
        tool_names,
    )
    # The last message is the model's reply. FastAPI sends the dictionary as JSON.
    return {"answer": run["messages"][-1].text}
