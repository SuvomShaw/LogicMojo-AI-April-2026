# Module 17 — Agent reliability and evaluation

Start with [Sales API and evals](00_sales_agent_api_and_evals.ipynb). Send a sales question, check which tool ran, and use the result to build context for a follow-up.

## Setup

Use Python 3.11 or later. Open a terminal in this folder and install the packages. All sample data is included.

```text
python -m pip install -r requirements.txt
```

Start the API using **one** of these options.

- **Ollama** runs the model locally. [Install and open Ollama](https://ollama.com/download), then run the commands below.
- **Groq** runs the model remotely. Follow [the Groq setup](04_sales_agent_api.md#groq) and leave that server terminal open.

```text
ollama pull qwen3.5:4b
python -m uvicorn sales_agent_api:app --port 8000
```

In a second terminal, use the same Python environment and run `python -m jupyterlab`. Open [Sales API and evals](00_sales_agent_api_and_evals.ipynb), select that environment as the kernel, and run the cells from the top. The first check should return `200` and `{"status": "ok"}`. For server commands and common errors, see [Run the sales API](04_sales_agent_api.md).

The [API script](sales_agent_api.py) is used by the notebook. Follow `chat()` into `run_agent()` to see how the server validates a question, runs the agent, and returns the answer.

## Assignment

After the walkthrough, try [Add a regional sales lookup](08_sales_agent_assignment.ipynb). Extend the API, check the response and tool use, build context for a follow-up, and sketch an AWS deployment.

## Optional labs

Each lab includes its own examples. Only the memory lab needs the sales API. For any lab that lists Ollama, open Ollama and pull `qwen3.5:4b` using the command above.

| Open | What you'll do | Requirements |
|---|---|---|
| [01 · Agent evals and context](01_agent_evals_and_context.ipynb) | Compare two identical replies and find the unnecessary tool call. | Python and Jupyter |
| [02 · Build and evaluate a sales agent](02_sales_agent_design_and_evals.ipynb) | Write a sales tool and test the model's arguments and answer. | Ollama |
| [03 · Retries and human approval](03_agent_reliability.ipynb) | Pause a refund for approval and test a tool timeout. | Installed Python packages |
| [05 · Context and memory](05_context_memory_and_agent_harness.ipynb) | Combine current sales facts, conversation state, and a saved preference. | Sales API and Ollama |
| [06 · Invoice review](06_invoice_review_agent_case_study.ipynb) | Match an invoice against a purchase order and delivery receipt. | Ollama |
| [07 · AWS deployment](07_deploying_agentic_systems_on_aws.md) | Map the API, tools, and model to an AWS deployment. | Reading only |
