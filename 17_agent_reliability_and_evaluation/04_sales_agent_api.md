# Run the sales agent as an API

A browser or Python client calls the agent by sending JSON with a `question` field. On a successful response, read the `answer` field.

Install the [module packages](README.md#setup) first. Start one server using [Ollama](#ollama) or [Groq](#groq), then try the requests below.

## Ollama

Use the [module setup](README.md#setup). With Ollama running, open a terminal in this folder.

```bash
MODEL_PROVIDER=ollama python -m uvicorn sales_agent_api:app
```

`sales_agent_api:app` points to the `app` object inside [sales_agent_api.py](sales_agent_api.py). FastAPI connects URLs to Python functions and validates the JSON request. Uvicorn listens for HTTP requests and passes them to FastAPI. `MODEL_PROVIDER` defaults to `ollama`, so a fresh PowerShell window can start the same app with `python -m uvicorn sales_agent_api:app`.

Open [the interactive API docs](http://127.0.0.1:8000/docs). Try `GET /health` first. A response of `{"status":"ok"}` shows the API process is responding. A question about March also checks that the app can reach Ollama.

## Send a question and follow its tool call

Under `POST /chat`, choose **Try it out**, paste this JSON, and select **Execute**.

```json
{"question": "Which region had the highest sales in March?"}
```

Look for HTTP `200` and an `answer` that names North and $1,200. The Uvicorn terminal shows which tool ran and how long the agent took.

| Route | Agent | Tool choices |
|---|---|---|
| `POST /chat` | Sales agent | `sales_by_region` |
| `POST /report` | Report agent | `list_report_months`, `compare_months` |

Try `POST /report` with “Which months are available in the sales report?” and “Compare total sales in February and March.” The first request should use `list_report_months`. The second should use `compare_months` and return February at $1,800 and March at $2,700. The route picks the agent, and the model picks from that agent's tools.

## Groq

Groq runs the model on its servers. Your FastAPI process still runs on your computer and executes the Python tools there. It sends the question, instructions, and tool results needed for the answer to the hosted model. This example uses the sample sales data included in the script.

1. Create your own key in the [Groq console](https://console.groq.com/keys).
2. Open a terminal in this folder. Run `python -m pip install -r requirements.txt` if you haven't installed the dependencies yet.
3. Stop the Ollama-backed API with **Ctrl+C** if it is already using port `8000`. On macOS or Linux, put `GROQ_API_KEY=your_actual_key` in a local `.env` file and run the commands below. The included `.gitignore` keeps `.env` out of Git.

```bash
source ./.env
export GROQ_API_KEY
MODEL_PROVIDER=groq python -m uvicorn sales_agent_api:app
```

`source` runs the assignment in your current shell. Only source files you trust. `export` makes the variable available to the Python process. The `MODEL_PROVIDER=groq` prefix sets a separate environment variable for the server command.

On Windows PowerShell, these commands let you enter the key at a hidden prompt.

```powershell
$key = Read-Host "Groq API key" -AsSecureString
$env:GROQ_API_KEY = [System.Net.NetworkCredential]::new("", $key).Password
$env:MODEL_PROVIDER = "groq"
python -m uvicorn sales_agent_api:app
```

`build_model()` reads `MODEL_PROVIDER` at startup. For Groq, `ChatGroq` reads the key from the server environment and uses `openai/gpt-oss-20b` by default. You can set `GROQ_MODEL` before starting the server to use a different [Groq model that supports tool calling](https://console.groq.com/docs/tool-use/overview). Restart the API after changing these settings.

Send the March question again in `/docs`. Expect North and $1,200, with `sales_by_region` in the server log. The wording may change. The URL, request body, and `answer` field work the same way. The notebook or browser only needs the local API URL. Keep the Groq key in the backend environment.

If you want both servers open, keep Ollama on port `8000` and add `--port 8001` to the Groq command. Open [Groq-backed API docs](http://127.0.0.1:8001/docs) for the second instance.

## Call the API from Python or a terminal

These examples use port `8000`. Change it to `8001` if you started Groq on a separate port.

```bash
curl -i http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"Which region led?"}'
```

The question is valid JSON but leaves out a month. Expect `200` and `{"answer":"Which month should I check?"}`.

Try a different request from a Jupyter cell. April is a full month name, so the request reaches the agent.

```python
import httpx

response = httpx.post(
    "http://127.0.0.1:8000/chat",
    json={"question": "Which region had the highest sales in April?"},
    timeout=40,
)
print(response.status_code, response.json())
```

April is a valid month with no rows in the report. Expect `200` and a reply explaining that April data is unavailable.

## Check errors and time limits

These are standard HTTP status codes. The table shows situations that produce them in this app. For example, if Groq sends our backend `429` for rate limiting, the backend returns `503` to its own client because the model service is temporarily unavailable to it.

| Try this | Expected response |
|---|---|
| Send `{}` or `{"question":" "}` to `/chat` | `422`. The request needs a non-blank question. |
| Visit `/chatt` | `404`. That path doesn't exist. |
| Send `GET /chat` | `405`. That route expects POST. |
| Send a March question while the selected model service is unreachable | `503`. The API can't reach Ollama or Groq. |
| Groq rejects the request because its rate limit was reached | `503`. Wait before trying again. The terminal logs the rate limit. |
| Temporarily set `AGENT_TIMEOUT_SECONDS = 0.001`, restart the API, then send a March question | `504` if the wait budget expires. Restore the value to `30` and restart after the check. |

The API maps a missing model or a rejected Groq key to `502`. Check the provider and HTTP status in the server terminal. If Groq mode stops at startup with a message about `GROQ_API_KEY`, set it in the terminal that starts Uvicorn. For an unexpected `500`, check the traceback in that terminal.

Look at `run_agent`. While it awaits the model call, the server can handle another request. Try sending the March question and opening `/health` in a second tab while you wait. `recursion_limit` caps graph steps. `asyncio.wait_for` sets the wait budget in seconds. The log measures elapsed seconds by subtracting the starting `perf_counter()` reading from a later reading.

Continue in [Sales API and evals](00_sales_agent_api_and_evals.ipynb) for the March, missing-month, and blank-input checks. Keep the server terminal open. If you used port `8001`, use that port in the notebook's API URL.

[Groq quickstart](https://console.groq.com/docs/quickstart) · [LangChain ChatGroq setup](https://docs.langchain.com/oss/python/integrations/chat/groq)
