# Multi-agent systems and MCP

One person, two drawers, then a receptionist, then a plug.

Read this page once. Then open the demo folders in order. Each folder has a short README: how to run it, a picture of the code, and a checklist of which functions to open.

## The story

**Meera** answers the customer. She has two drawers.

| Drawer | Demo | What is inside |
|---|---|---|
| Paper | `01_document_desk` | Figures and tables in real PDFs and one Word file |
| Bank book | `02_bank_book` | Counts, card types, and other facts in a real bank database |

**Ravi** stands at the door in `03_nova_support`. He does not answer. He points a specialist at one desk. Billing cannot open the help-center search. Technical help cannot move money.

**The plug** is `04_mcp_plug`. It is a standard socket so a second app can call the same tool. The plug does not know more facts than Meera.

## Meera's four rules

1. Open the right drawer. If the fact is in neither drawer, say so.
2. Do not invent a number.
3. Do not delete.
4. If the answer is a measured number, wait for a person to sign it.

```
Customer
   |
 Meera
   |-- hello -----------------> she answers, the drawer stays shut
   |-- a figure or a table ---> paper drawer
   |-- a count or a card -----> bank book
   |-- not in the drawer -----> she says she does not have it
   |-- "delete ..." -----------> she refuses
   |-- the answer is a number -> she waits for a signature
```

When one shop has several jobs that must not share tools, Ravi stands in front:

```
Customer
   |
  Ravi                 points only
   |
   +-- billing         plan, invoices, refund
   +-- technical       search the help center, open a ticket
   +-- account         login, password, seats
```

The model may ask for a refund. The program decides whether that refund is allowed. In Nova that rule is one number in `03_nova_support/config.py`.

## What you will run

Do them in this order. Run one app at a time. Streamlit uses port 8501, so a second app would take the same port.

| Order | Folder | You will see |
|---|---|---|
| 1 | [01_document_desk](01_document_desk/README.md) | A greeting searches nothing. A missing policy is refused. A figure shows the page. A table waits for a signature. |
| 2 | [02_bank_book](02_bank_book/README.md) | The same rules on a database. A delete never becomes SQL. A card type is sent. A count waits. |
| 3 | [03_nova_support](03_nova_support/README.md) | Ravi picks billing, technical, or account. A large refund pauses. |
| 4 | [04_mcp_plug](04_mcp_plug/README.md) | Ten lines that publish two tools. Read the file. You do not need to start it. |
| later | [05_agent_patterns](05_agent_patterns/README.md) | Eight small graphs, with toy tools. Optional. |

## Before you run anything

Ollama must be running, with these models:

```bash
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:4b
ollama pull qwen3-embedding:0.6b
```

`qwen3.5:4b` writes. The document desk embeds pages with `qwen3-embedding:4b`. Nova embeds help articles with `qwen3-embedding:0.6b`.

Use the same Python environment as module 15. In this course that environment is often named `rag_env`. If `python` is the system Python, activate the course environment first, then install inside the demo folder:

```bash
python -m pip install -r requirements.txt
```

The PDFs and the bank file are not in git. Each of the first two demos has `download_corpus.py`. Run it once from that demo's folder. It needs the internet. If Hugging Face asks you to accept the dataset terms, do that, then run the script again.

## What to click

Use the buttons on the screen. The buttons are the prepared questions, so a typo does not hide the path.

| Customer says | What should happen |
|---|---|
| Good morning | She says hello. Nothing is searched. |
| How many HarborDesk leave days? | She says she does not have it. Try this on both desks. |
| What do the red dots show? | She opens the paper drawer and shows the page. |
| What is the PNC loss percent? | She drafts it, then waits. Read the page before you approve. The amount column and the rate column are different. |
| Delete every loan with status D | She refuses. No delete is written. |
| What card does client 13539 have? | A word comes back. No signature. |
| How many accounts are in Pisek? | A count comes back, then she waits. |

Fill these in while you watch:

- Client 13539's card type: ________
- Accounts in Pisek: ________

On Nova, stay with **Priya Sharma** first.

| You type | Watch the right-hand panel |
|---|---|
| Good morning | Ravi sends it straight through. No specialist. |
| What plan am I on? | Billing. She calls the plan tool. She does not search the help center. |
| My board is not syncing | Technical. She must search the help center before she talks. |
| Please refund my last charge of $30 | At or under the limit in `config.py`, the program approves it. |
| Please refund $400 | The graph pauses. You approve or deny. The model does not decide. |

## One person, or a receptionist?

```
One job, one drawer             -> one Meera, one tool, a fixed path
One desk, a few tools           -> one agent
Two jobs that must stay apart   -> Ravi points, a specialist answers
A second app needs the same tool -> a plug (MCP)
A signed answer, still true     -> a cache, later
```

Start with one person. Add Ravi when one person would be holding tools that must not mix.

```
Streamlit  --\
Slack      ---+-->  one plug  -->  one tool
Dashboard  --/
```

Build the plug when a second app needs the same tool.

## How to read the code

Each demo README has the checklist. The habit is the same in every folder:

1. Run the app and click the buttons.
2. Open the README picture.
3. Search the script for `START HERE`.
4. Read the functions in the checklist order. Stop when the picture matches the screen.

The long helper functions exist so the demo can use real files. They are not the lesson. The lesson is the path from the question to the drawer, and the place where a person signs.

## Folder map

```
16_multiagent_mcp/
  README.md                  this page
  01_document_desk/          paper drawer, Streamlit app + 3 notebooks
  02_bank_book/              bank book, Streamlit app + 3 notebooks
  03_nova_support/           Ravi and three specialists
  04_mcp_plug/               the ten-line plug
  05_agent_patterns/         optional notebook of eight small graphs
```
