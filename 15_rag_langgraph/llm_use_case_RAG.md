# LLM Use Cases: Vector Databases, RAG


## Two colleagues, two problems

| Person | Role | Problem |
|---|---|---|
| **Arjun** | Engineer at a bank | His chatbot answers customer questions about products. The answers sound sure and are often wrong or out of date. |
| **Priya** | Product manager at a food-delivery company | She needs live numbers every day. Each question waits two or three days in the data-team queue. |

You will build the reasoning for both solutions. Arjun needs documents. Priya needs tables.

---

# Part 1 — The fundamental problem

## Why a language model gives wrong answers about your company

Arjun deploys a chatbot for bank customers. A customer asks:

> What is the interest rate on your new FlexiSave account launched last week?

The chatbot replies: **“The bank offers savings accounts at 3–4% interest.”**

That number is old. FlexiSave pays a different rate. In banking, a wrong number is a legal problem, not a small mistake.

The model learned during training, and training ended months or years ago. It has never read the FlexiSave document. When it has no fact to use, it still produces the most plausible next words. That behaviour is called **hallucination**.

**The knowledge inside the weights is frozen at the training cutoff.** GPT-4’s cutoff is April 2023. Gemini Flash’s is early 2024. A product launched after that date is invisible to the model. So are internal documents, pricing, and policies. The model has never seen them.

Three responses are usually considered. Only the third stays cheap and current.

```text
Option 1 — Retrain the model
  Feed the bank documents into a new training run.
  Cost: on the order of $500,000. Time: months.
  The moment training finishes, next week's launch is missing again.

Option 2 — Paste every document into the prompt
  "Here are our 50,000 product documents…"
  Cost: on the order of $500 per question.
  The context window overflows, and the model loses focus.

Option 3 — Vector database + RAG
  Index the documents once.
  On each question, fetch only the 3 or 4 most relevant pieces.
  Put those pieces into the prompt.
  Cost: fractions of a cent per question.
  Update a document, and the next question sees it. No retraining.
```

Option 3 is what production systems use. The rest of these notes builds it, then adds the database path Priya needs.

---

# Part 2 — Vector databases

## 2.1 What a vector is

A **vector** here is a list of numbers. It is a point in a high-dimensional space, and it stands for the meaning of a piece of text.

```text
"Dog"              → [0.92, 0.01, 0.45, 0.33, -0.12, ...]   384 numbers
"Cat"              → [0.89, 0.02, 0.44, 0.35, -0.10, ...]   384 numbers
"Python language"  → [0.12, 0.78, 0.23, 0.91,  0.67, ...]   384 numbers
"Stock market"     → [0.05, 0.11, 0.88, 0.02,  0.43, ...]   384 numbers
```

Texts with similar meanings receive numerically similar vectors. Dog and cat are both domestic pets, so their vectors sit near each other. Dog and the Python programming language are unrelated, so their vectors sit far apart.

**Check yourself.** If you embed “physician” and “doctor”, will the vectors be close or far apart? What about “physician” and “hospital”? What about “physician” and “restaurant”?

Physician and doctor are synonyms, so the vectors are very close. Physician and hospital share a domain, so they are somewhat close. Physician and restaurant are far apart.

That property does work that keyword search cannot do.

```text
Keyword search
  SELECT * FROM documents WHERE text LIKE '%home loan%'
  Finds documents that contain the exact words "home loan".
  Misses "housing mortgage", "property financing", and "EMI for house purchase".

Semantic search
  FIND documents WHERE meaning_similar_to("home loan") > 0.7
  Finds documents about the same idea, including those three phrases.
```

## 2.2 The embedding model

The component that turns text into a vector is an **embedding model**. It is a neural network trained for this job, separate from the chat model that writes the final answer.

```text
"FlexiSave offers 7% interest on balances above ₹1 lakh"
                        │
                        ▼
            Embedding model: all-MiniLM-L6-v2
                        │
                        ▼
        [0.234, -0.891, 0.445, 0.123, ..., -0.567]    384 numbers
```

| Model | Dimensions | Speed | Cost | Use when |
|---|---:|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Very fast | Free | Learning, prototypes, and many production systems |
| `nomic-embed-text` (Ollama) | 768 | Fast | Free, local | The text must stay on your machine |
| `text-embedding-3-small` (OpenAI) | 1536 | Fast | Paid per token | Retrieval quality is the priority |
| `text-embedding-004` (Google) | 768 | Fast | Free tier available | You are already on Google Cloud |

Start with `all-MiniLM-L6-v2`. It is about 80 MB, it runs on CPU, and it is strong enough for a large share of real retrieval tasks. The HarborDesk notebooks in this folder use a local Ollama embedding model for the same reason: the documents never leave the machine.

## 2.3 Cosine similarity

To rank documents against a question, measure the angle between the two vectors.

```text
cosine similarity = (A · B) / (|A| × |B|)

A · B     dot product: multiply matching elements, then add
|A|, |B|  magnitudes: the length of each vector

 1.0   identical direction (the same text twice)
 0.7+  highly similar (same topic, different words)
 0.3   loosely related
 0.0   unrelated
-1.0   opposite direction (uncommon for ordinary text embeddings)
```

```text
Query:      "What is the EMI for a home loan?"
Embedding:  [0.8, 0.3, 0.6, 0.1, ...]

Document A: "Home loan EMI calculator — 20 year tenure"
Embedding:  [0.79, 0.28, 0.61, 0.09, ...]
Similarity: 0.97    same topic

Document B: "Car insurance premium calculation"
Embedding:  [0.2, 0.9, 0.1, 0.7, ...]
Similarity: 0.12    different topic
```

Cosine similarity uses the angle, so two texts about the same topic can match even when one vector is longer than the other. A short sentence and a long paragraph can point the same way. Euclidean distance mixes that angle with length, which is why text search standardises on cosine.

## 2.4 Index once, query many times

A vector database stores a large number of vectors and returns the nearest ones to a query in milliseconds.

```text
INDEXING — once, when you load or update documents

Raw documents (PDFs, pages, wikis)
        │
        ▼
   Text splitter          break each document into chunks
        │
        ▼
   Embedding model        one vector per chunk
        │
        ▼
   Vector database        vector + original text + metadata
        │
   ChromaDB / Pinecone / Qdrant / FAISS


QUERYING — every user question

User question
        │
        ▼
   Embedding model        the same model you used while indexing
        │
        ▼
   Vector database        top-K nearest vectors
        │
        ▼
   Retrieved chunks       these go into the prompt
```

Use one embedding model for both phases. A store built with `all-MiniLM-L6-v2` (384 dimensions) must be queried with that same model. A different model lives in a different vector space, even when the dimension count happens to match. Search results from a mismatched pair are noise.

## 2.5 Chunking

Before indexing, split each document into **chunks**. This split decides what the search is allowed to find. It is the decision teams underestimate most often.

Chunk for three reasons:

1. Embedding quality drops on very long text, because the vector has to represent too many ideas at once.
2. A short, focused chunk retrieves a precise passage.
3. The prompt can hold only a few chunks. You want those chunks to be the right ones.

Five ways to cut the same note, ranked by how carefully they keep meaning:

1. **Fixed size.** Cut every N characters. Simplest to write, and the weakest result, because the cut ignores sentences.
2. **Recursive character split.** Try a paragraph, then a sentence, then a word, then a character. This is the usual LangChain default, and it is good enough for most documents.
3. **Sentence-based.** Split on full stops, question marks, and exclamation marks. A sound default for clean prose, FAQs, and policies.
4. **Semantic chunking.** Start a new chunk where the meaning changes, detected by embedding similarity. Best quality, and slower, because every sentence is embedded.
5. **Sliding window with overlap.** Repeat a strip of the previous chunk at the start of the next one, so a fact is not stranded on a boundary. Use this together with a fixed or recursive split.

The FlexiSave note below is 805 characters. The samples use a short window so the cut is visible. On a real policy, the same methods use a window of about 500 characters.

> The FlexiSave account offers 7% interest on balances above ₹1 lakh. Balances of ₹1 lakh or below earn the standard rate of 3.5% per annum. The minimum balance for FlexiSave is ₹10,000. If the balance stays below ₹10,000, the non-maintenance fee is ₹600 per quarter. FlexiSave holders get free NEFT, free RTGS, and unlimited ATM withdrawals at HDFC ATMs. Interest is credited at the end of each quarter. To open FlexiSave you need a PAN card, an Aadhaar card, and one cancelled cheque. A joint FlexiSave account can have two holders, and both must complete KYC. FlexiSave does not offer a cheque book. Use NEFT or RTGS for large transfers. The 7% rate applies only to the amount above ₹1 lakh, not to the whole balance. Closing FlexiSave in the first year costs ₹500. After the first year, closure is free.

### 1. Fixed size

Cut every 80 characters, even in the middle of a word.

```text
[0–80]    The FlexiSave account offers 7% interest on balances above ₹1 lakh. Balances of
[80–160]  ₹1 lakh or below earn the standard rate of 3.5% per annum. The minimum balance f
[160–240] or FlexiSave is ₹10,000. If the balance stays below ₹10,000, the non-maintenance
```

The first cut falls after “Balances of”. The rate 3.5% is in the next chunk, so the first chunk never says what those balances earn. The second cut leaves a leftover “f”, and the third chunk starts with “or FlexiSave”.

### 2. Recursive character split

Try a paragraph boundary, then a sentence, then a word, then a character. Cut a word only when one sentence is still too long. This is the usual LangChain default. With a limit of about 180 characters, the note becomes:

```text
[1] The FlexiSave account offers 7% interest on balances above ₹1 lakh.
    Balances of ₹1 lakh or below earn the standard rate of 3.5% per annum.

[2] The minimum balance for FlexiSave is ₹10,000.
    If the balance stays below ₹10,000, the non-maintenance fee is ₹600 per quarter.

[3] FlexiSave holders get free NEFT, free RTGS, and unlimited ATM withdrawals at HDFC ATMs.
    Interest is credited at the end of each quarter.
```

Each piece ends on a full stop. “3.5% per annum” stays with the sentence that introduces it.

### 3. Sentence-based

One chunk is two sentences. Nothing ends mid-sentence.

```text
[1] The FlexiSave account offers 7% interest on balances above ₹1 lakh.
    Balances of ₹1 lakh or below earn the standard rate of 3.5% per annum.

[2] The minimum balance for FlexiSave is ₹10,000.
    If the balance stays below ₹10,000, the non-maintenance fee is ₹600 per quarter.

[3] FlexiSave holders get free NEFT, free RTGS, and unlimited ATM withdrawals at HDFC ATMs.
    Interest is credited at the end of each quarter.
```

This is a sound default for a policy or an FAQ. A rule that needs three sentences to make sense can still be split across two chunks.

### 4. Semantic

Read sentence by sentence and start a new chunk where the topic changes. The break is detected by embedding similarity: when the embedding of the next sentence moves far from the current chunk, start a new chunk.

```text
interest rates
  The FlexiSave account offers 7% interest on balances above ₹1 lakh.
  Balances of ₹1 lakh or below earn the standard rate of 3.5% per annum.

minimum balance
  The minimum balance for FlexiSave is ₹10,000.
  If the balance stays below ₹10,000, the non-maintenance fee is ₹600 per quarter.

payments
  FlexiSave holders get free NEFT, free RTGS, and unlimited ATM withdrawals at HDFC ATMs.

opening the account
  To open FlexiSave you need a PAN card, an Aadhaar card, and one cancelled cheque.
  A joint FlexiSave account can have two holders, and both must complete KYC.

closure
  Closing FlexiSave in the first year costs ₹500.
  After the first year, closure is free.
```

“7% interest…” and “3.5% per annum” stay together because they are one topic. The minimum-balance rule starts its own chunk. Finding those breaks means embedding every sentence, so this costs more than the other methods. It pays off on long documents whose topics shift inside a section.

### 5. Sliding window with overlap

Use the same 80-character cut, and let each new chunk repeat the last 20 characters of the previous one.

```text
Chunk A  0–80
The FlexiSave account offers 7% interest on balances above ₹1 lakh. Balances of

Chunk B  60–140
1 lakh. Balances of ₹1 lakh or below earn the standard rate of 3.5% per annum. T
         ^^^^^^^^^^^^^^^^^^^
         these characters are also at the end of Chunk A

Chunk C  120–200
of 3.5% per annum. The minimum balance for FlexiSave is ₹10,000. If the balance
```

“Balances of ₹1 lakh” was sliced in Chunk A. In Chunk B the same words appear again, followed by “earn the standard rate of 3.5%”. The rate is no longer stranded. The cost is storing that repeated text twice.

On a long policy the same pattern is a 500-character window with 100 characters of overlap: 0–500, then 400–900, then 800–1300.

### Practical rules of thumb

| Document | Chunk size | Overlap |
|---|---|---|
| Short FAQ answers | 128–256 tokens | small |
| Policy documents | 400–600 tokens | about 80 tokens |
| Code | split by function or class, not by character count | — |
| Legal documents | 600–1000 tokens | about 150 tokens |

## 2.6 Popular vector databases

| Database | Type | Production use | When to choose it |
|---|---|---|---|
| **ChromaDB** | Open source, embedded | Small to medium scale | Learning, prototypes, and early startups. It runs inside the same Python process. |
| **FAISS** | Open source, in memory | Research | You need raw speed and can rebuild the index when the process restarts. |
| **Qdrant** | Open source, production server | Self-hosted production | You want full control and a store you run yourself. |
| **Pinecone** | Managed cloud | Large scale | You want zero operations and will pay per use. |
| **Weaviate** | Open source and cloud | Multimodal | You store text and images together. |
| **pgvector** | PostgreSQL extension | Teams already on Postgres | You want vectors in the same database as the rest of the product. |

Teams usually start with ChromaDB, because it needs no separate server. They move to Qdrant (self-hosted, often in Docker) or Pinecone (managed) when they need:

- the index to survive a new deploy;
- separate data for each tenant;
- filters over millions of documents;
- a service-level guarantee.

---

# Part 3 — RAG: retrieval-augmented generation

## 3.1 The idea

Arjun now has the bank’s product documents in ChromaDB. RAG is how the chatbot uses them.

**Before answering, search for the relevant pages, then ask the model to answer from those pages.**

Treat it as an open-book exam. A plain model answers from memory and can be wrong. A RAG system reads the specific pages first, then writes.

```text
Retrieval    find the relevant chunks
Augmented    add those chunks to the prompt
Generation   the model writes the answer from that prompt
```

## 3.2 The complete RAG pipeline

```text
INDEXING (done once)
  Your documents → split into chunks → embed → store in the vector database

QUERYING (every user question)

  User: "What is the interest rate on FlexiSave?"
          │
          ▼
    1. RETRIEVE
       Embed the question with the same model used at index time.
       Search the vector database.
       Take the top 3 chunks.
          │
          ▼
    2. AUGMENT
       Build this prompt:

       ┌──────────────────────────────────────────────────┐
       │ CONTEXT FROM THE PRODUCT DOCUMENTS:              │
       │ [Chunk 1: FlexiSave overview — 7% on ₹1 lakh+]   │
       │ [Chunk 2: FlexiSave charges and features]        │
       │ [Chunk 3: New products, January 2025]            │
       │                                                  │
       │ Answer ONLY from the context above.              │
       │ Never invent numbers or policies.                │
       │                                                  │
       │ Question: What is the FlexiSave interest rate?   │
       └──────────────────────────────────────────────────┘
          │
          ▼
    3. GENERATE
       The model reads that prompt and writes a grounded answer:
       "According to the FlexiSave product guide, the account
        offers 7% per annum on balances above ₹1 lakh."
```

## 3.3 Grounding

Retrieved pages are not enough on their own. If the prompt only says “you are a helpful bank assistant”, the model mixes those pages with facts from training. Part of the answer can be right and part invented, and a reader cannot see the join.

A **grounding** instruction tells the model to stay inside the retrieved text and to refuse when the text is silent.

```text
You are a helpful bank assistant.
Answer only using the information in the context provided.
If the answer is not in the context, say:
"I don't have that information. Please call our helpline at 1800-202-6161."
Do not invent interest rates, fees, or eligibility rules.
```

A strict prompt is a strong request. It is still a request. Evaluation, in section 3.5, is how you check that the model obeyed it.

## 3.4 Naive RAG vs production RAG vs advanced RAG

Most tutorials teach **naive RAG**. A system you ship uses **production RAG**. Teams that need higher recall add the steps under **advanced RAG**.

### Naive RAG (what you learn first)

```text
User question → embed → search → top 3 chunks → prompt → answer
```

- Simple, and fast to implement.
- A short or vague question retrieves poorly.
- Different vocabulary from the document means the right page is missed.
- No citation tracking.
- No check on the quality of the answer.

### Production RAG (what you ship)

```text
User question → embed → search → top-K chunks → format with citations → prompt → answer
```

What this adds:

- a composable chain, such as LangChain LCEL or the LangGraph flow in notebook 2, so each step can be tested on its own;
- source metadata kept on every chunk;
- a citation in every answer;
- chunking with overlap;
- conversation history, so a follow-up still has the earlier turns.

### Advanced RAG (what stronger systems add)

**1. Query rewriting**

```text
User:     "EMI?"
Rewriter: "What is the EMI calculation for a home loan?"
```

“EMI?” is a weak search string. The rewritten question uses the words the documents actually contain, so retrieval improves.

**2. HyDE (hypothetical document embeddings)**

Documents look like answers. A short question does not. Generate a hypothetical answer and embed that, instead of embedding the question.

```text
Question:  "What is the FlexiSave interest rate?"
HyDE text: "The FlexiSave account offers 7% per annum on balances above ₹1 lakh..."
Embed the HyDE text, then search.
```

The hypothetical answer sits closer to the real document vectors than the question does.

**3. Multi-query retrieval**

Generate three phrasings of the same question and search with each one. Merge the hits and drop duplicates.

```text
"What is the FlexiSave interest rate?"
"How much does a FlexiSave account pay?"
"FlexiSave account returns percentage"
```

A document that matches one phrasing and misses the others is still retrieved. Recall often rises by about 15–20%.

**4. Re-ranking**

```text
Step 1  Retrieve the top 20 candidates with a fast approximate search.
Step 2  Re-rank those 20 with a cross-encoder, which is slower and more precise.
Step 3  Keep the top 3 from the re-ranked list.
```

A **bi-encoder** (the ordinary embedding model) turns the question into numbers and each document into numbers separately. Search asks which numbers are closest. A **cross-encoder** reads the question and one document together and scores that pair. It can see that “7% above ₹1 lakh” answers “FlexiSave interest rate”. It is too slow to run on every document, so it only re-scores the 20 candidates.

**5. Parent-document retrieval**

```text
Index:   small chunks (about 128 tokens) → precise retrieval
Store:   large parent chunks (about 512 tokens)
Return:  the parent chunk to the model when a small chunk matches
```

Small chunks make the search precise. The larger parent gives the model enough surrounding text to answer. You get both.

## 3.5 How you know it works

RAGAS is a common evaluation set. Three scores matter on every project. Each runs from 0 to 1.

**Faithfulness.** Is the answer supported by the retrieved context?

```text
Answer:  "FlexiSave pays 7% per annum on balances above ₹1 lakh."
Context: "FlexiSave offers 7% p.a. on balances above ₹1 lakh."
Faithfulness: 0.98

Answer:  "FlexiSave is the best savings account in India."
Context: features of FlexiSave, with no comparison
Faithfulness: 0.2     the model added an opinion
```

**Answer relevance.** Does the answer address the question?

```text
Question: "What documents do I need for a home loan?"
Answer:   "The bank offers home loans at competitive rates."
Answer relevance: 0.05     the question was about documents
```

**Context precision.** Were the retrieved chunks the right ones?

```text
Five chunks retrieved: 3 about home loans, 2 about car loans
Context precision: 3/5 = 0.60
```

A practical bar for a system you would ship:

| Metric | Target |
|---|---|
| Faithfulness | above 0.85 |
| Answer relevance | above 0.80 |
| Context precision | above 0.75 |

When a score is below its bar, change chunking and retrieval first. Swapping the chat model rarely fixes a search that returned the wrong pages.

When an answer is wrong, name which stage failed:

1. Search returned the wrong page.
2. Search returned the right page and the model ignored it.
3. No page contained the fact, and the model answered anyway.

---

# Part 4 — Text-to-SQL

## 4.1 Priya’s queue

Every number Priya needs follows the same path:

1. She writes the question in Slack.
2. The data team triages it, writes SQL, tests it, and runs it.
3. Two or three days later she receives a CSV.
4. She has a follow-up. The path starts again.

Text-to-SQL turns the English question into a query and runs it.

```text
Priya: "Show me the top 5 restaurants by order count in Mumbai last month."

SELECT r.name, COUNT(o.id) AS order_count
FROM orders o
JOIN restaurants r ON o.restaurant_id = r.id
WHERE r.city = 'Mumbai'
  AND strftime('%Y-%m', o.created_at) = strftime('%Y-%m', DATE('now', '-1 month'))
GROUP BY r.name
ORDER BY order_count DESC
LIMIT 5;
```

The result comes back in a fraction of a second. The SQL is shown with the answer so Priya can see what was counted.

## 4.2 What the model needs in the prompt

The model can write SQL. It cannot see your tables unless you put the schema in the prompt.

```text
You are a SQL expert. Write accurate SQLite queries.

DATABASE SCHEMA
Table: orders
  id            INTEGER PRIMARY KEY
  customer_id   INTEGER   -- FK to customers.id
  total_amount  REAL      -- order value in rupees
  status        TEXT      -- 'placed', 'delivered', and so on
  created_at    TEXT      -- timestamp YYYY-MM-DD HH:MM

RULES
1. Write only SELECT queries.
2. Use SQLite date functions.
3. Return only the SQL, with no explanation.

User: How many orders were placed today?
Model: SELECT COUNT(*) FROM orders WHERE DATE(created_at) = DATE('now');
```

The comments on each column matter as much as the types. They are how the model learns your names for things.

## 4.3 Five problems that break a first version

**1. The schema is too big.** A real database has 100 to 500 tables. Pasting every table into one prompt hits the token limit, and the model stops attending to the tables that matter.

Use **schema RAG**. Index a short description of each table in a vector database. For each question, retrieve the three to five relevant tables and put only those in the prompt. This is the same retrieve-then-read pattern from Part 2.

**2. The user’s words are not the column names.** Priya says “GMV”. The column is `total_amount`. A model that guesses `SUM(gmv)` fails, because that column does not exist.

Put the business name in the column comment:

```text
total_amount REAL  -- gross merchandise value (GMV) in rupees
```

The model can now map “GMV” to `total_amount`.

**3. The SQL dialect is wrong.** The same date idea has different syntax:

```text
MySQL:       DATE_FORMAT(created_at, '%Y-%m')
PostgreSQL:  TO_CHAR(created_at, 'YYYY-MM')
SQLite:      strftime('%Y-%m', created_at)
BigQuery:    FORMAT_DATE('%Y-%m', DATE(created_at))
```

State the engine in the system prompt: “This is a SQLite database. Use SQLite syntax.”

**4. The query can be dangerous.** “Delete all orders from last year” and “show me all customer passwords” are both plain English. A system that runs whatever it generates will run those too.

Check the SQL before it executes:

- it must start with `SELECT`;
- reject `INSERT`, `UPDATE`, `DELETE`, `DROP`, and `ALTER`;
- open a read-only database connection;
- set a timeout, for example 5 seconds.

**5. The model invents a column or puts it on the wrong table.** It writes `orders.city = 'Mumbai'` when `city` lives on `restaurants`.

Repair it in a short loop:

1. Run the query.
2. If the database returns an error such as `sqlite3.OperationalError`, send that error back to the model with the schema.
3. Retry once. One retry is usually enough.

## 4.4 A production layout

```text
User question
      │
      ▼
1. Schema RAG
   Embed the question, search the schema index, keep the relevant tables.
      │
      ▼
2. SQL generation
   Those tables + a few worked examples + the rules → the model → one query.
      │
      ▼
3. Validation
   SELECT only. No forbidden keywords. Reject anything else before it runs.
      │
      ▼
4. Execution
   Read-only connection, 5-second timeout.
   On error: send the error back, retry once.
      │
      ▼
5. Interpretation
   The model explains the table in plain language.
   "Mumbai had the highest GMV at ₹45.2 lakh, up 12% from last month."
      │
      ▼
The user receives the answer, the explanation, and the SQL.
```

Returning the SQL is part of the product. Priya can check the join, and an auditor can replay the question.

## 4.5 Few-shot examples

Three to five worked question-and-SQL pairs in the prompt often raise accuracy by 30–50% on complex queries. Pick examples that match the questions people actually ask: a count, a rate, and a join with a date.

```python
FEW_SHOT_EXAMPLES = [
    {
        "question": "How many orders were placed today?",
        "sql": "SELECT COUNT(*) AS total FROM orders WHERE DATE(created_at) = DATE('now');",
    },
    {
        "question": "Cancellation rate by payment method",
        "sql": """SELECT payment_mode,
       ROUND(100.0 * SUM(CASE WHEN status='cancelled' THEN 1 ELSE 0 END) / COUNT(*), 2)
       AS cancellation_pct
FROM orders GROUP BY payment_mode ORDER BY cancellation_pct DESC;""",
    },
    {
        "question": "Average delivery time in minutes by city",
        "sql": """SELECT r.city,
       ROUND(AVG((julianday(o.delivered_at) - julianday(o.created_at)) * 24 * 60), 1)
       AS avg_minutes
FROM orders o JOIN restaurants r ON o.restaurant_id = r.id
WHERE o.status = 'delivered' AND o.delivered_at IS NOT NULL
GROUP BY r.city ORDER BY avg_minutes;""",
    },
]
```

Few-shot examples do not teach SQL. The model already knows SQL. They teach your conventions: which columns you join on, how you compute dates, and how you order the result. Two or three examples are enough to shift the style.

---

# Part 5 — Documents and tables in one system

In production the two pipelines sit behind one question.

```text
                User question
                      │
                      ▼
              Intent classifier
             "Documents or data?"
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
      RAG pipeline        SQL pipeline
      policies            live counts
      product pages       reports
      FAQs
            │                   │
            └─────────┬─────────┘
                      ▼
               Write the answer
          (and attach sources or SQL)
```
