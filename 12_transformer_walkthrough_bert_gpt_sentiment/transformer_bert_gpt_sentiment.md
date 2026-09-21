# Transformer Walkthrough: BERT, GPT, and Encoder-Decoder Models

---

## 1. The big picture

A Transformer turns each input token into a context-aware vector. A word's final vector represents the word **in this sentence**, not just its dictionary meaning.

```text
Text
  │
  ▼
Tokenizer ──► token IDs
  │
  ▼
token embeddings + position embeddings
  │
  ▼
Transformer blocks (attention + FFN, repeated)
  │
  ▼
context-aware vectors
  │
  ├──► BERT: use vectors to understand or classify complete text
  ├──► GPT: use the latest vector to predict the next token
  └──► Encoder-decoder: use source vectors to generate a new sequence
```

### Why position embeddings are needed

Attention alone sees a set of vectors. It does not know whether `dog bites man` or `man bites dog` came first. Adding a different position vector to every token preserves order.

```text
Text:             The      movie      was       not      boring
Token vector:       e1        e2        e3        e4        e5
Position vector:    p1        p2        p3        p4        p5
                    │         │         │         │         │
Input vector:     e1+p1     e2+p2     e3+p3     e4+p4     e5+p5
```

The input tensor has shape `(batch size, sequence length, model dimension)`. For example, a batch of 16 sentences padded to 64 tokens, with hidden size 768, has shape `(16, 64, 768)`.

---

## 2. Worked example: “The movie was not boring.”

The word `not` changes the meaning of `boring`. A useful representation of `boring` must gather information from `not`.

### Self-attention in three steps

Each token creates three learned vectors:

```text
Query (Q): what context am I looking for?
Key   (K): what kind of context can I provide?
Value (V): what information should I contribute if selected?
```

To update `boring`, its query is compared with every token's key. The scores become attention weights, and those weights blend the value vectors.

```text
                    value vector          attention weight

The               [ v1 ]  ─────────────────── 0.05 ─┐
movie             [ v2 ]  ─────────────────── 0.10 ─┤
was               [ v3 ]  ─────────────────── 0.08 ─┤
not               [ v4 ]  ─────────────────── 0.52 ─┤──► new vector for “boring”
boring            [ v5 ]  ─────────────────── 0.25 ─┘
```

These numbers are illustrative. During training, the model learns which words deserve attention. Here, the high weight on `not` lets the new `boring` vector include negation.

```text
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V
                      score → normalize → blend
```

`√dₖ` keeps large dot products from making softmax too sharp. The idea stays the same: score relevant tokens, convert scores to weights, then blend information.

### One attention map

Every row is computed in parallel. A row asks: “which tokens should this token consult?”

```text
                         keys / tokens consulted
                  The   movie   was    not   boring
query: The        [ .      .      .      .      .  ]
       movie      [ .      .      .      .      .  ]
       was        [ .      .      .      .      .  ]
       not        [ .      .      .      .      .  ]
       boring     [ .      .      .    0.52    0.25]
```

### Multi-head attention

One attention head can learn one type of relationship. Multiple heads work in parallel so different heads can learn different patterns.

```text
same input
   │
   ├──► head 1: possible negation relationship
   ├──► head 2: possible subject–verb relationship
   ├──► head 3: possible topic relationship
   └──► head h: another learned relationship
                 │
                 ▼
         concatenate + learned output projection
```

Heads are not manually assigned jobs. The useful patterns are learned from training data.

---

## 3. Inside one encoder block

An encoder block has two main transformations: attention mixes information **between** token positions; a feed-forward network (FFN) transforms information **within** each position.

```text
X: token + position vectors
 │
 ├──────────────── residual shortcut ────────────────┐
 ▼                                                    │
multi-head self-attention ─────────────────────────► Add ─► LayerNorm ─► H

H
 │
 ├──────────────── residual shortcut ────────────────┐
 ▼                                                    │
FFN: Linear → activation → Linear ─────────────────► Add ─► LayerNorm ─► Y
```

| Part | Plain-language job |
|---|---|
| Self-attention | Let each token gather relevant context from other tokens. |
| Residual connection | Keep the earlier information available and help gradients flow through deep stacks. |
| LayerNorm | Keep each token's feature values at a stable scale. |
| FFN | Process the context-aware features at each token position. |

The shape stays the same across the block:

```text
input X:  (batch, sequence length, d_model)
output Y: (batch, sequence length, d_model)
```

Stacking blocks repeatedly refines the context. After several encoder blocks, `boring` represents “boring after the word not,” rather than just the word by itself.

---

## 4. BERT and GPT use attention differently

The key difference is **visibility**.

### BERT-style encoder: complete-input understanding

The entire input is already known, so every real input token may attend to every other real input token.

```text
The   movie   was   not   boring
 ↑      ↑      ↑      ↑       ↑
 └──── every input token can use every other input token ────┘
```

BERT is therefore natural for tasks such as sentiment classification, document retrieval, token tagging, and intent classification.

### GPT-style decoder: next-token generation

When GPT predicts the next token, it may use only the tokens already written. A causal mask hides future positions.

```text
To predict the next token after “was”:

The   movie   was   [next?]   not   boring
 ✓      ✓      ✓       ?       ✗      ✗
```

This rule prevents a model from cheating during training and makes generation possible one token at a time.

```text
prompt → predict one token → append it → predict again → stop
```

GPT is natural for chat, writing, code completion, and any free-form continuation.

---

## 5. Encoder-decoder walkthrough: English to Hindi

Encoder-decoder models are useful when one complete sequence must become another sequence, such as translation or summarization.

```text
Source:  I       love      tea       <EOS>
Target:  मुझे    चाय      पसंद      है       <EOS>
```

### Step A: encode the source once

```text
SOURCE: I       love      tea       <EOS>
          │        │        │
          └── token + position vectors ──┐
                                         ▼
                                    encoder block 1
                                         ▼
                                    encoder block 2
                                         ▼
MEMORY M: [m_I,  m_love,  m_tea,  m_EOS]
```

`M` is called **encoder memory**. It is not one summary vector: it contains one contextual vector for every source token.

### Step B: shift the target during training

The decoder must predict a token without seeing that same token in its own input.

```text
target labels:  [मुझे] [चाय] [पसंद] [है] [<EOS>]
decoder input:  [<BOS>] [मुझे] [चाय] [पसंद] [है]
                  │       │      │       │
                  └──── predict the next target token ────┘
```

This is teacher forcing: correct earlier target tokens are supplied during training, while a causal mask hides later target tokens.

### Step C: use cross-attention to read the source

Each decoder block first looks at earlier target tokens, then looks at encoder memory.

```text
decoder states
      │
      ▼
masked self-attention
      │
      ▼
cross-attention ◄──────── encoder memory M
      │                     keys and values
      ▼
FFN → vocabulary scores → next target token
```

The data-flow rule is important:

```text
Q = decoder states
K = encoder memory M
V = encoder memory M
```

For example, while generating `चाय`, cross-attention may concentrate on the source token `tea`.

```text
source memory token:        I     love    tea   <EOS>
cross-attention weight:   0.04   0.20   0.72   0.04
```

The model learns these weights; people do not hard-code them.

### Step D: generate one token at a time

At inference, the target sentence is unknown.

```text
[I, love, tea, <EOS>] → encoder memory M

[<BOS>]                         → predict मुझे
[<BOS>, मुझे]                   → predict चाय
[<BOS>, मुझे, चाय]              → predict पसंद
[<BOS>, मुझे, चाय, पसंद]        → predict है
...                              → predict <EOS> and stop
```

The encoder processes source tokens in parallel. The decoder still produces new output tokens one at a time.

---

## 6. Which architecture fits the job?

| Need | Good starting architecture | Why |
|---|---|---|
| One fixed label from complete text | BERT-style encoder | Every input token can use the full input context. |
| Free-form text, chat, or code | GPT-style decoder | It is trained to continue text token by token. |
| Translation or summarization | Encoder-decoder | It reads a source sequence fully and generates a different target sequence. |

```text
Complete input → fixed label?       choose an encoder.
Prompt → next token / continuation? choose a decoder.
Source sequence → target sequence?  choose an encoder-decoder model.
```

---
