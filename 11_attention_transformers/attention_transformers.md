# Attention and Transformers

## 1. Why attention?

An RNN, LSTM, or GRU reads text one word at a time. To process word 50, it must first process words 1 to 49. Its information about the past is carried through a single evolving hidden state.

Attention changes the question from “what information survived the journey?” to “which words are relevant right now?” A token can look directly at any other relevant token.

```text
Sentence: France   is   known   for   its   French   language

Sequential recurrence:
France ──► is ──► known ──► for ──► its ──► French
             information must travel step by step

Self-attention:
French ──────────────────────────────────────────► France
        can directly gather useful context
```

This direct connection is useful for long-distance relationships such as negation, a pronoun and its noun, or a verb and its subject.

---

## 2. Query, Key, and Value

Self-attention gives every token three learned views of itself:

```text
Query (Q): “What information am I looking for?”
Key   (K): “How can other tokens find me?”
Value (V): “What information do I contribute if selected?”
```

For an input embedding `xᵢ`, learned weight matrices create these vectors:

```text
qᵢ = xᵢ W_Q
kᵢ = xᵢ W_K
vᵢ = xᵢ W_V
```

The same token embedding is projected in three different ways because asking, matching, and contributing are different jobs.

### Worked example

Suppose the current token is `French`. It compares its query with the keys for `France`, `is`, and `French`.

```text
Step 1 — score each possible source token

Q_French Kᵀ = [ 0.83,  0.51, -1.27 ]
               France   is   French

Step 2 — normalize scores using softmax

softmax(scores) = [ 0.54,  0.39,  0.07 ]
                  France   is   French

Step 3 — blend the value vectors using those weights

new vector for French
  = 0.54 × V_France + 0.39 × V_is + 0.07 × V_French
```

`France` receives the largest weight, so its information has the largest influence on the new vector for `French`.

The complete formula is:

```text
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V
                      score   normalize  blend
```

The division by `√dₖ` prevents large vector dimensions from creating scores so large that softmax becomes too sharp to learn well.

---

## 3. Self-attention sees a whole sequence at once

For a sequence of `n` tokens, attention produces an `n × n` matrix. Each row sums to 1.

```text
                        keys / tokens consulted
                 France    is    French
query France     [ 0.40   0.30   0.30 ]
      is         [ 0.25   0.50   0.25 ]
      French     [ 0.54   0.39   0.07 ]
```

Rows are queries: “what should this token read?” Columns are keys: “how much should this token be consulted?” All rows can be calculated in parallel, unlike sequential recurrence.

---

## 4. Multi-head attention

One attention calculation may learn one useful type of relationship. Multi-head attention runs several calculations in parallel and combines their outputs.

```text
input vectors
   │
   ├──► head 1: may learn nearby grammar patterns
   ├──► head 2: may learn a long-distance reference
   ├──► head 3: may learn topical similarity
   └──► head h: may learn another useful pattern
                  │
                  ▼
       concatenate head outputs → output projection
```

The labels above are examples, not hand-written rules. During training, each head learns patterns that help the final task.

---

## 5. Attention needs position information

Attention does not know token order by itself. If all token and position information were shuffled together, the attention operation would not know which word originally came first.

```text
dog bites man  ≠  man bites dog
```

Transformers add a position vector to every token embedding before the first attention block.

```text
Token:       dog       bites      man
embedding:    e1         e2        e3
position:     p1         p2        p3
              │          │         │
input:      e1+p1      e2+p2     e3+p3
```

Some models use fixed sine/cosine position vectors; many modern models learn position embeddings. The essential idea is the same: token identity tells the model **what** a word is, and position tells it **where** the word is.

---

## 6. The Transformer block

A Transformer block combines attention with a small per-token neural network. Residual connections and normalization make deep stacks trainable.

```text
input vectors (+ positions)
      │
      ├──────── residual shortcut ──────────────┐
      ▼                                         │
multi-head self-attention ───────────────────► Add ─► LayerNorm
                                                       │
                                                       ▼
                                      FFN: Linear → activation → Linear
                                                       │
      ┌──────── residual shortcut ────────────────────┘
      ▼
     Add ─► LayerNorm ─► output vectors
```

| Component | Job |
|---|---|
| Multi-head self-attention | Mix context across tokens. |
| Residual connection | Preserve earlier information and provide a short path for gradients. |
| LayerNorm | Keep feature scales stable for each token. |
| Feed-forward network (FFN) | Transform each token's context-aware features independently. |

One block preserves shape, so blocks can be stacked:

```text
(batch, sequence length, d_model)
              │
              ▼
(batch, sequence length, d_model)
```

---

## 7. Encoder attention and decoder attention

The same building blocks can be used with different visibility rules.

### Encoder: full-context attention

An encoder processes a complete input. Every real input token can use every other real input token.

```text
The   movie   was   not   boring
 ↑      ↑      ↑      ↑       ↑
 └──── every input token can consult every other input token ────┘
```

This fits understanding tasks such as classification, retrieval, and tagging.

### Decoder: causal attention

A decoder generates text one token at a time. It can use earlier tokens but not future tokens.

```text
To predict the next token after “was”:

The   movie   was   [next?]   not   boring
 ✓      ✓      ✓       ?       ✗      ✗
```

The causal mask hides future positions. This is why decoder models can be used for chat, continuation, and code generation.

---

## 8. From Transformer blocks to pretrained models

Large Transformer models are usually first trained on huge amounts of text, then adapted to a new task.

```text
large text collection
        │
        ▼
pretraining: learn general language patterns
        │
        ▼
pretrained Transformer
        │
        ├──► use directly for an appropriate task
        └──► adapt with labelled examples for a specific task
```

Always pair a pretrained model with its matching tokenizer. The tokenizer decides token IDs, special tokens, and subword splits; changing it changes what the model receives.

---
