# Decoder fine-tuning with LoRA and QLoRA


## The task

**PulseDesk** is a support desk. Short public posts arrive all day. Each post must be tagged with one of six dataset emotions: sadness, joy, love, anger, fear, or surprise. A human then scans the queue. The model must print **one label word**. A sentence, even a sensible sentence, is a failed request. These labels are dataset annotations. They are not a diagnosis.

```text
post → prompt → frozen Qwen + small LoRA adapter → one word, or INVALID
```

One real training row:

```text
text:  ive been feeling a little burdened lately wasnt sure why that was
gold:  sadness
```

A reader might also hear fatigue. The grade still uses `sadness`, because that is the annotation in the dataset.

If the model prints `This sounds joyful`, PulseDesk stores nothing useful. The tag has to be the word `joy`.

## Three ways to change what a model does

A pretrained model can be steered in more than one way. The difference is whether any weight changes, and what the model is allowed to see.

| Method | What goes in | Do weights change? | What comes out here |
|---|---|---|---|
| Prompting | Instruction, the current post, and maybe a few examples in this request | No | A generated label |
| BERT fine-tuning | The whole post, with attention in both directions | Yes, including a classifier head | Class scores |
| Decoder fine-tuning with LoRA | Earlier tokens only. A small adapter learns. | Only the adapter | The next tokens of the answer |

**Prompting** puts new words in the request. The weights stay the same. Three labelled examples inside one prompt are context for that request. When the request ends, nothing was saved. No parameter was updated.

**BERT fine-tuning** reads the whole post at once. A new classifier head learns to map the text to a class. That is a different architecture from the decoder used here.

**Decoder fine-tuning with LoRA** keeps next-token prediction. The original weights stay frozen. Two small matrices beside chosen layers learn the labelling habit.

A saved LoRA folder is not a model that can run by itself. It stores the update and the settings. Tagging a post still requires the same base model and the same tokenizer.

### Which method fits which job

1. **Three posts and a demo in an hour.** Use a clear prompt and check a few held-out lines. There is not enough labelled data, and there is not enough time, to justify a training run.
2. **Every ticket must be one of six words, and there are thousands of labelled posts.** Fine-tuning is a fair experiment. Compare it with the prompted base model on the same posts. The experiment can still lose.
3. **The right answer changes whenever the refund policy PDF changes.** Fine-tuning does not keep facts current. Retrieval belongs in that design. Training can still be used separately for a style or a label format.

Choosing LoRA is a hypothesis. The measurement decides whether the adapter helped. A lower training loss is not that measurement.

## Definitions

**Fine-tuning** means adapting a pretrained model to a task by updating some parameters on labelled examples.

**Full fine-tuning** updates the original weights.

**PEFT** means parameter-efficient fine-tuning. A small set of new parameters learns. The pretrained weights stay frozen. PEFT is the family, not a single algorithm.

**LoRA** (low-rank adaptation) is one PEFT method. Beside a frozen weight matrix `W`, it trains two small matrices `A` and `B`. Their product is a low-rank correction added to `W`.

**QLoRA** is LoRA with the frozen base stored in 4-bit form. The adapter still learns in a higher-precision compute dtype. QLoRA is a memory strategy. It does not promise a higher score than ordinary LoRA.

**An adapter** is the small trained add-on. In this lab it is the LoRA matrices, their scale, and the list of modules they attach to.

Other members of the same family, not trained in this lab:

| Method | What learns | Idea |
|---|---|---|
| LoRA | `A` and `B` beside a frozen matrix | Change how a selected layer transforms a vector |
| Prompt tuning | A short learned prefix | Learn the task prefix instead of typing it |
| IA3 | Small scale vectors | Turn channels up or down |

`LoraConfig(...)` does not train anything. It is the plan: rank, scale, dropout, and which module names receive an adapter. The trainer is what computes the loss and updates weights.

The libraries split the work:

| Library | Role |
|---|---|
| `transformers` | Pretrained Qwen model and tokenizer |
| `datasets` | The emotion posts and their labels |
| `peft` | Builds and saves the adapter |
| `trl` | Supervised fine-tuning trainer |

## Where the adapter sits

Qwen is a stack of decoder blocks. One block looks like this:

```text
token ids
   │
   ▼
embeddings
   │
   ▼
┌──────────── one decoder block ────────────┐
│ input layer norm                          │
│ self-attention                            │
│   q_proj   what this token looks for      │
│   k_proj   what earlier tokens can match  │
│   v_proj   the information that is mixed  │
│   o_proj   mix the heads back to width    │
│ residual add                              │
│ MLP: gate_proj, up_proj, down_proj        │
│ residual add                              │
└───────────────────────────────────────────┘
   │
   ▼
more blocks → final norm → lm_head → next-token scores
```

`target_modules` chooses the kind of module, such as every `q_proj`. `layers_to_transform` chooses which block numbers. Those are different controls. Naming `q_proj` and `v_proj` attaches a separate adapter pair in every block that has those names, unless the block index is also restricted.

LoRA on `q_proj` can change what this position looks for. LoRA on `v_proj` can change the information attention mixes in. That is a possible mechanism. A short training run does not prove that a head learned to look at the word `burdened`.

Standard LoRA needs a weight matrix. A GELU and a residual add have no weight matrix, so the low-rank update has nowhere to sit. Layer norms and ordinary bias vectors stay frozen in this lab.

| Module | Role | Used in this lab? |
|---|---|---|
| `q_proj` | Queries: what each position looks for | Yes |
| `k_proj` | Keys: what earlier positions can match | No |
| `v_proj` | Values: information attention mixes | Yes |
| `o_proj` | Maps attention output back to the hidden width | No |
| `gate_proj`, `up_proj`, `down_proj` | MLP transformation of each position | No |
| `embed_tokens`, `lm_head` | Token lookup, and scores over the vocabulary | No |

The same idea can sit on an encoder, an encoder–decoder, or a vision model. The module names change. Copying `q_proj` onto another architecture without checking its code attaches the adapter to nothing.

### How many modules to adapt

| Situation | First choice | Reason |
|---|---|---|
| One Colab GPU and a short run | `q_proj`, `v_proj` | Small, easy to inspect, and enough to see the method |
| Validation says attention is the weak spot | all four attention projections | More parameters. Measure it. Do not assume a win |
| The task needs more than attention | `all-linear` | Also hits the MLP. Watch memory |
| Only the last blocks should change | the same module names, plus `layers_to_transform` | Module type and depth are separate |
| A new head was randomly initialized | `modules_to_save` on that head | Different from LoRA targets. Watch tied weights |

This lab uses the first row. A larger target list is a later experiment.

For one token in one block, with LoRA only on query and value:

```text
Q = Wq h + (alpha / r) Bq (Aq h)
V = Wv h + (alpha / r) Bv (Av h)
K = Wk h
```

`h` is the hidden state of that token, a vector of numbers. It is not the word. `Wq` is frozen. `Aq` and `Bq` learn. `Aq` is not `Av`. Every targeted projection in every block has its own pair.

## The arithmetic

The forward pass adds a correction. It does not replace `W`.

```text
output = W h + (alpha / r) B (A h)
```

`r` is the rank: the width of the small path. `lora_alpha / r` is the scale on the correction. This lab uses `r = 8` and `lora_alpha = 16`, so the scale is `2`.

### A rank-1 example

The toy uses rank 1 and scale 1 so the arithmetic is visible. `W` is the identity, which is a teaching choice. Real `q_proj` is a learned matrix, not the identity. Real PEFT also starts the adapter near zero, so the first step adds about nothing. These nonzero toy values are for hand math only.

```text
A = [1, 0, -1, 2]                 shape (1, 4)
B = [0.1, 0, -0.2, 0.3]           shape (4, 1)
h = [1, 2, 0, -1]
```

`A` compresses 4 numbers into 1. `B` expands that 1 number into a correction of length 4. The product `BA` is 4×4, and every row is a scaled copy of `A`. That is what rank 1 means: one shared direction.

```text
z = A h = 1×1 + 0×2 + (−1)×0 + 2×(−1) = −1
correction = B z = [−0.1, 0, 0.2, −0.3]
frozen output = W h = [1, 2, 0, −1]
adapted output = [0.9, 2, 0.2, −1.3]
```

The frozen output is still there. The adapter added `[-0.1, 0, 0.2, -0.3]`.

### What 1.56% means

A 1,024×1,024 matrix has `1,048,576` values. Rank 8 adds `8×1,024 + 1,024×8 = 16,384` values, about **1.56% of that one matrix**.

That percent is not the trainable fraction of Qwen. Only the targeted layers get adapters. Embeddings, the language-model head, layer norms, and the untargeted projections stay frozen. The trainer prints the real fraction after the model is wrapped. In this setup the trainable names contain `lora_`. If a base weight is trainable, the run is not the experiment described here, and it should be stopped.

Freezing `W` means the optimizer does not update `W` and does not keep optimizer state for `W`. The forward pass still computes `W h`. The base weights, the activations, and the temporary tensors remain in GPU memory.

Rank 1 on one projection looks like two vectors. Rank 8 is a matrix with eight directions, and every targeted `q_proj` and `v_proj` in every block has its own pair. The model is not trained as two vectors.

## What the loss is allowed to see

A decoder predicts the next token from earlier tokens only. Training can pack a whole sentence into one tensor. The causal mask still hides the future. At generation time the model emits one new token at a time, because that token does not exist yet.

Four positions, left to right. A check means this query may look at that key.

```text
A sees only A
B sees A and B
C sees A, B, and C
D sees A, B, C, and D
```

`D` is the future for `C`, so `C` cannot see it.

Three different masks are easy to mix up:

| Mask | Hides | Does not hide |
|---|---|---|
| Causal attention | Future tokens | The prompt, and tokens already produced |
| Padding mask | Pad positions | Real tokens |
| Label `-100` | That position’s contribution to the loss | Attention. The token stays in context |

The prompt is labelled `-100`. The answer can still read the prompt. `-100` skips the grade. It does not delete context.

The position that predicts `sadness` cannot see the end-of-turn token that comes after it. That end token is the future.

```text
prompt: I got the job!     visible, loss skipped with -100
answer: joy + end token    graded
```

### Loss is a probability

One graded token contributes `-ln(P(correct))`.

| Probability of the correct token | Loss |
|---:|---:|
| 0.60 | about 0.51 |
| 0.10 | about 2.30 |
| 0.80 | about 0.22 |

Lower loss means the correct token was more likely. The average uses only the unmasked answer tokens. A prompt position labelled `-100` is absent from that average.

If the graded completion is two tokens with probabilities `0.60` and `0.80`, the average loss is about `(0.51 + 0.22) / 2 = 0.37`.

A falling training loss means the adapter assigned more probability to answer tokens it has already seen. It does not mean new posts are solved. Validation loss is a separate split, and a short curve can wobble. The held-out comparison is the measurement that matters.

If training loss falls while validation loss rises, the adapter is fitting rows it has seen. That can be overfitting. Read examples before treating the curve as a verdict. Do not pick a winner from training loss.

## The data

`dair-ai/emotion`, configuration `split`, has official train, validation, and test sets. Each row is a short post plus an integer label. The integer becomes one of: sadness, joy, love, anger, fear, surprise.

| Split | Full size | Used here |
|---|---:|---|
| Train | 16,000 | 1,200 rows, seed 42, to fit the adapter |
| Validation | 2,000 | 240 rows, to watch the loss |
| Test | 2,000 | Left untouched until the evaluation notebook |

The lab is sized so the path load → train → save → score is visible. It is not a benchmark of Qwen or of the dataset.

Class shares matter because accuracy can look kind while a rare emotion is ignored. Joy is common in this dataset. Surprise is thin. Macro-F1, defined below, gives each emotion an equal vote.

Some posts fit two emotions in ordinary reading. The dataset label is still the grade. Quote the text and call the row ambiguous. Do not invent a hidden attention story to explain the miss.

An instruct model is used because it already follows a chat template. The hour can then be about the adapter. Gains over that base may be small. A small gain is a result.

The label is generated as text, rather than produced by a new classifier head, so the answer stays next-token prediction. Six legal words make the mask, the parser, and the errors visible.

## The prompt contract

The prompt ends where the assistant begins. The completion is the label plus the end token. The same prompt is used later, when the model generates.

```text
A. prompt = instruction + post             completion = sadness
B. prompt = instruction + post + sadness   completion = sadness
C. prompt = instruction + post             completion = sadness + end token
```

**B leaks the answer.** The gold word is inside the context, so the model can copy it. That row does not test whether the model knows the label.

**C is the training contract.** The label and the stop token are the graded completion. The prompt does not contain the gold word.

**A is incomplete** unless the trainer adds the end token. The model needs a stop token in the graded completion so generation knows where the answer ends.

In the token print, prompt tokens are marked ignored and the label plus the end token are marked graded. Ignored means skip the loss. The model can still read that token. TRL builds this mask from the prompt and completion fields. The labels are not shifted a second time by hand.

A reply such as `The emotion is joy` is invalid. The product contract is the exact label word, after trim and lowercase. Searching inside a sentence for a label word is a different parser, invented after seeing the answer.

## Training settings

```text
examples → prompt visible, answer graded → loss
        → gradients into A and B only → W stays put
        → save the small adapter
```

| Setting | Meaning |
|---|---|
| `r=8` | Eight-wide bottleneck on each adapter |
| `lora_alpha=16` | Scale is `16/8 = 2` |
| `lora_dropout=0.05` | Dropout on the adapter path during training |
| `target_modules=["q_proj", "v_proj"]` | Query and value in every block |
| Batch size 2, accumulation 8 | One optimizer step sees 16 posts. Memory holds 2 at a time |
| `max_steps=60` | A time limit. Not an epoch. Not a promise of quality |
| `learning_rate=2e-4` | Step size for the adapter weights |
| `warmup_steps=3` | The step size ramps over the first 3 of 60 updates |
| `max_length=256` | Longer rows are dropped before training |
| `completion_only_loss=True` | Grade the answer. Keep the prompt as context |
| `fp16` or `bf16` | Match the dtype the GPU supports |
| `gradient_checkpointing=True` | Recompute some activations to save memory |

Sixty steps with 16 posts per step touch on the order of a thousand training rows, depending on shuffle. The step cap is a clock.

Before training, the notebook checks that `q_proj` and `v_proj` exist in this Qwen model, then estimates adapter size for three target choices. The trainer uses the Q-and-V row. The count is a parameter count, not peak GPU memory.

The 12-row preview uses the same validation posts, the same prompt, and greedy decoding, before and after training. It is a glance. It is not the test set. A finished training call can still make some of those posts worse. Improvements and regressions are both counted.

## What gets saved

Colab forgets `/content` when the runtime ends. The Drive folder is the handoff.

The folder holds `adapter_config.json`, `adapter_model.safetensors`, the tokenizer, and `lesson_manifest.json`. The files are megabytes. A full-model-sized folder means the wrong object was saved.

The folder is small because it stores the update and the recipe. Qwen itself stays the base model. With the base model not loaded, the folder cannot tag a post. Evaluation attaches the adapter to the same Qwen checkpoint with `PeftModel.from_pretrained`. One base stays in memory. A second full copy of Qwen can exhaust a Colab GPU.

Merging the adapter into the base makes a full weight set and loses the small-file advantage on that copy. This teaching artifact stays unmerged.

The adapter attaches to a different Qwen checkpoint only when the architecture and the module names match. The config names the base. A mismatched base is not a fair reload.

## How the adapter is scored

The evaluation changes one thing: adapter off, then adapter on. Same posts, same order, same prompt, same greedy decoding, same parser. Every row stays in the denominator.

A training log means the optimizer ran. The score asks whether that adapter helped on posts it did not train on.

These comparisons are not the same experiment:

| Mistake | What actually changed |
|---|---|
| Score the 12 preview rows from training time | Those rows were already seen during development, and 12 is too small |
| Give the adapter a sharper instruction than the base | The prompt changed, so the adapter is no longer the only difference |
| Temperature for one system, greedy decoding for the other | Randomness is a second change |
| If the base writes `It sounds surprising`, search the string and mark it `surprise` | The parser was invented after seeing the answer |
| Drop invalid rows, then divide by the rows that remain | 80 correct and 10 invalid out of 100 is 80%, not 80/90 |

```text
gold surprise
base:    "It sounds surprising."  → INVALID
adapter: "surprise"              → correct
```

The base may have understood the mood as a reader. It still failed the product contract. PulseDesk cannot file a sentence into a six-way dashboard.

Read the raw strings before any metric. Parsing too early hides format failures. `INVALID` is not a seventh emotion. It is a format failure, and it stays in accuracy and in the confusion matrix.

### Three scores

**Accuracy** is the fraction of rows whose parsed text is exactly the gold label. An invalid string counts as wrong. All rows stay in the denominator.

**Valid-label rate** is the fraction of rows whose parsed text is exactly one of the six words, whether or not that word is correct.

**Macro-F1** is the F1 of each emotion, then the average of those six F1 scores. A rare class has the same vote as a common class. Precision and recall for one emotion both treat an invalid prediction on that gold label as a miss. The invalid row does not vanish.

Worked counts:

```text
                 pred joy   pred sadness   INVALID
gold joy              8            2          1
gold sadness          1            7          1
```

There are 20 rows. Correct predictions are `8 + 7 = 15`, so accuracy is `15/20 = 0.75`. Valid predictions are 18, so the valid-label rate is `0.90`. Joy recall uses 11 gold-joy rows. The invalid joy row is a miss, so joy recall is `8/11`, not `8/10`.

A second worked comparison:

| Gold | Base | Adapter |
|---|---|---|
| joy | joy | joy |
| sadness | The label is sadness | sadness |
| fear | fear | surprise |

Accuracy is `2/3` for both. It did not improve. Format did improve: the base valid rate is `2/3`, and the adapter valid rate is `3/3`. The adapter also turned a correct `fear` into `surprise`. One number does not tell that story.

If accuracy rises and macro-F1 falls, a common class got better or a small class got worse. The per-class report is the next place to look.

### Paired changes

```text
                    adapter correct     adapter wrong
base correct         both fine           regression
base wrong           improvement         both wrong
```

Accuracy change equals improvements minus regressions, divided by the number of rows. That identity is a check that the two result tables describe the same rows.

How to read a real run:

- Overall accuracy can rise while fear gets worse. Report both. If fear is the queue that matters, the average is not permission to ship.
- One dramatic post getting fixed, while the score falls, does not make the adapter better. The paired count is the evidence.
- A genuinely ambiguous post is not automatically a bug. The gold label is one annotation. Quote the text.
- Two runs with different seeds and different slices are not the same exam. On 96 rows, one row is about one percentage point. Small gaps are noisy.
- The test set is touched once, at the end. The next change is chosen on validation.

The classroom measurement uses 96 untouched test posts, 60 optimizer steps, one seed, and one small instruct model. If the adapter does not win, the numbers still stand. Check the trainable count, the loss curve, the prompt, and the confusing posts. Changing the test slice after seeing the score is a different exam.

A higher valid-label rate with unchanged accuracy means the format got cleaner and the label choice did not. Both facts belong in the report.

Validation loss can fall while test accuracy does not. The model assigned more probability to answer tokens without producing more exact labels. Read the raw strings.

## QLoRA

| Method | Base storage | Base updated? | What learns |
|---|---|---|---|
| Full fine-tune | ordinary precision | Yes | the original weights |
| LoRA | ordinary precision | No | small `A` and `B` |
| QLoRA | often 4-bit | No | small `A` and `B` |

Four bits is how the frozen base is stored. The layer math is dequantized up to a compute dtype such as float16 or bfloat16. The adapter is not trained with 4-bit gradients.

Raw storage for the weights of a 0.5B model is about 1.0 GiB in 16-bit and about 0.25 GiB in 4-bit. Metadata, activations, and temporary tensors are still there. Quarter-size weights are not quarter-size training memory.

| Situation | Choice |
|---|---|
| The 0.5B model and plain LoRA fit the GPU | Run plain LoRA and measure the task. QLoRA is optional complexity here |
| An 8B model does not fit in 16 GB with ordinary weights | Try a 4-bit base plus LoRA, then measure quality. Fitting is not the same as matching full-precision accuracy |
| QLoRA fits, but each step is slow | Memory was only one constraint. Compute and activations remain |
| An optional cell prints a trainable percentage for `all-linear` on a quantized base | That cell builds a new untrained adapter. It is not the score of the Q-and-V adapter trained above |

QLoRA does not always beat LoRA on accuracy. Quantization can change the numbers. Measure both when the comparison matters.

## DoRA, the optional extension

DoRA uses the same posts, the same prompt, the same rank 8, and the same 60 steps. LoRA’s one update can change two things at once: the direction of a weight row, and the length of that row. DoRA splits them. `A` and `B` move the direction. A vector `m` sets the length. The frozen base stays frozen.

```text
LoRA:  W0 + scaled B A
DoRA:  m * (W0 + scaled B A) / length of that sum
```

`use_dora=True` is a flag on the same LoRA config. It is not a new decoder.

At step 0, `B` starts at zero and `m` starts as the row length of `W0`, so the result equals `W0`. The original model is unchanged until training moves `A`, `B`, or `m`.

A 2×2 picture:

```text
W0 = [3  0]
     [0  4]
sideways nudge on the first row: [0  0.6]
```

LoRA’s first row becomes `[3, 0.6]`. Its length grows past 3, to `sqrt(9 + 0.36) ≈ 3.06`. DoRA tilts that row and, while `m[0]` is still 3, the length stays 3. Adding 0.5 to `m[0]` makes the length 3.5. Direction and length are separate knobs. `W0` is still inside the sum. It is not a trainable parameter.

The magnitude is one number per output row, not a second full matrix. On a 1,024-wide layer, rank 8 is 16,384 LoRA values plus 1,024 magnitudes.

The third trainable tensor on `q_proj` is the magnitude. `A` and `B` are still trained. The original `q_proj` weight is not. The prompt, the `-100` rule, and the untouched test split stay the same. DoRA does not change which tokens are graded.

A trainable list with no name containing `magnitude` is plain LoRA. A falling training loss does not retire the earlier LoRA folder. The DoRA adapter is saved under its own folder, `decoder_dora_emotion`, and it still needs the original Qwen model.

DoRA can be slower than LoRA because of the norm. That is expected. QDoRA would put this same adapter on a 4-bit frozen base. That combination is not trained in this lab.
