# Loss Functions, Optimizers, and the Training Loop


## 1. The three-part decomposition

Every supervised deep learning system is exactly three separable decisions. Keeping them separate in your head is most of the skill.

```text
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │    MODEL     │    │     LOSS     │    │  OPTIMIZER   │
   ├──────────────┤    ├──────────────┤    ├──────────────┤
   │ What CAN be  │    │ What "good"  │    │ How we MOVE  │
   │ represented  │    │ means        │    │ toward good  │
   ├──────────────┤    ├──────────────┤    ├──────────────┤
   │ nn.Linear    │    │ BCEWithLogits│    │ SGD          │
   │ nn.ReLU      │    │ CrossEntropy │    │ Adam / AdamW │
   │ nn.Conv2d    │    │ MSE / Huber  │    │ lr, schedule │
   └──────────────┘    └──────────────┘    └──────────────┘
          │                    │                   │
          └────────────────────┴───────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   TRAINING LOOP     │
                    │ forward → loss →    │
                    │ zero → backward →   │
                    │ step                │
                    └─────────────────────┘
```

A useful diagnostic habit: when a model underperforms, ask *which of the three* is at fault.

| Symptom | Usually the fault of |
|---|---|
| Train loss will not go down | optimizer (LR) or a **bug** |
| Train loss low, valid loss high | model capacity / regularisation |
| Loss low but the business metric is bad | **loss** — you optimised the wrong thing |
| Training unstable, spiky, NaN | optimizer (LR too high) or data scale |

### Loss vs metric — never confuse them

| | Loss | Metric |
|---|---|---|
| Purpose | give gradients | tell humans if the model is useful |
| Must be | differentiable, smooth | anything |
| Examples | BCE, cross-entropy, MSE | accuracy, F1, PR-AUC, revenue |
| Used by | the optimizer | you, and the business |

Accuracy has zero gradient almost everywhere — nudging a weight slightly usually flips no prediction, so the derivative is 0. That is exactly why we cannot train on accuracy and must train on a smooth surrogate like cross-entropy.

---

## 2. Loss functions for regression

Predicting a number: ticket resolution time, house price, demand.

### MSE — `nn.MSELoss`

$$L = \frac{1}{N}\sum (\hat y - y)^2$$

Squaring means an error of 10 costs 100× an error of 1. **MSE cares intensely about outliers.**

### MAE — `nn.L1Loss`

$$L = \frac{1}{N}\sum |\hat y - y|$$

Linear cost. **Robust to outliers**, but its gradient is constant `±1`, which makes fine convergence near the optimum harder.

### Huber / Smooth L1 — `nn.HuberLoss(delta=1.0)`

Quadratic near zero, linear far away. The best of both.

$$
L_\delta(e)=
\begin{cases}
\tfrac12 e^2 & |e|\le\delta\\[4pt]
\delta(|e|-\tfrac12\delta) & |e|>\delta
\end{cases}
$$

### Diagram: cost shape

```text
 cost
   ^
   │        MSE  ╱             ← explodes on outliers
   │           ╱
   │          ╱
   │        ╱ ╱  Huber         ← quadratic inside ±δ, linear outside
   │      ╱ ╱
   │    ╱╱────────  MAE        ← constant slope
   │  ╱╱
   └──┴────┴─────────────────> error
     -δ    δ
```

| Loss | Optimises toward the | Use when |
|---|---|---|
| MSE | **mean** | errors are Gaussian, outliers are real signal |
| MAE | **median** | outliers are noise / data-entry errors |
| Huber | mean, robustly | you are unsure — good default |

> **Interview favourite:** "Why does MSE predict the mean and MAE the median?" Because minimising $\sum(c-y_i)^2$ over a constant $c$ gives $c=\bar y$, while minimising $\sum|c-y_i|$ gives $c=\text{median}(y)$. That single fact explains all their differing behaviour.

**Consider scaling a target with a very large range.** It can make optimisation easier when values are extremely large or have a wide spread. Keep the target in its original units when the scale is already manageable and those units make evaluation clearer. If you scale a target, invert the scaling before reporting predictions and MAE/RMSE.

---

## 3. Loss functions for classification

### Why not accuracy?

```text
 accuracy as a function of a weight        cross-entropy as a function of a weight

 1.0 ┤    ┌────────                        3 ┤╲
     │    │                                  │ ╲
 0.5 ┤────┘                                1 ┤  ╲___
     │                                       │      ╲____
 0.0 ┤                                     0 ┤           ╲___
     └──────────────> w                       └──────────────> w

  flat everywhere → gradient = 0            smooth everywhere → gradient always useful
  cannot be optimised by gradient descent
```

### Binary cross-entropy

For a true label $y\in\{0,1\}$ and predicted probability $p$:

$$L = -\big[\,y\log p + (1-y)\log(1-p)\,\big]$$

Read it as two halves; exactly one is active per example:

- If $y=1$: $L=-\log p$. Predict $p=0.99 \Rightarrow L=0.01$. Predict $p=0.01 \Rightarrow L=4.6$.
- If $y=0$: $L=-\log(1-p)$, mirrored.

**The cost of being confidently wrong grows without bound.** That property is what forces the model to become calibrated rather than merely correct.

| True label | Predicted p | Loss | Comment |
|---|---|---|---|
| 1 | 0.99 | 0.01 | confident and right — nearly free |
| 1 | 0.60 | 0.51 | right but unsure — small penalty |
| 1 | 0.50 | 0.69 | no information — `ln 2` |
| 1 | 0.10 | 2.30 | wrong |
| 1 | 0.01 | 4.61 | confidently wrong — very expensive |

> `0.693` is the loss of a model that outputs 0.5 for everything. On a **roughly balanced** binary dataset, a loss that stays near `0.693` can mean the model is not learning. On an imbalanced dataset, first compare with the baseline that always predicts the positive-class rate; the expected BCE is not necessarily `0.693`.

### Categorical cross-entropy

For K classes with softmax probabilities $p_k$ and true class $c$:

$$L = -\log p_c$$

Only the true class's probability enters the loss. But because softmax normalises across classes, pushing $p_c$ up necessarily pushes the others down.

$$p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

### Diagram: logits → softmax → loss

```text
   logits          softmax            true class = 1
   z = [2.0,        p = [0.24,
        3.0,             0.66,   ──>  L = -log(0.66) = 0.42
        1.0]             0.09]

   raise z[1] to 5.0:
   z = [2.0,        p = [0.05,
        5.0,             0.94,   ──>  L = -log(0.94) = 0.06
        1.0]             0.02]
```

### The stability argument — why logits, not probabilities

`BCEWithLogitsLoss` and `CrossEntropyLoss` fuse the squashing with the log using the **log-sum-exp trick**:

$$\log\sum_j e^{z_j} = m + \log\sum_j e^{z_j-m},\quad m=\max_j z_j$$

Subtracting the max keeps every exponential in a safe range. Compute softmax separately and a logit of 30 gives `p = 1.0` exactly in float32, then `log(1 - 1.0) = -inf`.

```text
✅  logits ──> BCEWithLogitsLoss / CrossEntropyLoss
✅  logits ──> sigmoid / softmax ──> metrics and thresholds   (after the loss)
❌  sigmoid ──> BCEWithLogitsLoss     squashed twice, silent, trains badly
❌  softmax ──> CrossEntropyLoss      same mistake
```

### The full loss selection table

| Task | Output shape | Target | Loss | PyTorch |
|---|---|---|---|---|
| Regression | `(N, 1)` | float `(N, 1)` | MSE / MAE / Huber | `nn.MSELoss()` |
| Binary | `(N, 1)` logits | float `(N, 1)` | BCE | `nn.BCEWithLogitsLoss()` |
| K-class, one label | `(N, K)` logits | long `(N,)` | CE | `nn.CrossEntropyLoss()` |
| K-class, multi-label | `(N, K)` logits | float `(N, K)` | BCE per class | `nn.BCEWithLogitsLoss()` |
| Ranking / similarity | embeddings | pairs/triplets | contrastive, triplet | `nn.TripletMarginLoss()` |
| Sequence labelling | `(N, T, K)` | long `(N, T)` | CE over flattened | `nn.CrossEntropyLoss(ignore_index=pad)` |

> For sequences you must reshape: `CrossEntropyLoss` wants `(N*T, K)` and `(N*T,)`. Use `logits.reshape(-1, K)` and `targets.reshape(-1)`. This is the standard pattern in every LLM training script.

### Label smoothing

```python
nn.CrossEntropyLoss(label_smoothing=0.1)
```

Instead of a hard target of 1.0 for the true class, use `1 - ε + ε/K`. This prevents the model driving logits to infinity, improves calibration, and usually adds a fraction of a point of accuracy. Standard in modern vision and NLP training.

---

## 4. Class imbalance

99% of tickets are normal, 1% urgent. A model that says "normal" always gets 99% accuracy and is worthless.

### Three levers

| Lever | How | Trade-off |
|---|---|---|
| **Loss weighting** | `pos_weight` (binary), `weight=` (CE) | no data change; can hurt calibration |
| **Resampling** | oversample minority / undersample majority | changes the effective data distribution |
| **Threshold tuning** | train normally, move the decision threshold | **often the best first move** |

### `pos_weight` for binary

```python
n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
pos_weight = (n_neg / n_pos)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
```

`pos_weight=4` means *"a mistake on a positive example costs four times as much"*. It multiplies **only** the $y\log p$ term, so the loss on a negative example is unchanged.

### `weight` for multi-class

```python
counts = torch.bincount(y_train)
weights = counts.sum() / (len(counts) * counts.float())   # inverse frequency
loss_fn = nn.CrossEntropyLoss(weight=weights)
```

### Diagram: what weighting actually does

```text
  Without pos_weight              With pos_weight = 10

  loss                            loss
   ^  positive errors             ^  positive errors
   │  ▁▁▁▁▁                       │  ██████████
   │  negative errors             │  negative errors
   │  ██████████                  │  ██████████
   └──────────────                └──────────────
   optimiser spends its           optimiser now cares about
   effort on the majority         the rare class too
```

> **Warning:** weighting distorts predicted probabilities. If downstream systems consume `p` as a real probability (expected-value calculations, ranking against a cost), prefer threshold tuning and keep the model calibrated.

### Metrics under imbalance

Accuracy is meaningless here. Use:

| Metric | Formula | Answers |
|---|---|---|
| Precision | TP / (TP + FP) | "when we say urgent, how often are we right?" |
| Recall | TP / (TP + FN) | "of all urgent tickets, how many did we catch?" |
| F1 | harmonic mean | single balanced number |
| PR-AUC | area under precision-recall | threshold-free, good for rare positives |
| ROC-AUC | area under TPR/FPR | can look deceptively good when positives are rare |

**Under heavy imbalance prefer PR-AUC over ROC-AUC.** ROC-AUC is dominated by the large negative class and stays high even for a poor model.

---

## 5. Gradient descent: the base algorithm

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L$$

Read it in English: *move each parameter a little bit in the direction that makes the loss go down.* The gradient points **uphill**, hence the minus sign.

### Diagram: descending a loss surface

```text
  loss
    │  ●  start
    │   ╲
    │    ●   step = -lr * gradient
    │     ╲
    │      ●
    │       ╲___●___●     small steps near the bottom
    │              ‾‾●‾   (gradient is small there)
    └──────────────────────> parameter
```

### The three flavours

| Variant | Batch for one update | Pros | Cons |
|---|---|---|---|
| Batch GD | whole dataset | smooth, stable | slow, may not fit in memory |
| **Mini-batch SGD** | 32–512 examples | good compromise — **what everyone uses** | some noise |
| Stochastic GD | 1 example | very noisy | slow, no vectorisation |

Confusingly, `torch.optim.SGD` is mini-batch by default — the "S" is historical.

The **noise in mini-batch SGD is a feature**, not a bug: it helps escape sharp minima and saddle points, and acts as a mild regulariser. This is part of why very large batches sometimes generalise *worse*.

---

## 6. The optimizer family

Every modern optimizer answers one of two questions:

1. **Should the step remember where it has been?** → momentum
2. **Should each parameter get its own learning rate?** → adaptive scaling

```text
                      SGD
                  θ -= lr * g
                       │
        ┌──────────────┴──────────────┐
        │ remember direction          │ scale per-parameter
        ▼                             ▼
   SGD + momentum                  RMSProp
   v = βv + g                      s = βs + g²
   θ -= lr * v                     θ -= lr * g / √s
        │                             │
        └──────────────┬──────────────┘
                       ▼
                     ADAM
              (momentum + RMSProp)
                       │
                       ▼
                     ADAMW
        (decoupled weight decay — the modern default)
```

### SGD with momentum

$$v_t = \beta v_{t-1} + g_t,\qquad \theta_{t+1}=\theta_t-\eta v_t$$

A ball rolling downhill. It builds speed in consistent directions and damps oscillation across a narrow ravine. `β = 0.9` is the near-universal choice — it corresponds to averaging roughly the last 10 gradients.

```text
  Without momentum          With momentum
  ╲ ↗╲ ↗╲ ↗                ╲
   ╲↙  ╲↙  ╲↙               ╲__
  zig-zags across            ╲___→  glides along the valley floor
  the ravine walls
```

### RMSProp

$$s_t=\beta s_{t-1}+(1-\beta)g_t^2,\qquad \theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{s_t}+\epsilon}g_t$$

Divides by a running RMS of recent gradients: parameters with consistently large gradients get smaller steps, rarely-updated parameters get larger ones. Crucial for sparse features and embeddings.

### Adam = momentum + RMSProp

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t \quad\text{(momentum)}
$$
$$
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2 \quad\text{(scale)}
$$
$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\quad \hat v_t=\frac{v_t}{1-\beta_2^t} \quad\text{(bias correction)}
$$
$$
\theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{\hat v_t}+\epsilon}\hat m_t
$$

Defaults `β₁=0.9, β₂=0.999, ε=1e-8` work almost everywhere. The bias correction exists because $m$ and $v$ start at zero and would otherwise be biased toward zero for the first few hundred steps.

### AdamW — use this one

In Adam, L2 regularisation added to the loss gets divided by $\sqrt{\hat v}$ along with the gradient, so parameters with large gradients receive *less* regularisation — the opposite of the intent. **AdamW decouples it**, applying weight decay directly to the parameter:

$$\theta_{t+1}=\theta_t-\eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}+\lambda\theta_t\right)$$

Every modern transformer is trained with AdamW.

### Choosing

| Optimizer | Typical LR | Use when |
|---|---|---|
| `SGD(momentum=0.9, nesterov=True)` | 1e-2 – 1e-1 | vision with a good schedule; often the best *final* accuracy |
| `Adam` | 1e-3 | fast, forgiving prototyping baseline |
| **`AdamW`** | 1e-3 (3e-5 – 1e-4 for fine-tuning) | **default for transformers, NLP, most work** |
| `RMSprop` | 1e-3 | RNNs, some RL |
| `Adagrad` | 1e-2 | very sparse features (LR decays monotonically to zero) |

> **Practical advice:** start with AdamW at `1e-3`. It converges fast and rarely fails. Switch to tuned SGD+momentum only when you are chasing the last point of accuracy on a vision benchmark and have a schedule.

---

## 7. Learning rate: the one hyperparameter that matters most

If you can only tune one thing, tune this.

```text
  lr TOO SMALL          lr GOOD              lr TOO LARGE         lr WAY TOO LARGE
    ●                     ●                     ●   ●                ●
     ●                     ╲                   ╱ ╲ ╱ ╲              ╱ ╲
      ●                     ●                 ●   ●   ●            ╱   ╲
       ●                     ╲               oscillates           ╱     ╲
        ●                     ●___           around the min      NaN / diverges
   crawls, never             finds it
   converges in time         quickly
```

### Symptoms and cures

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss decreases painfully slowly | LR too small | ×10 |
| Loss decreases then plateaus high | LR too large for fine convergence | add a decay schedule |
| Loss oscillates, does not settle | LR too large | ÷3 |
| Loss becomes `NaN` at some step | LR far too large, or `log(0)`, or bad data | ÷10, add grad clipping, check the data |
| Loss flat near `0.693` on balanced binary data | model predicts about 0.5 for everything | check the pipeline, then LR |

### The LR range test

Train for a few hundred steps while increasing the LR exponentially from `1e-7` to `1`. Plot loss against LR on a log axis.

```text
 loss
   │╲                        ╱
   │ ╲______________       ╱
   │                ╲____╱
   └────────────────┬─┬──────────> lr (log)
                    │ └── loss explodes
                    └──── steepest descent  ← pick roughly here (or 10x lower)
```

Pick the LR where the curve is *steepest downward*, not the minimum — at the minimum you are already close to instability. A common rule is one order of magnitude below the explosion point.

---

## 8. Learning rate schedules

Large steps early to explore, small steps late to settle.

| Schedule | Shape | Use when |
|---|---|---|
| `StepLR(step_size, gamma)` | staircase drops | classic vision recipes |
| `MultiStepLR(milestones)` | drops at chosen epochs | when you know the plateaus |
| `ExponentialLR(gamma)` | smooth decay | simple, general |
| `CosineAnnealingLR(T_max)` | smooth cosine to ~0 | **modern default** |
| `ReduceLROnPlateau` | drop when valid stops improving | metric-driven, no epoch guessing |
| `OneCycleLR` | warm up then anneal | fast convergence in few epochs |
| linear warmup + cosine | ramp then decay | **transformers, always** |

```text
 lr
  │      ┌──────┐
  │      │      └──┐              StepLR
  │      │         └────
  │
  │   ╱‾‾‾‾╲                      warmup + cosine  (transformer standard)
  │  ╱      ╲___
  │ ╱            ‾‾‾───___
  └────────────────────────> step
```

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

for epoch in range(EPOCHS):
    train_one_epoch(...)
    metrics = evaluate(...)
    scheduler.step()                       # per epoch for most schedulers
    # scheduler.step(metrics["loss"])      # ReduceLROnPlateau needs the metric
```

> **Ordering rule:** `optimizer.step()` comes **before** `scheduler.step()`. Most schedulers step once per **epoch**; `OneCycleLR` and warmup schedulers step once per **batch**. Getting this wrong silently ruins the schedule.

### Why warmup?

At step 0 Adam's variance estimate $\hat v$ is based on almost no data and is unreliable, producing wildly wrong step sizes. Ramping the LR from 0 over the first few hundred steps lets the moment estimates stabilise. Essential at large batch sizes and for transformers.

---

## 9. Regularisation: weight decay, dropout, early stopping

| Technique | Mechanism | Typical value |
|---|---|---|
| **Weight decay** | penalise large weights | `1e-2` (AdamW), `1e-4` (SGD) |
| **Dropout** | randomly zero activations while training | `0.1`–`0.5` |
| **Early stopping** | stop when validation stops improving | patience 5–15 epochs |
| **Data augmentation** | more effective data | task specific |
| **Batch/Layer norm** | stabilise activation scale | architectural |

### Weight decay

Adds $\lambda\|\theta\|^2$ pressure, shrinking weights toward zero. Smaller weights mean a smoother function and less overfitting.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

> **Do not decay biases and normalisation parameters.** They have no capacity to overfit and decaying them measurably hurts. Split parameter groups:

```python
decay, no_decay = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)

optimizer = torch.optim.AdamW([
    {"params": decay,    "weight_decay": 1e-2},
    {"params": no_decay, "weight_decay": 0.0},
], lr=1e-3)
```

### Dropout

During training each activation is zeroed with probability `p` and the rest are scaled by `1/(1-p)` (inverted dropout), so the expected value is unchanged. At eval time it is a no-op — this is why `model.eval()` matters.

### Early stopping

```text
 loss
   │╲
   │ ╲___ train  (keeps falling)
   │     ‾‾‾‾───___
   │╲
   │ ╲___  valid
   │      ╲___
   │          ●  ← minimum: STOP HERE and restore this checkpoint
   │           ╲___
   │               ‾‾‾  valid rises: overfitting from here on
   └──────────────────────> epoch
```

Save the checkpoint at the best validation score and restore it at the end. Without restoring, early stopping gives you the model from `patience` epochs *after* the best one.

### Gradient clipping

Stops a single pathological batch destroying training. Standard for RNNs and transformers.

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # AFTER backward, BEFORE step
optimizer.step()
```

---

## 10. A production training loop

Everything assembled.

```python
def fit(model, train_loader, valid_loader, *, epochs=100, lr=1e-3,
        weight_decay=1e-2, patience=10, clip=1.0, device="cpu"):

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss, best_state, bad_epochs = float("inf"), None, 0
    history = []

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_loader.dataset)

        # ---- validate ----
        valid_loss, valid_acc = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step()
        history.append({"epoch": epoch, "train": train_loss,
                        "valid": valid_loss, "acc": valid_acc,
                        "lr": optimizer.param_groups[0]["lr"]})

        # ---- early stopping on the BEST checkpoint ----
        if valid_loss < best_loss - 1e-4:
            best_loss, bad_epochs = valid_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stop at epoch {epoch}; best valid loss {best_loss:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)      # restore the BEST, not the last
    return history
```

### Checklist for any training script

- [ ] Seed set, device chosen
- [ ] Split **before** any preprocessing; fit scalers on train only
- [ ] Output shape and loss agree (logits!)
- [ ] `assert pred.shape == target.shape` while developing
- [ ] `zero_grad → backward → step` in that order
- [ ] `model.train()` / `model.eval()` in the right places
- [ ] Tiny-subset overfit test passes
- [ ] Best checkpoint saved **and restored**
- [ ] Test set touched exactly **once**, at the very end

---

## 11. Reading loss curves

```text
(A) HEALTHY                 (B) OVERFITTING            (C) UNDERFITTING
 │╲                          │╲                         │
 │ ╲___ train                │ ╲___ train               │────────  both flat
 │  ╲                        │     ‾‾‾───___            │────────  and high
 │   ╲__ valid               │ ╲___                     │
 │      ‾‾───                │     ╲__●___              │
 └──────────> epoch          └──────────> epoch         └──────────> epoch
 both fall, small gap        valid turns UP             model too small,
 → keep training             → regularise, more data,   LR too small, or
                               early stop at ●          features uninformative

(D) LR TOO HIGH             (E) NaN                    (F) VALID < TRAIN
 │ ╱╲  ╱╲  ╱╲                │╲                         │╲
 │╱  ╲╱  ╲╱  ╲               │ ╲                        │ ╲___ valid
 │                           │  ╲___ NaN ✖              │  ╲___ train
 └──────────> epoch          └──────────> epoch         └──────────> epoch
 spiky, no progress          exploded at one step       normal! dropout is ON
 → divide LR by 3–10         → clip grads, ÷10 LR,        during train, OFF in
                               check for log(0)           eval. Not a bug.
```

Case (F) surprises people every time: validation loss **below** training loss is usually correct behaviour when dropout is active, because training loss is measured with dropout on and validation with it off.

### Systematic diagnosis order

1. **Does the tiny-subset overfit test pass?** No → it is a bug. Stop and fix.
2. **Is training loss falling?** No → LR or capacity.
3. **Is validation tracking training?** No → regularisation / more data.
4. **Is the metric acceptable?** No → wrong loss, wrong threshold, or wrong problem framing.

---

## 12. Metrics and thresholds

Training gives you a **score**. It does not give you a **decision**. The threshold is a separate, deliberate choice.

```text
                        threshold
      p=0                  0.5                   p=1
      ├────────────────────┼────────────────────┤
       predict NORMAL       predict URGENT

  move LEFT   → more urgent predictions → recall ↑, precision ↓
  move RIGHT  → fewer urgent predictions → precision ↑, recall ↓
```

| Situation | Move the threshold | Because |
|---|---|---|
| Missing an urgent ticket is very costly | left (e.g. 0.3) | maximise recall |
| Escalating wrongly is expensive | right (e.g. 0.7) | maximise precision |
| Balanced costs | tune F1 on validation | |

### The rule

**Choose the threshold on the validation set. Touch the test set exactly once, at the very end.** If you tune the threshold on test, your reported number is optimistic and will not survive production.

### Confusion matrix

```text
                    predicted
                 normal    urgent
        normal │   TN    │   FP   │  ← false alarm
 actual        ├─────────┼────────┤
        urgent │   FN    │   TP   │  ← FN = missed urgent ticket
               └─────────┴────────┘

 precision = TP/(TP+FP)     "when we shout, are we right?"
 recall    = TP/(TP+FN)     "did we catch them all?"
 F1        = 2PR/(P+R)
```

---

## 13. Q&A and revision

**Q1. Why can we not train directly on accuracy?**
Accuracy is piecewise constant — its gradient is zero almost everywhere. A tiny weight change usually flips no prediction, so there is no signal. Cross-entropy is a smooth surrogate that is minimised by the same predictions.

**Q2. When would you choose MAE over MSE?**
When large errors are data-entry noise rather than real signal. MSE's squared cost lets a handful of outliers dominate the gradient. MAE optimises toward the median, MSE toward the mean.

**Q3. Explain Adam in one sentence.**
Momentum (an EMA of gradients) for direction, plus RMSProp (an EMA of squared gradients) for a per-parameter step size, with bias correction because both EMAs start at zero.

**Q4. What is the actual difference between Adam and AdamW?**
Adam adds L2 to the loss, so the penalty flows through the adaptive `1/√v̂` scaling and parameters with large gradients get *less* regularisation — backwards from the intent. AdamW applies decay directly to the parameter, decoupled from the gradient scaling.

**Q5. My loss is NaN at epoch 3. Diagnose.**
LR too high (÷10 and add `clip_grad_norm_`), or `log(0)` from manually applying sigmoid/softmax before the loss, or NaN/inf in the input data, or an unscaled target producing enormous gradients. Check the data first — it is free.

**Q6. Binary loss stuck near 0.693. What does that number mean?**
`ln 2` — the loss of a model outputting p = 0.5 for everything. On balanced data, it often means the model is not learning. On imbalanced data, compare it with the prevalence baseline first. Then check that the optimiser has the parameters, targets vary, and the learning rate is not zero.

**Q7. Why does warmup help transformers?**
Adam's second-moment estimate is unreliable in the first few hundred steps (built from almost no samples), producing erratic step sizes that can wreck the initialisation. Ramping the LR from ~0 lets those estimates stabilise first.

**Q8. Batch size 32 vs 512 — what changes?**
Large batches give less gradient noise, better hardware utilisation, and fewer updates per epoch — but often worse generalisation (they find sharper minima) and they need a larger LR plus warmup. The common heuristic is to scale LR linearly with batch size.

**Q9. Validation loss is *below* training loss. Is that a bug?**
Usually not. Dropout is active during training and disabled during evaluation, so the training number is measured on a handicapped model. It also happens when the validation split is easier by chance. Investigate only if the gap is large.

**Q10. Order of `optimizer.step()` and `scheduler.step()`?**
Optimizer first, scheduler second. Most schedulers step once per epoch; `OneCycleLR` and warmup schedulers step once per batch.

**Q11. Should weight decay apply to biases?**
No. Biases and normalisation parameters have negligible capacity to overfit, and decaying them measurably hurts. Split into parameter groups.

**Q12. Why does `pos_weight` distort probabilities?**
It changes the loss so the optimum is no longer the true conditional probability — the model is now calibrated to a re-weighted distribution. If you need real probabilities, prefer threshold tuning, or recalibrate afterwards (Platt scaling / isotonic).

**Q13. What is gradient clipping and where does it go?**
Rescales the gradient vector if its norm exceeds a maximum, capping the damage from one pathological batch. It goes **after** `loss.backward()` and **before** `optimizer.step()`.

**Q14. Your model has 99% accuracy on a 1%-positive dataset. React.**
That is exactly what predicting the majority class always gives. Report precision, recall, and PR-AUC, look at the confusion matrix, and check whether the model ever predicts positive at all.

### Exercises

1. Plot BCE loss against predicted probability for `y=1` and `y=0` on the same axes. Mark the value at `p=0.5`.
2. Implement `MSELoss` and `CrossEntropyLoss` from scratch. Match PyTorch's output to 6 decimals.
3. Implement SGD-with-momentum manually and reproduce `torch.optim.SGD(momentum=0.9)` step for step.
4. Train the same model at LR `1e-1, 1e-2, 1e-3, 1e-4` and plot all four loss curves together.
5. Compare `CosineAnnealingLR`, `StepLR`, and no scheduler over 60 epochs. Plot both LR and loss.
6. Create 95/5 class imbalance. Train once plain and once with `pos_weight`. Compare precision, recall, and the predicted-probability histograms.
7. Sweep the decision threshold from 0.05 to 0.95 and plot precision, recall, and F1. Choose the operating point for "missing an urgent ticket costs 5× a false alarm".
8. Run the LR range test and identify the steepest-descent region.

<details>
<summary>Answer hints</summary>

1. `-log(p)` and `-log(1-p)`; both equal `0.693` at `p=0.5`.
2. `((pred-y)**2).mean()`; for CE use `-(logits.log_softmax(1).gather(1, y[:,None])).mean()`.
3. `v = momentum*v + g; p -= lr*v`. PyTorch's default (non-Nesterov) matches this exactly.
4. `1e-1` is unstable, `1e-2` fastest, `1e-4` too slow to converge in the budget.
6. `pos_weight` raises recall and lowers precision; the probability histogram shifts right — that is the calibration cost.
7. Weighted F-beta with `beta=√5 ≈ 2.24`, or minimise `5·FN + 1·FP` directly.

</details>

---

## Official references

- [`torch.nn` loss functions](https://docs.pytorch.org/docs/stable/nn.html#loss-functions)
- [`torch.optim`](https://docs.pytorch.org/docs/stable/optim.html)
- [Learning-rate schedulers](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [Adam](https://arxiv.org/abs/1412.6980) · [AdamW / decoupled weight decay](https://arxiv.org/abs/1711.05101)
- [Cyclical learning rates (the LR range test)](https://arxiv.org/abs/1506.01186)
- [Label smoothing / when does it help](https://arxiv.org/abs/1906.02629)
