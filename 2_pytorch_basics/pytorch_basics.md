# PyTorch Basics — Tensors, Autograd, and Modules

> **PyTorch in one sentence:** a **tensor** holds numbers, **autograd** remembers how those numbers were combined so it can compute gradients, and an **`nn.Module`** packages tensors into a model you can train.



## 1. Why PyTorch exists

In the last class we built a neuron by hand: multiply inputs by weights, add a bias, apply an activation, compute a loss. Doing that by hand for a 50-layer network is impossible — not because the math is hard, but because computing **derivatives by hand** for millions of parameters is hopeless.

PyTorch gives us three things NumPy does not:

| Need | NumPy | PyTorch |
|---|---|---|
| Fast array math | yes | yes |
| Run on GPU | no | yes (`.to("cuda")`) |
| Automatic derivatives | no | yes (autograd) |
| Ready-made layers, losses, optimisers | no | yes (`torch.nn`, `torch.optim`) |

### Diagram: what PyTorch automates

```text
        WHAT YOU WRITE                    WHAT PYTORCH DOES FOR YOU
   ┌────────────────────────┐        ┌──────────────────────────────┐
   │ model = nn.Linear(4,3) │  ──>   │ allocate weight + bias        │
   │ y = model(x)           │  ──>   │ matmul + record the operation │
   │ loss = loss_fn(y, t)   │  ──>   │ record the operation          │
   │ loss.backward()        │  ──>   │ chain rule through everything │
   │ optimizer.step()       │  ──>   │ update every parameter        │
   └────────────────────────┘        └──────────────────────────────┘
```

You describe the **forward** computation. PyTorch derives the **backward** computation.

---

## 2. Tensors: the data container

A tensor is an n-dimensional array of numbers. The number of dimensions is called the **rank**.

```text
rank 0  scalar     3.7                       shape ()
rank 1  vector     [1, 2, 3]                 shape (3,)
rank 2  matrix     [[1,2,3],[4,5,6]]         shape (2, 3)
rank 3  cube       a batch of matrices       shape (2, 2, 3)
rank 4  images     batch of colour images    shape (N, C, H, W)
```

### Diagram: how a batch of data is laid out

```text
 A batch of 4 support tickets, 3 features each  →  shape (4, 3)

        feature0   feature1   feature2
        (age_hrs)  (replies)  (neg_score)
ticket0 [  2.0   ,   1.0    ,   0.8     ]
ticket1 [ 30.0   ,   6.0    ,   0.9     ]     shape = (4, 3)
ticket2 [  1.0   ,   0.0    ,   0.1     ]     dim 0 = batch  (rows)
ticket3 [ 12.0   ,   3.0    ,   0.4     ]     dim 1 = feature(cols)
```

**Rule to memorise: dimension 0 is almost always the batch dimension.**

### Ways to create tensors

```python
import torch

torch.tensor([[1.0, 2.0], [3.0, 4.0]])   # from a Python list (copies data)
torch.zeros(3, 4)                        # all zeros
torch.ones(2, 5)                         # all ones
torch.full((2, 2), 7.0)                  # filled with a value
torch.arange(0, 10, 2)                   # 0, 2, 4, 6, 8
torch.linspace(0, 1, 5)                  # 5 evenly spaced values in [0, 1]
torch.randn(3, 4)                        # normal(0, 1) — used to init weights
torch.rand(3, 4)                         # uniform [0, 1)
torch.eye(3)                             # identity matrix
torch.zeros_like(other)                  # same shape/dtype/device as `other`
```

`torch.manual_seed(42)` before random creation makes runs reproducible. Always seed in teaching and in experiments.

### Converting to and from NumPy

```python
import numpy as np
a = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(a)   # SHARES memory — editing one edits the other
back = t.numpy()          # also shares memory
safe = torch.tensor(a)    # copies — use when you want independence
```

> ⚠️ `from_numpy` sharing memory surprises people. If you later modify the NumPy array in place, your tensor silently changes too.

---

## 3. The three contracts: shape, dtype, device

Ninety percent of beginner PyTorch errors are **one of three contracts being broken**. Learn to check all three before debugging anything else.

```python
print(x.shape, x.dtype, x.device)
```

### Diagram: the three contracts

```text
                        ┌──────────────────────┐
   ┌──────────┐         │  SHAPE               │  Do the axes mean what
   │          │────────>│  (4, 3)              │  the next layer expects?
   │  TENSOR  │         ├──────────────────────┤
   │          │────────>│  DTYPE               │  float32 for features,
   │          │         │  torch.float32       │  int64 for class labels
   │          │────────>├──────────────────────┤
   └──────────┘         │  DEVICE              │  Model and data must be
                        │  cpu / cuda / mps    │  on the SAME device
                        └──────────────────────┘
```

### Shape

| Data | Conventional shape | Note |
|---|---|---|
| Tabular batch | `(N, F)` | N rows, F features |
| Binary target | `(N, 1)` float | must match model output shape |
| Multi-class target | `(N,)` int64 | class **indices**, not one-hot |
| Multi-class logits | `(N, K)` | one raw score per class |
| Image batch | `(N, C, H, W)` | PyTorch is channels-**first** |
| Text batch | `(N, T)` int64 | T = sequence length, token ids |

### What each letter means

| Symbol | Meaning | Example |
|---|---|---|
| `N` | batch size — how many examples in this batch | 32 tickets, 4 flowers |
| `F` | number of features per example | 3 (age, replies, negativity) |
| `K` | number of classes | 3 (setosa, versicolor, virginica) |
| `C` | number of image channels | 3 for RGB, 1 for grayscale |
| `H` | image height in pixels | |
| `W` | image width in pixels | |
| `T` | sequence length (tokens per example) | |

### What each row means

- **Tabular batch `(N, F)`** — a batch of rows-and-columns data, like a spreadsheet. This is what a model like `TicketMLP` takes as input.
- **Binary target `(N, 1)` float** — the correct yes/no answer per example, kept as a column so it lines up with the model's `(N, 1)` logit output. Float because `BCEWithLogitsLoss` expects `0.0`/`1.0`, not integers.
- **Multi-class target `(N,)` int64** — the correct class per example, but as a single number, not a column — notice the shape has no second dimension. It holds the class **index** (`2` for virginica), not a one-hot vector (`[0,0,1]`). `int64` because `CrossEntropyLoss` needs it as an index to look up, not a decimal to compare.
- **Multi-class logits `(N, K)`** — what the *model* outputs for a multi-class task: one raw score per class, per example. This is what gets compared against the `(N,)` target inside `CrossEntropyLoss`.
- **Image batch `(N, C, H, W)`** — "channels-first" means channels come right after the batch dimension. Other libraries (NumPy, PIL) usually store images "channels-last" as `(H, W, C)` — a common source of shape-mismatch bugs when moving data between them.
- **Text batch `(N, T)` int64** — each row is a sentence turned into a sequence of token ids, so the values are indices into a vocabulary, not the words themselves.

> The general pattern: a shape like `(N, something)` holds a score or continuous value per example. A shape like `(N,)` holds one discrete label per example.

### Dtype

| Dtype | Use for |
|---|---|
| `torch.float32` | default for features, weights, activations |
| `torch.float64` | rarely; slow on GPU |
| `torch.float16` / `bfloat16` | mixed-precision training (later class) |
| `torch.int64` (`long`) | class labels, token ids, indices |
| `torch.bool` | masks |

Two gotchas:

```python
torch.tensor([1, 2, 3]).dtype        # torch.int64  ← integer literals!
torch.tensor([1., 2., 3.]).dtype     # torch.float32
nn.Linear(3, 1)(torch.tensor([[1,2,3]]))   # RuntimeError: expected Float
```

Fix with `.float()`, `.long()`, or `dtype=torch.float32` at creation.

> ⚠️ "Float for features, int64 for labels" is a rule of thumb for *this* class's examples, not a law. Dtype depends on what a tensor represents, not on whether it's a "feature" or a "label":
> - Token ids and categorical indices going into `nn.Embedding` are `int64`, even though they're "features".
> - Binary and multi-label targets are `float`, even though they're "labels" — only single-label multi-class `CrossEntropyLoss` wants `int64`.
> - Images often arrive as `uint8` and get cast to `float32` only after normalising.
>
> The real rule: `float` for continuous values, `long` for discrete indices.

### Device

```python
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
model = model.to(device)          # moves in place for nn.Module
x = x.to(device)                  # returns a NEW tensor — must reassign
```

> ⚠️ `model.to(device)` mutates the model. `tensor.to(device)` does **not** mutate the tensor — it returns a copy. Forgetting to reassign is a classic bug.

---

## 4. Tensor operations and broadcasting

### Shape-changing operations

| Operation | Meaning | Example |
|---|---|---|
| `reshape(a, b)` | change shape, copy if needed | `(6,) → (2, 3)` |
| `view(a, b)` | change shape, requires contiguous memory | faster, stricter |
| `unsqueeze(1)` | insert a size-1 axis at position 1 | `(N,) → (N, 1)` |
| `squeeze(1)` | remove a size-1 axis at position 1 | `(N, 1) → (N,)` |
| `permute(0, 3, 1, 2)` | reorder axes | `NHWC → NCHW` |
| `transpose(0, 1)` | swap two axes | `(2, 3) → (3, 2)` |
| `cat([a, b], dim=0)` | join along an existing axis | stack batches |
| `stack([a, b], dim=0)` | join along a **new** axis | list → batch |

`-1` means "infer this dimension": `x.reshape(-1, 4)` keeps 4 columns and computes the rows.

> Use bare `squeeze()` carefully. With batch size 1 it removes the batch axis too and produces very confusing downstream errors. Prefer `squeeze(1)`.

### Broadcasting

Broadcasting lets tensors of different shapes combine without writing loops. Compare shapes **right to left**; each pair of dimensions must be equal, or one of them must be 1.

```text
    (4, 3)          batch of 4 tickets
  +    (3,)         one bias per feature
  ---------
    (4, 3)          bias is reused for every row  ✅

    (4, 3)
  +    (4,)         ❌ 3 vs 4 → RuntimeError
```

### Diagram: broadcasting a bias across a batch

```text
  x                          bias           result
 ┌────┬────┬────┐          ┌────┬────┬────┐
 │ 2.0│ 1.0│ 0.8│          │ 0.1│ 0.2│ 0.3│
 ├────┼────┼────┤    +     └────┴────┴────┘   =  same bias row added
 │30.0│ 6.0│ 0.9│              (3,)              to every one of the
 ├────┼────┼────┤        broadcast to (4,3)      4 rows
 │ 1.0│ 0.0│ 0.1│
 ├────┼────┼────┤
 │12.0│ 3.0│ 0.4│
 └────┴────┴────┘
      (4, 3)
```

The classic broadcasting bug:

```python
pred   = torch.randn(32, 1)     # model output
target = torch.randn(32)        # labels forgot unsqueeze
loss   = ((pred - target) ** 2).mean()   # broadcasts to (32, 32)! No error, wrong answer.
```

**This is silent.** No exception is raised, the loss is nonsense, and the model never learns. Always check that prediction and target shapes are identical.

### Reductions

```python
x.sum(), x.mean(), x.max(), x.min(), x.std()
x.sum(dim=0)      # collapse the batch  → (F,)
x.sum(dim=1)      # collapse features   → (N,)
x.argmax(dim=1)   # index of the largest value per row → class prediction
x.sum(dim=1, keepdim=True)   # keeps a size-1 axis → (N, 1)
```

`dim=k` means "**collapse** axis k". Say it out loud when you write it.

### Matrix multiplication

```python
A = torch.randn(4, 3)
B = torch.randn(3, 2)
A @ B            # shape (4, 2) — inner dims (3) must match
```

A `nn.Linear(3, 2)` layer is exactly `x @ W.T + b` with `W` of shape `(2, 3)`.

---

## 5. Autograd: how gradients are computed

### The idea

When a tensor has `requires_grad=True`, every operation performed on it is recorded in a **computation graph**. Calling `.backward()` on a scalar walks that graph backwards, applying the chain rule, and stores the result in each leaf tensor's `.grad`.

### Diagram: the computation graph

```text
      FORWARD  (build the graph)              BACKWARD (chain rule)

  w ──┐                                   w.grad ◄─┐
      ├──> z = w * x ──> L = (z - y)²              │  dL/dw = dL/dz · dz/dw
  x ──┘         │              │                   │
                │              │             dL/dz ┘
             (recorded)    (recorded)

  loss.backward()  walks right → left and fills  w.grad
```

```python
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)                   # no grad needed for data
loss = (w * x - 10) ** 2                # loss = (6 - 10)^2 = 16
loss.backward()
print(w.grad)                           # dL/dw = 2*(wx-10)*x = 2*(-4)*3 = -24
```

Verify by hand — students should always be able to check one gradient manually.

### The four rules of autograd

| Rule | Meaning |
|---|---|
| Only **float** tensors can require grad | integers have no meaningful derivative |
| `.backward()` needs a **scalar** | otherwise pass a gradient argument; in practice always `loss.mean()` |
| Gradients **accumulate** into `.grad` | you must zero them each step |
| The graph is **freed** after backward | call `backward(retain_graph=True)` to reuse (rare) |

### Why gradients accumulate

This is deliberate, not a bug. It lets you simulate a large batch on small memory:

```text
  Normal step                       Gradient accumulation (batch 128 on small GPU)

  zero_grad                         zero_grad
  forward(batch of 128)             for 4 micro-batches of 32:
  backward                              forward → backward   (grads pile up)
  step                              step                     (one update)
```

So in ordinary training you must clear them yourself:

```python
optimizer.zero_grad(set_to_none=True)   # set_to_none is faster than filling zeros
loss.backward()
optimizer.step()
```

Forget `zero_grad` and your gradients are the sum of every batch so far — training diverges or crawls.

### Turning autograd off

| Tool | What it does | When |
|---|---|---|
| `torch.no_grad()` | stops recording the graph | evaluation inside training |
| `torch.inference_mode()` | stricter and faster than `no_grad` | pure inference / serving |
| `tensor.detach()` | cuts one tensor out of the graph | logging, stop-gradient tricks |
| `param.requires_grad = False` | freeze a parameter | transfer learning |

```python
@torch.inference_mode()
def predict(model, x):
    model.eval()
    return model(x)
```

> `model.eval()` and `torch.no_grad()` are **different things** and you usually need both.
> `eval()` changes *layer behaviour* (Dropout off, BatchNorm uses running stats).
> `no_grad()` changes *memory and speed* (no graph is built).
> Neither implies the other.

### `.item()` — leaving the graph

```python
total_loss += loss.item()          # ✅ Python float, graph is freed
total_loss += loss                 # ❌ keeps the whole graph alive → memory leak
```

---

## 6. `nn.Module`: packaging a model

An `nn.Module` is a class that (a) holds parameters and (b) defines a `forward` method. PyTorch tracks any tensor assigned as an `nn.Parameter` or any sub-module assigned as an attribute.

```python
from torch import nn

class TicketMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=16):
        super().__init__()                      # never forget this line
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),           # 1 raw logit
        )

    def forward(self, x):
        return self.net(x)                      # return LOGITS, not probabilities

model = TicketMLP()
print(model)
print(sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable params")
```

### Diagram: what `nn.Module` gives you

```text
                     ┌───────────────────────────┐
                     │        nn.Module          │
                     ├───────────────────────────┤
 .parameters()  <────│ every nn.Parameter found  │──> feed to the optimiser
 .state_dict()  <────│ name → tensor dictionary  │──> save / load
 .to(device)    <────│ moves every parameter     │
 .train()/.eval()<───│ flips Dropout & BatchNorm │
 model(x)       <────│ calls forward(x)          │
                     └───────────────────────────┘
```

Call `model(x)`, **never** `model.forward(x)` — the `__call__` path runs registered hooks that `forward` alone skips.

### Common layers

| Layer | Purpose |
|---|---|
| `nn.Linear(in, out)` | fully connected: `xW^T + b` |
| `nn.ReLU()` | `max(0, x)` — default hidden activation |
| `nn.Sigmoid()` | squashes to (0,1) — probability, **not** before BCEWithLogits |
| `nn.Tanh()` | squashes to (-1,1) |
| `nn.Dropout(p)` | randomly zeroes p fraction during training only |
| `nn.BatchNorm1d(F)` | normalises each feature across the batch |
| `nn.Sequential(...)` | runs layers in order |
| `nn.Flatten()` | `(N,C,H,W) → (N, C*H*W)` |

### `nn.Parameter` vs buffer vs plain tensor

```python
class Demo(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(3))          # trained, in state_dict
        self.register_buffer("mean", torch.zeros(3))   # saved, NOT trained
        self.scale = torch.tensor(2.0)                 # ❌ not moved by .to(), not saved
```

If a tensor must move with the model but never be trained (running statistics, positional encodings), register it as a **buffer**.

### Output-and-loss pairing — the single most important table today

| Task | Model outputs | Target | Loss | Convert for metrics |
|---|---|---|---|---|
| Regression | `(N, 1)` values | float `(N, 1)` | `MSELoss` / `L1Loss` / `HuberLoss` | none |
| Binary | `(N, 1)` **logits** | float `(N, 1)` of 0/1 | `BCEWithLogitsLoss` | `sigmoid()` then threshold |
| K-class | `(N, K)` **logits** | long `(N,)` indices | `CrossEntropyLoss` | `argmax(dim=1)` |
| Multi-label | `(N, K)` **logits** | float `(N, K)` | `BCEWithLogitsLoss` | `sigmoid()` per class |

```text
  ✅ RIGHT                                ❌ WRONG
  logits ──> BCEWithLogitsLoss            sigmoid ──> BCEWithLogitsLoss
  logits ──> CrossEntropyLoss             softmax ──> CrossEntropyLoss
  logits ──> sigmoid ──> threshold        (double-applying the squash:
             (metrics only)                trains badly, silently)
```

Both `BCEWithLogitsLoss` and `CrossEntropyLoss` apply the squashing internally using a numerically stable formulation (log-sum-exp trick). Applying it yourself first is mathematically wrong *and* less stable.

---

## 7. Dataset and DataLoader

`Dataset` answers "how do I get example *i*?". `DataLoader` answers "how do I get batch *b*?".

```python
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Quickest path when everything already fits in memory:
train_ds = TensorDataset(X_train, y_train)

# Custom dataset — the pattern you will reuse for images and text:
class TicketDataset(Dataset):
    def __init__(self, features, labels):
        self.features, self.labels = features, labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  drop_last=False)
valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False)
```

### Diagram: the data pipeline

```text
  raw data ──> Dataset ──> Sampler ──> collate_fn ──> batch ──> .to(device) ──> model
              __getitem__   decides     stacks the
              one example   the order   examples
```

| DataLoader argument | Meaning |
|---|---|
| `batch_size` | examples per step |
| `shuffle=True` | **train only** — never shuffle validation/test |
| `num_workers` | parallel loading processes (0 on Windows/notebooks is safest) |
| `pin_memory=True` | faster CPU→GPU copy, CUDA only |
| `drop_last=True` | drop a ragged final batch (useful with BatchNorm) |

---

## 8. The training loop skeleton

Memorise this order. Every PyTorch training script you will ever write is this loop with more logging around it.

```text
  for each epoch:
      for each batch:
          1. move batch to device
          2. forward   →  logits = model(x)
          3. loss      →  loss = loss_fn(logits, y)
          4. zero      →  optimizer.zero_grad(set_to_none=True)
          5. backward  →  loss.backward()
          6. step      →  optimizer.step()
      evaluate on the validation set
```

```python
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)


@torch.inference_mode()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += loss_fn(logits, y).item() * x.size(0)
        pred = (logits.sigmoid() >= 0.5)
        correct += (pred == y.bool()).sum().item()
        total += y.numel()
    return loss_sum / len(loader.dataset), correct / total
```

Note `loss.item() * x.size(0)` then divide by dataset size: this gives a correct average even when the last batch is smaller.

---

## 9. Saving and loading

Save the **state dict**, not the model object.

```python
torch.save({
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "val_metrics": metrics,
}, "ticket_model.pt")

ckpt = torch.load("ticket_model.pt", map_location=device, weights_only=True)
model.load_state_dict(ckpt["model_state"])
optimizer.load_state_dict(ckpt["optimizer_state"])
```

| Approach | Pros | Cons |
|---|---|---|
| `state_dict` (recommended) | portable, small, version-tolerant | you must recreate the class first |
| `torch.save(model)` (pickle) | one line | breaks when your code moves/renames |

`weights_only=True` restricts unpickling. **Only load checkpoints you trust** — a pickle file can execute arbitrary code.

---

## 10. Debugging checklist

### Diagram: the debugging decision tree

```text
                    Model is not working
                            │
              ┌─────────────┴─────────────┐
        Raises an error              Runs, but loss is flat/NaN
              │                              │
   print(shape, dtype, device)        ┌──────┴──────┐
   for x, y, and model output      NaN loss     Flat loss
              │                        │             │
   ┌──────────┼──────────┐      lower LR,     check output/loss
 shape      dtype     device       check for      pairing, check
 mismatch   mismatch  mismatch     log(0) /       params are in the
   │          │          │         div by 0       optimiser
unsqueeze   .float()  .to(device)                       │
reshape     .long()                          TINY-SUBSET OVERFIT TEST
```

### The tiny-subset overfit test

Before tuning anything, take **16–64 examples** and train until the model nearly memorises them.

- **Cannot memorise them** → you have a *bug*. Wrong loss pairing, targets not connected, parameters not in the optimiser, learning rate absurd.
- **Memorises them but fails on validation** → the pipeline is correct; you now have a *generalisation* problem (more data, regularisation, simpler model).

This one test separates "my code is broken" from "my model is weak", and saves hours.

### Common errors and what they mean

| Error message | Real cause | Fix |
|---|---|---|
| `expected scalar type Float but found Long` | integer features into `nn.Linear` | `x.float()` |
| `Expected all tensors on the same device` | forgot `.to(device)` on batch or model | move both |
| `mat1 and mat2 shapes cannot be multiplied` | wrong `in_features` | print shape before the layer |
| `Target size must be the same as input size` | `(N,)` target vs `(N,1)` output | `y.unsqueeze(1)` |
| `0D or 1D target tensor expected` (CrossEntropy) | one-hot targets | pass class indices, `.long()` |
| `element 0 of tensors does not require grad` | broke the graph (`.detach()`, `no_grad`, numpy round-trip) | keep the graph intact |
| `Trying to backward a second time` | reused a freed graph | recompute forward, or `retain_graph=True` |
| loss stuck at exactly the same value | `zero_grad` missing, or LR = 0, or model params not passed to optimiser | check `optimizer.param_groups` |

---

## 11. Q&A and revision



**Q1. What is the difference between `torch.tensor(x)` and `torch.from_numpy(x)`?**
`torch.tensor` copies the data. `from_numpy` shares memory with the NumPy array, so mutating one mutates the other.

**Q2. Why does `nn.Linear` reject `torch.tensor([[1, 2, 3]])`?**
Integer literals create an `int64` tensor. Linear layers hold `float32` weights. Use `.float()`.

**Q3. What does `requires_grad=True` actually do?**
It marks the tensor as a leaf to differentiate with respect to, and makes PyTorch record every operation applied to it into a graph.

**Q4. Why must `.backward()` be called on a scalar?**
The gradient of a vector with respect to parameters is a Jacobian, not a vector. A scalar loss makes `dL/dθ` well defined. This is why losses reduce with `mean()` by default.

**Q5. Why does PyTorch accumulate gradients instead of overwriting?**
So you can split a large batch into micro-batches (gradient accumulation), and so multiple losses can contribute to the same parameters. The price is that you must call `zero_grad()`.

**Q6. `model.eval()` vs `torch.no_grad()` — do I need both?**
Yes. `eval()` changes layer behaviour (Dropout off, BatchNorm uses running statistics). `no_grad()`/`inference_mode()` stops graph building for speed and memory. They are orthogonal.

**Q7. Why should the model return logits and not probabilities?**
`BCEWithLogitsLoss` and `CrossEntropyLoss` fuse the sigmoid/softmax with the log for numerical stability. Applying sigmoid first and then `BCEWithLogitsLoss` squashes twice — it still runs, but trains badly.

**Q8. My loss does not change at all. First three checks?**
(1) Is `optimizer.step()` being called and are `model.parameters()` actually passed to the optimiser? (2) Do output shape and target shape match exactly? (3) Run the tiny-subset overfit test.

**Q9. What is the difference between `view` and `reshape`?**
`view` requires contiguous memory and returns a view; `reshape` returns a view when it can and copies when it cannot. When unsure, use `reshape`.

**Q10. Why is `shuffle=True` wrong for the validation loader?**
It changes nothing about the metric value but destroys reproducible per-example inspection, and with `drop_last` it can silently drop different examples each run. Keep evaluation deterministic.

**Q11. What is `set_to_none=True` in `zero_grad`?**
Instead of writing zeros into every gradient buffer, it sets `.grad = None`. It is faster and uses less memory. It is the default since PyTorch 2.0.

**Q12. Where does a broadcasting bug hide?**
Between `(N, 1)` predictions and `(N,)` targets. It produces an `(N, N)` result with no error. Always assert `pred.shape == target.shape`.

### Exercises

1. Create a `(6, 4)` float32 tensor on the correct device. Print shape, dtype, device. Convert it to `(3, 8)` two different ways.
2. Compute `w.grad` by hand for `L = (3w + 1)^2` at `w = 2`, then verify with autograd.
3. Take a model with a `(N, 1)` output and a `(N,)` target. Show the silent broadcasting bug numerically, then fix it.
4. Build a 3-layer MLP as an `nn.Module`. Print the total number of trainable parameters and verify the count by hand.
5. Write a custom `Dataset` over a NumPy array and iterate one batch from its `DataLoader`. Print the batch shapes.
6. Deliberately remove `optimizer.zero_grad()` from a training loop. Plot the loss and explain the shape of the curve.
7. Run the tiny-subset overfit test on 32 examples. Report the epoch at which training accuracy reaches 100%.

<details>
<summary>Answer hints</summary>

1. `torch.randn(6, 4, dtype=torch.float32, device=device)`; `x.reshape(3, 8)` and `x.view(3, 8)`.
2. `dL/dw = 2(3w+1)·3 = 6(3·2+1) = 42`.
3. `(pred - target)` becomes `(N, N)`; fix with `target.unsqueeze(1)` or `pred.squeeze(1)`.
4. For `Linear(a, b)` the count is `a*b + b`. Sum over layers.
6. Gradients sum over batches, so the effective step size grows through the epoch — loss becomes unstable or diverges.

</details>

---

## Official references

- [PyTorch — Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [`nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [`torch.utils.data`](https://docs.pytorch.org/docs/stable/data.html)
- [Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
- [Serialization and `weights_only`](https://docs.pytorch.org/docs/stable/notes/serialization.html)
