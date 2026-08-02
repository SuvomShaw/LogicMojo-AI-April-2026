# Deep Learning Foundations

> A neural network learns by making a prediction, measuring its mistake, and changing its weights to make the next prediction better.


## Contents

1. [The main idea](#1-the-main-idea)
2. [One neuron](#2-one-neuron)
3. [Layers and activation functions](#3-layers-and-activation-functions)
4. [Tensors and shapes](#4-tensors-and-shapes)
5. [Loss functions](#5-loss-functions)
6. [How the model learns](#6-how-the-model-learns)
7. [Training terms and optimisers](#7-training-terms-and-optimisers)
8. [Training, validation, and test data](#8-training-validation-and-test-data)
9. [A small PyTorch training pattern](#9-a-small-pytorch-training-pattern)
10. [Which network should we use?](#10-which-network-should-we-use)
11. [Quick revision and practice](#11-quick-revision-and-practice)

---

## 1. The main idea

A neural network is a function with numbers that it can learn. These learnable numbers are called **parameters**. Weights and biases are parameters.

### Diagram: complete learning cycle

```text
┌───────────────┐     ┌────────────────┐     ┌────────────┐
│ Input features│ ──> │ Neural network │ ──> │ Prediction │
└───────────────┘     └────────────────┘     └─────┬──────┘
                                                   │
                                                   v
┌───────────────┐     ┌────────────────┐     ┌────────────┐
│ Update weights│ <── │   Gradients    │ <── │    Loss    │
│   and bias    │     │                │     │            │
└───────┬───────┘     └────────────────┘     └─────▲──────┘
        │                                          │
        └──────── back to neural network           │
                                      Correct answer
```

Read the diagram from left to right:

1. Give input data to the model.
2. The model makes a prediction.
3. The loss tells us how wrong the prediction is.
4. Gradients tell us how to change the parameters.
5. The optimiser updates the parameters.

Three ideas must stay separate:

| Idea | Simple meaning | Example |
|---|---|---|
| **Model** | Makes a prediction | Predict whether a ticket is urgent. |
| **Loss** | Gives a training error value | Give a larger error when the model misses an urgent ticket. |
| **Metric** | Tells us how useful the model is | Check precision, recall, and accuracy. |

A low loss is helpful, but it is not the complete goal. The model must also work well on new data and meet the real need of the system.

### When should we use Deep Learning?

| Situation | Good first choice | Why? |
|---|---|---|
| Images, audio, or long text | A pretrained Deep Learning model | It already knows many useful patterns. |
| A large dataset with complex patterns | Neural network and one simple model for comparison | The network may learn useful feature combinations. |
| A small table of rows and columns | Tree model or linear model first | It is often faster and easier. |
| The answer follows clear rules | Rule-based code | Training may not be needed. |
| The result must be very easy to explain | A simpler model first | Simpler models are easier to inspect. |

Do not choose Deep Learning only because it is popular. First ask whether it gives enough value for its data, compute, and maintenance cost.

### Useful symbols

| Symbol | Meaning |
|---|---|
| \(x\) | input feature or input vector |
| \(w\) | weight |
| \(b\) | bias |
| \(z\) | raw score, also called a logit |
| \(\hat y\) | model prediction |
| \(y\) | correct answer or label |
| \(L\) | loss |
| \(\eta\) | learning rate |

---

## 2. One neuron

A neuron does three simple steps:

1. Multiply each input by a weight.
2. Add the results and the bias.
3. Pass the result through an activation function when needed.

For two inputs:

\[
z = w_1x_1 + w_2x_2 + b
\]

### Diagram: calculation inside one neuron

```text
Input x1 ──> multiply by weight w1 ──┐
                                    │
Input x2 ──> multiply by weight w2 ──┼──> add ──> raw score z
                                    │              │
Bias b ─────────────────────────────┘              v
                                           activation function
                                                   │
                                                   v
                                               prediction
```

### Meaning of each part

- **Input:** information given to the model.
- **Weight:** tells the model how important an input is.
- **Bias:** moves the final score up or down.
- **Logit:** the raw score before it becomes a probability.
- **Activation:** changes the score into a useful output or hidden feature.

### Small example

Suppose we use two ticket features:

- negative-word score = `0.8`
- waiting-time score = `0.6`

Use these parameters:

- negative-word weight = `2.0`
- waiting-time weight = `1.0`
- bias = `-1.0`

\[
z = (0.8 \times 2.0) + (0.6 \times 1.0) - 1.0 = 1.2
\]

The logit is `1.2`. A logit is not yet a probability.

For binary classification, sigmoid changes the logit into a number between `0` and `1`:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

\[
\sigma(1.2) \approx 0.77
\]

The model gives about `77%` probability to the urgent class. We still need a threshold to make a final class decision.

---

## 3. Layers and activation functions

A layer contains several neurons. Each neuron learns a different combination of the inputs.

### Diagram: a small neural network

```text
Input layer             Hidden layer               Output layer

negative score ──┐      ┌──────────┐
                 ├────> │ neuron 1 │ ──┐
waiting score  ──┘      │ neuron 2 │   │
                        │ neuron 3 │   ├──> ReLU ──> one urgency logit
                        │ neuron 4 │   │
                        └──────────┘ ──┘
```

### Input, hidden, and output layers

- **Input layer:** receives the feature values.
- **Hidden layer:** learns useful combinations of features.
- **Output layer:** produces the value needed for the task.

### Why do we need an activation function?

Without an activation function, many linear layers still behave like one linear calculation. The network cannot learn enough curved or complex patterns.

ReLU is a common hidden-layer activation:

\[
\operatorname{ReLU}(x) = \max(0, x)
\]

It changes negative values to `0` and keeps positive values.

| Input | ReLU output |
|---:|---:|
| -3 | 0 |
| -1 | 0 |
| 0 | 0 |
| 2 | 2 |
| 5 | 5 |

### Common output choices

| Task | Final model output | Loss |
|---|---|---|
| Predict one number | One raw number | MAE or MSE |
| Binary classification | One raw logit | `BCEWithLogitsLoss` |
| One class from many classes | One logit per class | `CrossEntropyLoss` |
| Several independent labels | One logit per label | `BCEWithLogitsLoss` |

For BCE and cross-entropy, return raw logits from the model. The loss function handles sigmoid or softmax safely.

---

## 4. Tensors and shapes

A PyTorch tensor is a table of numbers with one or more dimensions. It is similar to a NumPy array, but it can also use a GPU and calculate gradients.

### Diagram: examples, features, and shape

```text
                         Feature 1       Feature 2
                      negative score   waiting score
                     ┌───────────────┬───────────────┐
Ticket 1             │     0.90      │     0.80      │
                     ├───────────────┼───────────────┤
Ticket 2             │     0.10      │     0.20      │
                     └───────────────┴───────────────┘

Tensor shape = (2, 2)
               │  └── 2 features
               └───── 2 examples
```

Check these four properties:

| Property | Example | Simple meaning |
|---|---|---|
| Shape | `(32, 10)` | 32 examples and 10 features per example |
| Dtype | `torch.float32` | The values are decimal numbers |
| Device | `cpu`, `cuda`, or `mps` | Where the tensor is stored |
| Gradient tracking | `requires_grad=True` | PyTorch should calculate gradients for it |

```python
import torch

x = torch.tensor(
    [[0.9, 0.8], [0.1, 0.2]],
    dtype=torch.float32,
)

print(x.shape)   # 2 examples, 2 features
print(x.dtype)   # torch.float32
print(x.device)  # cpu by default
```

### Batch shape

For a dense layer:

| Tensor | Example shape | Meaning |
|---|---:|---|
| Input `X` | `(32, 10)` | 32 examples, 10 features |
| Weight `W` | `(64, 10)` | 64 neurons, 10 weights each |
| Bias `b` | `(64,)` | One bias for each neuron |
| Output | `(32, 64)` | 64 new features for each example |

### Device rule

The model and its input must be on the same device.

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

x = x.to(device)
model = model.to(device)
```

Use a GPU when the model and data are large enough to benefit from it. CPU is fine for small examples.

---

## 5. Loss functions

The loss gives one number that tells us how wrong the model is during training. A smaller loss is usually better.

### Diagram: where loss is used

```text
Model output ────────┐
                     ├──> Loss function ──> One error value ──> Backward pass
Correct answer ──────┘
```

### Choose loss from the task

| Task | Good starting loss | Useful metric |
|---|---|---|
| Predict a number | MAE, MSE, or Huber | MAE or RMSE |
| Binary classification | BCE with logits | Precision, recall, F1, PR-AUC |
| One class from many | Cross-entropy | Accuracy and per-class F1 |
| Several labels at once | BCE with logits for each label | Per-label precision and recall |

### MAE and MSE

Use these for regression.

- **MAE:** takes the average absolute error.
- **MSE:** squares each error before taking the average.

MSE gives much more importance to large mistakes.

Example:

| Correct value | Prediction | Absolute error | Squared error |
|---:|---:|---:|---:|
| 5 | 7 | 2 | 4 |
| 5 | 15 | 10 | 100 |

MAE is `(2 + 10) / 2 = 6`.

MSE is `(4 + 100) / 2 = 52`.

### Binary Cross-Entropy

Use Binary Cross-Entropy for a two-class problem such as urgent or normal.

The model returns one logit. `BCEWithLogitsLoss` compares that logit with a label of `0` or `1`.

```python
loss_function = torch.nn.BCEWithLogitsLoss()
loss = loss_function(logits, labels)
```

Important:

```python
# Correct: pass raw logits
loss = loss_function(logits, labels)

# Incorrect: do not apply sigmoid before BCEWithLogitsLoss
# loss = loss_function(torch.sigmoid(logits), labels)
```

Use sigmoid later when you want to display a probability or apply a threshold.

### Rare positive examples

Suppose urgent tickets are rare. A model may predict every ticket as normal and still get high accuracy.

Check these values instead:

- **Recall:** how many urgent tickets did we catch?
- **Precision:** how many predicted urgent tickets were really urgent?
- **False negatives:** which urgent tickets did we miss?
- **False positives:** how many normal tickets did we send for extra review?

`pos_weight` can give more importance to positive labels during training:

```python
loss_function = torch.nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([4.0])
)
```

Here, a mistake on a true positive example gets four times the normal loss. The label stays `1`; it does not become `4`.

A common starting value is:

\[
\text{positive weight} = \frac{\text{number of negative examples}}{\text{number of positive examples}}
\]

This is only a starting point. Choose the final value by checking validation results.

### Custom loss

Start with a standard loss. Add a custom rule only when the real requirement is clear.

For example, a model may learn urgency as its main task and ticket category as a smaller helper task:

\[
\text{total loss} = \text{urgency loss} + 0.2 \times \text{category loss}
\]

Before keeping a custom loss, check:

1. Does it produce valid gradients?
2. Are all loss parts on reasonable scales?
3. Does it beat the standard loss on validation data?
4. Does it improve the metric that matters for the real task?

---

## 6. How the model learns

Training repeats four main actions:

### Diagram: one training step

```text
┌──────────────┐    ┌────────────────┐    ┌───────────────┐
│ Forward pass │ ─> │ Calculate loss │ ─> │ Backward pass │
└──────▲───────┘    └────────────────┘    └───────┬───────┘
       │                                           │
       │          ┌───────────────────┐            │
       └───────── │ Update parameters │ <──────────┘
                  └───────────────────┘
```

### Forward pass

The model uses the current weights and bias to make a prediction.

### Backward pass

The backward pass calculates gradients. A gradient tells us how a small change in a parameter will change the loss.

PyTorch does this with:

```python
loss.backward()
```

The gradient for each parameter is stored in its `.grad` value.

### Update step

The optimiser changes the parameters in the direction that should reduce the loss:

\[
\text{new parameter} = \text{old parameter} - \text{learning rate} \times \text{gradient}
\]

The learning rate controls the size of the update.

- Too large: training may jump around or fail.
- Too small: training may be very slow.

### One PyTorch training step

```python
optimizer.zero_grad()
logits = model(features)
loss = loss_function(logits, labels)
loss.backward()
optimizer.step()
```

Remember the order:

```text
clear old gradients -> predict -> calculate loss -> backward -> update
```

---

## 7. Training terms and optimisers

### Diagram: batch, step, and epoch

```text
One epoch = one complete pass through the training data

Training data
├── Batch 1 ──> forward + loss + backward + update ──> Step 1
├── Batch 2 ──> forward + loss + backward + update ──> Step 2
├── Batch 3 ──> forward + loss + backward + update ──> Step 3
└── Batch 4 ──> forward + loss + backward + update ──> Step 4

After Batch 4, one epoch is complete.
```

| Term | Simple meaning |
|---|---|
| Batch | A small group of examples used for one update |
| Step or iteration | One forward pass, backward pass, and update |
| Epoch | One complete pass through the training data |

### Why use batches?

Sending all data at once may use too much memory. Sending one example at a time may be slow. A batch gives a useful middle option.

### Optimiser

The optimiser uses gradients to update weights and biases.

Common choices:

- **Adam or AdamW:** a good starting point for many tasks.
- **SGD with momentum:** simple and useful for comparison.

There is no optimiser that is best for every problem.

### Common training problems

| What you see | First things to check |
|---|---|
| Loss becomes `NaN` | Lower the learning rate and check invalid values. |
| Loss does not change | Check labels, shapes, loss choice, and gradients. |
| Training improves but validation gets worse | Check data leakage and overfitting. |
| Both losses stay high | Check data, features, model size, and learning rate. |

A useful test is to train on 10 to 50 examples. The model should almost memorise this tiny set. If it cannot, check the data and training code.

---

## 8. Training, validation, and test data

### Diagram: purpose of each data split

```text
                         ┌─> Training data   ─> learn weights and bias
                         │
All available data ──────┼─> Validation data ─> choose settings and threshold
                         │
                         └─> Test data       ─> final check
```

| Split | What it is used for |
|---|---|
| Training | Update weights and biases |
| Validation | Choose model settings, loss weights, and threshold |
| Test | Check the final model after choices are fixed |

Do not keep checking the test set while changing the model. If you do, the test set is no longer a fair final check.

### Avoid data leakage

Data leakage means the model receives information that it should not have.

Common examples:

- the same user appears in both training and validation data;
- duplicate rows appear in different splits;
- a feature uses information created after the prediction time;
- normalisation values are calculated using all data.

### Normalisation

Normalisation keeps feature values on similar scales. Calculate the mean and standard deviation from training data only.

```python
mean = train_x.mean(dim=0, keepdim=True)
std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)

train_x = (train_x - mean) / std
val_x = (val_x - mean) / std
test_x = (test_x - mean) / std
```

### Underfitting and overfitting

- **Underfitting:** the model does not learn the training pattern well.
- **Overfitting:** the model learns the training data but performs poorly on new data.

If the model overfits:

1. Check data leakage and labels first.
2. Add more useful training data if possible.
3. Use a smaller model.
4. Try weight decay or dropout.
5. Stop training when validation performance stops improving.

---

## 9. A small PyTorch training pattern

### Diagram: training code order

```text
Move data to device
        │
        v
Clear old gradients
        │
        v
Run the model
        │
        v
Calculate loss
        │
        v
Run backward
        │
        v
Update the model
```

```python
import torch
from torch import nn

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = nn.Sequential(
    nn.Linear(num_features, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
).to(device)

loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for features, labels in train_loader:
    features = features.to(device, dtype=torch.float32)
    labels = labels.to(device, dtype=torch.float32).view(-1, 1)

    model.train()
    optimizer.zero_grad()
    logits = model(features)
    loss = loss_function(logits, labels)
    loss.backward()
    optimizer.step()

model.eval()
with torch.inference_mode():
    validation_logits = model(validation_features.to(device))
    validation_probabilities = torch.sigmoid(validation_logits)
```

### Why use `model.eval()`?

It changes layers such as dropout into evaluation mode.

### Why use `torch.inference_mode()`?

It stops gradient tracking because we are not training. This uses less memory and work.

### Common PyTorch mistakes

| Mistake | Fix |
|---|---|
| Model is on GPU but data is on CPU | Move both to the same device. |
| BCE labels are integers | Convert them to floating-point values. |
| Sigmoid is used before `BCEWithLogitsLoss` | Pass raw logits to the loss. |
| Softmax is used before `CrossEntropyLoss` | Pass raw class logits to the loss. |
| Old gradients are not cleared | Call `optimizer.zero_grad()` before `backward()`. |
| Validation runs in training mode | Use `model.eval()` and `torch.inference_mode()`. |

### Before training

- [ ] Check feature and label shapes.
- [ ] Check dtypes and label values.
- [ ] Put the model and tensors on the same device.
- [ ] Make sure the model output shape matches the loss.
- [ ] Fit preprocessing using training data only.
- [ ] Try to overfit a tiny set of examples.
- [ ] Choose metrics that match the real cost of mistakes.

---

## 10. Which network should we use?

| Data type | Common starting model |
|---|---|
| Rows and columns | MLP, compared with a tree model |
| Images | CNN or Vision Transformer |
| Text | Transformer |
| Time series | Temporal CNN, RNN, or Transformer |
| Graph data | Graph Neural Network |

### Diagram: choose a model from the data type

```text
Data
├── Rows and columns ──> MLP or tree model
├── Images ────────────> CNN or Vision Transformer
├── Text ──────────────> Transformer
├── Time series ───────> Temporal CNN, RNN, or Transformer
└── Graph data ────────> Graph Neural Network
```

The architecture is only one part of the solution. Data quality, a suitable loss, a fair split, and useful metrics are equally important.

---

## 11. Quick revision and practice

### Diagram: the complete prediction path

```text
Raw ticket data
      │
      v
Features stored in a tensor
      │
      v
Linear layer + ReLU
      │
      v
Urgency logit
      │
      v
Sigmoid probability
      │
      v
Chosen threshold
      │
      ├── probability below threshold ──> normal queue
      └── probability at or above threshold ──> urgent review queue
```

### Quick revision

```text
Feature:       one input value used by the model
Label:         the correct answer
Weight:        importance learned for an input
Bias:          value that moves the score up or down
Logit:         raw model score
Probability:   sigmoid or softmax output used to read the model score
Loss:          training error value
Gradient:      direction and amount of parameter change
Optimiser:     applies parameter updates
Epoch:         one full pass through training data
Threshold:     converts a probability into a decision
```

### Practice questions

1. Why is a logit not the same as a probability?
2. What do weights and bias do?
3. Why do hidden layers need activation functions?
4. Which loss would you use for urgent or normal classification?
5. Why should sigmoid not be used before `BCEWithLogitsLoss`?
6. What is the difference between training, validation, and test data?
7. Why can high accuracy be misleading when urgent tickets are rare?
8. What should you check when training loss falls but validation loss rises?

<details>
<summary>Answers</summary>

1. A logit is an unrestricted raw score. Sigmoid changes it into a value between 0 and 1.
2. Weights control input importance. Bias moves the score up or down.
3. Activations help the network learn patterns that are not straight lines.
4. One raw logit with `BCEWithLogitsLoss`.
5. The loss already applies sigmoid safely.
6. Training updates parameters, validation chooses settings, and test checks the final model.
7. The model can predict the common class every time and still get high accuracy.
8. Check leakage and the split first, then check for overfitting.

</details>

### Further study

- [Google ML Crash Course: Neural networks](https://developers.google.com/machine-learning/crash-course/neural-networks)
- [Google: accuracy, precision, and recall](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall)
- [PyTorch: BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- [PyTorch: CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
