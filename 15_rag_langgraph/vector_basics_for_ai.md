# Vector basics for AI



You need six ideas: a vector, its shape, the dot product, length, cosine similarity, and a matrix–vector multiply. Those six show up in linear regression, logistic regression, neural nets, image models, embeddings, search, and LoRA.
## 1. A vector is an ordered list of numbers

A **scalar** is one number: a learning rate `0.01`, a loss `0.37`.

A **vector** is an ordered list. Order matters. `[2500, 3, 2]` is not the same house as `[3, 2500, 2]`.

```text
One house
  area        = 2500
  bedrooms    = 3
  bathrooms   = 2

  x = [2500, 3, 2]
```

In a model, this list is the **feature vector** of one example. A word embedding, a user profile, and a flattened image are the same kind of object: a list of numbers.

A **matrix** is a table of vectors. One row is one example. One column is one feature.

```text
X = [
  [2500, 3, 2],    row 0: house A
  [1800, 2, 1],    row 1: house B
]
```

**Shape** is `(rows, columns)`.

| Object | Shape | Meaning |
|---|---|---|
| one house `x` | `(3,)` | 3 features |
| two houses `X` | `(2, 3)` | 2 examples, 3 features |
| a weight vector `w` | `(3,)` | one weight per feature |
| a batch of 32 images, 224×224, 3 colours | `(32, 3, 224, 224)` | a tensor: a vector with more axes |

When code fails with a shape error, two lists were combined in an order the sizes do not allow. Check the shapes before checking the model.

## 2. Add and scale

Add matching positions. Scale every position by one number.

```text
a = [1, 2]
b = [10, 20]

a + b     = [11, 22]
3 * a     = [3, 6]
```

Training uses this on every step. The optimizer takes the current weights and steps a small distance along the gradient:

```text
new weights = old weights - learning_rate * gradient
```

That is vector addition and scaling. You do not need a new idea for it.

## 3. The dot product

This is the calculation inside a linear model and inside one neuron.

Multiply matching positions, then add.

```text
x = [2500, 3, 2]          features
w = [0.05, 10, 8]         weights

w · x = (2500 × 0.05) + (3 × 10) + (2 × 8)
      = 125 + 30 + 16
      = 171
```

Add a bias, which is a single number:

```text
price_score = w · x + b
```

If `b = 20`, the score is `191`.

Read each term as “how much this feature pushes the answer.” Area contributes `125`. Bedrooms contribute `30`. The dot product is the sum of those pushes.

A logistic model uses the same score, then squashes it into a probability:

```text
z = w · x + b
probability = 1 / (1 + e^(-z))
```

`z` is the dot product. The S-shaped function only rescales it into a number between 0 and 1.

One neuron in a network is the same line:

```text
z = w · x + b
output = activation(z)
```

A layer with many neurons is many dot products, one per neuron.

### Direction, in one sentence

```text
w · x = (length of w) × (length of x) × cos(angle between them)
```

| Angle | Dot product | Plain meaning |
|---|---|---|
| same direction | large and positive | the features line up with the weights |
| right angle | 0 | no alignment |
| opposite direction | large and negative | the features push the score down |

You will use the number. The angle is the reason similar things score as similar.

## 4. Length and distance

The **length** of a vector (L2 norm) is the straight-line distance from the origin:

```text
v = [3, 4]
length(v) = sqrt(3² + 4²) = sqrt(9 + 16) = 5
```

The **distance** between two examples is the length of their difference:

```text
a = [1, 2]
b = [4, 6]
b - a = [3, 4]
distance = 5
```

k-nearest neighbours classifies a new point by this distance. A regression error is a distance between predicted numbers and true numbers. Ridge regression adds the squared length of the weight vector to the loss so the weights stay small.

## 5. Cosine similarity

Two texts about the same topic can have embeddings of different lengths. Cosine ignores length and keeps the angle.

```text
cosine(a, b) = (a · b) / (length(a) × length(b))
```

The result is between `-1` and `1`. Closer to `1` means “same direction.”

```text
a = [1, 0]
b = [2, 0]     same direction, different length
c = [0, 1]     a right angle from a

cosine(a, b) = 1
cosine(a, c) = 0
```

Search and RAG use this. A question and a useful note point a similar way in embedding space, so their cosine is high. A note about a different topic points another way.

Use cosine when you care about meaning or direction. Use straight-line distance when the actual sizes of the numbers matter, as in “which house is closest in square feet and rooms.”

## 6. A matrix–vector multiply is many dot products

One weight vector makes one score. A weight **matrix** makes several scores at once. Each column (or each row, depending on how the code is written) is the weight vector of one neuron.

Shape rule: the inner sizes must match.

```text
X is (2 houses, 3 features)
W is (3 features, 1 score)

X W is (2, 1)     one score per house
```

```text
X = [[2500, 3, 2],
     [1800, 2, 1]]

W = [[0.05],
     [10],
     [8]]

house A = 125 + 30 + 16 = 171
house B =  90 + 20 +  8 = 118
scores  = [171, 118]
```

A mini-batch of 32 examples is the same multiply with 32 rows. A GPU is fast because it does these multiplies together.

A later layer repeats the pattern: output of layer 1 becomes the vector for layer 2.

LoRA uses the same multiply on purpose. A large weight matrix stays frozen. The trainable change is a product of two thin matrices, so the update has a small shape and few new numbers. If you can read shapes, you can read a LoRA layer.

## 7. An embedding vector stands for one thing

A house vector `[2500, 3, 2]` has names you chose: area, bedrooms, bathrooms. An **embedding** is a vector the model learned. No single position is “bedrooms.” The whole list stands for a word, a sentence, a chunk of a document, or a photo.

```text
"dog"   → [0.82, 0.11, 0.40]     a toy 3-number embedding
"cat"   → [0.79, 0.14, 0.37]
"loan"  → [0.05, 0.88, 0.12]
```

Dog and cat point a similar way, so their cosine is high. Loan points another way. That is why search can match “home loan” with a note that only says “housing mortgage.” The words differ. The directions do not.

A real embedding is the same idea with a longer list. A sentence model often uses 384 or 768 numbers. OpenAI’s small text embedding uses 1536. You still compare them with cosine.

In RAG, each chunk of a document becomes one embedding and is stored. The question is embedded with the **same** model. You keep the chunks with the highest cosine. Two different embedding models do not share a space, so a cosine between them does not mean “similar.”

| What you embed | One vector stands for | A common length |
|---|---|---|
| a token | one word-piece, such as `bank` | 256–768 |
| a chunk | a paragraph in the knowledge base | 384–1536 |
| a question | the user’s query, at search time | same length as the chunks |
| a face or product photo | that image | 128–512 |

## 8. Embedding + position makes a new vector

A token embedding has the meaning and no place. “bank by the river” and “river by the bank” would be the same bag of vectors. The model adds a second vector, one per seat in the sentence, and that sum is a **new** vector.

One token, one seat. Toy length 3 so the addition is visible. A real model uses 768 numbers and the same steps.

```text
sentence:  bank   by   the   river

1. Token ids
   bank → 17     by → 4     the → 9     river → 52

2. Embedding lookup  (meaning, same vector wherever the word sits)
   17 → [0.20, 0.90, 0.10]

3. Position lookup   (place, different vector for each seat)
   seat 0 → [0.01, 0.00, 0.02]

4. Add matching positions → the new vector
   0.20 + 0.01 = 0.21
   0.90 + 0.00 = 0.90
   0.10 + 0.02 = 0.12

   new vector at seat 0 = [0.21, 0.90, 0.12]
```

```text
token embedding  +  position embedding  =  new input vector
[0.20, 0.90, 0.10] + [0.01, 0.00, 0.02] = [0.21, 0.90, 0.12]
     (3,)                  (3,)                 (3,)
```

The same word in another seat gets the same embedding and a different position vector, so the new vector changes.

```text
"bank" at seat 3

embedding   [0.20, 0.90, 0.10]
position 3  [0.40, 0.05, 0.00]
new vector  [0.60, 0.95, 0.10]
```

Do that once per token, then stack the new vectors. Attention reads the stack, not the two tables it was built from.

```text
bank  at 0   [0.20, 0.90, 0.10] + [0.01, 0.00, 0.02] = [0.21, 0.90, 0.12]
by    at 1   [0.10, 0.20, 0.05] + [0.02, 0.01, 0.00] = [0.12, 0.21, 0.05]
the   at 2   [0.05, 0.05, 0.70] + [0.00, 0.03, 0.01] = [0.05, 0.08, 0.71]
river at 3   [0.70, 0.10, 0.20] + [0.40, 0.05, 0.00] = [1.10, 0.15, 0.20]

input matrix shape: (4 tokens, 3 numbers)
real sentence:      (4, 768)
```

The position vectors are learned, or they are a fixed sine-and-cosine pattern. Either way the flow is lookup, lookup, add.

## 9. PCA, in one short step

A 768-number embedding cannot be drawn. **PCA** builds a shorter vector that keeps the directions where the points actually spread out.

Picture a cloud of dots, stretched along one diagonal. The long direction of the cloud is the first new axis. The next direction, at a right angle to the first, is the second axis. Directions where every point sits in almost the same place are dropped. Those numbers were not separating anything.

```text
one chunk embedding      (768,)
PCA down to a plot       (2,)

1,000 chunks             (1000, 768)
after PCA                (1000, 2)    one dot per chunk
```

Chunks about the same topic land near each other on that scatter plot. The plot is a picture of the search space. Retrieval still uses the full 768-number embedding. The 2-number vector is only for your eyes. PCA drops detail on purpose, so the short vector is the wrong thing to store in the vector database.

Each new axis is itself a vector: a direction in the original space. The first axis is the direction of greatest spread. You can use the plot without the proof.

## Where you will meet this

| Idea | Where it appears |
|---|---|
| Feature vector | one row of a table, one embedding, one image turned into numbers |
| Dot product `w · x + b` | linear regression, logistic regression, every dense neuron |
| Matrix multiply | a whole layer, a whole batch |
| Length / distance | kNN, error, weight penalty (Ridge) |
| Cosine similarity | word embeddings, semantic search, RAG |
| Embedding vector | a learned list for a token, a chunk, a question, or an image |
| Positional embedding | added to the token vector so the model knows the word’s place |
| PCA | a short vector for a plot of embeddings; search keeps the long one |
| Shape | every PyTorch or NumPy bug that says the sizes do not match |

Trees still read a feature vector. A tree asks “is feature 2 greater than 10?” instead of multiplying by a weight. The data is a vector either way.
