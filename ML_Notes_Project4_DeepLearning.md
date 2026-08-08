# 📚 ML Notes — Project 4: Digit Recognizer (Deep Learning)
**By Arpan | TCS System Engineer**

First deep learning project — moving from classical ML to neural networks with PyTorch.

---

## Table of Contents
1. [Classical ML vs Deep Learning](#1-classical-ml-vs-deep-learning)
2. [What is a Neural Network?](#2-what-is-a-neural-network)
3. [The MNIST Dataset](#3-the-mnist-dataset)
4. [Data Loading in PyTorch](#4-data-loading-in-pytorch)
5. [Building the Network](#5-building-the-network)
6. [Activation Functions](#6-activation-functions)
7. [Loss Functions](#7-loss-functions)
8. [Optimizers](#8-optimizers)
9. [The Training Loop](#9-the-training-loop)
10. [Saving and Loading Models](#10-saving-and-loading-models)
11. [Inference — torch.no_grad()](#11-inference--torchno_grad)
12. [Real World Applications](#12-real-world-applications)
13. [Key Takeaways](#13-key-takeaways)

---

## 1. Classical ML vs Deep Learning

### The Fundamental Shift

**Classical ML (Phase 1):**
```
Raw Data → YOU engineer features → Model learns from features

Example — Spam Classifier:
  Raw email → YOU extract word counts (TF-IDF) → Naive Bayes classifies
```

**Deep Learning (Phase 2):**
```
Raw Data → Model learns features AND classification automatically

Example — Digit Recognizer:
  Raw pixels → Neural network learns edges, curves, shapes → Predicts digit
```

### Why Deep Learning?

| Aspect | Classical ML | Deep Learning |
|---|---|---|
| Feature engineering | Manual | Automatic |
| Data needed | Small-medium | Large |
| Performance on images | Poor | Excellent |
| Interpretability | High | Low |
| Training time | Seconds | Minutes-hours |
| Hardware | CPU fine | GPU preferred |

### When to Use Which

```
Tabular data (numbers, categories) → Classical ML (faster, interpretable)
Images, audio, text sequences      → Deep Learning (far more powerful)
```

---

## 2. What is a Neural Network?

### Biological Inspiration
Your brain has 86 billion neurons. Each neuron receives signals, processes them, and fires or doesn't fire. Neural networks are a mathematical imitation:

```
Biological Neuron:
  Receive signals → Process → Fire or don't fire

Artificial Neuron:
  Receive numbers → Multiply by weights → Sum → Activation → Output
```

### One Artificial Neuron

```
Inputs:    x1=0.5   x2=0.3   x3=0.8
Weights:   w1=0.4   w2=0.7   w3=0.2
Bias:      b=0.1

Step 1 — Weighted sum:
  z = (x1×w1) + (x2×w2) + (x3×w3) + b
  z = (0.5×0.4) + (0.3×0.7) + (0.8×0.2) + 0.1
  z = 0.2 + 0.21 + 0.16 + 0.1 = 0.67

Step 2 — Activation:
  output = ReLU(0.67) = 0.67
```

This is exactly what Linear Regression does in Step 1. The **activation function** is what makes neural networks different — it adds non-linearity.

### A Full Neural Network — Layer by Layer

```
INPUT LAYER     HIDDEN LAYER    OUTPUT LAYER
(784 neurons)   (128 neurons)   (10 neurons)

  ●                ●               ● → P(digit=0)
  ●                ●               ● → P(digit=1)
  ●       →        ●      →        ● → P(digit=2)
  ●                ●               ● → ...
  ●                ●               ● → P(digit=9)
 ...              ...

Every ● = one neuron
Every → = one weight (learnable number)
```

**Data flows left to right (forward pass)**
**Gradients flow right to left (backward pass / backprop)**

### Why Layers?

Each layer learns increasingly abstract features:
```
Layer 1 (close to input):  learns edges and pixel patterns
Layer 2 (middle):          learns curves and shapes
Layer 3 (close to output): learns complete digit structures
```

This hierarchical feature learning is what makes neural networks so powerful.

---

## 3. The MNIST Dataset

### What It Is
60,000 training + 10,000 test images of handwritten digits (0-9).
The "Hello World" of deep learning — used since 1998.

### Image Format
```
Each image:
- 28 × 28 pixels
- Grayscale (1 channel) — not RGB
- Each pixel = 0 (black) to 255 (white)
- Total = 784 numbers per image

Tensor shape: [1, 28, 28]
              ↑   ↑   ↑
           channel h   w
```

### After Normalization
Raw pixels (0-255) are normalized to roughly (-1, +1):
```
transform = transforms.Compose([
    transforms.ToTensor(),                     # 0-255 → 0.0-1.0
    transforms.Normalize((0.1307,), (0.3081,)) # subtract mean, divide by std
])

mean=0.1307 and std=0.3081 are the pre-calculated
statistics of the entire MNIST training set.
```

Why normalize?
- Neural networks train faster with zero-centered inputs
- Prevents any single pixel from dominating due to scale

### The MNIST Convention for Handwritten Input
MNIST digits are NOT just any 28×28 image. They follow a specific format:
```
1. Digit occupies roughly 20×20 pixels
2. Digit is centered in a 28×28 black frame
3. Background is black (0), digit is white (255)
```

This is why preprocessing canvas drawings correctly matters — if you don't match this convention, the model sees something it was never trained on.

---

## 4. Data Loading in PyTorch

### Dataset vs DataLoader

**Dataset** — holds all the data, handles transforms
```python
train_dataset = torchvision.datasets.MNIST(
    root='./data',      # where to save
    train=True,         # training set
    download=True,      # download if not present
    transform=transform # apply normalization
)
```

**DataLoader** — batches the data, shuffles, handles parallelism
```python
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,   # 64 images per batch
    shuffle=True     # shuffle order each epoch
)
```

### Why Batches?

```
Option 1 — Process all 60,000 images at once:
  → Needs huge memory
  → Slow to compute
  → One weight update per epoch

Option 2 — Process 1 image at a time:
  → Very noisy gradient estimates
  → Very slow (60,000 updates per epoch)

Option 3 — Batches of 64 (sweet spot ✅):
  → Memory efficient
  → Stable gradient estimates
  → 938 weight updates per epoch
```

### Math
```
60,000 images ÷ 64 per batch = 937.5 → 938 batches
938 batches × 5 epochs = 4,690 total weight updates
```

### Why shuffle=True for training?

Without shuffling, the model sees all 0s first, then all 1s, etc.
It forgets earlier classes while learning new ones (catastrophic forgetting).
Shuffling ensures each batch has a mix of all digit classes.

---

## 5. Building the Network

### PyTorch Model Structure

Every PyTorch model:
1. Inherits from `nn.Module`
2. Defines layers in `__init__`
3. Defines forward pass in `forward`

```python
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()          # 28×28 → 784
        self.fc1     = nn.Linear(784, 128)   # hidden layer
        self.relu    = nn.ReLU()             # activation
        self.fc2     = nn.Linear(128, 10)    # output layer

    def forward(self, x):
        x = self.flatten(x)   # [batch, 1, 28, 28] → [batch, 784]
        x = self.fc1(x)       # [batch, 784] → [batch, 128]
        x = self.relu(x)      # [batch, 128] → [batch, 128] (same shape)
        x = self.fc2(x)       # [batch, 128] → [batch, 10]
        return x              # raw scores (logits) for each of 10 classes
```

### nn.Linear — What It Does

`nn.Linear(in_features, out_features)` creates a fully connected layer:
```
fc1 = nn.Linear(784, 128)

Internally stores:
  weight matrix: shape [128, 784]  ← 100,352 learnable numbers
  bias vector:   shape [128]       ← 128 learnable numbers

Forward computation:
  output = input @ weight.T + bias
  [batch, 784] @ [784, 128] + [128] = [batch, 128]
```

### Total Parameters
```
fc1: 784 × 128 + 128 = 100,480
fc2: 128 × 10  + 10  = 1,290
Total: 101,770 learnable parameters
```

Each of these 101,770 numbers gets updated 4,690 times during training.

### nn.Flatten — Image to Vector
```
Input:  [64, 1, 28, 28]   (batch of 64 images)
Output: [64, 784]          (batch of 64 flat vectors)

Removes channel dimension + flattens spatial dimensions:
[[0, 0, 128, 255, ...],    →    [0, 0, 128, 255, ..., 0, 64, 200]
 [0, 64, 200, ...  ],                    784 numbers
 ...]
```

---

## 6. Activation Functions

### Why Activation Functions?

Without activation functions, stacking multiple linear layers is mathematically identical to a single linear layer:

```
Two linear layers without activation:
  y = W2(W1x + b1) + b2
  y = (W2·W1)x + (W2·b1 + b2)
  y = Wx + b    ← still just a linear function!

No matter how many layers you add — without activation,
it's still just linear regression. Useless for complex patterns.
```

Activation functions add **non-linearity** — allowing the network to learn complex curved decision boundaries.

### ReLU — Rectified Linear Unit

```
ReLU(x) = max(0, x)

x = -2.5 → output = 0    (negative → zero)
x =  0.0 → output = 0
x =  0.3 → output = 0.3  (positive → unchanged)
x =  1.7 → output = 1.7
```

Graph:
```
output
  |        /
  |       /
  |      /
  |_____/_________ input
  0
```

**Why ReLU is the default choice:**
- Simple and fast to compute
- No vanishing gradient problem (unlike Sigmoid/Tanh)
- Works well in practice for most deep networks

### Softmax — Output Layer

ReLU is used in hidden layers. The output layer uses **Softmax** to convert raw scores (logits) into probabilities:

```
Raw output (logits): [-1.2,  0.3,  2.1, -0.5,  0.1,
                      -0.8,  0.4, -0.2,  1.3,  0.0]
                        0     1    2     3     4
                        5     6    7     8     9

After Softmax:       [0.01, 0.02, 0.24, 0.01, 0.02,
                      0.01, 0.02, 0.01, 0.13, 0.02]
                      Sum = 1.0  ← always sums to 1 (probabilities)

Prediction: digit 2 (highest probability: 24%)
```

**Important:** In PyTorch, `CrossEntropyLoss` applies Softmax internally.
So during training you don't apply Softmax yourself.
During inference (prediction), you apply it manually with `torch.softmax()`.

---

## 7. Loss Functions

### CrossEntropyLoss — For Classification

Measures how wrong the model's probability distribution is:

```
Formula: Loss = -log(probability assigned to correct class)

Model predicts digit "7" with these probabilities:
  P(digit=7) = 0.92  → Loss = -log(0.92) = 0.083  (low loss, good prediction)
  P(digit=7) = 0.02  → Loss = -log(0.02) = 3.912  (high loss, bad prediction)
```

Key properties:
- Loss = 0 when prediction is perfect
- Loss → ∞ when model is completely wrong
- Heavily penalizes confident wrong predictions

### Why Not MSE for Classification?

MSE treats all wrong answers equally:
```
Predicting "3" when answer is "7":
  MSE sees: (3-7)² = 16  (just a number difference)

CrossEntropy sees: probability assigned to class 7
  If P(7)=0.01 → very confident wrong answer → high penalty
  If P(7)=0.45 → uncertain → lower penalty
```

CrossEntropy better reflects the nature of classification problems.

---

## 8. Optimizers

### What is an Optimizer?

After backprop calculates gradients (direction to improve), the optimizer decides **how much to change each weight**.

```
gradient = direction of steepest increase in loss
weight update = move in OPPOSITE direction (decrease loss)

new_weight = old_weight - learning_rate × gradient
```

### Adam — Adaptive Moment Estimation

The most popular optimizer. Improves on basic gradient descent by:

1. **Momentum** — considers past gradients, not just current
   ```
   Like a ball rolling downhill — builds up speed in consistent directions
   ```

2. **Adaptive learning rates** — different lr for each parameter
   ```
   Parameters with big gradients → smaller steps
   Parameters with small gradients → larger steps
   ```

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

**`lr=0.001`** — learning rate controls step size:
```
lr too high (0.1):  overshoots minimum → loss bounces, never converges
lr too low (0.00001): takes forever, may get stuck
lr=0.001:           Adam's sweet spot for most problems
```

### Visualizing Gradient Descent
```
Loss
 │
 │    ●                    ← starting point (random weights)
 │      ●
 │        ●
 │          ●
 │            ●●●●●        ← converged (minimum loss)
 └─────────────────────── Weight value

Each ● = one optimizer.step()
```

---

## 9. The Training Loop

### The 3 Sacred Lines

Every neural network training loop in existence has these 3 lines:

```python
optimizer.zero_grad()   # Step 1 — clear old gradients
loss.backward()         # Step 2 — calculate new gradients
optimizer.step()        # Step 3 — update weights
```

### Why `zero_grad()` First?

PyTorch **accumulates** gradients by default:
```
Batch 1 gradient for weight[0]: +0.3
Batch 2 gradient for weight[0]: +0.2

Without zero_grad:  gradient used = 0.3 + 0.2 = 0.5  ← wrong!
With zero_grad:     gradient used = 0.2              ← correct
```

If you forget `zero_grad()`, gradients from all previous batches
pile up and weight updates become completely wrong.

### Full Training Loop Explained

```python
for epoch in range(num_epochs):
    model.train()              # training mode (affects dropout, batchnorm)
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):

        # ── Forward pass ──────────────────────────────
        outputs = model(images)           # predict
        loss    = criterion(outputs, labels)  # measure error

        # ── Backward pass ─────────────────────────────
        optimizer.zero_grad()   # clear gradients from last batch
        loss.backward()         # calculate gradients (backprop)
        optimizer.step()        # update weights

        # ── Tracking ──────────────────────────────────
        running_loss += loss.item()   # .item() → Python float

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")
```

### What `loss.backward()` Actually Does

During forward pass, PyTorch secretly builds a computation graph:
```
images → flatten → fc1 → relu → fc2 → output → loss
  ↑         ↑       ↑      ↑      ↑       ↑       ↑
  PyTorch records every operation here
```

`loss.backward()` walks this graph **backwards** using the chain rule:
```
∂loss/∂fc2.weight = ?
∂loss/∂fc1.weight = ?
∂loss/∂fc1.bias   = ?
... for every learnable parameter
```

These derivatives (gradients) tell each weight:
> "Increase by this much and the loss goes up/down by that much"

### One Epoch — The Numbers

```
60,000 training images
÷ 64 per batch
= 938 batches

Per batch:
  1 forward pass
  1 loss calculation
  1 backward pass
  1 weight update

Per epoch: 938 weight updates
× 5 epochs = 4,690 total updates to 101,770 parameters
```

### Our Loss Curve

```
Epoch 1: 0.2518  ← large random errors, steep learning
Epoch 2: 0.1093  ← majority of patterns learned
Epoch 3: 0.0775  ← refinement
Epoch 4: 0.0584  ← diminishing returns
Epoch 5: 0.0465  ← converged

Healthy curve: drops fast early, flattens later
Unhealthy: stays flat (lr too low), bounces (lr too high),
           goes up after going down (overfitting)
```

---

## 10. Saving and Loading Models

### Two Approaches

**Approach 1 — Save weights only (state_dict) ✅ recommended**
```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model = DigitClassifier()                    # rebuild architecture
model.load_state_dict(torch.load('model.pth'))  # pour weights in
model.eval()                                 # set to inference mode
```

**Approach 2 — Save full model**
```python
# Save
torch.save(model, 'model.pth')

# Load
model = torch.load('model.pth')
```

### Why state_dict() is Better

| | state_dict() | Full Model |
|---|---|---|
| File size | Small | Large |
| PyTorch version dependency | None | High |
| Architecture flexibility | ✅ Can modify | ❌ Locked |
| Industry standard | ✅ Yes | ❌ No |

`state_dict()` is just a Python dictionary of tensors:
```python
{
  'fc1.weight': tensor([[...], [...]]),  # shape [128, 784]
  'fc1.bias':   tensor([...]),           # shape [128]
  'fc2.weight': tensor([[...], [...]]),  # shape [10, 128]
  'fc2.bias':   tensor([...])            # shape [10]
}
```

Clean, portable, version-independent.

### Compared to joblib (Phase 1)
```
sklearn:  joblib.dump(model, 'model.pkl')  ← saves full object
PyTorch:  torch.save(model.state_dict())   ← saves weights only

sklearn models are simple Python objects — no separation needed
PyTorch models are complex — separation matters for portability
```

---

## 11. Inference — torch.no_grad()

### What Happens During Forward Pass

PyTorch secretly records every operation in a computation graph
(needed for backpropagation during training):

```
input → flatten → fc1 → relu → fc2 → output
  ↑         ↑       ↑      ↑      ↑      ↑
  [PyTorch records all of this in memory]
```

### The Problem During Prediction

During inference you don't need:
- Backpropagation
- Gradient calculation
- Weight updates

But PyTorch still records everything by default — wasting memory and time.

### The Solution

```python
with torch.no_grad():
    outputs       = model(tensor)             # no recording
    probabilities = torch.softmax(outputs, 1) # no recording
    pred          = torch.max(probabilities)   # no recording
```

`torch.no_grad()` tells PyTorch:
> "I'm not calling .backward() — don't build the computation graph"

### Impact
```
With gradients:    ~2× memory, ~20% slower
Without gradients: lean, fast, production-ready

For MNIST: barely noticeable
For large production models serving 1000s of users: critical
```

### model.eval() vs torch.no_grad()

These are different things — both needed during inference:

```python
model.eval()          # changes behavior of dropout, batchnorm layers
torch.no_grad()       # disables gradient computation

# Always use both during inference:
model.eval()
with torch.no_grad():
    outputs = model(input)
```

---

## 12. Real World Applications

### Where Digit/Character Recognition Is Used

**Banking & Finance**
- Cheque processing — read handwritten amounts automatically
- Tax form digitization — auto-read handwritten numbers
- Signature verification systems

**Postal & Logistics**
- ZIP code recognition on envelopes
- Handwritten address parsing
- Package sorting automation

**Healthcare**
- Digitize handwritten doctor prescriptions
- Convert paper medical records to digital
- Read handwritten lab result forms

**Education**
- Auto-grade handwritten exam answers
- Digitize student submission forms
- Accessibility tools for dyslexic students

**Transportation**
- License plate recognition
- Toll booth automation
- Speed camera systems

**Mobile & Consumer**
- Handwriting-to-text keyboard input
- Document scanning apps
- Signature capture pads

### The Broader Impact — What This Architecture Powers

The fully connected network we built is the **foundation** of all modern AI:

```
Our network:  image → flatten → FC layers → digit
                                   ↑
                          This concept scales to:

GPT/ChatGPT:  tokens → embeddings → many FC + attention layers → text
DALL-E:       text   → embeddings → many layers → image
AlphaFold:    amino acids → many layers → protein structure
Tesla FSD:    camera frames → many layers → steering decisions
```

The math is the same — just more layers, more parameters, better architectures.

---

## 13. Key Takeaways

### Concepts to Remember Forever

**1. Deep learning learns features automatically**
No TF-IDF, no manual feature engineering. Raw pixels in, prediction out.

**2. Neural networks are stacked linear + non-linear transformations**
Without activation functions, any depth is equivalent to one linear layer.

**3. ReLU is the default activation — use it unless you have reason not to**
Simple, fast, no vanishing gradient problem.

**4. CrossEntropyLoss for classification, MSE for regression**
CrossEntropy penalizes confident wrong predictions more severely.

**5. The 3 sacred training lines — memorize this order:**
```python
optimizer.zero_grad()  # always first
loss.backward()        # always second
optimizer.step()       # always third
```

**6. Forget zero_grad() → gradients accumulate → wrong weight updates**
This is the most common beginner mistake in PyTorch.

**7. Save state_dict(), not the full model**
Portable, version-independent, industry standard.

**8. torch.no_grad() during inference — always**
Saves memory and compute. No gradients needed when not training.

**9. Batch size balances memory and gradient quality**
Too small = noisy. Too large = memory issues. 32-256 is usually right.

**10. Watching the loss curve is how you debug training**
Flat → lr too low. Bouncing → lr too high. Going up → overfitting.

---

## 📈 What's Next — Project 5: Image Classifier (CIFAR-10)

The step up from MNIST:

```
MNIST                         CIFAR-10
─────                         ────────
28×28 pixels                  32×32×3 pixels (color!)
1 class type (digits)         10 classes (cars, birds, planes...)
Simple shapes                 Complex real-world objects
FC network → 97.78%           FC network → ~50% (too hard!)
                              CNN → 90%+  ← what we'll build
```

**New concepts coming:**
- Convolutional layers — scan for local patterns (edges, textures)
- Pooling layers — reduce spatial dimensions
- Dropout — randomly disable neurons to prevent overfitting
- Data augmentation — artificially expand dataset
- Google Colab — free GPU for faster training

---

*Notes compiled during hands-on project building — every concept here was learned by doing, not just reading.*
