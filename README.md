# 🔢 Digit Recognizer — Neural Network on MNIST

A deep learning web application that recognizes handwritten digits (0-9) using a fully connected neural network trained on the MNIST dataset. Draw any digit on the canvas and get an instant prediction with confidence scores.

🔗 **[Live Demo](https://your-app-url.streamlit.app)** ← replace with your URL

---

## 📌 Project Overview

This is the first deep learning project — moving from classical ML (scikit-learn) to neural networks (PyTorch). A simple 3-layer fully connected network is trained from scratch on 60,000 handwritten digit images and achieves **97.78% test accuracy** in just 5 epochs.

---

## 🏗️ Model Architecture

```
Input Image (28×28 pixels)
        ↓
   Flatten Layer
   (784 neurons)
        ↓
  Hidden Layer (fc1)
   (128 neurons + ReLU)
        ↓
  Output Layer (fc2)
   (10 neurons → digits 0-9)
        ↓
   Softmax → Probabilities
```

**Total parameters:** 101,770 weights and biases

---

## 📊 Training Results

| Metric | Value |
|---|---|
| Test Accuracy | **97.78%** |
| Final Training Loss | 0.0465 |
| Epochs | 5 |
| Training Time (CPU) | ~2 minutes |

### Loss Progression

| Epoch | Avg Loss | Meaning |
|---|---|---|
| 1 | 0.2518 | Learning fast — big weight updates |
| 2 | 0.1093 | Slowing down — patterns learned |
| 3 | 0.0775 | Refining fine details |
| 4 | 0.0584 | Diminishing returns |
| 5 | 0.0465 | Converged — ready to stop |

---

## ⚙️ Training Setup

| Parameter | Value | Why |
|---|---|---|
| Loss Function | CrossEntropyLoss | Best for multi-class classification |
| Optimizer | Adam (lr=0.001) | Adaptive, fast convergence |
| Batch Size | 64 | Memory efficient, stable gradients |
| Epochs | 5 | Sufficient for MNIST convergence |
| Normalization | mean=0.1307, std=0.3081 | MNIST dataset standard |

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.x |
| Deep Learning | PyTorch |
| Data | torchvision (MNIST) |
| Web App | Streamlit |
| Canvas | streamlit-drawable-canvas |
| Deployment | Streamlit Cloud |

---

## 🗂️ Project Structure

```
digit-recognizer/
│
├── app.py              # Streamlit web application
├── train.ipynb         # Training notebook
├── model.pth           # Saved model weights
├── requirements.txt    # Dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/digit-recognizer.git
cd digit-recognizer

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

**requirements.txt:**
```
torch
torchvision
streamlit
streamlit-drawable-canvas
matplotlib
pandas
pillow
```

---

## 📈 App Features

**Tab 1 — Predict**
- Drawable canvas (280×280 black background)
- Preprocesses drawing to match MNIST format (crop → 20×20 → center in 28×28)
- Shows prediction + confidence score
- Probability bar chart for all 10 digits
- Preview of what the model actually sees

**Tab 2 — Model Info**
- Network architecture explanation
- Training setup details
- Performance metrics
- How softmax converts raw scores to probabilities

**Tab 3 — Training Details**
- Loss curve chart
- Epoch-by-epoch loss breakdown
- Sample training output logs
- Explanation of what dropping loss means

---

## 💡 What I Learned

- How neural networks work — neurons, weights, layers
- Forward pass — data flows input → hidden → output
- Backpropagation — gradients flow output → input to update weights
- The 3 sacred training lines: `zero_grad` → `backward` → `step`
- Why `optimizer.zero_grad()` is needed every batch
- CrossEntropyLoss — combines softmax + log + negative loss
- Adam optimizer — adaptive learning rate per parameter
- `state_dict()` vs full model save — and why weights-only is better
- `torch.no_grad()` — disables gradient recording during inference
- MNIST preprocessing convention — 20×20 digit centered in 28×28 frame
- Batch training — 938 batches per epoch, 4,690 total weight updates

---

## 🔮 Future Improvements

- [ ] Add CNN layers — should push accuracy to 99%+
- [ ] Add per-digit accuracy breakdown
- [ ] Show misclassified examples from test set
- [ ] Add data augmentation (rotation, shift) for robustness
- [ ] Deploy model to Azure/AWS instead of reloading from file

---

## 👤 Author

**Arpan** 
