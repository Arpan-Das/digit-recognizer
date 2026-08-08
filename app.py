import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ── Model Definition ──────────────────────────────────────
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(28 * 28, 128)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ── Load Model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = DigitClassifier()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ── Preprocessing ─────────────────────────────────────────
def preprocess_canvas_image(image_data):
    image = Image.fromarray(image_data.astype("uint8"), "RGBA").convert("L")
    bbox  = image.getbbox()
    if bbox is None:
        return None
    digit = image.crop(bbox)
    digit.thumbnail((20, 20), Image.LANCZOS)
    canvas28 = Image.new("L", (28, 28), color=0)
    paste_x  = (28 - digit.width)  // 2
    paste_y  = (28 - digit.height) // 2
    canvas28.paste(digit, (paste_x, paste_y))
    return canvas28

# ── Page Config ───────────────────────────────────────────
st.set_page_config(page_title="Digit Recognizer", page_icon="🔢", layout="wide")
st.title("🔢 Handwritten Digit Recognizer")
st.markdown(
    "A fully connected neural network trained on **60,000 MNIST images** — "
    "achieving **97.78% test accuracy** in just 5 epochs on CPU."
)

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["✏️ Predict", "🤖 Model Info", "📈 Training Details"])

# ════════════════════════════════════════════════════════════
# TAB 1 — Predict
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Draw a digit (0–9) on the canvas")

    col_canvas, col_result = st.columns([1, 1])

    with col_canvas:
        canvas_result = st_canvas(
            fill_color   = "black",
            stroke_width = 20,
            stroke_color = "white",
            background_color = "black",
            height = 280,
            width  = 280,
            drawing_mode = "freedraw",
            key    = "canvas",
        )

        col_pred, col_clear = st.columns(2)

        predict_clicked = col_pred.button("🔍 Predict", use_container_width=True)
        clear_clicked   = col_clear.button("🗑️ Clear", use_container_width=True)

        if clear_clicked:
            st.rerun()

    with col_result:
        if predict_clicked:
            if canvas_result.image_data is not None and canvas_result.image_data.sum() > 0:
                image = preprocess_canvas_image(canvas_result.image_data)

                if image is None:
                    st.warning("Please draw a digit first.")
                else:
                    tensor = transform(image).unsqueeze(0)

                    with torch.no_grad():
                        outputs       = model(tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, dim=1)

                    pred_digit = predicted.item()
                    conf       = confidence.item() * 100
                    probs      = probabilities[0].numpy() * 100

                    # Result
                    st.markdown(f"## Prediction: **{pred_digit}**")

                    if conf >= 90:
                        st.success(f"✅ Confidence: {conf:.1f}% — Very confident")
                    elif conf >= 70:
                        st.warning(f"⚠️ Confidence: {conf:.1f}% — Fairly confident")
                    else:
                        st.error(f"❌ Confidence: {conf:.1f}% — Not sure")

                    st.divider()

                    # What model saw
                    st.subheader("👁️ What the model saw")
                    st.image(image, width=140,
                             caption="Your drawing → 28×28 grayscale")

                    st.divider()

                    # Probability bar chart for all 10 digits
                    st.subheader("📊 Confidence for each digit")
                    prob_df = pd.DataFrame({
                        'Digit'      : [str(i) for i in range(10)],
                        'Confidence' : probs
                    })

                    fig, ax = plt.subplots(figsize=(6, 3))
                    colors  = ['steelblue'] * 10
                    colors[pred_digit] = 'green'
                    bars = ax.bar(prob_df['Digit'], prob_df['Confidence'],
                                  color=colors, edgecolor='white')
                    ax.set_xlabel('Digit')
                    ax.set_ylabel('Confidence (%)')
                    ax.set_title('Model Confidence per Digit')
                    ax.set_ylim(0, 100)

                    for bar, val in zip(bars, probs):
                        if val > 2:
                            ax.text(bar.get_x() + bar.get_width()/2,
                                    bar.get_height() + 1,
                                    f'{val:.1f}%', ha='center', fontsize=7)
                    st.pyplot(fig)
            else:
                st.warning("Please draw a digit first.")
        else:
            st.info("👈 Draw a digit on the canvas and click **Predict**")

            # Show example digits
            st.divider()
            st.subheader("💡 Tips for best results")
            st.markdown("""
            - Draw the digit **large** — fill most of the canvas
            - Use **thick strokes** for clearer edges
            - Draw in the **center** of the canvas
            - The model was trained on simple handwritten digits —
              keep it clean and simple
            - Try digits 0–9 and see how confident the model is!
            """)

# ════════════════════════════════════════════════════════════
# TAB 2 — Model Info
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🤖 Model — Fully Connected Neural Network")

    st.markdown("""
    ### What is a Neural Network?
    A neural network is a mathematical system inspired by the human brain.
    It consists of layers of **neurons** — each neuron receives numbers,
    multiplies them by learned **weights**, sums them up, and passes the
    result to the next layer.

    The network learns by adjusting these weights thousands of times
    until predictions become accurate.

    ### Why Neural Networks for Digit Recognition?
    - ✅ Raw pixels in → prediction out (no manual feature engineering)
    - ✅ Learns complex patterns automatically from examples
    - ✅ Generalizes across different handwriting styles
    - ✅ Fast inference — prediction in milliseconds
    """)

    st.divider()

    # Architecture diagram
    st.subheader("🏗️ Network Architecture")

    st.markdown("""
    ```
    Input Image (28×28 pixels)
           ↓
      Flatten Layer
      (784 neurons)
           ↓
    Hidden Layer — fc1
    (128 neurons + ReLU activation)
           ↓
    Output Layer — fc2
    (10 neurons → one per digit)
           ↓
    Softmax → Probabilities
    ```
    """)

    arch_data = {
        'Layer'       : ['Input', 'Flatten', 'Hidden (fc1)', 'ReLU', 'Output (fc2)', 'Softmax'],
        'Shape'       : ['[1, 28, 28]', '[784]', '[128]', '[128]', '[10]', '[10]'],
        'Parameters'  : ['-', '-', '784×128 + 128 = 100,480', '-', '128×10 + 10 = 1,290', '-'],
        'Purpose'     : [
            'Raw grayscale pixel values',
            'Convert 2D image to 1D vector',
            'Learn abstract features from pixels',
            'Add non-linearity (max(0, x))',
            'One score per digit class',
            'Convert scores to probabilities'
        ]
    }
    st.dataframe(pd.DataFrame(arch_data), hide_index=True)

    st.caption("**Total learnable parameters: 101,770** — all updated during training")

    st.divider()

    # Performance
    st.subheader("📈 Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("Test Accuracy",  "97.78%", "on 10,000 unseen images")
    col2.metric("Training Time",  "~2 min", "CPU only")
    col3.metric("Parameters",     "101,770", "learnable weights")

    st.markdown("""
    **What 97.78% means:**
    The model correctly identifies **9,778 out of 10,000** handwritten digit images
    it has never seen before. Only 222 mistakes across all 10 digit classes.

    **Limitation:** This is a simple fully connected network. A CNN (Convolutional
    Neural Network) would push this above **99%** by learning spatial patterns
    like edges and curves — which is exactly what Project 5 covers.
    """)

    st.divider()

    # Real world applications
    st.subheader("🌍 Real World Applications")

    apps = {
        'Industry'       : ['Banking', 'Postal Services', 'Healthcare',
                            'Transportation', 'Mobile/Consumer', 'Education'],
        'Application'    : [
            'Cheque processing — read handwritten amounts automatically',
            'ZIP code recognition on envelopes for sorting',
            'Digitize handwritten doctor prescriptions and forms',
            'License plate recognition, toll booth automation',
            'Handwriting-to-text keyboard input, document scanning',
            'Auto-grade handwritten exam answer sheets'
        ],
        'Scale'          : ['Millions/day', 'Millions/day', 'Hospitals globally',
                           '24/7 automated', 'Billions of users', 'Universities']
    }
    st.dataframe(pd.DataFrame(apps), hide_index=True)

    st.divider()

    # Key concepts
    st.subheader("🧠 Key Concepts")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **ReLU Activation**
        ```
        ReLU(x) = max(0, x)

        Negative → 0
        Positive → unchanged
        ```
        Adds non-linearity so the network
        can learn complex patterns.
        Without it, any number of layers
        is mathematically just one layer.

        **Softmax Output**
        Converts raw scores to probabilities
        that sum to 1.0:
        ```
        [2.1, -0.5, 1.3, ...]
               ↓ softmax
        [0.24, 0.02, 0.13, ...]
        ```
        """)

    with col2:
        st.markdown("""
        **state_dict — How Model is Saved**

        Only the learned weights are saved
        (not the full model object):
        ```python
        torch.save(model.state_dict(),
                   'model.pth')
        ```
        Portable, version-independent,
        industry standard approach.

        **torch.no_grad() — Inference Mode**

        Disables gradient recording during
        prediction — saves memory and runs
        faster since backprop isn't needed.
        ```python
        with torch.no_grad():
            outputs = model(tensor)
        ```
        """)

# ════════════════════════════════════════════════════════════
# TAB 3 — Training Details
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📈 How the Model Was Trained")

    # Training setup
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Dataset**
        - 60,000 training images
        - 10,000 test images
        - 28×28 grayscale (784 pixels)
        - 10 classes (digits 0–9)
        - Balanced — ~6,000 per class
        """)

    with col2:
        st.markdown("""
        **Training Setup**
        - Loss: CrossEntropyLoss
        - Optimizer: Adam (lr=0.001)
        - Batch size: 64
        - Epochs: 5
        - Hardware: CPU
        """)

    st.divider()

    # Loss curve
    st.subheader("📉 Training Loss Curve")

    epochs = [1, 2, 3, 4, 5]
    losses = [0.2518, 0.1093, 0.0775, 0.0584, 0.0465]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, losses, marker='o', color='steelblue',
            linewidth=2, markersize=8, label='Training Loss')
    ax.fill_between(epochs, losses, alpha=0.1, color='steelblue')

    for x, y in zip(epochs, losses):
        ax.annotate(f'{y}', (x, y),
                    textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Loss')
    ax.set_title('Training Loss over 5 Epochs')
    ax.set_xticks(epochs)
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.markdown("""
    **Reading the loss curve:**
    - **Epoch 1→2:** Biggest drop — model learns most patterns quickly
    - **Epoch 2→3:** Still significant learning
    - **Epoch 3→5:** Diminishing returns — model has converged
    - **Healthy curve:** drops fast early, flattens later ✅
    """)

    st.divider()

    # Training loop explanation
    st.subheader("🔄 The Training Loop — 3 Sacred Lines")

    st.code("""
for epoch in range(5):
    for images, labels in train_loader:    # 938 batches of 64 images

        # Forward pass — make predictions
        outputs = model(images)
        loss    = criterion(outputs, labels)

        # Backward pass — learn from mistakes
        optimizer.zero_grad()   # 1. clear old gradients
        loss.backward()         # 2. calculate new gradients (backprop)
        optimizer.step()        # 3. update weights
    """, language='python')

    st.markdown("""
    **Why this exact order?**

    | Line | What it does | Why it's needed |
    |---|---|---|
    | `zero_grad()` | Clears gradients from last batch | PyTorch accumulates gradients — must reset each batch |
    | `backward()` | Calculates how to improve each weight | Backpropagation through the computation graph |
    | `step()` | Moves each weight in the right direction | Applies the gradient update: `w = w - lr × gradient` |
    """)

    st.divider()

    # Numbers
    st.subheader("🔢 Training by the Numbers")

    numbers_data = {
        'Metric'     : ['Images per batch', 'Batches per epoch', 'Total epochs',
                        'Weight updates', 'Parameters updated each step', 'Final accuracy'],
        'Value'      : ['64', '938', '5', '4,690', '101,770', '97.78%'],
        'Meaning'    : [
            'Processed together in one forward pass',
            '60,000 ÷ 64 = 937.5 → 938',
            'Complete passes through training data',
            '938 batches × 5 epochs',
            'Every weight touched every update',
            '9,778 / 10,000 test images correct'
        ]
    }
    st.dataframe(pd.DataFrame(numbers_data), hide_index=True)

    st.divider()

    # Sample training output
    st.subheader("💻 Sample Training Output")
    st.code("""
Epoch 1/5 | Batch   0/938 | Loss: 2.2837  ← random weights, high loss
Epoch 1/5 | Batch 200/938 | Loss: 0.2121  ← learning fast
Epoch 1/5 | Batch 400/938 | Loss: 0.0842
Epoch 1/5 | Batch 800/938 | Loss: 0.2086
Epoch 1 complete. Average loss: 0.2518

Epoch 2/5 | Batch   0/938 | Loss: 0.0269  ← already much lower
Epoch 2/5 | Batch 200/938 | Loss: 0.1857
Epoch 2 complete. Average loss: 0.1093

Epoch 5/5 | Batch   0/938 | Loss: 0.0132  ← converged
Epoch 5/5 | Batch 800/938 | Loss: 0.0358
Epoch 5 complete. Average loss: 0.0465

Test Accuracy: 97.78%
""", language='text')
