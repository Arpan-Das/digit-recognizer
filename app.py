import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# ----------------------------------------------------------------
# 1. Model definition — MUST match the architecture used in training
# ----------------------------------------------------------------
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ----------------------------------------------------------------
# 2. Load the trained model once (cached across Streamlit re-runs)
# ----------------------------------------------------------------
@st.cache_resource
def load_model():
    model = DigitClassifier()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Same transform used during training — must match exactly
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ----------------------------------------------------------------
# 3. Preprocessing — matches MNIST's own convention:
#    crop to the digit's bounding box, resize to ~20x20 preserving
#    aspect ratio, then center it in a blank 28x28 black frame.
# ----------------------------------------------------------------
def preprocess_canvas_image(image_data):
    image = Image.fromarray(image_data.astype("uint8"), "RGBA").convert("L")

    bbox = image.getbbox()
    if bbox is None:
        return None  # nothing drawn yet

    digit = image.crop(bbox)
    digit.thumbnail((20, 20), Image.LANCZOS)

    canvas28 = Image.new("L", (28, 28), color=0)
    paste_x = (28 - digit.width) // 2
    paste_y = (28 - digit.height) // 2
    canvas28.paste(digit, (paste_x, paste_y))

    return canvas28

# ----------------------------------------------------------------
# 4. Streamlit UI — Tabbed layout for prediction, information, and training
# ----------------------------------------------------------------
st.set_page_config(page_title="Digit Recognizer", layout="wide")
st.title("Digit Recognizer")
st.markdown(
    "This app uses a simple fully connected neural network trained on MNIST digits. "
    "Use the tabs below to predict a digit, view the model architecture, and review the training details."
)

predict_tab, model_tab, training_tab = st.tabs(["Predict", "Model Info", "Training Details"])

with predict_tab:
    st.header("Draw and Predict")
    st.write(
        "Draw a digit from 0 to 9 on the canvas below. The app crops your drawing, centers it in a 28x28 frame, "
        "normalizes it, then predicts the digit using the saved PyTorch model."
    )

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Predict"):
        if canvas_result.image_data is not None and canvas_result.image_data.sum() > 0:
            image = preprocess_canvas_image(canvas_result.image_data)

            if image is None:
                st.warning("Please draw a digit first.")
            else:
                tensor = transform(image).unsqueeze(0)  # shape: (1, 1, 28, 28)

                with torch.no_grad():
                    outputs = model(tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, dim=1)

                st.success(f"Prediction: {predicted.item()}")
                st.write(f"Confidence: {confidence.item() * 100:.2f}%")
                st.image(image, caption="What the model saw (28x28)", width=140)
        else:
            st.warning("Please draw a digit first.")

with model_tab:
    st.header("Model Architecture")
    st.markdown(
        "- Input: 28x28 grayscale image from MNIST or the drawing canvas.\n"
        "- Flatten layer: converts the 28x28 image to 784 features.\n"
        "- Hidden layer: fully connected (Linear) layer with 128 neurons.\n"
        "- Activation: ReLU adds non-linearity to the network.\n"
        "- Output layer: fully connected layer with 10 neurons, one for each digit class (0-9)."
    )

    st.subheader("Training setup")
    st.markdown(
        "- Loss function: CrossEntropyLoss (good for multi-class classification).\n"
        "- Optimizer: Adam with learning rate 0.001.\n"
        "- Epochs: 5.\n"
        "- Batch size: 64 during training.\n"
        "- Data normalization: mean=0.1307, std=0.3081 (MNIST standard)."
    )

    st.subheader("Performance")
    st.markdown(
        "- Test Accuracy: **97.78%**.\n"
        "- This means the network correctly classifies about 98 out of every 100 handwritten digits on the MNIST test set.\n"
        "- Because this model is a simple fully-connected network, it is fast and easy to understand, but not as powerful as convolutional models for more complex handwriting tasks."
    )

    st.info(
        "The model outputs raw scores (logits). A softmax operation converts these scores into probabilities, and the highest probability gives the predicted digit."
    )

with training_tab:
    st.header("Training Process")
    st.markdown(
        "The training process uses the MNIST dataset and the following workflow:\n"
        "1. Load training and test images as grayscale tensors.\n"
        "2. Normalize each image using the MNIST mean and standard deviation.\n"
        "3. Feed batches of 64 images through the network.\n"
        "4. Compute loss with CrossEntropyLoss.\n"
        "5. Use Adam optimizer to update the network weights.\n"
        "6. Repeat for 5 epochs."
    )

    st.subheader("Loss progression")
    st.markdown(
        "- Epoch 1 average loss: 0.2518\n"
        "- Epoch 2 average loss: 0.1093\n"
        "- Epoch 3 average loss: 0.0775\n"
        "- Epoch 4 average loss: 0.0584\n"
        "- Epoch 5 average loss: 0.0465\n"
    )

    st.subheader("Sample training output")
    st.code(
        "Epoch 1/5 | Batch 0/938 | Loss: 2.2837\n"
        "Epoch 1/5 | Batch 200/938 | Loss: 0.2121\n"
        "Epoch 1/5 | Batch 400/938 | Loss: 0.0842\n"
        "Epoch 1/5 | Batch 600/938 | Loss: 0.1605\n"
        "Epoch 1/5 | Batch 800/938 | Loss: 0.2086\n"
        "Epoch 1 complete. Average loss: 0.2518\n\n"
        "Epoch 2/5 | Batch 0/938 | Loss: 0.0269\n"
        "Epoch 2/5 | Batch 200/938 | Loss: 0.1857\n"
        "Epoch 2/5 | Batch 400/938 | Loss: 0.1528\n"
        "Epoch 2/5 | Batch 600/938 | Loss: 0.0474\n"
        "Epoch 2/5 | Batch 800/938 | Loss: 0.1721\n"
        "Epoch 2 complete. Average loss: 0.1093\n\n"
        "Epoch 3/5 | Batch 0/938 | Loss: 0.0234\n"
        "Epoch 3/5 | Batch 200/938 | Loss: 0.0271\n"
        "Epoch 3/5 | Batch 400/938 | Loss: 0.0147\n"
        "Epoch 3/5 | Batch 600/938 | Loss: 0.0801\n"
        "Epoch 3/5 | Batch 800/938 | Loss: 0.0618\n"
        "Epoch 3 complete. Average loss: 0.0775\n\n"
        "Epoch 4/5 | Batch 0/938 | Loss: 0.0681\n"
        "Epoch 4/5 | Batch 200/938 | Loss: 0.0116\n"
        "Epoch 4/5 | Batch 400/938 | Loss: 0.0185\n"
        "Epoch 4/5 | Batch 600/938 | Loss: 0.0453\n"
        "Epoch 4/5 | Batch 800/938 | Loss: 0.0476\n"
        "Epoch 4 complete. Average loss: 0.0584\n\n"
        "Epoch 5/5 | Batch 0/938 | Loss: 0.0132\n"
        "Epoch 5/5 | Batch 200/938 | Loss: 0.0341\n"
        "Epoch 5/5 | Batch 400/938 | Loss: 0.0225\n"
        "Epoch 5/5 | Batch 600/938 | Loss: 0.0118\n"
        "Epoch 5/5 | Batch 800/938 | Loss: 0.0358\n"
        "Epoch 5 complete. Average loss: 0.0465"
    )

    st.write(
        "The loss values show the network learning quickly during the first epoch and then refining its weights over the next epochs. "
        "By epoch 5, the model is well-trained for standard MNIST digit recognition."
    )
