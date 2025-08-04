# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Title
st.title("🧠 Handwritten Digit Recognizer")
st.write("This app predicts digits (0–9) using a trained neural network.")

# Load the pre-trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Load MNIST test data
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0

# Choose test image
index = st.slider("Pick a test image (0–9999)", 0, 9999, 0)
st.image(x_test[index], width=150, caption=f"Actual: {y_test[index]}")

# Predict digit
img_input = np.array([x_test[index]])
prediction = model.predict(img_input)
predicted_digit = np.argmax(prediction)

# Show prediction
st.success(f"✅ Predicted Digit: {predicted_digit}")
