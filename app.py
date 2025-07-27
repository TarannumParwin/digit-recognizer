import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# Title
st.title("🧠 Handwritten Digit Recognizer")

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Build model
@st.cache_resource
def train_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train_cat, epochs=5, verbose=0)
    return model

model = train_model()

# Select test image
index = st.slider("Select a test image index", 0, 9999, 0)

# Show image
st.image(x_test[index], width=150, caption=f"Actual: {y_test[index]}")

# Predict
prediction = model.predict(np.array([x_test[index]]))
predicted_digit = np.argmax(prediction)

# Show result
st.success(f"Predicted Digit: {predicted_digit}")
