import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
print("Training started...")
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
print("Training completed!")

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy}")

# Save the model
model.save("model.h5")
print("✅ Model saved as model.h5")

# Make predictions on test data
predictions = model.predict(x_test)

# Get the first prediction
predicted_digit = np.argmax(predictions[0])
actual_digit = np.argmax(y_test[0])

# Show the result
print(f"Predicted digit: {predicted_digit}")
print(f"Actual digit: {actual_digit}")

# Display the image
plt.imshow(x_test[0], cmap='gray')
plt.title(f"Predicted: {predicted_digit} | Actual: {actual_digit}")
plt.show()
