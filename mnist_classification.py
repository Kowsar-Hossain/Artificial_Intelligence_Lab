'''
MNIST classification using Sequential API.
'''

#======================= Necessary Imports =========================
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

#======================= Load and Preprocess =========================
(trainX, trainY), (testX, testY) = mnist.load_data()

# Normalize pixel values
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# One-hot encode labels
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

#======================= Model Construction =========================
model = Sequential([
    Flatten(input_shape=(28, 28), name="flatten"),
    Dense(64, activation="relu", name="dense_1"),
    Dense(128, activation="relu", name="dense_2"),
    Dense(64, activation="relu", name="dense_3"),
    Dense(10, activation="softmax", name="output_layer")
])

# Show model summary
model.summary()

#======================= Compile & Train =========================
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(trainX, trainY, validation_split=0.1, epochs=10, verbose=1)

#======================= Evaluate =========================
test_loss, test_acc = model.evaluate(testX, testY, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

#======================= Visualization =========================
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

