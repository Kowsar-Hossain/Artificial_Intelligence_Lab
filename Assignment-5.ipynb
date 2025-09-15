
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

def main():
    (trainX, trainY), (testX, testY) = load_data()

    trainX = trainX.astype('float32') / 255.0
    testX = testX.astype('float32') / 255.0

    trainY_cat = to_categorical(trainY, num_classes=10)
    testY_cat = to_categorical(testY, num_classes=10)

    print("Train shape:", trainX.shape, trainY_cat.shape)
    print("Test shape:", testX.shape, testY_cat.shape)

    model = build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


     # ✅ Train and capture history
    history = model.fit(trainX, trainY_cat, epochs=10, batch_size=32, validation_data=(testX, testY_cat))

    # ✅ Evaluate model
    loss, accuracy = model.evaluate(testX, testY_cat)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    plt.figure(figsize=(4, 3))
    # ✅ Plot accuracy and loss
    plot_accuracy(history)
    plot_loss(history)

    # ✅ Predictions
    predictions = model.predict(testX)
    predicted_classes = np.argmax(predictions, axis=1)
    actual_classes = testY

    plot_predictions(testX, actual_classes, predicted_classes)

def build_model():
    num_classes=10
    img_size=(28,28,1)
    Inputs=Input(img_size)
    x=Conv2D(filters=8,kernel_size=(3,3),activation='relu')(Inputs)
    x=Conv2D(filters=16,kernel_size=(3,3),activation='relu')(x)
    x=Flatten()(x)
    x=Dense(64,activation='relu')(x)
    x=Dense(16,activation='relu')(x)
    outputs=Dense(num_classes,activation='softmax')(x)
    model=Model(Inputs,outputs)
    model.summary(show_trainable=True)
    return model



def plot_predictions(images, actual, predicted, num=25):
    plt.figure(figsize=(10, 15))
    for i in range(num):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"True: {actual[i]}\nPred: {predicted[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_accuracy(history):

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='x')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_loss(history):
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='x')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
