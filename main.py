import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

# Defines the architecture of neural network
model = keras.Sequential([
    # Input layer
    keras.layers.Flatten(input_shape=(28,28)),

    # Hidden layer with 128 neurons
    # Dense means fully connected
    keras.layers.Dense(128, activation="relu"),

    # Output layer
    # Softmax means values add up to one
    keras.layers.Dense(10, activation="softmax")
])

# Set up model parameters
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# Train the model
# Epoch is the number of time the model will see the information
model.fit(train_images, train_labels, epochs=1)

"""
test_lost, test_acc = model.evaluate(test_images, test_labels)

print("Test Acc:", test_acc)
"""

prediction = model.predict(test_images)

# Shows actual image as x label and picture, and prediction as title
# Shows actual image as x label and picture, and prediction as title
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])  # Fix here
    plt.show()






