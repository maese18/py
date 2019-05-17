import tensorflow as tf
import matplotlib.pyplot as plt
# mnist = tf.keras.datasets.mnist
# or
from tensorflow import keras
mnist = keras.datasets.mnist

# Load and prepare the MNIST dataset. Convert the samples from integers to floating-point numbers (Color values from 0-255)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# y_train is the corresponding label
print(train_labels[1])

plt.figure(figsize=(15,15))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
plt.show()


print("train_images shape", train_images.shape)
print("train_labels shape", train_labels.shape)
