from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

# Loading the dataset returns four NumPy arrays:
#
# The train_images and train_labels arrays are the training set—the data the model uses to learn.
# The model is tested against the test set, the test_images, and test_labels arrays.
# The images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255. The labels are an array of integers, ranging from 0 to 9. These correspond to the class of clothing the image represents:
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Each image is mapped to a single label. Since the class names are not included with the dataset, store them here to use later when plotting the images:

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Explore the data
print ("train_images.shape",train_images.shape)
print("train_labels count:",len(train_labels))
print ("train labels:",train_labels)

showImages=False
if showImages:
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

# Define 3 layers
# Layer 1: The first layer in this network, tf.keras.layers.Flatten, transforms the format of the images from a two-dimensional array
#          (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels). Think of this layer as unstacking rows of
#          pixels in the image and lining them up. This layer has no parameters to learn; it only reformats the data.
# Layer 2 and 3 are densely connected or fully connected, neural layers
# Layer 2: The first Dense layer has 128 nodes (or neurons)
# Layer 3:  The second (and last) layer is a 10-node softmax layer that returns an array of 10 probability scores that sum to 1

# Each node contains a score that indicates the probability that the current image belongs to one of the 10 classes.

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:
#
# Loss function —This measures how accurate the model is during training. We want to minimize this function to "steer" the model in the right direction.
# Optimizer —This is how the model is updated based on the data it sees and its loss function.
# Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.

# A loss function (or objective function, or optimization score function) is one of the two parameters required to compile a model:
# Available loss functions see https://keras.io/losses/
# Test accuracy with different loss functions:
# 0.8239 sparse_categorical_crossentropy
# 0.0 mean_absolute_percentage_error
# 0.0 kullback_leibler_divergence
# 0.0 poisson
# ==> sparse_categorical_crossentropy seeems to be the only reasonable cost function
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

#Next, compare how the model performs on the test dataset:

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\nTest accuracy:', test_acc)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Make predictions
# ---------------------------------------------------------------------------------------------------------------------------------------
predictions = model.predict(test_images)
print('\nPreditions[0]',predictions[0])
print('Highest confidence value:',np.argmax(predictions[0]))

#
# Plot the image
#
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

#
# Plot the probabilities
#
def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([0,1,2,3,4,5,6,7,8,9])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()


# Grab an image from the test dataset.
img = test_images[0]

print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)
predictions_single = model.predict(img)

print(predictions_single)
plot_value_array(0, predictions_single, test_labels,_ = plt.xticks(range(10), class_names, rotation=45))