import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# mnist = tf.keras.datasets.mnist
# or
from tensorflow import keras
mnist = keras.datasets.mnist

# Load and prepare the MNIST dataset. Convert the samples from integers to floating-point numbers (Color values from 0-255)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

print("x_train shape", train_images.shape)
print("y_train shape", train_labels.shape)
#Build the tf.keras.Sequential model by stacking layers. Choose an optimizer and loss function used for training:
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

#Train and evaluate model:
# Test accuracy with different loss functions
# 0.0011 kullback_leibler_divergence
# 0.0006 poisson
# 0.9803 sparse_categorical_crossentropy (see for explanation: https://www.reddit.com/r/MLQuestions/comments/93ovkw/what_is_sparse_categorical_crossentropy/)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ---------------------------------------------------------------------------------------------------------------------------------------
# Train the model
# ---------------------------------------------------------------------------------------------------------------------------------------
# Train the model
# Training the neural network model requires the following steps:
#
# 1. Feed the training data to the model. In this example, the training data is in the train_images and train_labels arrays.
# 2. The model learns to associate images and labels.
# 3. We ask the model to make predictions about a test set—in this example, the test_images array.
#    We verify that the predictions match the labels from the test_labels array.
#
# To start training, call the model.fit method—so called because it "fits" the model to the training data:
model.fit(train_images, train_labels, epochs=7)

# ---------------------------------------------------------------------------------------------------------------------------------------
# Evaluate accuracy
# ---------------------------------------------------------------------------------------------------------------------------------------
# Next, compare how the model performs on the test dataset:
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\nTest accuracy:', test_acc)

model.summary()

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

  plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                100*np.max(predictions_array),
                                true_label),
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

# Let's test the 10th image
i=10
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()
# predictions = model.predict(test_images)
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(test_images[i], cmap=plt.cm.binary)
#     plt.xlabel(predictions[i])
# plt.show()