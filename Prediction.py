import numpy as np
import tensorflow as tf

# Load the CIFAR-10 dataset
cifar10 = tf.keras.datasets.cifar10

# Get test and training data where x are the images and y are the labels
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the images to a pixel value range of [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Load the model from the checkpoint file where the ModelCheckpoint callback saved it to.
model = tf.keras.models.load_model("checkpoint.hdf5")

# Get an image from the test data to feed it into the network. Since the input of the network has to
# be 4-dimensional, we add a first dimension by reshaping the image.
first_image = x_test[0]
first_image_4d = np.reshape(first_image, (1, 32, 32, 3))

# Run the prediction on the loaded model
predicted_class_probabilities = model.predict(first_image_4d)

# Get the index of the class with the highest probability and print it.
predicted_class = np.argmax(predicted_class_probabilities)
print("Prediction: ", predicted_class, "  Expected: ", y_test[0])
