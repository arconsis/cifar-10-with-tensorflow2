import tensorflow as tf

# Load the CIFAR-10 dataset
cifar10 = tf.keras.datasets.cifar10

# Get test and training data where x are the images and y are the labels
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the images to a pixel value range of [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the Keras model.
# Please note, to keep this example simple, we use an overly simplified network here. This will not reach
# a good accuracy. For a better network please refer to the BetterNetwork.py script.
model = tf.keras.models.Sequential([
	tf.keras.layers.Conv2D(24, 3, activation='relu', padding='same', input_shape=(32, 32, 3)),  # (32, 32, 24)
	tf.keras.layers.Flatten(),  																# (24576)
	tf.keras.layers.Dense(128, activation='relu'),  											# (128)
	tf.keras.layers.Dense(10, activation='softmax') 											# (10)
])
# Print a summary of the model with parameter counts
model.summary()

# Define a ModelCheckpoint callback to persist the model after every epoch.
cp_callback = tf.keras.callbacks.ModelCheckpoint('checkpoint.hdf5', verbose=1, save_weights_only=False)

# Compile, i.e. configure the network with the Adam optimizer and the sparse_categorical_crossentropy loss.
# Furthermore, keep track of the model's accuracy metric and have it printed out during training.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Actually train the model for 15 epochs, i.e. show the model all the training data 15 times.
# To be able to judge the model's performance, also provide the test data. It will be evaluated after
# every epoch and the results will be printed to the console.
model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test), callbacks=[cp_callback])
