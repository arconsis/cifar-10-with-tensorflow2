# NOTE: You also have to install tensorflow_datasets for this code to work.

import tensorflow as tf
import tensorflow_datasets as tfds


def prepare_dataset(dataset, train=False, batch_size=512):
	""" Applies shuffling and augmentation (if train==True) + normalization and batching to the given dataset. """

	if train:
		# Shuffle the dataset with a buffer of 2048 and apply the augmentation
		# Prefetching adds an output buffer for the 16 augmentation threads
		dataset = dataset.shuffle(buffer_size=2048)
		dataset = dataset.map(augment, num_parallel_calls=16).prefetch(buffer_size=128)

	# normalize the images and batch them. Again add a buffer to prevent
	# unnecessary stopping of the batching or starvation of the GPU
	dataset = dataset.map(normalize, num_parallel_calls=8) \
		.batch(batch_size) \
		.prefetch(buffer_size=8)

	return dataset


@tf.function
def augment(image, label):
	""" Applies augmentations to the given image. For details on the augmentations please refer to the documentation. """
	# First convert image to floating point representation
	image = tf.image.random_flip_left_right(image)

	image = tf.image.random_brightness(image, 0.2)

	image = tf.image.random_contrast(image, 1 / 1.3, 1.3)

	image = tf.image.random_hue(image, 0.1)

	image = tf.image.random_saturation(image, 1 / 1.2, 1.2)

	# Randomly increase the size of the image slightly to then randomly crop a part out of it.
	# This is a way to get random scales + translations
	random_height = tf.random.uniform((), minval=32, maxval=40, dtype=tf.int32)
	random_width = tf.random.uniform((), minval=32, maxval=40, dtype=tf.int32)
	image = tf.image.resize(image, (random_height, random_width))
	image = tf.image.random_crop(image, (32, 32, 3))

	return image, label


@tf.function
def normalize(image, label):
	""" Apply per image standardisation to normalize the image """
	return tf.image.per_image_standardization(image), label


# Load the test and training data as datasets with tensorflow_datasets
train_dataset = prepare_dataset(tfds.load(name="cifar10", split=tfds.Split.TRAIN, as_supervised=True), True)
test_dataset = prepare_dataset(tfds.load(name="cifar10", split=tfds.Split.TEST, as_supervised=True), False)

# Define the model
model = tf.keras.models.Sequential([
	# As a further augmentation, apply a dropout
	tf.keras.layers.Dropout(0.1, input_shape=(32, 32, 3)),

	tf.keras.layers.Conv2D(filters=96, kernel_size=3, activation='relu', padding='same'),  # (32, 32, 96)
	tf.keras.layers.Conv2D(filters=96, kernel_size=3, activation='relu', padding='same'),  # (32, 32, 96)
	tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=2, activation='relu', padding='same'),  # (16, 16, 96)
	tf.keras.layers.Dropout(0.1),

	tf.keras.layers.Conv2D(filters=192, kernel_size=3, activation='relu', padding='same'),  # (16, 16, 192)
	tf.keras.layers.Conv2D(filters=192, kernel_size=3, activation='relu', padding='same'),  # (16, 16, 192)
	tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=2, activation='relu', padding='same'),  # (8, 8, 192)
	tf.keras.layers.Dropout(0.1),

	tf.keras.layers.Conv2D(filters=192, kernel_size=3, activation='relu', padding='same'),  # (8, 8, 192)
	tf.keras.layers.Conv2D(filters=192, kernel_size=1, activation='relu', padding='same'),  # (8, 8, 192)
	tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=2, activation='relu', padding='same'),  # (8, 8, 192)

	tf.keras.layers.Dropout(0.1),
	tf.keras.layers.Conv2D(filters=10, kernel_size=1, activation='relu', padding='same'),  # (8, 8, 10)

	tf.keras.layers.AveragePooling2D(pool_size=4, strides=4, padding='valid'),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Activation('softmax')
])
# Print model summary
model.summary()

# Create callbacks for model saving and TensorBoard
cp_callback = tf.keras.callbacks.ModelCheckpoint('checkpoint.hdf5', verbose=1, save_weights_only=False)
tb_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

# Compile and run the model for 50 epochs
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=50, validation_data=test_dataset, callbacks=[cp_callback, tb_callback])
