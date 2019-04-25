## CIFAR-10 Example with TensorFlow 2.0

This repository aims to provide a simple example for the CIFAR-10 dataset with TensorFlow 2.0 and the Keras API.
There are multiple example scripts in this repository, each illustrating a different aspect.

### Prerequisites
In order to run these scripts, you need to have Python 3.5, 3.6 or 3.7 installed.
In your Python environment, please install TensorFlow 2.0 and tensorflow-datasets. 
At the time of writing TensorFlow 2.0 is in alpha, therefore you can install the CPU version with the following pip command:

```pip install --upgrade tensorflow==2.0.0alpha0 tensorflow-datasets```

To install the GPU version of TensorFlow, please refer to the [official installation instructions](https://www.tensorflow.org/install/gpu).


### FirstNetwork
The `FirstNetwork.py` script shows how the CIFAR-10 dataset is loaded and trained with a very simple network.
Please note that this network is overly simplified for the ease of understanding the workflow.
To see a model with better performance, please refer to the `BetterNetwork.py` script.


### FirstNetworkWithTensorBoard
This script extends the `FirstNetwork.py` script by adding a `Tensorboard` callback. 
Please note that the callback has to be provided to `model.fit(...)`.

To view the logged data in TensorBoard, open a command line and enter

```tensorboard --logdir=<log_dir>```

while replacing `<log_dir>` with the directory provided in the `log_dir` parameter of the `TensorBoard` callback.


### Prediction / FirstNetworkWithPrediction
These two scripts show how to use the trained and saved model to predict the classes of the images.
To do this, the model is first loaded from the checkpoint and then fed an image to be predicted. 

Please note that the fed values have to be 4-dimensional.
This allows predicting multiple images in one go.


### BetterNetwork
The `BetterNetwork.py` script states a more complex model to achieve better results on the CIFAR-10 dataset (about 85% test accuracy).
Furthermore, it uses `tf.data.Dataset` to build an input pipeline using some data augmentation to further improve the training.
Augmentations increase the variance in the data by changing the images randomly every time.