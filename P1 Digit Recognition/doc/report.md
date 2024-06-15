# P1 Digit Recognition

**Name**: 赵子涵
**Major**: Computer Science
**Student ID**: 2023E8013282148

### Dataset

The MNIST consist 60000 28*28 grayscale images for train set, and 10000 for test set. The images are all handwritten digits, and lable ranges from 0 to 9.

### Model Design

We implemented a CNN for this task. Specifically, the CNN we build consist three basic components, the convolution layer, the poolling layer, and the linear layer. Here is a diagram of our network

```mermaid
graph TD;
input("Input [1*28*28]")
conv1["Conv(ReLU) [4*28*28]"]
pool1["MaxPool [4*14*14]"]
conv2["Conv(ReLU) [16*14*14]"]
pool2["MaxPool [16*7*7]"]
conv3["Conv(ReLU) [64*7*7]"]
lin["Linear [10]"]

input-->conv1;
conv1-->pool1;
pool1-->conv2;
conv2-->pool2;
pool2-->conv3;
conv3-->lin;
lin-->output("Output");

```

We choose maxpool as our pooling layer, and ReLU as our activation function. 

### Model Evaluation

We trained our model with batch_size=100, learning_rate=0.001 and epoch=5. We plot the loss and test accuracy as follows.

<img src="pic\training_loss.png" alt="training_loss" style="zoom:50%;" />

<img src="pic\test_accuracy.png" alt="training_loss" style="zoom:50%;" />

As shown above, the training loss decline rapidly, and the test accuracy reached 99%.