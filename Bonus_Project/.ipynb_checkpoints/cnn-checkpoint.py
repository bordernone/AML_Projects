import numpy as np
from scipy import signal
from keras.datasets import mnist
from PIL import Image

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class CNN:
    def __init__(self):
        self.input_shape = (28, 28, 1) # (height, width, channel)
        
        self.layer_1_kernel_size = (3, 3)
        self.layer_1_kernels = 2
        self.layer_1_stride = 1
        
        self.layer_1_kernels = np.random.randn(self.layer_1_kernel_size[0], self.layer_1_kernel_size[1], self.layer_1_kernels)
        
    def applyKernel(self, input, kernel):
        assert input.shape == self.input_shape, "Input shape is not correct"
        return signal.correlate2d(input, kernel, mode='valid')
        
    def forwardCNN(self, input):
        assert input.shape == self.input_shape, "Input shape is not correct"
        
        output_shape = (26, 26, 2)
        output = np.zeros(output_shape)
        
        for i in range(self.layer_1_kernels):
            output[:, :, i] = self.applyKernel(input[:, :, 0], self.layer_1_kernels[:, :, i])
        
        return output

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Display first image
Image.fromarray(x_train[0]).show()