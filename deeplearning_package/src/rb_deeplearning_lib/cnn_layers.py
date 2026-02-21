from .autogradient import Values
from math import ceil
import numpy as np

class Convo2D:
    def __init__(self, kernel_matrix, padding='valid', stride=1):
        self.kernel = kernel_matrix if isinstance(kernel_matrix, Values) else Values(kernel_matrix)
        self.padding = padding
        self.stride = stride
    
    def params(self):
        return self.kernel

    def __call__(self, x):
        self.kernel = x if isinstance(x, Values) else Values(x)
        input_height, input_width = x.shape
        kernel_height, kernel_width = self.kernel.shape

        pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0

        if self.padding == 'same':
            # Calculate output dimensions based on 'same' padding logic:
            # output_dim = ceil(input_dim / stride)
            # This is a common interpretation in frameworks like TensorFlow/Keras
            output_height_target = ceil(input_height / stride)
            output_width_target = ceil(input_width / stride)

            # Calculate total padding needed to achieve the target output dimensions
            total_padding_h = max(0, (output_height_target - 1) * stride + kernel_height - input_height)
            total_padding_w = max(0, (output_width_target - 1) * stride + kernel_width - input_width)

            # Distribute padding to top/bottom and left/right. If padding is odd, more goes to bottom/right.
            pad_top = total_padding_h // 2
            pad_bottom = total_padding_h - pad_top
            pad_left = total_padding_w // 2
            pad_right = total_padding_w - pad_left
        elif self.padding != 'valid':
            print("ERROR: Padding method does not exist")
            return None

        # Apply padding to the input matrix
        padded_input = x.pad(((pad_top, pad_bottom), (pad_left, pad_right)))

        # Calculate the actual output dimensions after padding and considering stride
        # Formula: (Padded_Input_Dimension - Kernel_Dimension) // Stride + 1
        output_height = (padded_input.shape[0] - kernel_height) // stride + 1
        output_width = (padded_input.shape[1] - kernel_width) // stride + 1

        # Initialize the output matrix with zeros
        output_matrix = Values(np.zeros((output_height, output_width)))

        # Perform the convolution operation
        for i in range(output_height):
            for j in range(output_width):
                # Extract the current window (receptive field) from the padded input
                window = padded_input[i * stride : i * stride + kernel_height,
                                    j * stride : j * stride + kernel_width]

                # Perform element-wise multiplication and sum (dot product)
                output_matrix[i, j] = (window * self.kernel).sum()

        return output_matrix
            

class Pooling:
    def __init__(self, pool_size, stride=1):
        self.size = pool_size
        self.stride = stride
    
    def __call__(self, x):
        return x

class MaxPooling(Pooling):
    def __call__(self, x):
        in_h, in_w = x.shape
        p_h, p_w = self.size
        stride = self.stride
        output_height = (in_h - p_h) // stride + 1
        output_width = (in_w - p_w) // stride + 1
        out = Values(np.zeros((output_height, output_width)))
        for i in range(output_height):
            for j in range(output_width):
                window = x[i*stride : i*stride+p_h, j*stride : j*stride+p_w]
                out[i,j] = window.max()
        return out

class AvgPooling(Pooling):
    def __call__(self, x):
        in_h, in_w = x.shape
        p_h, p_w = self.size
        stride = self.stride
        output_height = (in_h - p_h) // stride + 1
        output_width = (in_w - p_w) // stride + 1
        out = Values(np.zeros((output_height, output_width)))
        for i in range(output_height):
            for j in range(output_width):
                window = x[i*stride : i*stride+p_h, j*stride : j*stride+p_w]
                out[i,j] = window.mean()
        return out
