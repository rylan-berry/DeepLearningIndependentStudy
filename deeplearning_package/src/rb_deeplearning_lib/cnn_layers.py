from .autogradient import Values
from math import ceil
import numpy as np

class Reshape:
  def __init__(self, shape):
    self.shape = shape
  def __call__(self, x):
    out = x.reshape(self.shape)
    return out

class Convo2D:
    def __init__(self, kernel_matrix, padding='valid', stride=1):
        self.kernel = kernel_matrix if isinstance(kernel_matrix, Values) else Values(kernel_matrix)
        self.padding = padding
        self.stride = stride
    
    def params(self):
        return self.kernel

    def set_params(self, param):
        self.kernel = param if isinstance(param, Values) else Values(param)

    def __call__(self, _x):
        _x = _x if isinstance(_x, Values) else Values(_x)
        x = []
        if len(_x.shape) == 2:
          x = Values(np.zeros((1,_x.shape[0],_x.shape[1])))
          x[0,:,:] = _x
        else:
          x = _x
        
        x_len, input_height, input_width = x.shape
        kernel_height, kernel_width = self.kernel.shape
        stride = self.stride
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

        # Apply padding to the input matrix. The 0,0 at the front is so to avoid making padding for the inputs row
        padded_input = x.pad(((0,0),(pad_top, pad_bottom), (pad_left, pad_right)))

        # Calculate the actual output dimensions after padding and considering stride
        # Formula: (Padded_Input_Dimension - Kernel_Dimension) // Stride + 1
        output_height = (padded_input.shape[1] - kernel_height) // stride + 1
        output_width = (padded_input.shape[2] - kernel_width) // stride + 1

        # Initialize the output matrix with zeros
        output_matrix = Values(np.zeros((x_len, output_height, output_width)))

        # Perform the convolution operation
        for i in range(output_height):
            for j in range(output_width):
                # Extract the current window (receptive field) from the padded input
                window = padded_input[:,i * stride : i * stride + kernel_height,
                                    j * stride : j * stride + kernel_width]

                # Perform element-wise multiplication and sum (dot product)
                output_matrix[:, i, j] = (window * self.kernel).sum()

        return output_matrix
            

class Pooling:
    def __init__(self, pool_size, stride=1):
        self.size = pool_size
        self.stride = stride
    
    def __call__(self, x):
        x = x if isinstance(x, Values) else Values(x)
        return x

class MaxPooling(Pooling):
    def __call__(self, _x):
        _x = _x if isinstance(_x, Values) else Values(_x)
        x = []
        if len(_x.shape) == 2:
          x = Values(np.zeros((1,_x.shape[0],_x.shape[1])))
          x[0,:,:] = _x
        else:
          x = _x

        x_len, in_h, in_w = x.shape
        p_h, p_w = self.size
        stride = self.stride
        output_height = (in_h - p_h) // stride + 1
        output_width = (in_w - p_w) // stride + 1
        out = Values(np.zeros((x_len, output_height, output_width)))
        for i in range(output_height):
            for j in range(output_width):
                window = x[:,i*stride : i*stride+p_h, j*stride : j*stride+p_w]
                out[:,i,j] = window.max()
        return out

class in2col_Convo2D:
    def __init__(self, kernel_matrix, padding='valid', stride=1):
        k = kernel_matrix if isinstance(kernel_matrix, Values) else Values(kernel_matrix)
        shap = k.shape
        self.k_shape = shap
        self.kernel = k.reshape((-1, 1))
        self.padding = padding
        self.stride = stride
    
    def params(self):
        return self.kernel

    def set_params(self, param):
        self.kernel = param if isinstance(param, Values) else Values(param)


    def im2col(self, X):
        in_h, in_w = X.shape #expected 2d
        k_h, k_w = self.k_shape
        stride = self.stride
        out_h = (in_h - k_h) // stride + 1
        out_w = (in_w - k_w) // stride + 1

        col_idx=0
        im2col_matrix = Values(np.zeros((out_h * out_w, k_h * k_w)))
        for y in range(0, out_h * stride, stride):
          for x in range(0,out_w * stride, stride):
            patch = X[y: y+k_h, x:x+k_w]
            im2col_matrix[col_idx,:] = patch.reshape((k_w * k_h))
            col_idx += 1
        return im2col_matrix



    def __call__(self, _x):
        _x = _x if isinstance(_x, Values) else Values(_x)
        x = []
        if len(_x.shape) == 2:
          x = Values(np.zeros((1,_x.shape[0],_x.shape[1])))
          x[0,:,:] = _x
        else:
          x = _x
        
        batch, input_height, input_width = x.shape #only expected to be a 2d with batches
        kernel_height, kernel_width = self.k_shape
        stride = self.stride
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

        # Apply padding to the input matrix. The 0,0 at the front is so to avoid making padding for the inputs row
        padded_input = x.pad(((0,0),(pad_top, pad_bottom), (pad_left, pad_right)))

        # Calculate the actual output dimensions after padding and considering stride
        # Formula: (Padded_Input_Dimension - Kernel_Dimension) // Stride + 1
        output_height = (padded_input.shape[1] - kernel_height) // stride + 1
        output_width = (padded_input.shape[2] - kernel_width) // stride + 1

        # Initialize the output matrix with zeros
        output_matrix = Values(np.zeros((batch, output_height, output_width)))

        # Perform the convolution operation
        for i in range(batch):
          cur_im2col = self.im2col(padded_input[i])
          flat_out = cur_im2col @ self.kernel # cur_im2col (num_patches, kernel_size), self.kernel (kernel_size, 1)
                                            # Result flat_out (num_patches, 1)
          output_matrix[i] = flat_out.reshape((output_height,output_width)) # FIX: Squeeze before reshape


        return output_matrix

class AvgPooling(Pooling):
    def __call__(self, _x):
        _x = _x if isinstance(_x, Values) else Values(_x)
        x = []
        if len(_x.shape) == 2:
          x = Values(np.zeros((1,_x.shape[0],_x.shape[1])))
          x[0,:,:] = _x
        else:
          x = _x

        x_len, in_h, in_w = x.shape
        p_h, p_w = self.size
        stride = self.stride
        output_height = (in_h - p_h) // stride + 1
        output_width = (in_w - p_w) // stride + 1
        out = Values(np.zeros((x_len, output_height, output_width)))
        for i in range(output_height):
            for j in range(output_width):
                window = x[:,i*stride : i*stride+p_h, j*stride : j*stride+p_w]
                out[:,i,j] = window.mean()
        return out
