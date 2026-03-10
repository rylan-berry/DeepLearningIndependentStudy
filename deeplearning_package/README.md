# Deeplearning Package

## Overview

This package is designed to be similar to the PyTorch system of a building block system, providing functions that can be mixed, matched, and customized for any given model. It implements core deep learning concepts including automatic differentiation, sequential modeling, and various layer types.

## Modules

This project has six main modules:

* `autogradient.py`  
* `sequence.py`
* `optimizer.py`
* `neural_net.py`
* `cnn_layers.py`
* `rnn_layers.py`

All of which are automatically part of the initial import of the package.

## Detailed Module Descriptions

### `autogradient.py`

This module forms the core of the automatic differentiation system. It introduces the `Values` class, which tracks computational history to compute gradients.

*   **`Values` Class**: Encapsulates numerical values (`vals`) and their corresponding gradients (`grad`). 
*   **Broadcasting Support**: Includes a sophisticated `_broadcast_grad` static method that automatically handles gradient shape adjustments during backpropagation when operations involve broadcasted NumPy arrays (e.g., adding a bias vector to a matrix).
*   **Operations**: Supports a wide array of operations including arithmetic (`+`, `-`, `*`, `/`, `@`, `**`), activation functions (`relu`, `tanh`, `softmax`), and tensor manipulations (`reshape`, `transpose`, `pad`, `sum`, `mean`, `max`, and slicing/indexing).

### `rnn_layers.py` (New)

This module provides implementations for Recurrent Neural Networks, suitable for sequential data processing.

*   **`RNNCell`**: The fundamental building block for recurrent layers, performing the basic hidden state update: $h_t = \text{activation}(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})$.
*   **`RNN` Layer**: A high-level wrapper that processes entire sequences, maintaining hidden states across time steps and supporting both many-to-many and many-to-one architectures.

### `cnn_layers.py`

Implements fundamental building blocks for Convolutional Neural Networks.

*   **`Convo2D`**: Supports 'valid' and 'same' padding, custom strides, and automated gradient computation for kernels.
*   **Pooling**: Includes `MaxPooling` and `AvgPooling` for spatial down-sampling.

### `neural_net.py`

Defines high-level components for model construction.

*   **`Layer` & `Dense`**: Fully connected layers with customizable activations.
*   **`Dropout`**: Regularization technique to prevent overfitting by randomly zeroing activations during training.
*   **`Model`**: The central class for training. It handles the training loop, batching, and integration with loss functions (`cross_entropy_loss`, `mse_loss`) and penalty functions (`l1_reg`, `l2_reg`).

### `optimizer.py`

Implements parameter update algorithms: `Optim_SGD`, `Optim_SGD_Momentum`, `Optim_AdaGrad`, `Optim_RMSPropclass`, and `Optim_Adam`.

### `sequence.py`

Provides the `Sequence` container to stack layers and manage parameter retrieval across the model.

## Dependencies

The package relies heavily on `numpy` for numerical operations and gradient tracking.
