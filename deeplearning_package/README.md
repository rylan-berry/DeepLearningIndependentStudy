# Deeplearning Package

## Overview

This package is designed to be similar to the PyTorch system of a building block system, providing functions that can be mixed, matched, and customized for any given model. It implements core deep learning concepts including automatic differentiation, sequential modeling, various layer types (CNN, RNN, Attention), optimization algorithms, and utilities for model saving and loading.

## Modules

This project contains several key modules:

*   `autogradient.py`
*   `sequence.py`
*   `optimizer.py`
*   `neural_net.py`
*   `cnn_layers.py`
*   `rnn_layers.py`
*   `attention.py`
*   `save_load_funcs.py`

All of which are automatically part of the initial import of the package.

## Detailed Module Descriptions

### `autogradient.py`

This module forms the core of the automatic differentiation system. It introduces the `Values` class, which tracks computational history to compute gradients.

*   **`Values` Class**: Encapsulates numerical values (`vals`) and their corresponding gradients (`grad`). It supports a wide array of operations and tensor manipulations.
*   **Broadcasting Support**: Includes a sophisticated `_broadcast_grad` static method that automatically handles gradient shape adjustments during backpropagation when operations involve broadcasted NumPy arrays (e.g., adding a bias vector to a matrix or operations with varying input dimensions).
*   **Operations**: Supports a comprehensive set of operations including arithmetic (`+`, `-`, `*`, `/`, `@`, `**`), activation functions (`relu`, `tanh`, `softmax`, `log`, `exp`), and crucial tensor manipulations (`reshape`, `transpose`, `pad`, `sum`, `mean`, `max`, `abs`, and flexible slicing/indexing via `__getitem__` and `__setitem__`). It also provides direct access to the NumPy `T` (transpose) attribute.

### `rnn_layers.py`

This module provides implementations for Recurrent Neural Networks, suitable for sequential data processing.

*   **`FullRNNLayer`**: Implements a standard RNN layer with configurable input, hidden, and output dimensions. It supports different activation functions (`tanh` by default) and handles input sequence transposition for flexible batching. The `_rnn_cell_forward` method details the computation for a single time step, integrating input, previous hidden state, and biases to produce the current hidden state and output.

### `cnn_layers.py`

Implements fundamental building blocks for Convolutional Neural Networks.

*   **`Convo2D`**: Supports 2D convolution with configurable `kernel_matrix`, `padding` ('valid' or 'same'), and `stride`. It automatically calculates output dimensions and applies padding. Gradients are computed for the kernel during backpropagation.
*   **Pooling**: Includes `MaxPooling` and `AvgPooling` classes, which perform spatial down-sampling operations with specified `pool_size` and `stride`.

### `neural_net.py`

Defines high-level components for neural network construction, training, and regularization.

*   **`Layer`**: Represents a single fully connected layer with weights, biases, and a customizable activation function (e.g., `relu`, `softmax`, `tanh`) dynamically accessed via `getattr`.
*   **`Dense`**: A multi-layered dense block, composed of multiple `Layer` instances, allowing for easy creation of deep fully connected networks.
*   **`Dropout`**: A regularization technique to prevent overfitting by randomly setting a fraction of input units to zero during training.
*   **`Model`**: The central class for defining, training, and evaluating neural networks. It wraps a `Sequence` of blocks, manages an `optimizer`, and handles the training loop, batching, and integration with various `loss_fn` (e.g., `cross_entropy_loss`, `mse_loss`) and `pen_fn` (e.g., `l1_reg`, `l2_reg`).
*   **Loss Functions**: `cross_entropy_loss` and `mse_loss` are provided for common supervised learning tasks.
*   **Penalty Functions**: `l1_reg` (L1 regularization) and `l2_reg` (L2 regularization) are available to apply penalties to model weights during training, helping to prevent overfitting.

### `optimizer.py`

Implements various parameter update algorithms to minimize the loss function during training.

*   **`Optimizer` (Base Class)**: Defines the common interface for all optimizers.
*   **`Optim_SGD`**: Stochastic Gradient Descent with an optional learning rate scheduler (`finitters`, `fin_l_rate`).
*   **`Optim_SGD_Momentum`**: SGD with momentum to accelerate convergence and dampen oscillations.
*   **`Optim_AdaGrad`**: Adaptive Gradient algorithm, which scales learning rates for each parameter individually.
*   **`Optim_RMSPropclass`**: RMSProp optimizer, an adaptive learning rate method that divides the learning rate by an exponentially decaying average of squared gradients.
*   **`Optim_Adam`**: Adaptive Moment Estimation, combining ideas from Momentum and RMSProp for efficient optimization.

### `sequence.py`

Provides the `Sequence` container to stack layers and manage parameter retrieval and setting across the model. It allows for a straightforward, sequential construction of neural networks.

### `attention.py`

This module implements mechanisms for attention, crucial for modern sequence models like Transformers.

*   **`Embedding`**: Converts discrete tokens (elements) into dense vector representations. It manages a vocabulary (`encoder`, `decoder`) and provides functionality to `encode` elements to indices, `decode` indices back to elements, and generate embeddings for input sequences. It supports both raw elements and pre-encoded indices as input.
*   **`AttentionHead`**: The core component of self-attention. It projects input features into Query (Q), Key (K), and Value (V) representations, calculates attention scores (`S`) via dot product, normalizes them using `softmax` to get `A` (attention weights), and computes the weighted sum of Value vectors (`Y`).
*   **`AttentionMultiHead`**: Combines multiple `AttentionHead` instances in parallel. It concatenates the outputs of individual heads and projects them through a final linear layer (`w_0`), allowing the model to jointly attend to information from different representation subspaces at different positions.

### `save_load_funcs.py`

This module provides utility functions for saving and loading model parameters, enabling persistence of trained models.

*   **`getModelSave`**: Extracts the numerical values of all trainable parameters (weights and biases) from a given model and converts them into a list of standard Python lists (or NumPy arrays), suitable for serialization (e.g., to JSON, Pickle, or other formats).
*   **`modelLoadParams`**: Takes a list of parameter matrices (either NumPy arrays or `Values` objects) and loads them back into a `Model` instance, setting the model's weights and biases to the provided values. This allows for reloading previously saved model states.

## Dependencies

The package relies heavily on `numpy` for numerical operations and gradient tracking.
