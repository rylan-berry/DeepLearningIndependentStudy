# Deeplearning Package

## Overview

This package is designed to be similar to the PyTorch system of a building block system. Providing the functions that can be mixed, matched, and customized as pleased for any given model. This library is bare bones and only includes the few methods and ideas I learned about while studying *Deep Learning* by Ian Goodfellow et. al.. AI was used in the project, but it was used sparingly.

## Modules

This project has four main modules:

* `autogradient.py`  
* `sequence.py`
* `optimizer.py`
* `neural_net.py`

All of which are automatically part of the initial import of the package.

## Detailed Module Descriptions

### `autogradient.py`

This module forms the core of the automatic differentiation system, enabling the computation of gradients for complex mathematical operations. It introduces the `Values` class, which is central to tracking computational history.

*   **`Values` Class**: Encapsulates numerical values (`vals`) and their corresponding gradients (`grad`). It supports a wide array of arithmetic and mathematical operations (e.g., `+`, `-`, `*`, `/`, `@`, `exp`, `log`, `relu`, `abs`, `sum`, `softmax`, `mean`, `__pow__`, `__getitem__`, `T` for transpose). Each operation automatically builds a computational graph by defining a `_backward` function. A static method, `_broadcast_grad`, meticulously handles gradient broadcasting to correctly match original tensor shapes. The `backward()` method then leverages this graph to efficiently compute and propagate gradients.

### `sequence.py`

This module provides a mechanism to combine multiple operations or layers into a single, cohesive sequential model.

*   **`Sequence` Class**: Designed to take a list of callable objects (such as `Layer`, `Dense`, or `Dropout` instances). Its `__call__` method ensures that input data is passed through each item in the sequence in the defined order. The `params()` method is crucial for model training, as it gathers all trainable parameters (weights and biases) from its constituent layers, making them accessible to optimizers.

### `optimizer.py`

The `optimizer.py` module implements various algorithms used to update neural network parameters based on their computed gradients, facilitating the learning process.

*   **`Optimizer` Base Class**: Serves as the blueprint for all optimization algorithms, providing a `step` method that each specific optimizer overrides.
*   **Subclasses**: The module includes several widely used optimizers:
    *   `Optim_SGD`: Implements Stochastic Gradient Descent, with an optional learning rate scheduler that adjusts the rate over time.
    *   `Optim_SGD_Momentum`: Extends SGD by incorporating momentum, which helps accelerate convergence by considering an exponentially weighted average of past gradients.
    *   `Optim_AdaGrad`: An adaptive learning rate optimizer that adjusts the learning rate for each parameter individually based on the historical sum of its squared gradients.
    *   `Optim_RMSPropclass`: Similar to AdaGrad, RMSProp uses a moving average of squared gradients to normalize the learning rate, helping to mitigate issues with vanishing or exploding gradients.
    *   `Optim_Adam`: A powerful and popular optimizer that combines elements of both Momentum and RMSProp, utilizing moving averages of both the gradients and the squared gradients to provide efficient and robust parameter updates.

### `neural_net.py`

This module defines the fundamental building blocks for constructing neural networks, including various layers, network architectures, regularization techniques, loss functions, and a comprehensive model training framework.

*   **`Layer` Class**: Represents a single fully connected layer, equipped with trainable weights and biases. It supports different activation functions (e.g., `'relu'`, `'softmax'`), which can be specified as strings and dynamically called. The `params()` method provides access to its weights and biases for optimization.
*   **`Dense` Class**: Facilitates the creation of multi-layered perceptrons by stacking multiple `Layer` instances. It allows for detailed configuration, including the number of layers, sizes of input, middle, and output layers, and distinct activation functions for hidden and final layers.
*   **`Dropout` Class**: Implements the dropout regularization technique, a method to prevent overfitting during training. During training, it randomly sets a fraction of input units to zero at each update, while scaling up the remaining activations to maintain the expected output.
*   **Loss Functions**: Essential for quantifying the error of a model. The module includes:
    *   `cross_entropy_loss`: Calculates the categorical cross-entropy loss, primarily used for classification tasks.
    *   `mse_loss`: Computes the Mean Squared Error loss, a common choice for regression problems.
*   **`Model` Class**: Serves as the central orchestrator for the neural network training process. It takes a list of `blocks` (e.g., `Layer`, `Dense`, `Dropout` instances) to define the network's architecture. It integrates an `optimizer`, a `loss_fn`, and an optional `pen_fn` (penalty function) for regularization. The `train` method manages the entire training loop, including batching, forward passes, loss calculation, backward propagation (leveraging `autogradient`), and parameter updates via the chosen optimizer.
*   **Penalty Functions (Regularization)**: These functions are designed to prevent overfitting by adding a penalty term to the loss function.
    *   `l2_reg`: Implements L2 regularization (also known as weight decay), which adds a penalty proportional to the sum of the squared values of the weights.
    *   `l1_reg`: Implements L1 regularization, which adds a penalty proportional to the sum of the absolute values of the weights.

## Making and Running a Model

When creating a model, use the Model class, which runs most of the functions included in the package itself. The first argument is a list of layers or blocks, each element is the steps in the network. These steps can be a Dense, Layer, or Dropout blocks (more will be made), a Dense is just multiple layers stacked back to back.   
Training a model is done through: `def train(epochs, x_t, y_t, x_v, y_v, val_run=1, l_rate=0.01, _lambda=0.1, batch_size = None) ` 
Where epochs is the number of times you train through the data, the `#_t` means training data and `#_v` means validation data, `x` means input, `y` means output, `val_run` is the epochs between when you want to test the validation data, `l_rate` is the learn rate, `_lambda` is a hyperparameter that determines the strength of the penalty functions, and `batch_size` determines how large batches will be (if the batch size isn’t a multiple of the data size then it will still run, there is just a smaller batch then the others). 

## Dependencies

The auto gradient–which is used for back propagation–relies heavily on `numpy`.
