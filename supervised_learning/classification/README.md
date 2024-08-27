# Neural Networks Binary Classification

## Overview
This project implements a simple neural network for binary classification using Python and NumPy. The neural network is designed to classify data into one of two categories.

## Project Structure
- `supervised_learning/classification/5-neuron.py`: Contains the implementation of the `Neuron` class with methods for forward propagation, cost calculation, evaluation, and gradient descent.
- `supervised_learning/classification/6-neuron.py`: Extends the `Neuron` class with a training method to iteratively improve the model.
- `data/`: Directory containing training and testing datasets in `.npz` format.

## Neuron Class
The `Neuron` class is the core of the neural network. It includes:
- **Attributes**:
  - `W`: Weights vector.
  - `b`: Bias.
  - `A`: Activated output.
- **Methods**:
  - `__init__(self, nx)`: Initializes the neuron with `nx` input features.
  - `forward_prop(self, X)`: Computes the forward propagation of the neuron.
  - `cost(self, Y, A)`: Calculates the cost using logistic regression.
  - `evaluate(self, X, Y)`: Evaluates the neuron's predictions.
  - `gradient_descent(self, X, Y, A, alpha)`: Performs one pass of gradient descent on the neuron.
  - `train(self, X, Y, iterations, alpha)`: Trains the neuron over a specified number of iterations.

## Usage
1. **Initialize the Neuron**:
   ```python
   neuron = Neuron(nx)
   ```
   - `nx`: Number of input features.

2. **Forward Propagation**:
   ```python
   A = neuron.forward_prop(X)
   ```
   - `X`: Input data.

3. **Cost Calculation**:
   ```python
   cost = neuron.cost(Y, A)
   ```
   - `Y`: True labels.
   - `A`: Activated output.

4. **Evaluation**:
   ```python
   predictions, cost = neuron.evaluate(X, Y)
   ```

5. **Training**:
   ```python
   neuron.train(X, Y, iterations, alpha)
   ```
   - `iterations`: Number of training iterations.
   - `alpha`: Learning rate.

## Example
```python
import numpy as np
from supervised_learning.classification import Neuron

# Load data
lib_train = np.load('data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

# Initialize and train neuron
neuron = Neuron(X.shape[0])
neuron.train(X, Y, iterations=5000, alpha=0.05)

# Evaluate performance
predictions, cost = neuron.evaluate(X, Y)
print(f'Cost: {cost}')
print(f'Predictions: {predictions}')
```
