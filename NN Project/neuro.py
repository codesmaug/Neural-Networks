import numpy as np

class NeuralNetwork:
    """
    Neural Network Model for Classification
    ---------------------------------------
    Parameters
    ----------
    layers_size : list
        List containing the number of nodes in each layer, including the input layer and output layer.
    
    Methods
    -------
    forward_propagation(X):
        Computes the forward pass through the network.
    
    train(X, y, learning_rate=0.01, epochs=1000):
        Trains the neural network using the provided training data.
    
    predict(X):
        Predicts the output for the given input data.
    
    Returns
    -------
    self : object
    """
    
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i-1]) * np.sqrt(2 / layer_sizes[i-1]) 
                        for i in range(1, self.num_layers)]
        self.biases = [np.zeros((size, 1)) for size in layer_sizes[1:]]

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def forward_propagation(self, x):
        a = x
        activations = [x]
        zs = []
        for W, b in zip(self.weights, self.biases):
            z = np.dot(W, a) + b
            zs.append(z)
            a = self.relu(z) if W is not self.weights[-1] else self.sigmoid(z)
            activations.append(a)
        return activations, zs

    def backward_propagation(self, y, activations, zs):
        delta = activations[-1] - y
        dW = [np.dot(delta, activations[-2].T)]
        db = [np.mean(delta, axis=1, keepdims=True)]
        delta = np.dot(self.weights[-1].T, delta) * self.relu_derivative(zs[-2])
        for i in range(2, self.num_layers):
            dW.insert(0, np.dot(delta, activations[-i-1].T))
            db.insert(0, np.mean(delta, axis=1, keepdims=True))
            if i < self.num_layers - 1:
                delta = np.dot(self.weights[-i].T, delta) * self.relu_derivative(zs[-i-1])
        return dW, db

    def update_parameters(self, dW, db, learning_rate):
        for i in range(self.num_layers - 1):
            self.weights[i] -= learning_rate * dW[i]
            self.biases[i] -= learning_rate * db[i]

    def train(self, X, y, learning_rate=0.01, epochs=1000):
        m = X.shape[1]
        for epoch in range(epochs):
            activations, zs = self.forward_propagation(X)
            dW, db = self.backward_propagation(y, activations, zs)
            self.update_parameters(dW, db, learning_rate)
            
            # Print cost (optional)
            if epoch % 100 == 0:
                cost = np.mean(np.square(activations[-1] - y))
                print(f"Epoch {epoch}: Cost = {cost}")

    def predict(self, X):
        activations, _ = self.forward_propagation(X)
        return activations[-1]


