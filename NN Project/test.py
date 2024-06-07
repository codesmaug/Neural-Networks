import numpy as np
from neuro import NeuralNetwork

# Create a neural network with 2 input neurons, 3 neurons in one hidden layer, and 1 output neuron
layers_size = [2, 3, 1]
model = NeuralNetwork(layers_size)

# Example training data (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
y = np.array([[0, 1, 1, 0]])

# Train the model
model.train(X, y, learning_rate=0.1, epochs=1000)

# Make predictions
predictions = model.predict(X)
print("Predictions:", predictions)
