# Usage example:
if __name__ == "__main__":
    from neuro import NeuralNetwork
    import numpy as np

    # Create a neural network with 2 input neurons, 5 neurons in two hidden layers, and 1 output neuron
    layer_sizes = [2, 5, 5, 1]
    model = NeuralNetwork(layer_sizes)

    # Example training data (XOR problem)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
    y = np.array([[0, 1, 1, 0]])

    # Train the model
    model.train(X, y, learning_rate=0.1, epochs=2000)

    # Make predictions
    predictions = model.predict(X)
    print("Predictions:", predictions)
