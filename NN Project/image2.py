import matplotlib.pyplot as plt

"""
Draws a neural network diagram with the given layer sizes.
------------
Parameters:
layer_sizes: A list of integers representing the
number of nodes in each layer of the network.
------------
Methods:
draw_neural_network(layer_sizes)
------------
Returns:
None
"""

def draw_neural_network(layer_sizes, max_nodes_per_layer=20):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.axis('off')

    # Compute layer positions
    v_spacing = 1.0 / (max(min(max_nodes_per_layer, max(layer_sizes)), 1) + 1)
    h_spacing = 1.0 / (len(layer_sizes) + 1)

    # Nodes
    for i, layer_size in enumerate(layer_sizes):
        layer_size = min(layer_size, max_nodes_per_layer)
        layer_top = (1 - (layer_size - 1) * v_spacing) / 2
        for j in range(layer_size):
            circle = plt.Circle((i * h_spacing + h_spacing, layer_top + j * v_spacing), v_spacing / 4.0, color='b', ec='k', zorder=4)
            ax.add_artist(circle)
    
    # Edges
    for i, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_size_a = min(layer_size_a, max_nodes_per_layer)
        layer_size_b = min(layer_size_b, max_nodes_per_layer)
        layer_top_a = (1 - (layer_size_a - 1) * v_spacing) / 2
        layer_top_b = (1 - (layer_size_b - 1) * v_spacing) / 2
        for j in range(layer_size_a):
            for k in range(layer_size_b):
                line = plt.Line2D([i * h_spacing + h_spacing, (i + 1) * h_spacing + h_spacing],
                                  [layer_top_a + j * v_spacing, layer_top_b + k * v_spacing], c='k', alpha=0.1)
                ax.add_artist(line)

    plt.title(f'Neural Network Architecture: {"x".join(map(str, layer_sizes))}')
    plt.show()

if __name__ == "__main__":
    from neuro import NeuralNetwork

    # Example usage:
    layer_sizes = [2, 64, 64, 64, 64, 2]
    model = NeuralNetwork(layer_sizes)
    
    # Draw the neural network
    draw_neural_network(layer_sizes)
