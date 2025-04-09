import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 4))
ax = plt.gca()

# Layer configurations
layers = [4, 256, 128, 64, 12]  # Number of neurons per layer
layer_labels = ['Input\n(Φ, P, ...)', 'Hidden 1\n(256)', 'Hidden 2\n(128)', 'Hidden 3\n(64)', 'Output\n(T, H2, ...)']
colors = ['#66b3ff', '#99ff99', '#99ff99', '#99ff99', '#ff9999']  # Blue → Green → Red

# Draw layers as blocks (simplified representation)
for i, (n_neurons, label, color) in enumerate(zip(layers, layer_labels, colors)):
    # Draw a rectangle for each layer
    rect = plt.Rectangle((i - 0.4, 0.1), 0.8, 0.8, fc=color, ec='k', alpha=0.7)
    ax.add_patch(rect)
    plt.text(i, -0.1, label, ha='center', va='top', fontsize=9)

# Arrows between layers
for i in range(len(layers) - 1):
    plt.arrow(i + 0.2, 0.5, 0.6, 0, head_width=0.05, head_length=0.05, fc='k', ec='k')

# Annotations
plt.text(2, 1.1, 'ReLU Activation\nL₂ Regularization (α=0.001)', ha='center', fontsize=8)
plt.text(2, -0.3, 'Adam Optimizer (lr=0.001)', ha='center', fontsize=8)

plt.xlim(-0.5, len(layers) - 0.5)
plt.ylim(-0.5, 1.5)
plt.axis('off')
# plt.title('Simplified MLP Architecture for Combustion Modeling', fontsize=10)
plt.tight_layout()
plt.savefig('mlp_simplified.png', dpi=300, bbox_inches='tight')
plt.show()