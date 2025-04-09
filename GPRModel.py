import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

plt.figure(figsize=(12, 8))

# Model Architecture Components
plt.text(0.1, 0.8, "Input Features\n(Φ, P, MassFlow, N₂)", ha='center', va='center', 
         bbox=dict(facecolor='lightblue', alpha=0.5, boxstyle='round'))
plt.text(0.1, 0.6, "Feature Engineering\n• ln(P)\n• N₂/O₂ ratio\n• Interaction terms", 
         ha='center', va='center', bbox=dict(facecolor='lightgreen', alpha=0.5, boxstyle='round'))

# Kernel Function
plt.text(0.4, 0.7, "Composite Kernel\nRBF + Matérn (ν=2.5) + WhiteKernel", 
         ha='center', va='center', bbox=dict(facecolor='gold', alpha=0.5, boxstyle='round'))

# GPR Core
plt.text(0.7, 0.7, "Gaussian Process\n• Marginal likelihood optimization\n• Automatic relevance determination\n• Uncertainty quantification", 
         ha='center', va='center', bbox=dict(facecolor='salmon', alpha=0.5, boxstyle='round'))

# Outputs
plt.text(0.9, 0.8, "Temperature Prediction\n(T)", ha='center', va='center', 
         bbox=dict(facecolor='lightpink', alpha=0.5, boxstyle='round'))
plt.text(0.9, 0.6, "Species Concentrations\n(H₂, O₂, H₂O, CO, CO₂, NO, NO₂, N₂)\nRadicals (H, O, OH)", 
         ha='center', va='center', bbox=dict(facecolor='lightpink', alpha=0.5, boxstyle='round'))

# Arrows
arrows = [
    [(0.2, 0.8), (0.3, 0.7), "Feature\nTransformation"],
    [(0.5, 0.7), (0.6, 0.7), "Kernel\nLearning"],
    [(0.8, 0.7), (0.85, 0.75), "Mean\nPrediction"],
    [(0.8, 0.7), (0.85, 0.65), "Uncertainty\nBounds"]
]

for (x1, y1), (x2, y2), label in arrows:
    plt.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle="->", lw=1.5, color='k'))
    plt.text((x1+x2)/2, (y1+y2)/2, label, ha='center', va='center', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# Uncertainty Visualization
x = np.linspace(0.05, 0.35, 10)
y = 0.3 + 0.1*np.sin(10*x)
y_err = 0.05 + 0.03*np.random.rand(len(x))
plt.fill_between(x, y-y_err, y+y_err, color='gray', alpha=0.3)
plt.plot(x, y, 'r-', lw=2, label='Prediction ±95% CI')
plt.scatter([0.25], [0.25], c='blue', s=100, label='Training Data')

plt.text(0.2, 0.2, "Probabilistic Output\n(Mean ± Confidence Interval)", 
         ha='center', va='center')

plt.legend(loc='lower left')
plt.axis('off')
# plt.title("GPR Model Architecture for Hydrogen Combustion Prediction", pad=20, fontsize=14)
plt.tight_layout()
plt.savefig('gpr_model_architecture.png', dpi=300, bbox_inches='tight')
plt.show()