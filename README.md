# Predicting Hydrogen Combustion Behavior Using Advanced Machine Learning Models

This repository provides the code and dataset for predicting equilibrium combustion products and temperature of hydrogen/air mixtures using three machine learning architectures: Deep Neural Network (DNN), Multi-Layer Perceptron (MLP), and Gaussian Process Regression (GPR).

---

## Dataset Documentation

### Overview

The dataset (`Data-PreparedByMorteza.csv`) contains thermochemical simulation data for hydrogen combustion under varying operating conditions. Each row corresponds to one combustion simulation case with defined inlet conditions and the resulting equilibrium species concentrations and temperature.

**Total features used in models:** 4 input variables + 7 engineered features (see below)  
**Total output targets:** 12

---

### Input Features

| Column Name | Description | Unit | Typical Range |
|-------------|-------------|------|---------------|
| `Phi` | Equivalence ratio (fuel-to-air ratio normalized by stoichiometry) | Dimensionless | 0.3 – 3.0 |
| `P` | Combustion pressure | atm (or Pa — see note) | 1 – 50 atm |
| `MassFlow` | Reactant mass flow rate | kg/s | 0.001 – 1.0 |
| `N2frac` | Mole fraction of N₂ in the oxidizer stream | Dimensionless (0–1) | 0.0 – 0.79 |

> **Note on `Phi`:** Values below 1.0 indicate lean mixtures (excess air); values above 1.0 indicate rich mixtures (excess fuel); Phi = 1.0 is stoichiometric.

> **Note on `N2frac`:** Standard air contains approximately 0.79 N₂ by mole fraction. Values below this imply oxygen-enriched oxidizers; values above 0.79 imply nitrogen-diluted oxidizers.

---

### Engineered Features (Generated Internally in Code)

The following features are derived from the raw inputs inside `DNN.py` / `MLP.py` and are not columns in the CSV, but are part of the model input vector:

| Engineered Feature | Formula | Physical Meaning |
|--------------------|---------|-----------------|
| `log_P` | ln(P) | Log-scaled pressure to handle nonlinear pressure dependence |
| `N2_O2_ratio` | N2frac / 0.21 | Ratio of N₂ to nominal O₂ fraction in air |
| `Phi × P` | Phi · P | Interaction term capturing pressure-equivalence ratio coupling |
| `Phi × N2frac` | Phi · N2frac | Interaction term for dilution effect at varying richness |
| `P × N2frac` | P · N2frac | Interaction term for pressure effect under diluted oxidizer |

The full input vector to the models is therefore of dimension **9** (4 raw + 5 engineered features; note `log_P` and `N2_O2_ratio` each add 1).

---

### Output Targets

The model simultaneously predicts 12 combustion output quantities. Units and value ranges are defined in the table below.

| Target Name | Description | Unit | Approximate Range | Preprocessing |
|-------------|-------------|------|-------------------|---------------|
| `T` | Adiabatic flame temperature | K (Kelvin) | 300 – 3500 K | None (linear) |
| `H2` | Mole fraction of molecular hydrogen | Dimensionless (mol/mol) | 0.0 – 0.6 | None (linear) |
| `H` | Mole fraction of atomic hydrogen radical | Dimensionless (mol/mol) | 0.0 – 0.05 | log1p transform |
| `O` | Mole fraction of atomic oxygen radical | Dimensionless (mol/mol) | 0.0 – 0.02 | log1p transform |
| `O2` | Mole fraction of molecular oxygen | Dimensionless (mol/mol) | 0.0 – 0.40 | None (linear) |
| `OH` | Mole fraction of hydroxyl radical | Dimensionless (mol/mol) | 0.0 – 0.05 | log1p transform |
| `H2O` | Mole fraction of water vapor | Dimensionless (mol/mol) | 0.0 – 0.50 | None (linear) |
| `CO` | Mole fraction of carbon monoxide | Dimensionless (mol/mol) | ~0.0 (trace) | None (linear) |
| `CO2` | Mole fraction of carbon dioxide | Dimensionless (mol/mol) | ~0.0 (trace) | None (linear) |
| `NO` | Mole fraction of nitric oxide | Dimensionless (mol/mol) | 0.0 – 0.005 | log1p transform |
| `NO2` | Mole fraction of nitrogen dioxide | Dimensionless (mol/mol) | 0.0 – 0.001 | log1p transform |
| `N2` | Mole fraction of molecular nitrogen | Dimensionless (mol/mol) | 0.0 – 0.79 | None (linear) |

> **Note on log1p transform:** Targets for trace radicals (`H`, `O`, `OH`, `NO`, `NO2`) are internally transformed using `log1p(x) = ln(1 + x)` during training to improve numerical stability and model accuracy for small-valued species. Predictions are back-transformed using `expm1` before reporting metrics.

> **Note on CO/CO2:** Since the fuel is pure hydrogen (no carbon), CO and CO2 are expected to be near zero throughout the dataset. They are included for completeness and generalizability.

---

### Data Format

The dataset is stored as a CSV file with the first row as a header. All values are numeric. Missing values were dropped during preprocessing (`dropna()`).

**File:** `dataset/Data-PreparedByMorteza.csv`

**Example structure:**

```
Phi, P, MassFlow, N2frac, T, H2, H, O, O2, OH, H2O, CO, CO2, NO, NO2, N2
0.5, 1.0, 0.01, 0.79, 1820.3, 0.000, 0.00012, 0.00008, 0.152, 0.00310, 0.321, 0.0, 0.0, 0.00041, 0.00002, 0.526
...
```

---

### Data Splitting and Normalization

- **Train/Test split:** 80% training, 20% testing (`random_state=42`)
- **Train/Validation split:** 90% train, 10% validation (from the training set)
- **Input normalization:** MinMaxScaler applied to inputs (fit on training set only, applied to test set)
- **Output normalization:** MinMaxScaler applied to outputs (fit on training set only, applied to test set)

---

## Repository Structure

```
├── dataset/
│   └── Data-PreparedByMorteza.csv   # Full simulation dataset
├── DNN.py                            # Deep Neural Network model
├── MLP.py                            # Multi-Layer Perceptron model
├── GPR.py                            # Gaussian Process Regression model
├── LICENSE
└── README.md
```

---

## Model Architectures

### DNN (DNN.py)
- 4 fully connected layers: 256 → 128 → 64 → output
- Batch Normalization + Dropout (0.3, 0.3, 0.2) after each hidden layer
- Optimizer: Adam (lr=0.001)
- Loss: Mean Squared Error
- Callbacks: EarlyStopping (patience=20), ReduceLROnPlateau (factor=0.5, patience=10)
- Max epochs: 500, batch size: 64

### MLP (MLP.py)
- Standard multi-layer perceptron variant (see source for architecture details)

### GPR (GPR.py)
- Gaussian Process Regression (non-parametric probabilistic model)
- Suitable for smaller subsets of the data due to O(n³) complexity

---

## Reproducibility

To reproduce results:

1. Clone the repository
2. Install dependencies: `pip install numpy pandas tensorflow scikit-learn matplotlib`
3. Place `Data-PreparedByMorteza.csv` in the working directory (or adjust the path in each script)
4. Run any model script: `python DNN.py`

All random seeds are fixed (`random_state=42`) in train/test splitting to ensure reproducible data partitions.

---

## Citation

If you use this dataset or code, please cite the associated paper (citation to be added upon publication).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
