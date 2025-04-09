import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load and clean data
data = pd.read_csv('Data-PreparedByMorteza.csv')

# 1. Check for and handle NaN values
print(f"NaN values before cleaning: {data.isna().sum().sum()}")
data = data.dropna()  # Remove rows with NaN values
print(f"NaN values after cleaning: {data.isna().sum().sum()}")

def engineer_features(df):
    """Create physics-informed features for combustion prediction"""
    X = df[['Phi', 'P', 'MassFlow', 'N2frac']].values
    log_P = np.log(df['P'].values).reshape(-1, 1)
    N2_O2_ratio = (df['N2frac'] / 0.21).values.reshape(-1, 1)
    interaction_terms = np.hstack([
        (df['Phi'] * df['P']).values.reshape(-1, 1),
        (df['Phi'] * df['N2frac']).values.reshape(-1, 1),
        (df['P'] * df['N2frac']).values.reshape(-1, 1)
    ])
    return np.hstack([X, log_P, N2_O2_ratio, interaction_terms])

def process_targets(df):
    """Process target variables with appropriate transforms"""
    targets = []
    target_names = ['T', 'H2', 'H', 'O', 'O2', 'OH', 'H2O', 'CO', 'CO2', 'NO', 'NO2', 'N2']
    
    for name in target_names:
        if name in ['H', 'O', 'OH', 'NO', 'NO2']:  # Radicals and pollutants
            targets.append(np.log1p(df[name].values).reshape(-1, 1))
        else:  # Major species and temperature
            targets.append(df[name].values.reshape(-1, 1))
    
    return np.hstack(targets), target_names

# Prepare data
X = engineer_features(data)
y, target_names = process_targets(data)

# 2. Train-test split (without stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Target scaling (per-output)
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# MLP Model Configuration - CORRECTED ACTIVATION FUNCTION
mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64),  # 3 hidden layers
    activation='relu',                  # Changed from 'swish' to standard 'relu'
    solver='adam',
    learning_rate_init=0.001,
    max_iter=500,
    batch_size=64,
    early_stopping=True,
    validation_fraction=0.1,
    alpha=0.001,                        # L2 regularization
    random_state=42
)

# Training
print("\nTraining MLP model...")
history = mlp.fit(X_train_scaled, y_train_scaled)

# Predictions
y_pred_scaled = mlp.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Inverse transform for log-scaled targets
for i, name in enumerate(target_names):
    if name in ['H', 'O', 'OH', 'NO', 'NO2']:
        y_pred[:, i] = np.expm1(y_pred[:, i])
        y_test[:, i] = np.expm1(y_test[:, i])

# Evaluation
def evaluate_performance(y_true, y_pred, targets):
    """Calculate comprehensive performance metrics including MAPE and Precision"""
    metrics = {}
    for i, name in enumerate(targets):
        # Avoid division by zero in MAPE for targets that can be zero
        mask = y_true[:, i] != 0 if name in ['H', 'O', 'OH', 'NO', 'NO2'] else np.ones_like(y_true[:, i], dtype=bool)
        y_true_masked = y_true[:, i][mask]
        y_pred_masked = y_pred[:, i][mask]
        
        # Standard metrics
        rmse = np.sqrt(mean_squared_error(y_true_masked, y_pred_masked))
        mae = mean_absolute_error(y_true_masked, y_pred_masked)
        r2 = r2_score(y_true_masked, y_pred_masked)
        
        # MAPE (handle zeros carefully)
        if len(y_true_masked) > 0:
            mape = np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100
        else:
            mape = np.nan
        
        # Precision (fraction of predictions within 10% of true values)
        if len(y_true_masked) > 0:
            threshold = 0.1 * np.abs(y_true_masked)  # 10% error margin
            precision = np.mean(np.abs(y_true_masked - y_pred_masked) <= threshold) * 100
        else:
            precision = np.nan
        
        metrics[name] = {
            'MAE': mae,
            'MAPE (%)': mape,
            'RMSE': rmse,
            'R²': r2
        }
    return metrics

metrics = evaluate_performance(y_test, y_pred, target_names)

# Print performance summary
print("\nMLP Performance Summary:")
print(f"{'Target':<6}{'MAE':>12}{'MAPE (%)':>12}{'RMSE':>12}{'R²':>12}")
for target in target_names:
    print(f"{target:<6}{metrics[target]['MAE']:>12.4f}{metrics[target]['MAPE (%)']:>12.2f}{metrics[target]['RMSE']:>12.4f}{metrics[target]['R²']:>12.4f}")

# Visualization
plt.figure(figsize=(15, 12))
for i, name in enumerate(target_names[:12]):  # Plot first 9 targets
    plt.subplot(3, 4, i+1)
    plt.scatter(y_test[:, i], y_pred[:, i], alpha=0.6)
    plt.plot([min(y_test[:, i]), max(y_test[:, i])], 
             [min(y_test[:, i]), max(y_test[:, i])], 'k--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{name} (R²={metrics[name]["R²"]:.3f})')
plt.tight_layout()
plt.savefig('mlp_predictions.png', dpi=300)
plt.show()