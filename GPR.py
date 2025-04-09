import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, 
                            r2_score, 
                            mean_absolute_error,
                            mean_absolute_percentage_error)

# Load dataset
data = pd.read_csv('Data-PreparedByMorteza.csv')  # Ensure columns match your data

# Feature Engineering
def engineer_features(df):
    # Original features
    X = df[['Phi', 'P', 'MassFlow', 'N2frac']].values
    
    # Feature transformations
    log_P = np.log(df['P'].values).reshape(-1, 1)
    N2_O2_ratio = (df['N2frac'] / 0.21).values.reshape(-1, 1)  # Assuming air is 21% O2
    
    # Interaction terms
    phi_P = (df['Phi'] * df['P']).values.reshape(-1, 1)
    phi_N2 = (df['Phi'] * df['N2frac']).values.reshape(-1, 1)
    P_N2 = (df['P'] * df['N2frac']).values.reshape(-1, 1)
    
    # Combine all features
    X_engineered = np.hstack([X, log_P, N2_O2_ratio, phi_P, phi_N2, P_N2])
    return X_engineered

# Target processing (log transform for radical species)
def process_targets(df):
    targets = []
    target_names = ['T', 'H2', 'H', 'O', 'O2', 'OH', 'H2O', 'CO', 'CO2', 'NO', 'NO2', 'N2']
    
    for name in target_names:
        if name in ['H', 'O', 'OH', 'NO', 'NO2']:  # Radicals and pollutants
            targets.append(np.log1p(df[name].values).reshape(-1, 1))
        else:  # Major species and temperature
            targets.append(df[name].values.reshape(-1, 1))
    
    y = np.hstack(targets)
    return y, target_names

# Prepare data
X = engineer_features(data)
y, target_names = process_targets(data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Target scaling (separate scaler for each output)
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Define composite kernel
kernel = (ConstantKernel(1.0) * RBF(length_scale=[1.0]*X_train_scaled.shape[1]) 
          + Matern(length_scale=[1.0]*X_train_scaled.shape[1], nu=2.5) 
          + WhiteKernel(noise_level=0.1))

# Create and train GPR model
gpr = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-5,
    n_restarts_optimizer=10,
    normalize_y=True
)

gpr.fit(X_train_scaled, y_train_scaled)

# Predictions
y_pred_scaled, y_std_scaled = gpr.predict(X_test_scaled, return_std=True)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Inverse log transform for radical species
for i, name in enumerate(target_names):
    if name in ['H', 'O', 'OH', 'NO', 'NO2']:
        y_pred[:, i] = np.expm1(y_pred[:, i])
        y_test[:, i] = np.expm1(y_test[:, i])

# Calculate uncertainty bounds
y_lower = scaler_y.inverse_transform(y_pred_scaled - 1.96 * y_std_scaled)
y_upper = scaler_y.inverse_transform(y_pred_scaled + 1.96 * y_std_scaled)

# Enhanced Evaluation Metrics
def evaluate_model(y_true, y_pred, target_names):
    results = {}
    for i, name in enumerate(target_names):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        mape = mean_absolute_percentage_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        
        # Calculate precision (1 - coefficient of variation)
        std_dev = np.std(y_pred[:, i])
        mean_val = np.mean(y_pred[:, i])
        precision = 1 - (std_dev / (mean_val + 1e-10))  # Small epsilon to avoid division by zero
        
        results[name] = {
            'MAE': mae,
            'MAPE (%)': mape * 100,
            'RMSE': rmse,
            'R²': r2,
            'Precision': precision
        }
    return results

# Get evaluation metrics
metrics = evaluate_model(y_test, y_pred, target_names)

# Print comprehensive results
print("\n=== Model Performance Metrics ===")
print(f"{'Target':<6}{'MAE':>10}{'MAPE (%)':>12}{'RMSE':>12}{'R²':>12}{'Precision':>12}")
for target in target_names:
    print(f"{target:<6}{metrics[target]['MAE']:>10.4f}{metrics[target]['MAPE (%)']:>12.2f}"
          f"{metrics[target]['RMSE']:>12.4f}{metrics[target]['R²']:>12.4f}"
          f"{metrics[target]['Precision']:>12.4f}")

# Feature importance analysis (length scales)
print("\nOptimized kernel parameters:")
print(gpr.kernel_)

# Visualization: Actual vs Predicted for each target
plt.figure(figsize=(18, 15))
for i, name in enumerate(target_names):
    plt.subplot(4, 3, i+1)
    plt.scatter(y_test[:, i], y_pred[:, i], alpha=0.5)
    plt.plot([y_test[:, i].min(), y_test[:, i].max()], 
             [y_test[:, i].min(), y_test[:, i].max()], 'k--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{name} (R² = {metrics[name]["R²"]:.3f})')
plt.tight_layout()
plt.savefig('actual_vs_predicted_all_params.png', dpi=300)
plt.show()

# Uncertainty visualization
plt.figure(figsize=(18, 15))
for i, name in enumerate(target_names):
    plt.subplot(4, 3, i+1)
    plt.errorbar(range(len(y_test[:, i])), y_pred[:, i], 
                 yerr=(y_pred[:, i] - y_lower[:, i]), 
                 fmt='o', alpha=0.5, capsize=3)
    plt.plot(y_test[:, i], 'r.', markersize=4)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title(f'{name} with 95% CI')
plt.tight_layout()
plt.savefig('uncertainty_bounds_all_params.png', dpi=300)
plt.show()