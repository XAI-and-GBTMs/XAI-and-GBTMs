import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import os
import random

def set_seed(seed=42):
    # 1. System level
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Required for deterministic algorithms in newer PyTorch/CUDA
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    
    # 2. Python & NumPy
    random.seed(seed)
    np.random.seed(seed)
    
    # 3. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    # 4. Hardware/CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Forces PyTorch to use deterministic algorithms (will error if one isn't available)
    # torch.use_deterministic_algorithms(True) 

set_seed(25492)

def create_submission(log_preds, filename):
    final_prices = np.expm1(log_preds)
    submission = pd.DataFrame({
        'Id': test_df.index,
        'SalePrice': final_prices
    })
    submission.to_csv(filename, index=False)
    print(f"Saved: {filename}")

# 1. Load Data
train_df = pd.read_csv('data/train.csv', index_col='Id')
test_df = pd.read_csv('data/test.csv', index_col='Id')

# Target Transformation (Log transform for RMSLE)
y_train = np.log1p(train_df['SalePrice'])
train_df = train_df.drop('SalePrice', axis=1)

# 2. Define Feature Groups
nom_cols = [
    'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig', 
    'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
    'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 
    'Foundation', 'Heating', 'CentralAir', 'Electrical', 'GarageType', 
    'MiscFeature', 'SaleType', 'SaleCondition'
]

num_cols = [
    'LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 
    '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 
    'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 
    'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 
    'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 
    'MiscVal', 'MoSold', 'YrSold', 'OverallQual', 'OverallCond'
]

ord_cols = [
    'LotShape', 'Utilities', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 
    'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 
    'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 
    'GarageCond', 'PavedDrive', 'PoolQC', 'Fence'
]

# 3. Manual Ordinal Mapping (Fixes the "Cannot use median strategy" error)
# We define all specific scales found in the data description
mappings = {
    'qual': {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0},
    'shape': {'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1},
    'util': {'AllPub':4, 'NoSewr':3, 'NoSeWa':2, 'ELO':1},
    'slope': {'Gtl':3, 'Mod':2, 'Sev':1},
    'exp': {'Gd':4, 'Av':3, 'Mn':2, 'No':1, 'NA':0},
    'fintype': {'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1, 'NA':0},
    'func': {'Typ':8, 'Min1':7, 'Min2':6, 'Mod':5, 'Maj1':4, 'Maj2':3, 'Sev':2, 'Sal':1},
    'garfin': {'Fin':3, 'RFn':2, 'Unf':1, 'NA':0},
    'paved': {'Y':3, 'P':2, 'N':1},
    'fence': {'GdPrv':4, 'MnPrv':3, 'GdWo':2, 'MnWw':1, 'NA':0}
}

# Apply mappings to specific columns
def apply_ord_mappings(df):
    df = df.copy()
    # Map by category
    for col in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']:
        df[col] = df[col].map(mappings['qual'])
    df['LotShape'] = df['LotShape'].map(mappings['shape'])
    df['Utilities'] = df['Utilities'].map(mappings['util'])
    df['LandSlope'] = df['LandSlope'].map(mappings['slope'])
    df['BsmtExposure'] = df['BsmtExposure'].map(mappings['exp'])
    df['BsmtFinType1'] = df['BsmtFinType1'].map(mappings['fintype'])
    df['BsmtFinType2'] = df['BsmtFinType2'].map(mappings['fintype'])
    df['Functional'] = df['Functional'].map(mappings['func'])
    df['GarageFinish'] = df['GarageFinish'].map(mappings['garfin'])
    df['PavedDrive'] = df['PavedDrive'].map(mappings['paved'])
    df['Fence'] = df['Fence'].map(mappings['fence'])
    # Fill remaining NaNs in ordinal columns with 0
    df[ord_cols] = df[ord_cols].fillna(0)
    return df

train_df_mapped = apply_ord_mappings(train_df)
test_df_mapped = apply_ord_mappings(test_df)

# 4. Neural Network Pipeline (train_df_nn)
nn_numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

nn_cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor_nn = ColumnTransformer(transformers=[
    ('num', nn_numeric_transformer, num_cols + ord_cols),
    ('nom', nn_cat_transformer, nom_cols)
])

train_df_nn = preprocessor_nn.fit_transform(train_df_mapped)
test_df_nn = preprocessor_nn.transform(test_df_mapped)

# 5. GBTM Pipeline (train_df_gbtm)
gbtm_numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

gbtm_cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor_gbtm = ColumnTransformer(transformers=[
    ('num', gbtm_numeric_transformer, num_cols),
    ('cat', gbtm_cat_transformer, nom_cols + ord_cols)
])


#actually numpy array objects
train_df_gbtm = preprocessor_gbtm.fit_transform(train_df_mapped)
test_df_gbtm = preprocessor_gbtm.transform(test_df_mapped)

# 6. Verify Outputs
print(f"NN Training Shape: {train_df_nn.shape}")
print(f"GBTM Training Shape: {train_df_gbtm.shape}")


# For the Neural Network
X_train_nn, X_val_nn, y_train_split, y_val_split = train_test_split(
    train_df_nn, y_train, test_size=0.2, random_state=42
)

# For the GBTM (Use the same random_state to keep the rows aligned!)
X_train_gbtm, X_val_gbtm, _, _ = train_test_split(
    train_df_gbtm, y_train, test_size=0.2, random_state=42
)


###train using nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 7. Convert NumPy arrays/Pandas Series to PyTorch Tensors
# We reshape y to (-1, 1) to match the output dimension of the network
X_train_tensor = torch.tensor(X_train_nn, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_split.values, dtype=torch.float32).view(-1, 1)

X_val_tensor = torch.tensor(X_val_nn, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_split.values, dtype=torch.float32).view(-1, 1)

# Create Datasets and DataLoaders
batch_size = 100  
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
validation_dataset = TensorDataset(X_val_tensor, y_val_tensor)

g = torch.Generator()
g.manual_seed(25491)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g
)

validation_loader = DataLoader(validation_dataset, shuffle=False, batch_size=batch_size)

# 8. Define the Model Architecture
def build_model(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.2), # Prevents the model from memorizing noise
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

# Using the actual number of features from our preprocessing
input_features = X_train_nn.shape[1] 
model = build_model(input_features)
model.to(device)

# 9. Loss and Optimizer
# XGBoost's reg:squarederror minimizes the squared differences, which is mathematically 
# identical to nn.MSELoss() in PyTorch. 
criterion = nn.MSELoss()
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)

max_epochs = 100

# Early stopping parameters
max_iter_since_best_val_perf = 10
iter_since_best_val_perf = 0

PATH = './best_model_housing_nn.pth'
best_validation_loss = 1e9

# 10. Training Loop
print("Starting Neural Network Training...")
train_rmse_history = []
val_rmse_history = []

for epoch in range(max_epochs):
    # 1. Training Phase (Dropout ACTIVE)
    model.train()
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels) 
        loss.backward() 
        optimizer.step()
    
    # 2. Evaluation Phase (Dropout INACTIVE, No Gradients)
    model.eval()
    
    # Calculate "Clean" Training RMSE
    clean_train_loss = 0.0
    with torch.no_grad():
        for data in train_loader: # Use train_loader but in eval mode
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            clean_train_loss += criterion(outputs, labels).item() * inputs.size(0)
    
    clean_train_loss /= len(train_loader.dataset)
    train_rmse_history.append(np.sqrt(clean_train_loss))

    # Calculate Validation RMSE
    running_val_loss = 0.0
    with torch.no_grad():
        for data in validation_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            running_val_loss += criterion(outputs, labels).item() * inputs.size(0)

    running_val_loss /= len(validation_loader.dataset)
    val_rmse_history.append(np.sqrt(running_val_loss))

    # 3. Checkpoint & Early Stopping
    if (running_val_loss < best_validation_loss):
        best_validation_loss = running_val_loss
        torch.save(model.state_dict(), PATH)
        iter_since_best_val_perf = 0
    else:
        iter_since_best_val_perf += 1
        if (iter_since_best_val_perf >= max_iter_since_best_val_perf):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break


# --- PLOT: Main NN Training vs Epochs ---
plt.figure(figsize=(8, 5))
plt.plot(np.exp(train_rmse_history), label='Train exp(RMSE)', color='blue')
plt.plot(np.exp(val_rmse_history), label='Validation exp(RMSE)', color='orange')
plt.title('Main Neural Network: exp(RMSE) vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('exp(RMSE)')
plt.ylim(0, 10)
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("training_and_val_loss_NN_with_dropout_240_features.png")

# 11. Load best model and evaluate
final_model = build_model(input_features)
final_model.load_state_dict(torch.load(PATH, weights_only=True))
final_model.to(device)
final_model.eval()

# Calculate final validation RMSE (which corresponds to Kaggle's RMSLE)
print(f"The trained best model achieves a MSE loss of {best_validation_loss:.5f} on the validation dataset")
print(f"Validation RMSE (Kaggle Metric): {np.sqrt(best_validation_loss):.5f}")




##############xgboost training
# 12. Initialize XGBoost Regressor (Updated for XGBoost 2.0+)
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,  # Moved here!
    n_jobs=-1,
    random_state=42
)

# 13. Train XGBoost (Updated .fit call)
print("\nStarting XGBoost Training...")
xgb_model.fit(
    X_train_gbtm, 
    y_train_split,
    eval_set=[(X_val_gbtm, y_val_split)],
    verbose=100
    # early_stopping_rounds=50  <-- Remove it from here
)

# 14. Evaluate XGBoost
xgb_val_pred = xgb_model.predict(X_val_gbtm)
xgb_val_rmse = np.sqrt(mean_squared_error(y_val_split, xgb_val_pred))

print(f"\nXGBoost Validation RMSE: {xgb_val_rmse:.5f}")



print("\nGenerating final test predictions...")

# --- Neural Network Predictions ---
final_model.eval()
with torch.no_grad():
    # Convert the processed test data to a tensor
    X_test_tensor = torch.tensor(test_df_nn, dtype=torch.float32).to(device)
    # Get predictions and move back to CPU/NumPy
    nn_test_log_preds = final_model(X_test_tensor).cpu().numpy().flatten()

# --- XGBoost Predictions ---
# XGBoost handles the numpy array (test_df_gbtm) directly
xgb_test_log_preds = xgb_model.predict(test_df_gbtm)

# 17. Ensemble (Weighted Average)
# Since XGBoost was stronger, we'll give it 75% of the vote
# This usually balances the 'smoothness' of the NN with the 'accuracy' of the Trees
final_log_preds = (1 * xgb_test_log_preds) + (0 * nn_test_log_preds)

# 18. Inverse Transform (Log -> Dollars)
# np.expm1 is the inverse of np.log1p
final_prices = np.expm1(final_log_preds)

# 19. Format and Save for Kaggle
# We use the original test_df index to ensure IDs match exactly
submission = pd.DataFrame({
    'Id': test_df.index,
    'SalePrice': final_prices
})

submission.to_csv('submission.csv', index=False)



# 15. Find the optimal Ensemble weight 'p' using the Validation Set
print("\nOptimizing ensemble weights on validation data...")

# Get NN predictions for validation set (already have xgb_val_pred)
final_model.eval()
with torch.no_grad():
    nn_val_pred = final_model(X_val_tensor.to(device)).cpu().numpy().flatten()

best_p = 0
min_val_rmse = float('inf')

p_values = []
ensemble_val_rmses = []

# Search for the best p between 0 and 1
for p in np.linspace(0, 1, 101):
    
    # Correcting the weighted average logic:
    ensemble_val_pred = (p * xgb_val_pred) + ((1 - p) * nn_val_pred)
    
    current_rmse = np.sqrt(mean_squared_error(y_val_split, ensemble_val_pred))
    
    p_values.append(p)
    ensemble_val_rmses.append(current_rmse)

    if current_rmse < min_val_rmse:
        min_val_rmse = current_rmse
        best_p = p

print(f"Optimal weight p (XGBoost): {best_p:.2f}")
print(f"Optimal weight (1-p) (NN): {1-best_p:.2f}")
print(f"Best Validation RMSE: {min_val_rmse:.5f}")

# --- PLOT: Validation RMSE vs p ---
plt.figure(figsize=(8, 5))
plt.plot(p_values, np.exp(ensemble_val_rmses), label='Ensemble Val exp(RMSE)', color='purple')
plt.axvline(best_p, color='red', linestyle='--', label=f'Best p = {best_p:.2f}')
plt.title('Validation exp(RMSE) vs Ensemble Weight (p)')
plt.xlabel('Weight (p) for XGBoost')
plt.ylabel('Validation exp(RMSE)')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("Ensemble_p.png")

# 16. Generate Final Test Predictions using the optimized 'p'
print("\nGenerating final test predictions with optimized ensemble...")

# NN Predictions for Test
final_model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(test_df_nn, dtype=torch.float32).to(device)
    nn_test_log_preds = final_model(X_test_tensor).cpu().numpy().flatten()

# XGBoost Predictions for Test
xgb_test_log_preds = xgb_model.predict(test_df_gbtm)

# Apply the optimized p-factor
final_log_preds = (best_p * xgb_test_log_preds) + ((1 - best_p) * nn_test_log_preds)

# 17. Inverse Transform (Log -> Dollars)
final_prices = np.expm1(final_log_preds)

# 18. Save to CSV
submission = pd.DataFrame({
    'Id': test_df.index,
    'SalePrice': final_prices
})

submission.to_csv('submission.csv', index=False)
print(f"Submission saved! Using p={best_p:.2f}. Good luck on the leaderboard!")


import shap
import warnings
warnings.filterwarnings('ignore') # Suppress SHAP/Sklearn warnings for clean output

# ==============================================================================
# Helper Function for PyTorch Training (To avoid repeating the loop 20+ times)
# ==============================================================================
def train_and_evaluate_nn(model, X_train_np, y_train_np, X_val_np, y_val_np, epochs=300, lr=0.001):
    # Convert to tensors
    X_tr = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train_np.values if isinstance(y_train_np, pd.Series) else y_train_np, dtype=torch.float32).view(-1, 1).to(device)
    X_va = torch.tensor(X_val_np, dtype=torch.float32).to(device)
    y_va = torch.tensor(y_val_np.values if isinstance(y_val_np, pd.Series) else y_val_np, dtype=torch.float32).view(-1, 1).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # NEW LOGIC: Instead of early stopping, we use a scheduler to "refine" the fit
    # If the loss doesn't improve for 10 epochs, reduce LR by 50%
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    train_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tr)
        loss = criterion(outputs, y_tr)
        loss.backward()
        optimizer.step()
        
        train_history.append(np.sqrt(loss.item()))

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_va)
            val_loss = criterion(val_outputs, y_va).item()
        
        # Step the scheduler based on validation loss
        scheduler.step(val_loss)
            
        # We still track the best state to return the absolute best version found
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
                
    # Load best model found during the long run
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        final_train_rmse = np.sqrt(criterion(model(X_tr), y_tr).item())
        final_val_rmse = np.sqrt(criterion(model(X_va), y_va).item())
        
    return final_train_rmse, final_val_rmse, train_history

# Try to extract feature names, fallback to generic names if pipeline blocks it
try:
    gbtm_features = preprocessor_gbtm.get_feature_names_out()
    nn_features = preprocessor_nn.get_feature_names_out()
except:
    gbtm_features = np.array([f"GBTM_Feat_{i}" for i in range(X_train_gbtm.shape[1])])
    nn_features = np.array([f"OHE_Feat_{i}" for i in range(X_train_nn.shape[1])])

# ==============================================================================
# TASK 1: GBTM Data (79 Features) - TreeSHAP & Feature Selection
# ==============================================================================
print("\n" + "="*50)
print("TASK 1: GBTM Data (79 Features) - SHAP Analysis")
print("="*50)

# Calculate SHAP values on Validation set
explainer_gbtm = shap.TreeExplainer(xgb_model)
shap_values_gbtm = explainer_gbtm.shap_values(X_val_gbtm)

# Average absolute SHAP values per feature
mean_abs_shap_gbtm = np.abs(shap_values_gbtm).mean(axis=0)
top_gbtm_indices = np.argsort(mean_abs_shap_gbtm)[::-1]

print("\nTop 10 Most Important Features (GBTM - Ordinal):")
for i in range(10):
    idx = top_gbtm_indices[i]
    print(f"{i+1}. {gbtm_features[idx]} (Avg SHAP: {mean_abs_shap_gbtm[idx]:.4f})")

# Vary mst_imp_cnt from 5 to 15
best_gbtm_val_rmse = float('inf')
best_gbtm_cnt = 0


gbtm_sweep_k = []
gbtm_sweep_rmse = []

print("\n--- GBTM Feature Selection Sweep (XGBoost) ---")
for mst_imp_cnt in range(5, 16):
    selected_indices = top_gbtm_indices[:mst_imp_cnt]
    
    # Subset the data
    X_tr_sub = X_train_gbtm[:, selected_indices]
    X_va_sub = X_val_gbtm[:, selected_indices]
    
    # Train XGBoost on subset
    xgb_sub = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
    xgb_sub.fit(X_tr_sub, y_train_split, eval_set=[(X_va_sub, y_val_split)], verbose=False)
    
    # Evaluate
    train_rmse = np.sqrt(mean_squared_error(y_train_split, xgb_sub.predict(X_tr_sub)))
    val_rmse = np.sqrt(mean_squared_error(y_val_split, xgb_sub.predict(X_va_sub)))
    
    gbtm_sweep_k.append(mst_imp_cnt)
    gbtm_sweep_rmse.append(val_rmse)

    print(f"Features: {mst_imp_cnt:2d} | Train RMSE: {train_rmse:.5f} | Val RMSE: {val_rmse:.5f}")
    
    if val_rmse < best_gbtm_val_rmse:
        best_gbtm_val_rmse = val_rmse
        best_gbtm_cnt = mst_imp_cnt

print(f"\n>> Best GBTM XGBoost Val RMSE: {best_gbtm_val_rmse:.5f} (using {best_gbtm_cnt} features)")


# ==============================================================================
# TASK 2: OHE Data (240 Features) - Baselines & Overfit/Underfit Demos
# ==============================================================================
print("\n" + "="*50)
print("TASK 2: OHE Data (240 Features) - Baseline Models")
print("="*50)

# 1. XGBoost on OHE Data
xgb_ohe = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
xgb_ohe.fit(X_train_nn, y_train_split, eval_set=[(X_val_nn, y_val_split)], verbose=False)
ohe_xgb_train_rmse = np.sqrt(mean_squared_error(y_train_split, xgb_ohe.predict(X_train_nn)))
ohe_xgb_val_rmse = np.sqrt(mean_squared_error(y_val_split, xgb_ohe.predict(X_val_nn)))

print(f"XGBoost (Full 240 Features)     | Train RMSE: {ohe_xgb_train_rmse:.5f} | Val RMSE: {ohe_xgb_val_rmse:.5f}")

# 2. Neural Network - Overfit (Huge capacity, NO dropout)
def build_overfit_nn(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, 1)
    ).to(device)

# 3. Neural Network - Underfit (Tiny capacity, NO dropout)
def build_underfit_nn(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 4), nn.ReLU(),
        nn.Linear(4, 1)
    ).to(device)

overfit_model = build_overfit_nn(X_train_nn.shape[1])
of_train_rmse, of_val_rmse, of_history = train_and_evaluate_nn(overfit_model, X_train_nn, y_train_split, X_val_nn, y_val_split, epochs=500)
print(f"NN Overfit (High Neurons)       | Train RMSE: {of_train_rmse:.5f} | Val RMSE: {of_val_rmse:.5f}")

underfit_model = build_underfit_nn(X_train_nn.shape[1])
uf_train_rmse, uf_val_rmse, uf_history = train_and_evaluate_nn(underfit_model, X_train_nn, y_train_split, X_val_nn, y_val_split, epochs=500)
print(f"NN Underfit (Low Neurons)       | Train RMSE: {uf_train_rmse:.5f} | Val RMSE: {uf_val_rmse:.5f}")

# --- PLOT: Overfit vs Underfit Training Curves ---
plt.figure(figsize=(8, 5))
plt.plot(np.exp(of_history), label='Overfit NN Train exp(RMSE)', color='red')
plt.plot(np.exp(uf_history), label='Underfit NN Train exp(RMSE)', color='blue')
plt.title('Training exp(RMSE) vs Epochs (OHE Data)')
plt.xlabel('Epoch')
plt.ylabel('Train exp(RMSE)')
plt.ylim(0, 10)
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("training_loss_of_uf_OHE.png")

# ==============================================================================
# TASK 3: OHE Data - TreeSHAP & Feature Selection (XGB & Normal NN)
# ==============================================================================
print("\n" + "="*50)
print("TASK 3: OHE Data (240 Features) - SHAP Analysis & Selection")
print("="*50)

# Calculate SHAP values for the OHE XGBoost model
explainer_ohe = shap.TreeExplainer(xgb_ohe)
shap_values_ohe = explainer_ohe.shap_values(X_val_nn)

mean_abs_shap_ohe = np.abs(shap_values_ohe).mean(axis=0)
top_ohe_indices = np.argsort(mean_abs_shap_ohe)[::-1]

print("\nTop 10 Most Important Features (OHE Data):")
for i in range(10):
    idx = top_ohe_indices[i]
    print(f"{i+1}. {nn_features[idx]} (Avg SHAP: {mean_abs_shap_ohe[idx]:.4f})")

# Define a Normal NN (No dropout, standard size) for the feature sweep
def build_normal_nn(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 1)
    ).to(device)

best_ohe_xgb_rmse = float('inf')
best_ohe_xgb_cnt = 0
best_ohe_nn_rmse = float('inf')
best_ohe_nn_cnt = 0

ohe_sweep_k = []
ohe_xgb_sweep_rmse = []
ohe_nn_sweep_rmse = []

print("\n--- OHE Feature Selection Sweep (XGBoost vs Normal NN) ---")
for mst_imp_cnt in range(5, 16):
    selected_indices = top_ohe_indices[:mst_imp_cnt]
    
    # Subset Data
    X_tr_sub = X_train_nn[:, selected_indices]
    X_va_sub = X_val_nn[:, selected_indices]
    
    # 1. Train XGBoost
    xgb_sub = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
    xgb_sub.fit(X_tr_sub, y_train_split, eval_set=[(X_va_sub, y_val_split)], verbose=False)
    xgb_tr_rmse = np.sqrt(mean_squared_error(y_train_split, xgb_sub.predict(X_tr_sub)))
    xgb_va_rmse = np.sqrt(mean_squared_error(y_val_split, xgb_sub.predict(X_va_sub)))
    
    gbtm_sweep_k.append(mst_imp_cnt)
    gbtm_sweep_rmse.append(xgb_va_rmse)

    # 2. Train Normal NN
    set_seed(25492)
    normal_nn = build_normal_nn(mst_imp_cnt)
    nn_tr_rmse, nn_va_rmse, _ = train_and_evaluate_nn(normal_nn, X_tr_sub, y_train_split, X_va_sub, y_val_split, epochs=500)
    
    ohe_sweep_k.append(mst_imp_cnt)
    ohe_xgb_sweep_rmse.append(xgb_va_rmse)
    ohe_nn_sweep_rmse.append(nn_va_rmse)

    print(f"Features: {mst_imp_cnt:2d} | XGB Train: {xgb_tr_rmse:.4f}, Val: {xgb_va_rmse:.4f} | NN Train: {nn_tr_rmse:.4f}, Val: {nn_va_rmse:.4f}")
    
    # Track Bests
    if xgb_va_rmse < best_ohe_xgb_rmse:
        best_ohe_xgb_rmse = xgb_va_rmse
        best_ohe_xgb_cnt = mst_imp_cnt
        
    if nn_va_rmse < best_ohe_nn_rmse:
        best_ohe_nn_rmse = nn_va_rmse
        best_ohe_nn_cnt = mst_imp_cnt


# --- PLOT: Number of Features vs Validation RMSE ---
plt.figure(figsize=(8, 5))
plt.plot(gbtm_sweep_k, np.exp(gbtm_sweep_rmse), marker='o', label='GBTM Data (79 features) - XGBoost', color='green')
plt.plot(ohe_sweep_k, np.exp(ohe_xgb_sweep_rmse), marker='s', label='OHE Data (240 features)- XGBoost', color='orange')
plt.plot(ohe_sweep_k, np.exp(ohe_nn_sweep_rmse), marker='^', label='OHE Data (240 features) - Normal NN', color='blue')

plt.title('Validation exp(RMSE) vs Number of Selected Features (k)')
plt.xlabel('Number of Most Important Features Used (k)')
plt.ylabel('Validation exp(RMSE)')
plt.xticks(range(5, 16))
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("val_rmse_vs_num_features.png")

print(f"\n>> Best OHE XGBoost Val RMSE: {best_ohe_xgb_rmse:.5f} (using {best_ohe_xgb_cnt} features)")
print(f">> Best OHE Normal NN Val RMSE: {best_ohe_nn_rmse:.5f} (using {best_ohe_nn_cnt} features)")



print("\n" + "="*50)
print("GENERATING ALL SUBMISSION FILES")
print("="*50)

# 1. XGBoost (Full 79 Features)
xgb_79_preds = xgb_model.predict(test_df_gbtm)
create_submission(xgb_79_preds, "submission_xgb_79_features.csv")

# 2. NN with Dropout (240 Features)
final_model.eval()
with torch.no_grad():
    X_test_nn_tensor = torch.tensor(test_df_nn, dtype=torch.float32).to(device)
    nn_dropout_preds = final_model(X_test_nn_tensor).cpu().numpy().flatten()
create_submission(nn_dropout_preds, "submission_nn_dropout_240.csv")

# 3. Optimized Ensemble
ensemble_preds = (best_p * xgb_79_preds) + ((1 - best_p) * nn_dropout_preds)
create_submission(ensemble_preds, "submission_ensemble.csv")

# 4. Overfit NN
overfit_model.eval()
with torch.no_grad():
    nn_overfit_preds = overfit_model(X_test_nn_tensor).cpu().numpy().flatten()
create_submission(nn_overfit_preds, "submission_overfit_nn.csv")

# 5. Underfit NN
underfit_model.eval()
with torch.no_grad():
    nn_underfit_preds = underfit_model(X_test_nn_tensor).cpu().numpy().flatten()
create_submission(nn_underfit_preds, "submission_underfit_nn.csv")

# 6. XGBoost with Low Number of Features
print(f"Re-training best Low-Feature XGB (k={best_gbtm_cnt})...")
selected_gbtm_idx = top_gbtm_indices[:best_gbtm_cnt]
xgb_low = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
xgb_low.fit(X_train_gbtm[:, selected_gbtm_idx], y_train_split)
xgb_low_preds = xgb_low.predict(test_df_gbtm[:, selected_gbtm_idx])
create_submission(xgb_low_preds, f"submission_xgb_low_feat_{best_gbtm_cnt}.csv")

# 7. NN with Low Number of Features
print(f"Re-training best Low-Feature NN (k={best_ohe_nn_cnt})...")
selected_ohe_idx = top_ohe_indices[:best_ohe_nn_cnt]
nn_low = build_normal_nn(best_ohe_nn_cnt)
set_seed(25492) # Ensuring the re-train is seeded
train_and_evaluate_nn(nn_low, X_train_nn[:, selected_ohe_idx], y_train_split, 
                      X_val_nn[:, selected_ohe_idx], y_val_split, epochs=500)
nn_low.eval()
with torch.no_grad():
    X_test_low_tensor = torch.tensor(test_df_nn[:, selected_ohe_idx], dtype=torch.float32).to(device)
    nn_low_preds = nn_low(X_test_low_tensor).cpu().numpy().flatten()
create_submission(nn_low_preds, f"submission_nn_low_feat_{best_ohe_nn_cnt}.csv")

print("\nAll submissions generated. Ready for the leaderboard!")
