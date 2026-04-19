#High level pipeline
# Import modules and dataset
# Clean the features by encoding them into numbers. No missing value were found. 
# Split the dataset into 80, 20 train-test sets, making sure to keep the time order intact (no shuffling). 
# Train an XGBoost model on the training set, and test it on the test set. 
#Accuracy metrics
import matplotlib
matplotlib.use('Agg')
import pandas as pd 
import xgboost as xgb 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import shap

# Load dataset
df = pd.read_csv('SeoulBikeData.csv', encoding='latin1')
print(f"Number of columns: {df.shape[1]}")

df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['DayOfWeek'] = df['Date'].dt.dayofweek  # Monday=0, Sunday=6
# Encode categorical variables
df['Holiday'] = df['Holiday'].map({'Holiday': 1, 'No Holiday': 0})
df['Functioning Day'] = df['Functioning Day'].map({'Yes': 1, 'No': 0})
# One-hot encode seasons
df = pd.get_dummies(df, columns=['Seasons'], drop_first=False)

# --- Correlation Matrix ---

plt.figure(figsize=(14, 10))
# Calculate Pearson correlation
corr_matrix = df.corr()
# Plot the heatmap
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
            vmin=-1, vmax=1, square=True, linewidths=.5)
plt.title('Pearson Correlation Matrix')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
# Save the plot for the LaTeX appendix
plt.savefig('correlation_matrix.png', dpi=150)
plt.close()
print("Correlation matrix saved as 'correlation_matrix.png'.")


# 2. Combine Weekend + Holiday logic
# 3. Interaction: Peak hour is only relevant on working days
df['Is_Peak_Hour'] = df['Hour'].apply(lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0)
df['Is_Non_Working_Day'] = ((df['DayOfWeek'] >= 5) | (df['Holiday'] == 1)).astype(int)
df['Peak_Workday'] = df['Is_Peak_Hour'] * (1 - df['Is_Non_Working_Day'])



df=df.drop('Holiday', axis=1) if 'Holiday' in df.columns else df
df = df.drop('Date', axis=1)
df=df.drop({'Year'}, axis=1)
df = df.drop('Dew point temperature(°C)', axis=1)
df = df.drop('Day', axis=1) if 'Day' in df.columns else df
#df=df.drop(['Snowfall (cm)', 'Seasons_Summer','Wind speed (m/s)', 'Seasons_Spring', 'Visibility (10m)'], axis=1)
df = df[df['Functioning Day'] == 1].drop('Functioning Day', axis=1)


print("Average bikes rented:", df['Rented Bike Count'].mean())
print("Maximum bikes rented in one hour:", df['Rented Bike Count'].max())


# Check for missing values 
missing_values = df.isnull().sum()
print("Columns with missing values:")
print(missing_values[missing_values > 0])

# --- Time-safe train/test split (no shuffling) ---
def keeptime(X, y):
    split_index = int(len(X) * 0.8)
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test  = X.iloc[split_index:]
    y_test  = y.iloc[split_index:]
    return X_train, X_test, y_train, y_test

y = df['Rented Bike Count']
X = df.drop('Rented Bike Count', axis=1)
X_train, X_test, y_train, y_test = keeptime(X, y)



# --- Train XGBoost model ---
model = xgb.XGBRegressor(
    n_estimators=800,
    max_depth=6,           # slightly shallower = less overfitting
    learning_rate=0.05,    # slower + more trees = better generalization
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=3,    # avoids splits on tiny groups
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.5,
    random_state=42
)

model.fit(X_train, y_train)
print("XGBoost trained.")

#Test set predictions
y_pred = model.predict(X_test)

# --- Accuracy metrics ---
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"MAE: {mae:.1f} bikes")
print(f"RMSE: {rmse:.1f}")
print(f"R-squared: {r2*100:.1f}%")
print(f"Error as % of mean: {mae / df['Rented Bike Count'].mean() * 100:.1f}%")

# --- Plots ---
# --- Actual vs. Predicted ---
def plots():
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    plt.title('Actual vs. Predicted Bike Rentals')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig('actual_vs_predicted.png', bbox_inches='tight', dpi=100)
    plt.close()

    # --- Residuals Plot ---
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Residuals Plot')
    plt.xlabel('Predicted')
    plt.ylabel('Error (Actual - Predicted)')
    plt.savefig('residuals_plot.png', bbox_inches='tight', dpi=100)
    plt.close()

    print("Plots saved.")
plots()



# SHAP plots
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig('shap_bar.png', bbox_inches='tight', dpi=100)
plt.close()
# Waterfall for a single prediction
explanation = explainer(X_test)
shap.plots.waterfall(explanation[0], show=False)
plt.savefig('shap_waterfall.png', bbox_inches='tight', dpi=100)
plt.close()

print("SHAP plots saved.")


# --- Sequential Feature Elimination by SHAP Importance ---
# Remove features one-by-one starting from the LEAST important (per SHAP bar plot)
# Retrain XGBoost at each step and track RMSE
# Get mean absolute SHAP values per feature, ranked ascending (least → most important)

shap_importance = np.abs(shap_values).mean(axis=0)
shap_feature_order = np.argsort(shap_importance)           # ascending: least important first
feature_names_ordered = X_test.columns[shap_feature_order].tolist()

rmse_scores = []
features_removed = []
features_remaining_counts = []

current_features = list(X_train.columns)

for i, feat_to_remove in enumerate(feature_names_ordered):
    # Retrain on current feature set
    X_tr = X_train[current_features]
    X_te = X_test[current_features]

    m = xgb.XGBRegressor(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=42,
        verbosity=0
    )
    m.fit(X_tr, y_train)
    preds = m.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    rmse_scores.append(rmse)
    features_removed.append(feat_to_remove)
    features_remaining_counts.append(len(current_features))

    print(f"[{i+1}/{len(feature_names_ordered)}] Removed '{feat_to_remove}' | "
        f"Features left: {len(current_features)} | RMSE: {rmse:.1f}")

    # Remove the feature for the next iteration
    current_features.remove(feat_to_remove)

# --- Plot ---
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(range(len(rmse_scores)), rmse_scores, marker='o', color='steelblue',
        linewidth=2, markersize=5, label='RMSE')
ax.axhline(y=rmse_scores[0], color='gray', linestyle='--', alpha=0.6, label='Baseline RMSE (all features)')

# Annotate the minimum RMSE point
min_idx = np.argmin(rmse_scores)
ax.annotate(f'Min RMSE\n{rmse_scores[min_idx]:.1f}',
            xy=(min_idx, rmse_scores[min_idx]),
            xytext=(min_idx + 0.8, rmse_scores[min_idx] + 30),
            arrowprops=dict(arrowstyle='->', color='red'),
            color='red', fontsize=9)

ax.set_xticks(range(len(features_removed)))
ax.set_xticklabels(
    [f"{f}\n({n} left)" for f, n in zip(features_removed, features_remaining_counts)],
    rotation=45, ha='right', fontsize=7.5
)

ax.set_xlabel('Feature Removed (left → right: least → most important by SHAP)', fontsize=11)
ax.set_ylabel('RMSE (bikes)', fontsize=11)
ax.set_title('Sequential Feature Elimination: RMSE vs Features Removed\n(Least → Most SHAP-Important)', fontsize=13)
ax.legend()
ax.grid(axis='y', alpha=0.4)

plt.tight_layout()
plt.savefig('feature_elimination_rmse.png', bbox_inches='tight', dpi=120)
plt.close()
print("Feature elimination plot saved as 'feature_elimination_rmse.png'.")

