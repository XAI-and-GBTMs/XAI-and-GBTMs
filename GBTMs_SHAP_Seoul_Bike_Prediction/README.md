# Seoul Bike Demand Forecasting & Interpretability

This folder contains the code, dataset, and visual diagnostics for predicting hourly bike rental demand in Seoul using Gradient Boosted Tree Models (GBTMs) and SHAP.

## Project Overview
The goal of this section is to achieve a highly parsimonious predictive model. We used an XGBoost regressor trained on chronological, time-safe splits to prevent data leakage. We then utilized TreeSHAP to unpack the model's decision-making and systematically prune redundant features.

## Folder Structure
* `seoul_bike_gbtm_shap.py`: The main Python script containing the data pipeline, XGBoost training, and SHAP explainability generation.
* `SeoulBikeData.csv`: The raw dataset (8,761 hourly observations).
* `/plots`: Directory containing all generated visual diagnostics, including:
  * Global feature importance (SHAP bar plots)
  * Local explainability (SHAP waterfall plots)
  * Sequential feature elimination (RMSE curve)
  * Error diagnostics (Actual vs. Predicted, Residuals)

## Key Findings
1. **Dominant Predictors:** SHAP analysis revealed that the hour of the day and ambient temperature strongly dominated the model's predictive capabilities.
2. **Engineered Signals:** Engineered interaction features (like `Peak_Workday`) captured genuine behavioral signals and ranked highly in global importance.
3. **Parsimony:** By sequentially eliminating features based on SHAP importance, we reduced the feature space by 30% (from 17 to 12 features) while simultaneously minimizing the global RMSE to 207.6.
