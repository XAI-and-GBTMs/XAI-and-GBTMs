# House Prices Prediction: Reproduction Guide

Follow these steps to set up the environment and data required to run the models and generate submissions.

## 1. Prerequisites
Install the required Python libraries using pip:

```bash
pip install numpy pandas scikit-learn xgboost torch shap matplotlib kaggle
```

## 2. Dataset Download
Download the dataset from the Kaggle competition page:
[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

## 3. Data Setup
1. Create a folder named `data` in your project directory.
2. Place the downloaded files inside the `data` folder.
3. Ensure the files are named exactly `train.csv` and `test.csv`.

## 4. Project Directory Structure
Your project directory should look like this:

```text
.
├── data/
│   ├── train.csv
│   └── test.csv
└── gbtm_nn_comparison_house_prices_dataset.py
```

