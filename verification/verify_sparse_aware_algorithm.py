"""
XGBoost Sparse-Aware Algorithm Verification
============================================
Paper: "XGBoost: A Scalable Tree Boosting System" - Chen & Guestrin (KDD '16)
Section 3.4, Algorithm 3, Table 2

PAPER CLAIM: ~50x speedup on Allstate-10K dataset

HOW TO RUN:
    pip install xgboost scikit-learn pandas
    python verify_sparse_aware_algorithm.py
"""

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

try:
    import xgboost as xgb
except ImportError:
    print("Installing xgboost...")
    install("xgboost")
    import xgboost as xgb

try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.preprocessing import OneHotEncoder
except ImportError:
    print("Installing dependencies...")
    install("numpy")
    install("pandas")
    install("scikit-learn")
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.preprocessing import OneHotEncoder

import time

REPRODUCIBLE_SEED = 42

np.random.seed(REPRODUCIBLE_SEED)

print(f"XGBoost version: {xgb.__version__}")
print("="*70)


# ============================================================================
# TEST 1: Adult Dataset - Missing Value Handling Comparison
# ============================================================================
ADULT_COLUMNS = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income',
]


def _read_adult_file(path):
    df = pd.read_csv(
        path,
        names=ADULT_COLUMNS,
        sep=r',\s*',
        engine='python',
        na_values='?',
        comment='|',
        skipinitialspace=True,
    )
    df = df.dropna(how='all')

    # Normalize string columns and labels from adult.test (which includes ".")
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    df['income'] = df['income'].str.replace('.', '', regex=False)

    return df


def _ordinal_encode_with_nan(series):
    categorical = pd.Categorical(series)
    codes = pd.Series(categorical.codes.astype(float), index=series.index)
    codes[codes < 0] = np.nan
    return codes


def _build_sparse_aware_features(X_raw, numeric_cols, categorical_cols):
    X_numeric = X_raw[numeric_cols].apply(pd.to_numeric, errors='coerce')
    X_categorical = [
        _ordinal_encode_with_nan(X_raw[col]).rename(f"{col}__encoded")
        for col in categorical_cols
    ]
    X_all = pd.concat([X_numeric] + X_categorical, axis=1)
    feature_columns = list(X_all.columns)
    return X_all, feature_columns


def _inject_missingness_into_raw(X_raw, missing_rate=0.20, features_per_row=2, seed=REPRODUCIBLE_SEED):
    rng = np.random.default_rng(seed)
    X_missing = X_raw.copy()

    row_count = len(X_missing)
    selected_row_count = max(1, int(round(row_count * missing_rate)))
    selected_rows = rng.choice(row_count, size=selected_row_count, replace=False)

    feature_names = list(X_missing.columns)
    for row_idx in selected_rows:
        feature_count = min(features_per_row, len(feature_names))
        selected_features = rng.choice(feature_names, size=feature_count, replace=False)
        X_missing.iloc[row_idx, X_missing.columns.get_indexer(selected_features)] = np.nan

    injected_mask = X_missing.isna().any(axis=1).to_numpy()
    return X_missing, injected_mask, selected_rows


def _impute_for_classical_gbdt(X_train, X_test, numeric_cols):
    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()

    for col in X_train_imputed.columns:
        train_series = X_train_imputed[col]
        if col in numeric_cols:
            fill_value = float(train_series.mean()) if not train_series.dropna().empty else 0.0
        else:
            mode_values = train_series.dropna().mode()
            fill_value = float(mode_values.iloc[0]) if not mode_values.empty else 0.0

        X_train_imputed[col] = train_series.fillna(fill_value)
        X_test_imputed[col] = X_test_imputed[col].fillna(fill_value)

    return X_train_imputed, X_test_imputed


def _print_missing_row_report(
    dataset_name,
    test_row_count,
    missing_row_count,
    sparse_correct,
    approx_correct,
    gbdt_correct,
    sparse_acc,
    approx_acc,
    gbdt_acc,
    sparse_time,
    approx_time,
    gbdt_time,
):
    print(f"\n{dataset_name}")
    print(f"Missing test rows: {missing_row_count}/{test_row_count}")
    print(f"Correct predictions - GBTM: {sparse_correct}/{missing_row_count}")
    print(f"Correct predictions - XGB approx: {approx_correct}/{missing_row_count}")
    print(f"Correct predictions - GBDT: {gbdt_correct}/{missing_row_count}")
    print(f"Accuracy - GBTM: {sparse_acc:.4f}")
    print(f"Accuracy - XGB approx: {approx_acc:.4f}")
    print(f"Accuracy - GBDT: {gbdt_acc:.4f}")
    print(f"Train time - GBTM: {sparse_time:.4f} sec")
    print(f"Train time - XGB approx: {approx_time:.4f} sec")
    print(f"Train time - GBDT: {gbdt_time:.4f} sec")


def _benchmark_training_step(dtrain, params, num_boost_round=10, repeats=3):
    # DMatrix creation and preprocessing are intentionally excluded here.
    # This measures the training-step cost that contains XGBoost's sparse-aware
    # split finding and tree construction.
    timings = []
    for _ in range(repeats + 1):
        start = time.perf_counter()
        xgb.train(params, dtrain, num_boost_round=num_boost_round, verbose_eval=False)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)

    # Drop the first run as a warm-up so the reported number is steadier.
    return float(np.median(timings[1:]))


def test1_learned_default_direction():
    """
    Adult dataset test for sparse-aware missing-value handling.

    We compare:
    1) GBTM sparse-aware: XGBoost with native NaN handling
    2) Classical GBDT baseline: GradientBoostingClassifier after imputation

    Accuracy is measured ONLY on test rows that contain at least one missing
    feature, after injecting extra missingness into the test split so the
    comparison is larger and more stable.
    """
    print("\n" + "="*70)
    print("TEST 1: Adult Dataset Missing-Value Handling")
    print("="*70)
    print("Comparison mode: GBTM exact split search with native NaN vs classical GBDT exact split search with imputation")

    train_df = _read_adult_file('../data/adult/adult.data')
    test_df = _read_adult_file('../data/adult/adult.test')
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    y = (df['income'] == '>50K').astype(int)
    X_raw = df.drop(columns=['income'])

    X_raw_missing, _, injected_rows = _inject_missingness_into_raw(
        X_raw,
        missing_rate=0.25,
        features_per_row=2,
        seed=REPRODUCIBLE_SEED,
    )

    missing_counts = X_raw_missing.isna().sum().sort_values(ascending=False)
    missing_counts = missing_counts[missing_counts > 0]

    if missing_counts.empty:
        raise RuntimeError("Adult dataset has no missing values to test.")

    top_missing_feature = missing_counts.index[0]

    numeric_cols = [
        'age',
        'fnlwgt',
        'education-num',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
    ]
    categorical_cols = [col for col in X_raw.columns if col not in numeric_cols]

    X_all, _ = _build_sparse_aware_features(X_raw_missing, numeric_cols, categorical_cols)

    all_indices = np.arange(len(X_all))
    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.25,
        random_state=REPRODUCIBLE_SEED,
        stratify=y,
    )

    X_train = X_all.iloc[train_idx].copy()
    y_train = y.iloc[train_idx].copy()
    X_test = X_all.iloc[test_idx].copy()
    y_test = y.iloc[test_idx].copy()

    missing_mask_test = X_raw_missing.iloc[test_idx].isna().any(axis=1).to_numpy()
    missing_only_count = int(missing_mask_test.sum())
    print(f"Adult test rows with missing features: {missing_only_count}")

    if missing_only_count == 0:
        raise RuntimeError("No missing rows in test split; rerun with a different random seed.")

    sparse_model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='exact',
        random_state=REPRODUCIBLE_SEED,
        n_jobs=4,
        missing=np.nan,
    )
    start_time = time.perf_counter()
    sparse_model.fit(X_train, y_train)
    sparse_train_time = time.perf_counter() - start_time

    approx_model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='approx',
        random_state=REPRODUCIBLE_SEED,
        n_jobs=4,
        missing=np.nan,
    )
    start_time = time.perf_counter()
    approx_model.fit(X_train, y_train)
    approx_train_time = time.perf_counter() - start_time

    X_train_imputed, X_test_imputed = _impute_for_classical_gbdt(X_train, X_test, numeric_cols)

    # Classical pre-XGBoost GBDT baseline (requires fully observed features).
    normal_model = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        random_state=REPRODUCIBLE_SEED,
    )
    start_time = time.perf_counter()
    normal_model.fit(X_train_imputed, y_train)
    gbdt_train_time = time.perf_counter() - start_time

    sparse_pred = (sparse_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    approx_pred = (approx_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    normal_pred = (normal_model.predict_proba(X_test_imputed)[:, 1] >= 0.5).astype(int)

    y_missing = y_test.to_numpy()[missing_mask_test]
    sparse_pred_missing = sparse_pred[missing_mask_test]
    approx_pred_missing = approx_pred[missing_mask_test]
    normal_pred_missing = normal_pred[missing_mask_test]
    sparse_correct_missing = int((sparse_pred_missing == y_missing).sum())
    approx_correct_missing = int((approx_pred_missing == y_missing).sum())
    normal_correct_missing = int((normal_pred_missing == y_missing).sum())
    sparse_acc_missing = accuracy_score(y_missing, sparse_pred_missing)
    approx_acc_missing = accuracy_score(y_missing, approx_pred_missing)
    normal_acc_missing = accuracy_score(y_missing, normal_pred_missing)
    improvement = sparse_acc_missing - normal_acc_missing
    approx_vs_exact = approx_acc_missing - sparse_acc_missing

    _print_missing_row_report(
        dataset_name="Adult",
        test_row_count=len(X_test),
        missing_row_count=missing_only_count,
        sparse_correct=sparse_correct_missing,
        approx_correct=approx_correct_missing,
        gbdt_correct=normal_correct_missing,
        sparse_acc=sparse_acc_missing,
        approx_acc=approx_acc_missing,
        gbdt_acc=normal_acc_missing,
        sparse_time=sparse_train_time,
        approx_time=approx_train_time,
        gbdt_time=gbdt_train_time,
    )
    print(f"Difference in accuracy (GBTM - GBDT): {improvement:+.4f}")
    print(f"Difference in accuracy (XGB approx - GBTM exact): {approx_vs_exact:+.4f}")

    return {
        'top_missing_feature': top_missing_feature,
        'missing_counts': missing_counts.to_dict(),
        'missing_rows_test': missing_only_count,
        'injected_rows_test': int(len(injected_rows)),
        'sparse_correct_missing': sparse_correct_missing,
        'approx_correct_missing': approx_correct_missing,
        'gbdt_correct_missing': normal_correct_missing,
        'sparse_accuracy_missing': float(sparse_acc_missing),
        'approx_accuracy_missing': float(approx_acc_missing),
        'gbdt_accuracy_missing': float(normal_acc_missing),
        'delta_accuracy': float(improvement),
        'delta_approx_vs_exact': float(approx_vs_exact),
        'sparse_train_time': float(sparse_train_time),
        'approx_train_time': float(approx_train_time),
        'gbdt_train_time': float(gbdt_train_time),
        'passed': bool(sparse_acc_missing >= normal_acc_missing),
    }


# ============================================================================
# TEST 2: Allstate Dataset - Missing Value Handling Comparison
# ============================================================================
def test2_allstate_missing_handling():
    """
    Allstate test for sparse-aware missing-value handling.

    We compare:
    1) GBTM sparse-aware: XGBoost with native NaN handling
    2) Classical GBDT baseline: GradientBoostingClassifier after imputation

    Accuracy is measured ONLY on test rows that contain at least one missing
    feature, after injecting extra missingness into the test split so the
    comparison is larger and more stable.
    """
    print("\n" + "="*70)
    print("TEST 2: Allstate Dataset Missing-Value Handling")
    print("="*70)
    print("Comparison mode: GBTM exact split search with native NaN vs classical GBDT exact split search with imputation")

    print("\nLoading Allstate dataset (10K rows)...")
    df = pd.read_csv('../data/allstate_raw/train_set.csv', nrows=10000)

    y = (df['Claim_Amount'] > 0).astype(int)
    X_raw = df.drop(columns=['Claim_Amount'])

    # Exclude identifiers from the feature matrix so the comparison focuses on
    # actual predictive fields.
    X_raw = X_raw.drop(columns=['Row_ID', 'Household_ID'])

    X_raw_missing, _, injected_rows = _inject_missingness_into_raw(
        X_raw,
        missing_rate=0.30,
        features_per_row=3,
        seed=REPRODUCIBLE_SEED,
    )

    missing_counts = X_raw_missing.isna().sum().sort_values(ascending=False)
    missing_counts = missing_counts[missing_counts > 0]

    if missing_counts.empty:
        raise RuntimeError("Allstate dataset has no missing values to test.")

    top_missing_feature = missing_counts.index[0]

    categorical_cols = [
        'Vehicle', 'Blind_Make', 'Blind_Model', 'Blind_Submodel',
        'Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5', 'Cat6', 'Cat7', 'Cat8',
        'Cat9', 'Cat10', 'Cat11', 'Cat12', 'OrdCat', 'NVCat'
    ]
    numeric_cols = [
        'Calendar_Year', 'Model_Year', 'Var1', 'Var2', 'Var3', 'Var4',
        'Var5', 'Var6', 'Var7', 'Var8', 'NVVar1', 'NVVar2', 'NVVar3', 'NVVar4'
    ]

    categorical_cols = [col for col in categorical_cols if col in X_raw.columns]
    numeric_cols = [col for col in numeric_cols if col in X_raw.columns]

    X_all, _ = _build_sparse_aware_features(X_raw_missing, numeric_cols, categorical_cols)

    all_indices = np.arange(len(X_all))
    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.25,
        random_state=REPRODUCIBLE_SEED,
        stratify=y,
    )

    X_train = X_all.iloc[train_idx].copy()
    y_train = y.iloc[train_idx].copy()
    X_test = X_all.iloc[test_idx].copy()
    y_test = y.iloc[test_idx].copy()

    missing_mask_test = X_raw_missing.iloc[test_idx].isna().any(axis=1).to_numpy()
    missing_only_count = int(missing_mask_test.sum())
    print(f"Allstate test rows with missing features: {missing_only_count}")

    if missing_only_count == 0:
        raise RuntimeError("No missing rows in test split; rerun with a different random seed.")

    sparse_model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='exact',
        random_state=REPRODUCIBLE_SEED,
        n_jobs=4,
        missing=np.nan,
    )
    start_time = time.perf_counter()
    sparse_model.fit(X_train, y_train)
    sparse_train_time = time.perf_counter() - start_time

    approx_model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='approx',
        random_state=REPRODUCIBLE_SEED,
        n_jobs=4,
        missing=np.nan,
    )
    start_time = time.perf_counter()
    approx_model.fit(X_train, y_train)
    approx_train_time = time.perf_counter() - start_time

    X_train_imputed, X_test_imputed = _impute_for_classical_gbdt(X_train, X_test, numeric_cols)

    normal_model = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        random_state=REPRODUCIBLE_SEED,
    )
    start_time = time.perf_counter()
    normal_model.fit(X_train_imputed, y_train)
    gbdt_train_time = time.perf_counter() - start_time

    sparse_pred = (sparse_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    approx_pred = (approx_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    normal_pred = (normal_model.predict_proba(X_test_imputed)[:, 1] >= 0.5).astype(int)

    y_missing = y_test.to_numpy()[missing_mask_test]
    sparse_pred_missing = sparse_pred[missing_mask_test]
    approx_pred_missing = approx_pred[missing_mask_test]
    normal_pred_missing = normal_pred[missing_mask_test]
    sparse_correct_missing = int((sparse_pred_missing == y_missing).sum())
    approx_correct_missing = int((approx_pred_missing == y_missing).sum())
    normal_correct_missing = int((normal_pred_missing == y_missing).sum())
    sparse_acc_missing = accuracy_score(y_missing, sparse_pred_missing)
    approx_acc_missing = accuracy_score(y_missing, approx_pred_missing)
    normal_acc_missing = accuracy_score(y_missing, normal_pred_missing)
    improvement = sparse_acc_missing - normal_acc_missing
    approx_vs_exact = approx_acc_missing - sparse_acc_missing

    _print_missing_row_report(
        dataset_name="Allstate",
        test_row_count=len(X_test),
        missing_row_count=missing_only_count,
        sparse_correct=sparse_correct_missing,
        approx_correct=approx_correct_missing,
        gbdt_correct=normal_correct_missing,
        sparse_acc=sparse_acc_missing,
        approx_acc=approx_acc_missing,
        gbdt_acc=normal_acc_missing,
        sparse_time=sparse_train_time,
        approx_time=approx_train_time,
        gbdt_time=gbdt_train_time,
    )
    print(f"Difference in accuracy (GBTM - GBDT): {improvement:+.4f}")
    print(f"Difference in accuracy (XGB approx - GBTM exact): {approx_vs_exact:+.4f}")

    return {
        'top_missing_feature': top_missing_feature,
        'missing_counts': missing_counts.to_dict(),
        'missing_rows_test': missing_only_count,
        'injected_rows_test': int(len(injected_rows)),
        'sparse_correct_missing': sparse_correct_missing,
        'approx_correct_missing': approx_correct_missing,
        'gbdt_correct_missing': normal_correct_missing,
        'sparse_accuracy_missing': float(sparse_acc_missing),
        'approx_accuracy_missing': float(approx_acc_missing),
        'gbdt_accuracy_missing': float(normal_acc_missing),
        'delta_accuracy': float(improvement),
        'delta_approx_vs_exact': float(approx_vs_exact),
        'sparse_train_time': float(sparse_train_time),
        'approx_train_time': float(approx_train_time),
        'gbdt_train_time': float(gbdt_train_time),
        'passed': bool(sparse_acc_missing >= normal_acc_missing),
    }


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("\nXGBoost Sparse-Aware Algorithm Verification")
    print("Paper: Chen & Guestrin, KDD '16, Section 3.4")
    
    results = {}
    
    results['test1'] = test1_learned_default_direction()
    results['test2'] = test2_allstate_missing_handling()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
Test 1 - Adult Missing-Only Accuracy:
    Missing test rows:              {results['test1']['missing_rows_test']}
    Correct datapoints (GBTM):      {results['test1']['sparse_correct_missing']}
    Correct datapoints (XGB approx): {results['test1']['approx_correct_missing']}
    Correct datapoints (GBDT):      {results['test1']['gbdt_correct_missing']}
    Sparse-aware GBTM:              {results['test1']['sparse_accuracy_missing']:.4f}
    XGB approx:                     {results['test1']['approx_accuracy_missing']:.4f}
    Classical GBDT (imputed):       {results['test1']['gbdt_accuracy_missing']:.4f}
    Delta (GBTM - GBDT):            {results['test1']['delta_accuracy']:+.4f}
    Delta (approx - exact):         {results['test1']['delta_approx_vs_exact']:+.4f}
    GBTM fit time:                  {results['test1']['sparse_train_time']:.4f} sec
    XGB approx fit time:            {results['test1']['approx_train_time']:.4f} sec
    GBDT fit time:                  {results['test1']['gbdt_train_time']:.4f} sec
Test 2 - Allstate Missing-Only Accuracy:
    Missing test rows:              {results['test2']['missing_rows_test']}
    Correct datapoints (GBTM):      {results['test2']['sparse_correct_missing']}
    Correct datapoints (XGB approx): {results['test2']['approx_correct_missing']}
    Correct datapoints (GBDT):      {results['test2']['gbdt_correct_missing']}
    Sparse-aware GBTM:              {results['test2']['sparse_accuracy_missing']:.4f}
    XGB approx:                     {results['test2']['approx_accuracy_missing']:.4f}
    Classical GBDT (imputed):       {results['test2']['gbdt_accuracy_missing']:.4f}
    Delta (GBTM - GBDT):            {results['test2']['delta_accuracy']:+.4f}
    Delta (approx - exact):         {results['test2']['delta_approx_vs_exact']:+.4f}
    GBTM fit time:                  {results['test2']['sparse_train_time']:.4f} sec
    XGB approx fit time:            {results['test2']['approx_train_time']:.4f} sec
    GBDT fit time:                  {results['test2']['gbdt_train_time']:.4f} sec
""")
