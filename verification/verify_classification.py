"""
XGBoost Classification Performance Verification
================================================
Paper: "XGBoost: A Scalable Tree Boosting System" - Chen & Guestrin (KDD '16)
Section 6.1, Table 3

PAPER CLAIM (HIGGS-1M):
  - Test AUC: 0.8304
  - Time per tree: 0.6841 sec
  - 500 trees, exact greedy algorithm
"""

import subprocess
import sys
import random

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
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
except ImportError:
    install("numpy")
    install("pandas")
    install("scikit-learn")
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

import time

REPRODUCIBLE_SEED = 42

random.seed(REPRODUCIBLE_SEED)
np.random.seed(REPRODUCIBLE_SEED)


def verify_higgs_classification():
    """
    Paper Table 3: HIGGS-1M classification
    - XGBoost exact greedy: AUC 0.8304, 0.6841 sec/tree
    """
    print("\n" + "="*60)
    print("  CLASSIFICATION PERFORMANCE VERIFICATION (HIGGS-1M)")
    print("  Paper: Chen & Guestrin, KDD '16 - Table 3")
    print("="*60)
    
    print("\nLoading HIGGS dataset (1M rows)...")
    df = pd.read_csv('../data/HIGGS.csv', header=None, nrows=1_000_000)
    
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    
    print(f"Samples: {X.shape[0]:,}")
    print(f"Features: {X.shape[1]}")
    print(f"Positive rate: {y.mean():.2%}")
    
    # Train/test split (80/20 like paper)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=REPRODUCIBLE_SEED
    )
    
    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Paper parameters
    num_trees = 500
    base_params = {
        'max_depth': 6,
        'eta': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': REPRODUCIBLE_SEED,
    }

    def run_case(tree_method, title):
        params = dict(base_params)
        params['tree_method'] = tree_method

        print(f"\nTraining {num_trees} trees ({title})...")
        print("-" * 50)

        t1 = time.time()
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_trees,
            evals=[(dtest, 'test')],
            verbose_eval=100 if tree_method == 'exact' else False,
        )
        total_time = time.time() - t1
        time_per_tree = total_time / num_trees

        y_pred = model.predict(dtest)
        auc = roc_auc_score(y_test, y_pred)
        return auc, time_per_tree, total_time

    exact_auc, exact_time_per_tree, exact_total_time = run_case('exact', 'exact greedy')
    approx_auc, approx_time_per_tree, approx_total_time = run_case('approx', 'approx')
    
    # Paper results
    paper_auc = 0.8304
    paper_time = 0.6841
    
    print(f"""
+--------------------------------------------------+
|         RESULTS (HIGGS-1M, 500 trees)            |
+--------------------------------------------------+
|  Exact AUC:             {exact_auc:>8.4f}               |
|  Approx AUC:            {approx_auc:>8.4f}               |
|  Paper AUC:             {paper_auc:>8.4f}               |
+--------------------------------------------------+
|  Exact time/tree:       {exact_time_per_tree:>8.4f} sec          |
|  Approx time/tree:      {approx_time_per_tree:>8.4f} sec          |
|  Paper time/tree:       {paper_time:>8.4f} sec          |
+--------------------------------------------------+
|  Exact total time:      {exact_total_time:>8.1f} sec          |
|  Approx total time:     {approx_total_time:>8.1f} sec          |
+--------------------------------------------------+
""")
    
    # Verify
    auc_match = abs(exact_auc - paper_auc) < 0.01
    print(f"Exact AUC Match: {'YES' if auc_match else 'NO'} (within 0.01 of paper)")
    print(f"Approx vs Exact AUC delta: {approx_auc - exact_auc:+.4f}")
    
    return {
        'exact_auc': exact_auc,
        'approx_auc': approx_auc,
        'exact_time_per_tree': exact_time_per_tree,
        'approx_time_per_tree': approx_time_per_tree,
    }


if __name__ == "__main__":
    results = verify_higgs_classification()
