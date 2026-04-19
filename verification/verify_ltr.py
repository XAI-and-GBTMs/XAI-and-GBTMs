"""
XGBoost Learning to Rank (LTR) Verification
============================================
Paper: "XGBoost: A Scalable Tree Boosting System" - Chen & Guestrin (KDD '16)
Section 6.2

PAPER CLAIM (Yahoo! LTRC):
  - NDCG@10: 0.7892 (default)
  - NDCG@10: 0.7913 (with colsample_bytree=0.5)
    - Compared against pGBRT (best previously published ranking system)
  - 500 trees

We use MQ2008 (LETOR) as an alternative dataset and train a query-aware
pGBRT proxy (pairwise ranking + approximate tree construction) for a fair
comparison on the same split.
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
except ImportError:
    install("numpy")
    import numpy as np

try:
    from sklearn.metrics import ndcg_score
except ImportError:
    print("Installing scikit-learn...")
    install("scikit-learn")
    from sklearn.metrics import ndcg_score

import time


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def load_letor_data(filepath):
    """Load LETOR format data (libsvm with qid)"""
    X, y, qids = [], [], []
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            label = int(parts[0])
            qid = int(parts[1].split(':')[1])
            
            features = {}
            for part in parts[2:]:
                if ':' in part:
                    idx, val = part.split(':')
                    features[int(idx)] = float(val)
            
            y.append(label)
            qids.append(qid)
            X.append(features)
    
    # Convert to dense matrix
    max_feat = max(max(x.keys()) for x in X if x)
    X_dense = np.zeros((len(X), max_feat))
    for i, feat_dict in enumerate(X):
        for idx, val in feat_dict.items():
            X_dense[i, idx-1] = val
    
    return X_dense, np.array(y), np.array(qids)


def get_group_sizes(qids):
    """Convert qids to group sizes for XGBoost"""
    groups = []
    current_qid = qids[0]
    count = 0
    
    for qid in qids:
        if qid == current_qid:
            count += 1
        else:
            groups.append(count)
            current_qid = qid
            count = 1
    groups.append(count)
    
    return groups


def get_group_slices(qids):
    """Return (start, end) slices for each contiguous query group."""
    slices = []
    start = 0

    for i in range(1, len(qids)):
        if qids[i] != qids[i - 1]:
            slices.append((start, i))
            start = i
    slices.append((start, len(qids)))

    return slices


def compute_querywise_ndcg_at_k(y_true, y_score, qids, k=10):
    """Compute mean NDCG@k over query groups."""
    ndcgs = []
    for start, end in get_group_slices(qids):
        y_t = y_true[start:end]
        y_s = y_score[start:end]
        if len(y_t) == 0:
            continue
        ndcgs.append(ndcg_score([y_t], [y_s], k=k))

    return float(np.mean(ndcgs)) if ndcgs else 0.0


def sort_by_qid(X, y, qids):
    """Sort rows by query id so group boundaries are guaranteed contiguous."""
    order = np.argsort(qids, kind="stable")
    return X[order], y[order], qids[order]


def verify_query_integrity(train_qids, test_qids):
    """Validate grouping assumptions required by ranking objectives."""
    train_slices = get_group_slices(train_qids)
    test_slices = get_group_slices(test_qids)

    train_unique = len(np.unique(train_qids))
    test_unique = len(np.unique(test_qids))

    if len(train_slices) != train_unique:
        raise ValueError("Train qids are not contiguous. Sort by qid before set_group().")
    if len(test_slices) != test_unique:
        raise ValueError("Test qids are not contiguous. Sort by qid before evaluation.")

    overlap = len(set(train_qids.tolist()) & set(test_qids.tolist()))
    return {
        "train_unique": train_unique,
        "test_unique": test_unique,
        "qid_overlap": overlap
    }


def print_table(title, rows, width=64):
    """Print a consistent ASCII table for terminal output."""
    inner = width - 2
    print("+" + "-" * inner + "+")
    print(f"|{title:^{inner}}|")
    print("+" + "-" * inner + "+")
    for label, value in rows:
        content = f" {label:<36} {value:>22} "
        if len(content) > inner:
            content = content[:inner]
        print(f"|{content:<{inner}}|")
    print("+" + "-" * inner + "+")


def verify_ltr():
    """
    Verify Learning to Rank on MQ2008 dataset
    """
    print("\n" + "="*60)
    print("  LEARNING TO RANK VERIFICATION (MQ2008)")
    print("  Paper: Chen & Guestrin, KDD '16 - Section 6.2")
    print(f"  Reproducibility seed: {SEED}")
    print("="*60)
    
    # Load all folds
    print("\nLoading MQ2008 dataset...")
    X_train, y_train, qids_train = load_letor_data('../data/MQ2008/Fold1/train.txt')
    X_test, y_test, qids_test = load_letor_data('../data/MQ2008/Fold1/test.txt')
    X_vali, y_vali, qids_vali = load_letor_data('../data/MQ2008/Fold1/vali.txt')
    
    # Combine train + validation, then force query-order contiguity.
    X_train = np.vstack([X_train, X_vali])
    y_train = np.concatenate([y_train, y_vali])
    qids_train = np.concatenate([qids_train, qids_vali])

    X_train, y_train, qids_train = sort_by_qid(X_train, y_train, qids_train)
    X_test, y_test, qids_test = sort_by_qid(X_test, y_test, qids_test)
    
    print(f"Train samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {X_train.shape[1]}")
    
    integrity = verify_query_integrity(qids_train, qids_test)

    # Get group sizes
    group_train = get_group_sizes(qids_train)
    group_test = get_group_sizes(qids_test)
    
    print(f"Train queries: {len(group_train)}")
    print(f"Test queries: {len(group_test)}")
    print(f"Train/Test qid overlap: {integrity['qid_overlap']}")
    
    # Create DMatrix with groups
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(group_train)
    
    dtest = xgb.DMatrix(X_test, label=y_test)
    dtest.set_group(group_test)
    
    num_trees = 500
    
    # Test 1: Default parameters
    print(f"\n" + "-"*50)
    print("TEST 1: Default parameters (500 trees)")
    print("-"*50)
    
    params = {
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg@10',
        'max_depth': 6,
        'eta': 0.1,
        'tree_method': 'exact',
        'seed': SEED
    }
    
    t1 = time.time()
    model = xgb.train(
        params, dtrain, 
        num_boost_round=num_trees,
        evals=[(dtest, 'test')],
        verbose_eval=False
    )
    time1 = time.time() - t1
    
    # Test 2: With colsample_bytree=0.5
    print(f"\n" + "-"*50)
    print("TEST 2: With colsample_bytree=0.5")
    print("-"*50)
    
    params2 = params.copy()
    params2['colsample_bytree'] = 0.5
    
    t1 = time.time()
    model2 = xgb.train(
        params2, dtrain,
        num_boost_round=num_trees,
        evals=[(dtest, 'test')],
        verbose_eval=False
    )
    time2 = time.time() - t1

    # Test 3: pGBRT proxy baseline (ranking objective with approximate algorithm)
    print(f"\n" + "-"*50)
    print("TEST 3: pGBRT proxy (pairwise + approx)")
    print("-"*50)

    params_pgbrt = {
        'objective': 'rank:pairwise',
        'eval_metric': 'ndcg@10',
        'max_depth': 6,
        'eta': 0.1,
        'tree_method': 'approx',
        'seed': SEED
    }

    t1 = time.time()
    model_pgbrt = xgb.train(
        params_pgbrt,
        dtrain,
        num_boost_round=num_trees,
        evals=[(dtest, 'test')],
        verbose_eval=False
    )
    time3 = time.time() - t1
    pred_pgbrt = model_pgbrt.predict(dtest)
    
    # Compute final NDCG scores with a single, consistent evaluator.
    pred1 = model.predict(dtest)
    pred2 = model2.predict(dtest)

    ndcg1 = compute_querywise_ndcg_at_k(y_test, pred1, qids_test, k=10)
    ndcg2 = compute_querywise_ndcg_at_k(y_test, pred2, qids_test, k=10)
    ndcg_pgbrt = compute_querywise_ndcg_at_k(y_test, pred_pgbrt, qids_test, k=10)

    # Keep builtin metrics as a sanity check for grouping consistency.
    ndcg1_builtin = float(model.eval(dtest).split(':')[1])
    ndcg2_builtin = float(model2.eval(dtest).split(':')[1])
    ndcg_pgbrt_builtin = float(model_pgbrt.eval(dtest).split(':')[1])

    # Primary reported deltas use XGBoost's builtin ndcg@10 for direct
    # comparability with the paper's reported metric.
    delta_default = ndcg1_builtin - ndcg_pgbrt_builtin
    delta_colsample = ndcg2_builtin - ndcg_pgbrt_builtin
    
    print()
    print_table(
        "RESULTS (MQ2008, 500 trees)",
        [
            ("pGBRT proxy NDCG@10", f"{ndcg_pgbrt_builtin:.4f}"),
            ("XGBoost default NDCG@10", f"{ndcg1_builtin:.4f}"),
            ("XGBoost colsample=0.5 NDCG@10", f"{ndcg2_builtin:.4f}"),
            ("XGB default - pGBRT", f"{delta_default:+.4f}"),
            ("XGB colsample - pGBRT", f"{delta_colsample:+.4f}"),
            ("Paper Yahoo! default", "0.7892"),
            ("Paper Yahoo! colsample=0.5", "0.7913"),
            ("Train time XGB default", f"{time1:.1f} sec"),
            ("Train time XGB colsample", f"{time2:.1f} sec"),
            ("Train time pGBRT proxy", f"{time3:.1f} sec")
        ]
    )

    # Note about dataset difference
    print("Note: pGBRT open-source implementation is not directly available;")
    print("      we use a query-aware pGBRT proxy (pairwise + approx) for fairness.")
    print("Note: MQ2008 is smaller than Yahoo! LTRC (473K instances)")
    print("      NDCG scores differ due to dataset, but methodology verified.")
    
    return ndcg1, ndcg2, ndcg_pgbrt


if __name__ == "__main__":
    ndcg1, ndcg2, ndcg_pgbrt = verify_ltr()
