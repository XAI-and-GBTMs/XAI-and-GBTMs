"""
XGBoost Cache-Aware Access Verification
=======================================
Paper: "XGBoost: A Scalable Tree Boosting System" - Chen & Guestrin (KDD '16)
Section 4.1 - Cache-aware Access

PAPER CLAIM: Cache-aware prefetching gives ~2x speedup on large datasets
             by using optimal block size to reduce CPU cache misses.

The problem: During split finding, XGBoost must accumulate gradient statistics
in sorted order by feature value. This causes NON-CONTIGUOUS memory access
to the gradient/hessian arrays, causing CPU cache misses.

The solution: Prefetch gradient statistics into a contiguous buffer before
accumulation. This is O(1) extra memory but gives ~2x speedup.
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
except ImportError:
    install("numpy")
    install("pandas")
    import numpy as np
    import pandas as pd

import time


GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)


def test_cache_effect_demo():
    """
    Demonstrate WHY cache-aware access is faster.
    This simulates XGBoost's internal gradient accumulation.
    """
    print("\n" + "="*60)
    print("  CACHE-AWARE vs NAIVE - Memory Access Demo")
    print("  (Simulates XGBoost's gradient accumulation)")
    print("="*60)
    
    n = 10_000_000
    print(f"\nData size: {n:,} samples")
    
    # Gradient/hessian arrays (stored by sample order)
    rng = np.random.default_rng(GLOBAL_SEED)
    gradients = rng.standard_normal(n, dtype=np.float32)
    hessians = np.abs(rng.standard_normal(n, dtype=np.float32))
    
    # Sorted indices (sorted by feature value - random order relative to gradients)
    sorted_indices = rng.permutation(n).astype(np.int32)
    
    # NAIVE: Random memory access (causes cache misses)
    print("\n1. NAIVE (random memory access)...")
    t1 = time.time()
    G_naive, H_naive = 0.0, 0.0
    for idx in sorted_indices[:1_000_000]:
        G_naive += gradients[idx]
        H_naive += hessians[idx]
    naive_time = time.time() - t1
    
    # CACHE-AWARE: Prefetch into contiguous buffer
    print("2. CACHE-AWARE (prefetch into buffer)...")
    t1 = time.time()
    buffer_g = gradients[sorted_indices[:1_000_000]]
    buffer_h = hessians[sorted_indices[:1_000_000]]
    G_cache = buffer_g.sum()
    H_cache = buffer_h.sum()
    cache_time = time.time() - t1
    
    speedup = naive_time / cache_time
    
    print(f"""
+--------------------------------------------------+
|              RESULTS                             |
+--------------------------------------------------+
|  Naive (random access):     {naive_time:>8.3f} sec        |
|  Cache-aware (prefetch):    {cache_time:>8.3f} sec        |
+--------------------------------------------------+
|  SPEEDUP:                   {speedup:>8.1f}x           |
+--------------------------------------------------+
""")
    
    return speedup


def _measure_access_speedup(sample_size=1_000_000, seed=42):
    """
    Measure random-access vs prefetch-access ratio as a proxy for
    cache-miss overhead during gradient accumulation.
    """
    rng = np.random.default_rng(seed)
    gradients = rng.standard_normal(sample_size, dtype=np.float32)
    hessians = np.abs(rng.standard_normal(sample_size, dtype=np.float32))
    sorted_indices = rng.permutation(sample_size).astype(np.int32)

    t1 = time.time()
    g_naive, h_naive = 0.0, 0.0
    for idx in sorted_indices:
        g_naive += gradients[idx]
        h_naive += hessians[idx]
    naive_time = time.time() - t1

    t1 = time.time()
    buffer_g = gradients[sorted_indices]
    buffer_h = hessians[sorted_indices]
    _ = buffer_g.sum() + buffer_h.sum()
    cache_time = time.time() - t1

    speedup = naive_time / max(cache_time, 1e-9)
    return speedup


def test_cache_aware_higgs():
    """
    Paper Table 3 & Figure 5: HIGGS 10M with exact greedy
    """
    print("\n" + "="*60)
    print("  XGBOOST EXACT GREEDY (HIGGS 10M)")
    print("  Paper: Section 4.1, Figure 5")
    print("="*60)
    
    print("\nLoading HIGGS dataset (10M rows)...")
    df = pd.read_csv('HIGGS.csv', header=None, nrows=10_000_000)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    
    print(f"Samples: {X.shape[0]:,}")
    print(f"Features: {X.shape[1]}")
    
    dtrain = xgb.DMatrix(X, label=y)
    del X, df
    
    num_trees = 10
    params = {
        'max_depth': 6,
        'objective': 'binary:logistic',
        'tree_method': 'exact',
        'seed': GLOBAL_SEED,
        'nthread': 1,
    }
    
    print(f"\nTraining {num_trees} trees (cache-aware ON, exact greedy)...")
    t1 = time.time()
    xgb.train(params, dtrain, num_boost_round=num_trees, verbose_eval=False)
    total_time = time.time() - t1
    cache_time_per_tree = total_time / num_trees

    # XGBoost does not expose a runtime switch to disable cache-aware access.
    # We therefore estimate a naive baseline using measured memory-access ratio.
    access_speedup = _measure_access_speedup(sample_size=1_000_000, seed=GLOBAL_SEED)
    naive_time_per_tree_est = cache_time_per_tree * access_speedup
    
    print(f"""
+--------------------------------------------------+
|              RESULTS (HIGGS 10M)                 |
+--------------------------------------------------+
|  Cache-aware (measured):    {cache_time_per_tree:>8.2f} sec/tree |
|  Naive (estimated*):        {naive_time_per_tree_est:>8.2f} sec/tree |
+--------------------------------------------------+
|  Estimated speedup:             {access_speedup:>8.2f}x           |
+--------------------------------------------------+
* Estimated = cache-aware time multiplied by measured random-vs-prefetch
    memory-access penalty. XGBoost does not expose a switch to disable
    cache-aware access in training.
""")
    
    return {
        'cache_time_per_tree': cache_time_per_tree,
        'naive_time_per_tree_est': naive_time_per_tree_est,
        'estimated_speedup': access_speedup,
    }


def test_cache_aware_allstate():
    """
    Paper Figure 5: Allstate 10M with exact greedy
    """
    print("\n" + "="*60)
    print("  XGBOOST EXACT GREEDY (Allstate 10M)")
    print("  Paper: Section 4.1, Figure 5")
    print("="*60)
    
    from sklearn.preprocessing import OneHotEncoder
    
    print("\nLoading Allstate dataset (10M rows)...")
    df = pd.read_csv('allstate_raw/train_set.csv', nrows=10_000_000, low_memory=False)
    
    cat_cols = ['Blind_Make', 'Blind_Model', 'Blind_Submodel', 
                'Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5', 'Cat6', 
                'Cat7', 'Cat8', 'Cat9', 'Cat10', 'Cat11', 'Cat12', 
                'NVCat']
    
    for col in cat_cols:
        df[col] = df[col].astype(str)
    
    encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    X_sparse = encoder.fit_transform(df[cat_cols])
    y = (df['Claim_Amount'] > 0).astype(int).values
    
    print(f"Samples: {X_sparse.shape[0]:,}")
    print(f"Features: {X_sparse.shape[1]:,}")
    
    dtrain = xgb.DMatrix(X_sparse, label=y)
    del df, X_sparse
    
    num_trees = 10
    params = {
        'max_depth': 6,
        'objective': 'binary:logistic',
        'tree_method': 'exact',
        'seed': GLOBAL_SEED,
        'nthread': 1,
    }
    
    print(f"\nTraining {num_trees} trees (cache-aware ON, exact greedy)...")
    t1 = time.time()
    xgb.train(params, dtrain, num_boost_round=num_trees, verbose_eval=False)
    total_time = time.time() - t1
    cache_time_per_tree = total_time / num_trees

    # XGBoost does not expose a runtime switch to disable cache-aware access.
    # We therefore estimate a naive baseline using measured memory-access ratio.
    access_speedup = _measure_access_speedup(sample_size=1_000_000, seed=GLOBAL_SEED + 1)
    naive_time_per_tree_est = cache_time_per_tree * access_speedup
    
    print(f"""
+--------------------------------------------------+
|             RESULTS (Allstate 10M)               |
+--------------------------------------------------+
|  Cache-aware (measured):    {cache_time_per_tree:>8.2f} sec/tree |
|  Naive (estimated*):        {naive_time_per_tree_est:>8.2f} sec/tree |
+--------------------------------------------------+
|  Estimated speedup:             {access_speedup:>8.2f}x           |
+--------------------------------------------------+
* Estimated = cache-aware time multiplied by measured random-vs-prefetch
    memory-access penalty. XGBoost does not expose a switch to disable
    cache-aware access in training.
""")
    
    return {
        'cache_time_per_tree': cache_time_per_tree,
        'naive_time_per_tree_est': naive_time_per_tree_est,
        'estimated_speedup': access_speedup,
    }


if __name__ == "__main__":
    print(f"Reproducibility seed: {GLOBAL_SEED}")

    # First show WHY cache-aware is faster
    demo_speedup = test_cache_effect_demo()
    
    # Then run actual XGBoost tests
    higgs_result = test_cache_aware_higgs()
    allstate_result = test_cache_aware_allstate()
    
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"""
WHY Cache-Aware is Faster:
  Random memory access causes CPU cache misses.
  Prefetching into contiguous buffer gives {demo_speedup:.0f}x speedup!

XGBoost Results (10M rows, exact greedy):
    HIGGS (cache-aware measured): {higgs_result['cache_time_per_tree']:.2f} sec/tree
    HIGGS (naive estimated):      {higgs_result['naive_time_per_tree_est']:.2f} sec/tree
    Allstate (cache-aware measured): {allstate_result['cache_time_per_tree']:.2f} sec/tree
    Allstate (naive estimated):      {allstate_result['naive_time_per_tree_est']:.2f} sec/tree

Note: Modern XGBoost ALWAYS uses cache-aware mode.
HIGGS naive is estimated from measured memory-access ratio.
""")

