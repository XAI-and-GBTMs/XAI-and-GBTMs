"""
XGBoost Out-of-Core Verification
================================
Paper: "XGBoost: A Scalable Tree Boosting System" - Chen & Guestrin (KDD '16)
Section 4.3 - Out-of-core Computation

This run verifies only the practical parts we can test here:
    - In-memory scaling behavior
    - I/O vs compute tradeoff

We use Criteo day_2-5 dataset (4 files, ~6GB compressed).
Note: true disk-bound out-of-core behavior is not tested in this run.
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
except ImportError:
    install("numpy")
    install("pandas")
    import numpy as np
    import pandas as pd

import time
import os


def load_criteo_gz(filepath, nrows=None):
    """Load Criteo gzipped data"""
    col_names = ['label'] + [f'num_{i}' for i in range(13)] + [f'cat_{i}' for i in range(26)]
    df = pd.read_csv(filepath, sep='\t', header=None, names=col_names, 
                     nrows=nrows, compression='gzip')
    num_cols = [f'num_{i}' for i in range(13)]
    return df[num_cols].fillna(0).values, df['label'].values


def verify_out_of_core():
    """
    Verify practical out-of-core behavior that can be tested locally.
    """
    print("\n" + "="*60)
    print("  OUT-OF-CORE VERIFICATION")
    print("  Paper: Chen & Guestrin, KDD '16 - Section 4.3")
    print("="*60)
    
    gz_files = ['../data/day_2.gz', '../data/day_3.gz', '../data/day_4.gz', '../data/day_5.gz']
    gz_files = [f for f in gz_files if os.path.exists(f)]
    if not gz_files:
        print("\nNo day_2.gz/day_3.gz/day_4.gz/day_5.gz files found.")
        print("Cannot run this verification without at least one gz file.")
        return []

    total_gz = sum(os.path.getsize(f) for f in gz_files) / (1024**3)
    print(f"\nDatasets: {len(gz_files)} files, {total_gz:.2f} GB compressed")
    
    # Test 1: Compression benefit
    print("\n" + "-"*60)
    print("TEST 1: I/O vs Compute Tradeoff (2M rows)")
    print("-"*60)
    
    n = 2_000_000
    col_names = ['label'] + [f'num_{i}' for i in range(13)] + [f'cat_{i}' for i in range(26)]
    
    if os.path.exists('../data/day_2.tsv'):
        t1 = time.time()
        df = pd.read_csv('../data/day_2.tsv', sep='\t', header=None, names=col_names, nrows=n)
        uncomp_time = time.time() - t1
        print(f"  Uncompressed: {uncomp_time:.1f} sec")
    else:
        uncomp_time = 0
        print("  Uncompressed: (day_2.tsv not found)")
    
    t1 = time.time()
    df = pd.read_csv('../data/day_2.gz', sep='\t', header=None, names=col_names, nrows=n, compression='gzip')
    comp_time = time.time() - t1
    print(f"  Compressed:   {comp_time:.1f} sec")
    
    # Test 2: In-memory scaling across files
    print("\n" + "-"*60)
    print("TEST 2: In-memory Scaling (2M rows each)")
    print("-"*60)
    
    results = []
    rows_per_file = 2_000_000
    
    for num_files in [1, 2, len(gz_files)]:
        if num_files > len(gz_files):
            continue
        files_to_use = gz_files[:num_files]
        total_rows = num_files * rows_per_file
        
        print(f"\n{num_files} file(s), {total_rows:,} rows:")
        
        t1 = time.time()
        Xs, ys = [], []
        for f in files_to_use:
            X, y = load_criteo_gz(f, nrows=rows_per_file)
            Xs.append(X)
            ys.append(y)
        X = np.vstack(Xs)
        y = np.concatenate(ys)
        load_time = time.time() - t1
        
        dtrain = xgb.DMatrix(X, label=y)
        t1 = time.time()
        xgb.train({'max_depth': 6, 'objective': 'binary:logistic', 'tree_method': 'hist'}, 
                  dtrain, num_boost_round=10, verbose_eval=False)
        train_time = time.time() - t1
        
        throughput = total_rows / train_time / 1_000_000
        print(f"  Load: {load_time:.1f}s | Train: {train_time:.1f}s | {throughput:.2f} M rows/sec")
        
        results.append({'files': num_files, 'rows': total_rows, 'load': load_time, 'train': train_time})
        del X, y, Xs, ys, dtrain
    
    # Summary
    print("\nPart verified in this run:")
    print("  1) Verified in-memory scaling")
    print("  2) Observed I/O vs compute tradeoff")

    print(f"""
+----------------------------------------------------------+
|              RESULTS SUMMARY                             |
+----------------------------------------------------------+
|  Files | Rows       | Load Time | Train Time | Throughput|
+----------------------------------------------------------+""")
    for r in results:
        tp = r['rows'] / r['train'] / 1_000_000
        print(f"|  {r['files']}    | {r['rows']:>10,} |   {r['load']:>5.1f} s |    {r['train']:>5.1f} s | {tp:>5.2f} M/s |")
    print("+----------------------------------------------------------+")

    print("\n3) Could not test true disk-bound regime")
    
    return results


if __name__ == "__main__":
    results = verify_out_of_core()
