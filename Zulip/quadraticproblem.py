import numpy as np
import pandas as pd
import json
import math

def compute_R(P, v, M_param, C, p):
    """Reward for investigating claim i."""
    return P * (v * M_param * p + C + M_param) - C

def compute_B_lin(clusters, C, alpha, k):
    """B_c = (sum_{j in I_c} C_j) * alpha * tanh(|I_c| / k)"""
    K = len(clusters)
    B_lin = np.zeros(K)
    for c in range(K):
        sum_cj = sum(C[j] for j in clusters[c])
        card = len(clusters[c])
        B_lin[c] = sum_cj * alpha * np.tanh(card / k)
    return B_lin

def load_data(csv_path, clusters_json_path):
    df = pd.read_csv(csv_path).sort_values("claim_id").reset_index(drop=True)
    claim_to_idx = {cid: i for i, cid in enumerate(df["claim_id"])}
    with open(clusters_json_path) as f:
        raw_clusters = json.load(f)
    clusters, skipped = {}, []
    for c_str, claim_ids in raw_clusters.items():
        indices = [claim_to_idx[cid] for cid in claim_ids if cid in claim_to_idx]
        [skipped.append(cid) for cid in claim_ids if cid not in claim_to_idx]
        if indices:
            clusters[len(clusters)] = indices
    if skipped:
        print(f"[WARNING] {len(skipped)} claims not in CSV (ignored)")
    return df, clusters
