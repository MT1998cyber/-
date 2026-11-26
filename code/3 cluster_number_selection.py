import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from single_ct_utils import cluster_space, load_training_features, make_cluster_pipeline

# ---------- Manually set a directory containing TRAINING tables only ----------
data_dir = r"DATA_DIR"    
output_dir = None          # None = do not save figures/table
K_RANGE = list(range(2, 11))
SHOW_PLOTS = True
from config_loader import apply_config
apply_config("step3", globals())   # values from pipeline_config.py (if present)

if not SHOW_PLOTS:
    matplotlib.use("Agg")


def evaluate_clusters(train_data, k_range=range(2, 11)):
    unique_count = len(np.unique(train_data.to_numpy(), axis=0))
    scores = []
    for k in k_range:
        if k >= len(train_data) or k > unique_count:
            continue
        pipeline = make_cluster_pipeline(k)
        labels = pipeline.fit_predict(train_data)
        scaled = cluster_space(pipeline, train_data)
        if not 1 < len(np.unique(labels)) < len(train_data):
            continue
        scores.append((k, calinski_harabasz_score(scaled, labels), davies_bouldin_score(scaled, labels)))
        print(f"K={k}: CH={scores[-1][1]:.1f}  DB={scores[-1][2]:.3f}", flush=True)
    if not scores:
        raise ValueError("Insufficient distinct finite core voxels to evaluate K >= 2")
    return scores


def main():
    train_data, _ = load_training_features(data_dir)
    print("Training core voxel data:", train_data.shape)
    scores = evaluate_clusters(train_data, K_RANGE)
    ks, ch, db = np.asarray(scores).T
    print("Best K by CH (higher is better):", int(ks[np.argmax(ch)]))
    print("Best K by DB (lower is better):", int(ks[np.argmin(db)]))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame({"K": ks.astype(int), "calinski_harabasz": ch, "davies_bouldin": db}) \
            .to_csv(os.path.join(output_dir, "k_selection_scores.csv"), index=False)
    for values, ylabel, fname in [(ch, "Calinski-Harabasz Index", "k_selection_CH.png"),
                                  (db, "Davies-Bouldin Index", "k_selection_DB.png")]:
        plt.figure(figsize=(6, 4))
        plt.plot(ks, values, marker="o")
        plt.xticks(ks)
        plt.xlabel("Number of Clusters")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, fname), dpi=150)
        if SHOW_PLOTS:
            plt.show()
        plt.close()
    if output_dir:
        print(f"Saved K-selection figures and table to {output_dir}")


if __name__ == "__main__":
    main()
