import os
import joblib
import numpy as np

from single_ct_utils import (
    FEATURE_COLUMNS, SCHEMA_VERSION, load_training_features, make_cluster_pipeline,
)

# ---------- Manually edit paths; train_dir must exclude validation patients ----------
train_dir = r"TRAIN_DATA_DIR"    
model_dir = r"MODEL_DIR"    

# Select using step 3 and training data only. Retain the original K=3 default.
N_CLUSTERS = 3
from config_loader import apply_config
apply_config("step4", globals())   

def train_model(directory, n_clusters=N_CLUSTERS):
    train_data, thickness_mm = load_training_features(directory)
    if n_clusters < 2 or n_clusters > 254:
        raise ValueError("Use 2..254 core clusters; one additional label is reserved for rim")
    if len(np.unique(train_data.to_numpy(), axis=0)) < n_clusters:
        raise ValueError("Fewer distinct finite core feature vectors than requested clusters")
    pipeline = make_cluster_pipeline(n_clusters)
    pipeline.fit(train_data)
    print("Training core data shape:", train_data.shape)
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_columns": FEATURE_COLUMNS.copy(),
        "cluster_region": "core",
        "n_clusters": n_clusters,
        "rim_thickness_mm": thickness_mm,
        "partition_mode": train_data.attrs["partition_mode"],
        "pipeline": pipeline,
    }


def main():
    model = train_model(train_dir)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"joint_kmeans_k{N_CLUSTERS}.pkl")
    joblib.dump(model, model_path)
    print("Model saved (training-fitted scaler + single-CT K-means):", model_path)


if __name__ == "__main__":
    main()
