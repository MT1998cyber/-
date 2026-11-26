import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import SimpleITK as sitk

from single_ct_utils import (
    FEATURE_COLUMNS, SCHEMA_VERSION, core_features,
    geometry_and_coordinates, read_voxel_table, table_partition_settings,
)

# ---------- Manually edit paths for the cohort currently being predicted ----------
input_dir = r"INPUT_DIR"
model_path = r"MODEL_PATH"
output_dir = r"OUTPUT_DIR"
from config_loader import apply_config
apply_config("step5", globals())   

def generate_habitat(data, model):
    if not isinstance(model, dict) or model.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Legacy/incompatible model; retrain the single-CT model using step 4")
    if model.get("feature_columns") != FEATURE_COLUMNS or model.get("cluster_region") != "core":
        raise ValueError("Model feature schema or clustering region does not match")
    mode, _ = table_partition_settings(data)
    if mode != model.get("partition_mode", "inner"):
        raise ValueError("Patient partition mode differs from the training model")
    if not np.allclose(data["RimThicknessMM"], model["rim_thickness_mm"], rtol=0, atol=1e-6):
        raise ValueError("ROI rim thickness differs from the training model")
    geometry, coordinates = geometry_and_coordinates(data)
    n_clusters = model["n_clusters"]
    features = core_features(data)
    if features.empty:
        raise ValueError("No core voxel with all features finite")
    # Predict only: never refit the scaler or cluster centers on this cohort.
    labels = model["pipeline"].predict(features)
    habitat_labels = pd.Series(pd.NA, index=data.index, dtype="Int64")
    habitat_labels.loc[features.index] = labels + 1
    habitat_labels.loc[data["Region"] == "rim"] = n_clusters + 1

    data = data.copy()
    # Excel and NRRD both use core=1..K, rim=K+1. Invalid core voxels remain 0
    # in the NRRD and blank in Excel, never silently assigned to a habitat.
    data[f"cluster{n_clusters}_label"] = habitat_labels
    size = tuple(int(v) for v in geometry["Dimension"])
    habitat = np.zeros(size[::-1], dtype=np.uint8)
    valid = habitat_labels.notna().to_numpy()
    habitat[tuple(coordinates[valid].T)] = habitat_labels.loc[valid].to_numpy(dtype=np.uint8)
    image = sitk.GetImageFromArray(habitat)
    image.SetSpacing(tuple(geometry["Spacing"]))
    image.SetOrigin(tuple(geometry["Origin"]))
    image.SetDirection(tuple(geometry["Direction"]))
    image.SetMetaData("RimThicknessMM", str(model["rim_thickness_mm"]))
    image.SetMetaData("PartitionMode", mode)
    rim_label = "none" if mode == "none" else str(n_clusters + 1)
    image.SetMetaData("HabitatLabels", f"background=0; core=1..{n_clusters}; rim={rim_label}")
    image.SetMetaData("InvalidCoreVoxelCount", str(int(((data["Region"] == "core") & habitat_labels.isna()).sum())))
    return data, image

def main():
    model = joblib.load(model_path)
    paths = sorted(p for p in Path(input_dir).iterdir()
                   if p.suffix.lower() in (".xlsx", ".xls") and not p.name.startswith("~$"))
    if not paths:
        raise ValueError(f"No voxel tables in {input_dir}")
    os.makedirs(output_dir, exist_ok=True)
    for path in paths:
        data, image = generate_habitat(read_voxel_table(path), model)
        data.to_excel(path, index=False)
        sitk.WriteImage(image, os.path.join(output_dir, path.stem + ".nrrd"), True)
        print(f"Completed: {path.stem}")


if __name__ == "__main__":
    main()
