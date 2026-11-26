import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from single_ct_utils import (
    GEOMETRY_COLUMNS, FEATURE_COLUMNS, collect_images, make_case_id,
    patient_id, read_partition_settings, same_geometry, validate_partition_arrays,
)

# ---------- Manually edit center name and paths before running ----------
CENTER_NAME = "CENTER_NAME"
mask_dir = r"MASK_DIR"
feature_root = r"FEATURE_ROOT"
output_dir = r"OUTPUT_DIR"
from config_loader import apply_config
apply_config("step2", globals())   

def get_patient_id(filename):
    return patient_id(filename)

def find_map(directory, feature_name, case_id):
    prefix = feature_name + "_"
    matches = [p for p in (Path(directory) / feature_name).iterdir()
               if p.is_file() and p.name.startswith(prefix)
               and patient_id(p.name[len(prefix):]) == case_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one {feature_name} map for {case_id}, found {len(matches)}")
    return sitk.ReadImage(str(matches[0]))


def extract_voxels(mask_path, directory, patient_id, center_name=None):
    center_name = CENTER_NAME if center_name is None else center_name
    case_id = make_case_id(center_name, patient_id)
    mask = sitk.ReadImage(str(mask_path))
    maps = {name: find_map(directory, name, patient_id) for name in FEATURE_COLUMNS + ["Region"]}
    if not all(same_geometry(mask, image) for image in maps.values()):
        raise ValueError(f"{patient_id}: feature maps and full tumor mask must share geometry")
    roi = sitk.GetArrayFromImage(mask) > 0
    regions = sitk.GetArrayFromImage(maps["Region"])
    if not np.isin(regions, [0, 1, 2]).all():
        raise ValueError(f"{patient_id}: Region map must contain only 0,1,2")
    if not maps["Region"].HasMetaDataKey("RimThicknessMM"):
        raise ValueError("Missing partition thickness; regenerate maps with step 1")
    mode, thickness = read_partition_settings(maps["Region"], mask)
    validate_partition_arrays(regions == 1, regions == 2, roi, mode)
    coordinates = np.column_stack(np.where(roi))  # (z, y, x)
    data = pd.DataFrame({
        "Direction": [repr(tuple(mask.GetDirection()))] * len(coordinates),
        "Origin": [repr(tuple(mask.GetOrigin()))] * len(coordinates),
        "Spacing": [repr(tuple(mask.GetSpacing()))] * len(coordinates),
        "Dimension": [repr(tuple(mask.GetSize()))] * len(coordinates),
        "Coordinate": [repr(tuple(int(v) for v in row)) for row in coordinates],
    }, columns=GEOMETRY_COLUMNS)
    for name in FEATURE_COLUMNS:
        data[name] = sitk.GetArrayFromImage(maps[name])[roi]
    data["Region"] = np.where(regions[roi] == 1, "core", "rim")
    data["RimThicknessMM"] = thickness
    data["PartitionMode"] = mode
    data["Center"] = center_name
    data["PatientID"] = patient_id
    data["CaseID"] = case_id
    return data


def main():
    masks = collect_images(mask_dir)
    os.makedirs(output_dir, exist_ok=True)
    for patient_id, path in masks.items():
        data = extract_voxels(path, feature_root, patient_id)
        # Rebuild rather than silently reusing old dual-sequence tables.
        case_id = make_case_id(CENTER_NAME, patient_id)
        data.to_excel(os.path.join(output_dir, f"{case_id}.xlsx"), index=False)
        print(f"Completed: {case_id}")


if __name__ == "__main__":
    main()
