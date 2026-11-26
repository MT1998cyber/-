import json
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt

from single_ct_utils import (
    RIM_THICKNESS_MM, collect_images, image_from_array, partition_roi, require_expected_grid,
    same_geometry,
)

mask_dir = r"MASK_DIR"
image_dir = r"IMAGE_DIR"
PARTITION_MODE = "none" 
output_dir = r"OUTPUT_DIR"
core_dir = os.path.join(output_dir, "core")
rim_dir = os.path.join(output_dir, "rim")

# NEW (outer mode only): keep only rim voxels whose CT value lies in this HU range,
# so skull (>~100 HU) and air/scalp fat (<0 HU) never enter the perihematomal rim.
# Edema (~10-30 HU), brain and CSF are kept. Set None to reproduce the original behaviour.
# Use the SAME range for every center.
RIM_HU_RANGE = (-15.0, 100.0)
# Keep only ROI voxels whose CT value lies in this HU range BEFORE partitioning
# (removes skull / calcification captured by an imperfect mask; acute hematoma ~40-90 HU).
# None = original behaviour (ROI used as drawn).
CORE_HU_RANGE = None
from config_loader import apply_config
apply_config("step0", globals())   # values from pipeline_config.py (if present)


def partition_by_mode(mask_image, mode="none", thickness_mm=RIM_THICKNESS_MM,
                      ct_image=None, rim_hu_range=None, core_hu_range=None): 
    mode = mode.lower()
    if mode not in {"inner", "outer", "none"}:
        raise ValueError("PARTITION_MODE must be 'inner', 'outer' or 'none'")
    if mask_image.GetDimension() != 3:
        raise ValueError("A 3D ROI mask is required")
    values = sitk.GetArrayViewFromImage(mask_image)
    if not np.isfinite(values).all():
        raise ValueError("ROI mask contains NaN or Inf")
    roi = values > 0
    if not roi.any():
        raise ValueError("Empty tumor ROI")
    if core_hu_range is not None:
        if ct_image is None or not same_geometry(ct_image, mask_image):
            raise ValueError("CORE_HU_RANGE needs the matching CT on the same grid as the mask")
        lo, hi = core_hu_range
        ct_all = sitk.GetArrayViewFromImage(ct_image)
        roi = roi & (ct_all >= lo) & (ct_all <= hi)
        if not roi.any():
            raise ValueError("No ROI voxel inside CORE_HU_RANGE")
        mask_image = image_from_array(roi.astype(np.uint8), mask_image)
    if mode != "none" and (not np.isfinite(thickness_mm) or thickness_mm <= 0):
        raise ValueError("Thickness must be positive and finite for inner/outer mode")

    if mode == "inner":
        core_image, rim_image = partition_roi(mask_image, thickness_mm)
        core = sitk.GetArrayFromImage(core_image) > 0
        rim = sitk.GetArrayFromImage(rim_image) > 0
    elif mode == "outer":
        core = roi.copy()
        distance = distance_transform_edt(~roi, sampling=mask_image.GetSpacing()[::-1])
        rim = (~roi) & (distance <= thickness_mm)
        if rim_hu_range is not None:
            if ct_image is None or not same_geometry(ct_image, mask_image):
                raise ValueError("RIM_HU_RANGE needs the matching CT on the same grid as the mask")
            lo, hi = rim_hu_range
            ct = sitk.GetArrayViewFromImage(ct_image)
            rim = rim & (ct >= lo) & (ct <= hi)
        if not rim.any():
            raise ValueError("No external rim on this grid; inspect thickness, ROI and image field of view")
    else:
        core = roi.copy()
        rim = np.zeros_like(roi)

    analysis = core | rim
    effective_thickness = 0.0 if mode == "none" else float(thickness_mm)
    result = []
    for array, role in [(core, "core"), (rim, "rim"), (analysis, "analysis_roi")]:
        image = image_from_array(array.astype(np.uint8), mask_image)
        image.SetMetaData("RimThicknessMM", str(effective_thickness))
        image.SetMetaData("PartitionMode", mode)
        image.SetMetaData("PartitionRole", role)
        result.append(image)
    return tuple(result)


def main():
    mode = PARTITION_MODE.lower()
    if mode not in {"inner", "outer", "none"}:
        raise ValueError("PARTITION_MODE must be 'inner', 'outer' or 'none'")
    source, destination = Path(mask_dir).resolve(), Path(output_dir).resolve()
    if source == destination:
        raise ValueError("output_dir must differ from mask_dir to preserve original masks")
    if len({destination, Path(core_dir).resolve(), Path(rim_dir).resolve()}) != 3:
        raise ValueError("Analysis ROI, core and rim output directories must be distinct")
    masks = collect_images(mask_dir)
    use_hu = mode == "outer" and RIM_HU_RANGE is not None
    need_ct = use_hu or CORE_HU_RANGE is not None
    images = collect_images(image_dir) if need_ct else {}
    if need_ct and masks.keys() - images.keys():
        raise ValueError(f"Missing CT for masks: {sorted(masks.keys() - images.keys())}")
    config = {
        "mode": mode, "source_mask_dir": str(source),
        "configured_thickness_mm": float(RIM_THICKNESS_MM),
        "effective_rim_thickness_mm": 0.0 if mode == "none" else float(RIM_THICKNESS_MM),
        "rim_hu_range": list(RIM_HU_RANGE) if use_hu else None,
    }
    if CORE_HU_RANGE is not None:  
        config["core_hu_range"] = list(CORE_HU_RANGE)
    config_path = destination / "partition_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Output directory belongs to another partition setup; choose a separate output_dir")
    for directory in (output_dir, core_dir, rim_dir):
        Path(directory).mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    quality = []
    for case_id, path in masks.items():
        mask = sitk.ReadImage(str(path))
        require_expected_grid(mask)
        ct = sitk.ReadImage(str(images[case_id])) if need_ct else None
        core, rim, analysis = partition_by_mode(
            mask, mode, RIM_THICKNESS_MM, ct_image=ct,
            rim_hu_range=RIM_HU_RANGE if use_hu else None, core_hu_range=CORE_HU_RANGE)
        filename = case_id + ".nrrd"
        sitk.WriteImage(core, str(Path(core_dir) / filename), True)
        sitk.WriteImage(rim, str(Path(rim_dir) / filename), True)
        sitk.WriteImage(analysis, str(destination / filename), True)

        array = sitk.GetArrayViewFromImage(analysis)
        touches_boundary = any(
            np.take(array, 0, axis=axis).any() or np.take(array, -1, axis=axis).any()
            for axis in range(3)
        )
        if mode == "outer" and touches_boundary:
            warnings.warn(f"{case_id}: expanded ROI touches image boundary; external rim may be truncated", RuntimeWarning)
        quality.append({
            "PatientID": case_id, "Mode": mode,
            "EffectiveRimThicknessMM": config["effective_rim_thickness_mm"],
            "OriginalVoxels": int(np.count_nonzero(sitk.GetArrayViewFromImage(mask) > 0)),
            "CoreVoxels": int(np.count_nonzero(sitk.GetArrayViewFromImage(core))),
            "RimVoxels": int(np.count_nonzero(sitk.GetArrayViewFromImage(rim))),
            "AnalysisVoxels": int(np.count_nonzero(array)),
            "SpacingXYZ": str(mask.GetSpacing()),
            "TouchesImageBoundary": bool(touches_boundary),
            "ExternalRimMayBeTruncated": bool(mode == "outer" and touches_boundary),
        })
        print(f"Partitioned: {case_id}; mode={mode}; rim={config['effective_rim_thickness_mm']:g} mm")
    pd.DataFrame(quality).to_csv(destination / "partition_qc.csv", index=False, encoding="utf-8-sig")
    if mode == "none":
        print("No-expansion baseline: rim masks are intentionally empty; use the whole-tumor core branch downstream.")


if __name__ == "__main__":
    main()
