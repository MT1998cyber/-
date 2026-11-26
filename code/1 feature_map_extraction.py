import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

from single_ct_utils import (
    FEATURE_COLUMNS, VOXEL_FEATURE_SET, VOXEL_FEATURES, collect_images, image_from_array,
    image_stem, radiomic_voxel_features, read_partition_settings, require_expected_grid,
    same_geometry, validate_partition_arrays, voxel_column,
)

# ---------- Manually edit paths for ONE center before running ----------
image_dir = r"IMAGE_DIR"
mask_dir = r"MASK_DIR"
output_dir = r"OUTPUT_DIR"
params_file = r"PARAMS_FILE_DIR\voxel_radiomics.yaml"

MAX_CORE_NAN_FRACTION = 0.005
MAX_RIM_NAN_FRACTION = 0.05

N_WORKERS = 1
SKIP_EXISTING = True   
from config_loader import apply_config
apply_config("step1", globals())   

_EXTRACTOR = None

def build_extractor():
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
    extractor.addProvenance(False)
    classes = radiomic_voxel_features()
    extractor.disableAllFeatures()
    if classes:
        extractor.enableFeaturesByName(**classes)
    if extractor.settings.get("normalize", False):
        warnings.warn("voxel YAML has normalize: true – bins then differ between patients", RuntimeWarning)
    return extractor


def _init_worker():
    global _EXTRACTOR
    _EXTRACTOR = build_extractor()


def report_path_for(destination, case_name):
    return Path(destination) / "diagnostics" / f"{case_name}_voxel_qc.json"


def write_quality_report(destination, case_name, image, quality, mode, thickness, status):
    path = report_path_for(destination, case_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "case": case_name, "status": status, "spacing_mm": list(image.GetSpacing()),
        "voxel_feature_set": VOXEL_FEATURE_SET, "feature_columns": FEATURE_COLUMNS,
        "partition_mode": mode, "rim_thickness_mm": thickness,
        "max_core_nan_fraction": MAX_CORE_NAN_FRACTION,
        "max_rim_nan_fraction": MAX_RIM_NAN_FRACTION,
        "regions": quality,
    }, indent=2), encoding="utf-8")
    return path


def extract_case(image_path, mask_path, core_path, rim_path, destination, extractor):
    limits = {"core": MAX_CORE_NAN_FRACTION, "rim": MAX_RIM_NAN_FRACTION}
    for region, limit in limits.items():
        if not np.isfinite(limit) or not 0 <= limit < 1:
            raise ValueError(f"MAX_{region.upper()}_NAN_FRACTION must be in [0, 1)")
    case_name = image_stem(image_path)
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    core = sitk.ReadImage(str(core_path))
    rim = sitk.ReadImage(str(rim_path))
    if not all(same_geometry(image, item) for item in (mask, core, rim)):
        raise ValueError(f"{image_path}: CT and ROI partitions must share geometry")
    require_expected_grid(image)
    roi = sitk.GetArrayFromImage(mask) > 0
    core_array = sitk.GetArrayFromImage(core) > 0
    rim_array = sitk.GetArrayFromImage(rim) > 0
    if not all(region.HasMetaDataKey("RimThicknessMM") for region in (core, rim)):
        raise ValueError("Missing partition metadata; rerun step 0")
    mode, thickness = read_partition_settings(core, rim, mask)
    validate_partition_arrays(core_array, rim_array, roi, mode)

    radiomic = [f for f in VOXEL_FEATURES if f != "HU"]
    arrays = {voxel_column(f): np.full(roi.shape, np.nan, dtype=np.float32) for f in radiomic}
    quality = []
    # Core and rim get separate masked local neighbourhoods.
    for region_name, region_array in [("core", core_array), ("rim", rim_array)]:
        total = int(region_array.sum())
        if region_name == "rim" and mode == "none":
            quality.append({"region": "rim", "total_voxels": 0, "invalid_voxels": 0,
                            "invalid_fraction": 0.0, "per_feature_nonfinite": {},
                            "action": "disabled_none"})
            continue
        if radiomic:
            region_mask = image_from_array(region_array.astype(np.uint8), image)
            result = extractor.execute(image, region_mask, voxelBased=True)
            for f in radiomic:
                fmap = result.get("original_" + f)
                if not isinstance(fmap, sitk.Image):
                    raise ValueError(f"Voxel map original_{f} missing; check imageType Original in YAML")
                aligned = sitk.Resample(fmap, image, sitk.Transform(), sitk.sitkNearestNeighbor,
                                        float("nan"), sitk.sitkFloat32)
                arrays[voxel_column(f)][region_array] = sitk.GetArrayFromImage(aligned)[region_array]
        if arrays:
            stack = np.stack([a[region_array] for a in arrays.values()], axis=1)
            finite = np.isfinite(stack)
            invalid_rows = ~finite.all(axis=1)
            per_feature = {c: int((~finite[:, i]).sum()) for i, c in enumerate(arrays) if (~finite[:, i]).any()}
            has_inf = bool(np.isinf(stack).any())
        else:
            invalid_rows, per_feature, has_inf = np.zeros(total, bool), {}, False
        count = int(invalid_rows.sum())
        retain = (not has_inf) and 0 < count < total and count / total <= limits[region_name]
        quality.append({
            "region": region_name, "total_voxels": total, "invalid_voxels": count,
            "invalid_fraction": count / total if total else 0.0,
            "per_feature_nonfinite": per_feature, "contains_inf": has_inf,
            "action": "retained_nan" if retain else ("error" if count else "ok"),
        })
        if count:
            invalid = np.zeros(roi.shape, bool)
            invalid[region_array] = invalid_rows
            diag = Path(destination) / "diagnostics"
            diag.mkdir(parents=True, exist_ok=True)
            diag_path = diag / f"{case_name}_{region_name}_invalid_voxels.nrrd"
            sitk.WriteImage(image_from_array(invalid.astype(np.uint8), image), str(diag_path), True)
            msg = (f"Case={case_name}, region={region_name}: {count}/{total} ({count / total:.4%}) voxels "
                   f"with a non-finite feature; per feature: {per_feature}. Mask: {diag_path}. ")
            if retain:
                warnings.warn(msg + "Within tolerance: kept as NaN (core: excluded from K-means).",
                              RuntimeWarning, stacklevel=2)
            else:
                write_quality_report(destination, case_name, image, quality, mode, thickness, "error")
                raise ValueError(msg + "Above tolerance or Inf present; values were not replaced.")

    regions = core_array.astype(np.uint8) + 2 * rim_array.astype(np.uint8)
    region_map = image_from_array(regions, image)
    region_map.SetMetaData("RimThicknessMM", str(thickness))
    region_map.SetMetaData("PartitionMode", mode)
    maps = {name: image_from_array(a, image) for name, a in arrays.items()}
    if "HU" in VOXEL_FEATURES:
        maps[voxel_column("HU")] = sitk.Cast(image, sitk.sitkFloat32)  # raw HU, no normalization
    maps["Region"] = region_map
    for name, value in maps.items():
        directory = Path(destination) / name
        directory.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(value, str(directory / f"{name}_{case_name}.nrrd"), True)
    write_quality_report(destination, case_name, image, quality, mode, thickness, "done")


def run_one(case_id, image_path, mask_path):
    t0 = time.time()
    extract_case(image_path, mask_path,
                 Path(mask_dir) / "core" / (case_id + ".nrrd"),
                 Path(mask_dir) / "rim" / (case_id + ".nrrd"),
                 output_dir, _EXTRACTOR)
    return time.time() - t0


def already_done(case_id):
    path = report_path_for(output_dir, case_id)
    if not path.exists():
        return False
    try:
        rep = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return False
    return rep.get("status") == "done" and rep.get("feature_columns") == FEATURE_COLUMNS


def main():
    global _EXTRACTOR
    print(f"Voxel feature set: {VOXEL_FEATURE_SET} ({len(FEATURE_COLUMNS)} columns)")
    images = collect_images(image_dir)
    masks = collect_images(mask_dir)
    missing = images.keys() - masks.keys()
    if missing:
        raise ValueError(f"Missing tumor masks for patients: {sorted(missing)}")
    todo = [(cid, p) for cid, p in images.items() if not (SKIP_EXISTING and already_done(cid))]
    print(f"{len(images) - len(todo)} cases already done, {len(todo)} to process")
    failures = {}
    if N_WORKERS <= 1:
        _EXTRACTOR = build_extractor()
        for cid, path in todo:
            print(f"Processing: {cid}", flush=True)
            sec = run_one(cid, path, masks[cid])     # errors stop the run, as before
            print(f"Completed: {cid} ({sec:.0f} s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker) as pool:
            futures = {pool.submit(run_one, cid, path, masks[cid]): cid for cid, path in todo}
            for fut in as_completed(futures):
                cid = futures[fut]
                try:
                    print(f"Completed: {cid} ({fut.result():.0f} s)", flush=True)
                except Exception as exc:  # keep the other workers running
                    failures[cid] = f"{type(exc).__name__}: {exc}"
                    print(f"FAILED: {cid}: {exc}", flush=True)
    if failures:
        (Path(output_dir) / "diagnostics").mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "diagnostics" / "failed_cases.json").write_text(
            json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
        raise RuntimeError(f"{len(failures)} cases failed; see diagnostics/failed_cases.json")


if __name__ == "__main__":
    main()
