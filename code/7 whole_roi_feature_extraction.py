import hashlib
import json
import logging
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import radiomics
import SimpleITK as sitk
from radiomics import featureextractor


# ---------- Manually edit these paths before running ----------
image_dirs = {
    "CENTER_1": r"IMAGE_DIR",
    "CENTER_2": r"IMAGE_DIR",
    "CENTER_3": r"IMAGE_DIR",
    "CENTER_4": r"IMAGE_DIR",
}
mask_dirs = {
    "CENTER_1": r"MASK_DIR",
    "CENTER_2": r"MASK_DIR",
    "CENTER_3": r"MASK_DIR",
    "CENTER_4": r"MASK_DIR",
}
output_dir = r"OUTPUT_DIR"
params_file = r"PARAMS_FILE_DIR\regional_radiomics.yaml"
WRITE_EXCEL = True

# None: automatically accept exactly ONE positive foreground label, e.g. 1 or 255.
# Multiple positive labels are rejected to avoid silently merging different tissues.
# If your intended whole ROI is explicitly the union of labels 1 and 2, set (1, 2).
FOREGROUND_LABELS = None

FEATURE_PREFIX = "Whole"
IDENTITY_COLUMNS = ["CaseID", "Center", "PatientID"]
IMAGE_EXTENSIONS = (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd")
GEOMETRY_ATOL = 1e-6
from config_loader import apply_config
apply_config("step7", globals())   # values from pipeline_config.py (if present)


def collect_images(directory):
    """Match exact local basenames; never guess suffixes or use fuzzy matching."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError("Input directory does not exist: {}".format(directory))
    found, normalized = {}, {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file():
            continue
        extension = next((ext for ext in IMAGE_EXTENSIONS
                          if path.name.lower().endswith(ext)), None)
        if extension is None:
            continue
        patient_id = path.name[:-len(extension)]
        if not patient_id or patient_id.casefold() in normalized:
            raise ValueError("Empty/duplicate patient basename in {}: {}".format(directory, path.name))
        normalized[patient_id.casefold()] = path
        found[patient_id] = path
    return found


def same_geometry(image, mask):
    if image.GetDimension() != 3 or mask.GetDimension() != 3:
        return False
    if image.GetSize() != mask.GetSize():
        return False
    return all(np.allclose(a, b, atol=GEOMETRY_ATOL, rtol=0)
               for a, b in [(image.GetSpacing(), mask.GetSpacing()),
                            (image.GetOrigin(), mask.GetOrigin()),
                            (image.GetDirection(), mask.GetDirection())])


def make_whole_roi(image, mask, foreground_labels):
    """Validate the source mask before constructing an in-memory binary ROI."""
    if image.GetNumberOfComponentsPerPixel() != 1 or mask.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("CT and mask must both be scalar images")
    if not same_geometry(image, mask):
        raise ValueError("CT/mask must have identical 3D size, spacing, origin and direction; no automatic resizing")
    if not np.isfinite(sitk.GetArrayViewFromImage(image)).all():
        raise ValueError("CT contains NaN or Inf")
    values = sitk.GetArrayViewFromImage(mask)
    if not np.isfinite(values).all():
        raise ValueError("Mask contains NaN or Inf")
    if (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Mask must contain nonnegative integer labels, not probabilities")
    positive = [int(value) for value in np.unique(values) if value > 0]
    if not positive:
        raise ValueError("Whole-ROI mask is empty")
    if foreground_labels is None:
        if len(positive) != 1:
            raise ValueError("Mask has multiple foreground labels {}; explicitly set FOREGROUND_LABELS".format(positive))
        selected = positive
    else:
        selected = list(foreground_labels)
        if not selected or any(not np.isfinite(value) or value <= 0 or int(value) != value
                               for value in selected):
            raise ValueError("FOREGROUND_LABELS must be None or a nonempty sequence of positive integers")
        selected = sorted(set(int(value) for value in selected))
    foreground = np.isin(values, selected)
    voxel_count = int(np.count_nonzero(foreground))
    if voxel_count < 2:
        raise ValueError("Whole ROI contains fewer than two selected voxels")
    binary_mask = sitk.GetImageFromArray(foreground.astype(np.uint8))
    binary_mask.CopyInformation(mask)
    return binary_mask, {
        "MaskLabels": ";".join(str(value) for value in positive),
        "SelectedMaskLabels": ";".join(str(value) for value in selected),
        "VoxelCount": voxel_count,
        "VolumeMM3": voxel_count * float(np.prod(image.GetSpacing())),
        "SpacingXYZ": str(tuple(image.GetSpacing())),
    }


def extract_patient(image_path, mask_path, extractor, qc, foreground_labels):
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    whole_roi, details = make_whole_roi(image, mask, foreground_labels)
    qc.update(details)
    # Extract ALL selected foreground voxels as one ROI, independent of habitats.
    result = extractor.execute(image, whole_roi, label=1, voxelBased=False)
    features = {}
    for name, value in result.items():
        if name.startswith("diagnostics_"):
            continue
        scalar = np.asarray(value)
        if scalar.size != 1:
            raise ValueError("Expected scalar regional feature: {}".format(name))
        features["{}_{}".format(FEATURE_PREFIX, name)] = float(scalar.item())
    if not features:
        raise ValueError("No whole-ROI features returned; check the YAML")
    bad = [name for name, value in features.items() if not np.isfinite(value)]
    qc.update({"FeatureCount": len(features), "NonFiniteFeatures": len(bad),
               "NonFiniteFeatureNames": ";".join(bad),
               "Status": "nonfinite_features" if bad else "ok"})
    return {name: value if np.isfinite(value) else np.nan
            for name, value in features.items()}


def save_tables(rows, quality, destination, write_excel):
    feature_columns = sorted({key for row in rows for key in row
                              if key.startswith(FEATURE_PREFIX + "_")})
    # An omitted filter/feature must not look like a completely successful case.
    for row, qc in zip(rows, quality):
        missing = [name for name in feature_columns if name not in row]
        qc["MissingFeatureCount"] = len(missing)
        qc["MissingFeatureNames"] = ";".join(missing)
        if missing and qc["Status"] == "ok":
            qc["Status"] = "inconsistent_feature_schema"
            qc["Error"] = "Feature names differ across cases; inspect extraction.log for skipped filters"
    table = pd.DataFrame(rows).reindex(columns=IDENTITY_COLUMNS + feature_columns)
    qc_table = pd.DataFrame(quality)
    table.to_csv(destination / "whole_roi_features.csv", index=False, encoding="utf-8-sig")
    qc_table.to_csv(destination / "whole_roi_qc.csv", index=False, encoding="utf-8-sig")
    if write_excel:
        with pd.ExcelWriter(destination / "whole_roi_features.xlsx", engine="openpyxl") as writer:
            table.to_excel(writer, sheet_name="WholeROI", index=False)
            qc_table.to_excel(writer, sheet_name="QC", index=False)
    return table, qc_table


def main():
    if set(image_dirs) != set(mask_dirs) or not mask_dirs:
        raise ValueError("image_dirs and mask_dirs must configure the same nonempty set of centers")
    config_path = Path(params_file)
    extractor = featureextractor.RadiomicsFeatureExtractor(str(config_path))
    extractor.addProvenance(False)
    if extractor.settings.get("resampledPixelSpacing") is not None:
        raise ValueError("Do not resample again; use CT/mask on the same grid as existing features")
    if extractor.settings.get("correctMask", False):
        raise ValueError("correctMask must remain false; do not silently correct mismatched masks")
    if extractor.settings.get("voxelBased", False) or any(
            name in extractor.settings for name in ("kernelRadius", "maskedKernel", "voxelBatch")):
        raise ValueError("Use the supplied regional YAML, not a voxel-wise extraction configuration")

    cases = []
    for center in mask_dirs:
        images = collect_images(image_dirs[center])
        masks = collect_images(mask_dirs[center])
        local_ids = sorted(set(images) | set(masks))
        if not local_ids:
            raise ValueError("No supported CT/mask files for center {}".format(center))
        for patient_id in local_ids:
            # Keep the original script's identity convention, without inventing
            # a mapping to clinical center=1/2, group, outcome or downstream ID.
            identity = dict(zip(IDENTITY_COLUMNS,
                                [center + "__" + patient_id, center, patient_id]))
            cases.append((identity, images.get(patient_id), masks.get(patient_id)))

    destination = Path(output_dir)
    for source_dir in list(image_dirs.values()) + list(mask_dirs.values()):
        source_root = Path(source_dir).resolve()
        if destination.resolve() == source_root or source_root in destination.resolve().parents:
            raise ValueError("Output must be outside all source image/mask directories")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("Output is not empty; choose a NEW output_dir: {}".format(destination))
    destination.mkdir(parents=True, exist_ok=True)
    config_bytes = config_path.read_bytes()
    (destination / "regional_radiomics_used.yaml").write_bytes(config_bytes)
    (destination / "whole_roi_extraction_config.json").write_text(json.dumps({
        "python_version": platform.python_version(),
        "pyradiomics_version": radiomics.__version__,
        "simpleitk_version": sitk.Version_VersionString(),
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "params_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "image_dirs": image_dirs, "mask_dirs": mask_dirs,
        "foreground_labels": FOREGROUND_LABELS,
        "feature_prefix": FEATURE_PREFIX, "voxelBased": False,
        "roi_definition": "Original source mask foreground; no erosion/dilation/habitat subdivision",
        "feature_settings": extractor.settings,
        "enabled_image_types": extractor.enabledImagetypes,
        "enabled_feature_classes": extractor.enabledFeatures,
        "planned_cases": [{**identity, "CTPath": str(image_path) if image_path else "",
                           "MaskPath": str(mask_path) if mask_path else ""}
                          for identity, image_path, mask_path in cases],
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    log_handler = logging.FileHandler(str(destination / "extraction.log"), encoding="utf-8")
    logger = logging.getLogger("radiomics")
    logger.addHandler(log_handler)
    rows, quality = [], []
    try:
        for index, (identity, image_path, mask_path) in enumerate(cases, 1):
            print("[{}/{}] Whole ROI: {}".format(index, len(cases), identity["CaseID"]), flush=True)
            row = identity.copy()
            qc = {**identity, "Region": "WholeROI",
                  "CTPath": str(image_path) if image_path else "",
                  "MaskPath": str(mask_path) if mask_path else "",
                  "FeatureCount": 0, "NonFiniteFeatures": 0,
                  "NonFiniteFeatureNames": "", "Status": "ok", "Error": ""}
            try:
                if image_path is None or mask_path is None:
                    raise FileNotFoundError("Exact CT/mask basename match missing; CT={}, mask={}".format(image_path, mask_path))
                row.update(extract_patient(image_path, mask_path, extractor, qc, FOREGROUND_LABELS))
            except Exception as exc:
                qc["Status"] = "case_error"
                qc["Error"] = "{}: {}".format(type(exc).__name__, exc)
                logger.exception("Whole-ROI case failed: %s", identity["CaseID"])
            rows.append(row)
            quality.append(qc)
            print("  {}: {} features".format(qc["Status"], qc["FeatureCount"]), flush=True)
        table, qc_table = save_tables(rows, quality, destination, WRITE_EXCEL)
        failed_count = int((qc_table["Status"] != "ok").sum())
        summary = {"attempted_cases": len(table), "passed_cases": len(table) - failed_count,
                   "failed_cases": failed_count,
                   "feature_columns": len(table.columns) - len(IDENTITY_COLUMNS),
                   "all_cases_passed": failed_count == 0,
                   "status_counts": {str(key): int(value) for key, value
                                     in qc_table["Status"].value_counts().items()}}
        (destination / "whole_roi_run_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print("Saved {} rows, {} feature columns to {}".format(
            len(table), summary["feature_columns"], destination), flush=True)
        if failed_count:
            warnings.warn("{} cases need review; failed values remain missing. See whole_roi_qc.csv".format(failed_count))
            raise RuntimeError("Extraction/QC failed for some cases; partial tables were saved for review")
    finally:
        logger.removeHandler(log_handler)
        log_handler.close()


if __name__ == "__main__":
    main()
