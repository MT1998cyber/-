"""Extract patient-level regional radiomics from CT and habitat labels.

Labels are read from each habitat map's metadata written by step 5:
  core habitats 1..K  -> Hab1..HabK
  rim K+1             -> Rim   (absent when PartitionMode == "none")
so any K chosen in step 4 works without editing this file. All cases must share one K.

No outcome labels, feature selection, standardization across patients or model
training are performed. Missing/failed regions remain missing and are reported.
"""

import hashlib
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import radiomics
import SimpleITK as sitk
from radiomics import featureextractor

from single_ct_utils import collect_images, make_case_id, same_geometry, validate_partition_settings

# ---------- Manually edit paths before running ----------
# CT must be the SAME image used to construct these habitat masks.
# Only centers present in habitat_dir are read; edit/add dictionary entries.
image_dirs = {
    "CENTER_1": r"IMAGE_DIR",
    "CENTER_2": r"IMAGE_DIR",
    "CENTER_3": r"IMAGE_DIR",
    "CENTER_4": r"IMAGE_DIR",
}
habitat_dir = r"HABITAT_DIR"
output_dir = r"OUTPUT_DIR"
params_file = r"PARAMS_FILE_DIR\regional_radiomics.yaml"
WRITE_EXCEL = True

# K is read from the habitat maps. Set an int (e.g. 4) to additionally require that value;
# None = accept whatever step 4/5 used (still identical across all cases).
EXPECTED_N_CLUSTERS = None
IDENTITY_COLUMNS = ["CaseID", "Center", "PatientID"]
from config_loader import apply_config
apply_config("step6", globals())   # values from pipeline_config.py (if present)

def parse_case_id(case_id):
    parts = case_id.split("__")
    if len(parts) != 2 or make_case_id(*parts) != case_id:
        raise ValueError("Habitat filename must be <Center>__<PatientID>.nrrd (as saved by step 5)")
    return tuple(parts)


def habitat_layout(habitat):
    """Return (mode, K, regions) from step-5 metadata, e.g. 'background=0; core=1..4; rim=none'."""
    mode = habitat.GetMetaData("PartitionMode") if habitat.HasMetaDataKey("PartitionMode") else "inner"
    if mode not in {"inner", "outer", "none"}:
        raise ValueError("Unknown PartitionMode in habitat metadata")
    if not habitat.HasMetaDataKey("HabitatLabels"):
        raise ValueError("Habitat map has no HabitatLabels metadata; regenerate it with step 5")
    text = habitat.GetMetaData("HabitatLabels")
    m = re.fullmatch(r"background=0; core=1\.\.(\d+); rim=(none|\d+)", text.strip())
    if not m:
        raise ValueError(f"Cannot parse HabitatLabels metadata: {text!r}")
    k = int(m.group(1))
    rim = None if m.group(2) == "none" else int(m.group(2))
    if (mode == "none") != (rim is None) or (rim is not None and rim != k + 1):
        raise ValueError(f"Inconsistent metadata: PartitionMode={mode}, HabitatLabels={text!r}")
    if EXPECTED_N_CLUSTERS is not None and k != EXPECTED_N_CLUSTERS:
        raise ValueError(f"Habitat map has K={k}, but EXPECTED_N_CLUSTERS={EXPECTED_N_CLUSTERS}")
    regions = {i: f"Hab{i}" for i in range(1, k + 1)}
    if rim is not None:
        regions[rim] = "Rim"
    return mode, k, regions


def validate_pair(image, habitat):
    if not same_geometry(image, habitat):
        raise ValueError("CT and habitat mask must have identical 3D geometry; do not resize labels silently")
    if not np.isfinite(sitk.GetArrayViewFromImage(image)).all():
        raise ValueError("CT contains NaN or Inf")
    mode, k, regions = habitat_layout(habitat)
    values = sitk.GetArrayViewFromImage(habitat)
    allowed = [0] + list(regions)
    if not np.isfinite(values).all() or not np.isin(values, allowed).all():
        found = sorted(set(np.unique(values).tolist()) - set(allowed))
        raise ValueError(f"Habitat mask contains labels {found} outside {allowed} (K={k}, mode={mode})")
    if not (values > 0).any():
        raise ValueError("Habitat mask is entirely background")
    if habitat.HasMetaDataKey("RimThicknessMM"):
        validate_partition_settings(mode, float(habitat.GetMetaData("RimThicknessMM")))
    return mode, k, regions


def extract_patient(image_path, habitat_path, extractor):
    image = sitk.ReadImage(str(image_path))
    habitat = sitk.ReadImage(str(habitat_path))
    mode, k, regions = validate_pair(image, habitat)
    # Labels are categorical. Cast only after checking their exact values.
    labels = sitk.Cast(habitat, sitk.sitkUInt8)
    array = sitk.GetArrayViewFromImage(labels)
    features, quality = {}, []
    thickness = habitat.GetMetaData("RimThicknessMM") if habitat.HasMetaDataKey("RimThicknessMM") else ""
    for label, prefix in regions.items():
        voxel_count = int(np.count_nonzero(array == label))
        qc = {
            "HabitatLabel": label, "Region": prefix,
            "VoxelCount": voxel_count,
            "VolumeMM3": voxel_count * float(np.prod(image.GetSpacing())),
            "SpacingXYZ": str(tuple(image.GetSpacing())),
            "RimThicknessMM": thickness,
            "PartitionMode": mode,
            "FeatureCount": 0, "NonFiniteFeatures": 0,
            "Status": "ok", "Error": "", "NonFiniteFeatureNames": "",
        }
        if voxel_count == 0:
            qc["Status"] = "missing_label"
            qc["Error"] = "Label absent; no region features computed, not filled with zero"
        elif voxel_count == 1:
            qc["Status"] = "too_small"
            qc["Error"] = "Single-voxel region cannot support regional radiomics"
        else:
            try:
                result = extractor.execute(image, labels, label=label, voxelBased=False)
                region_features = {}
                for name, value in result.items():
                    if name.startswith("diagnostics_"):
                        continue
                    scalar = np.asarray(value)
                    if scalar.size != 1:
                        raise ValueError(f"Expected a scalar regional feature: {name}")
                    region_features[f"{prefix}_{name}"] = float(scalar.item())
                if not region_features:
                    raise ValueError("No regional features returned; check the YAML")
                bad = [name for name, value in region_features.items() if not np.isfinite(value)]
                # Preserve a missing value rather than converting an invalid feature to zero.
                features.update({name: value if np.isfinite(value) else np.nan
                                 for name, value in region_features.items()})
                qc["FeatureCount"] = len(region_features)
                qc["NonFiniteFeatures"] = len(bad)
                if bad:
                    qc["Status"] = "nonfinite_features"
                    qc["NonFiniteFeatureNames"] = ";".join(bad)
            except (RuntimeError, ValueError, OSError) as exc:
                qc["Status"] = "extraction_error"
                qc["Error"] = str(exc)
        quality.append(qc)
    return features, quality, k


def safe_write(write_fn, path):
    """Windows locks files open in Excel/WPS: fall back to a timestamped name instead of losing results."""
    try:
        write_fn(path)
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}")
        write_fn(alt)
        warnings.warn(f"{path.name} is open in another program (Excel?); saved as {alt.name} instead",
                      RuntimeWarning)
        return alt


def save_tables(rows, quality, destination, write_excel, regions):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    # Include the union of observed feature names, even when a patient lacks a label.
    feature_columns = []
    for prefix in regions.values():
        feature_columns.extend(sorted({key for row in rows for key in row if key.startswith(prefix + "_")}))
    all_features = pd.DataFrame(rows).reindex(columns=IDENTITY_COLUMNS + feature_columns)
    core_prefixes = tuple(p + "_" for p in regions.values() if p != "Rim")
    core = all_features[IDENTITY_COLUMNS + [name for name in feature_columns if name.startswith(core_prefixes)]]
    rim = all_features[IDENTITY_COLUMNS + [name for name in feature_columns if name.startswith("Rim_")]]
    qc = pd.DataFrame(quality)
    for filename, table in [
        ("regional_features_all.csv", all_features),
        ("core_features.csv", core), ("rim_features.csv", rim),
        ("regional_features_qc.csv", qc),
    ]:
        safe_write(lambda p, tb=table: tb.to_csv(p, index=False, encoding="utf-8-sig"), destination / filename)
    if write_excel:
        sheets = [("All", all_features), ("Core", core)] + ([("Rim", rim)] if "Rim" in regions.values() else []) + [("QC", qc)]

        def write_xlsx(p):
            with pd.ExcelWriter(p, engine="openpyxl") as writer:
                for name, table in sheets:
                    table.to_excel(writer, sheet_name=name, index=False)
        safe_write(write_xlsx, destination / "regional_features.xlsx")
    return all_features, qc


def main():
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
    extractor.addProvenance(False)
    if extractor.settings.get("resampledPixelSpacing") is not None:
        raise ValueError("Regional YAML must not resample again; use the same CT/mask grid as step 5")
    if extractor.settings.get("voxelBased", False):
        raise ValueError("Use the separate regional YAML, without voxelBased=true")
    if any(name in extractor.settings for name in ("kernelRadius", "maskedKernel", "voxelBatch")):
        raise ValueError("Use 6 regional_radiomics.yaml, not the voxel-wise JointEntropy YAML")
    habitats = collect_images(habitat_dir)
    # Validate identities before starting expensive extraction.
    identities = {}
    for case_id in habitats:
        center, local_id = parse_case_id(case_id)
        if center not in image_dirs:
            raise ValueError(f"Configure image_dirs for center {center}")
        identities[case_id] = (center, local_id)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    config_bytes = Path(params_file).read_bytes()
    (destination / "regional_radiomics_used.yaml").write_bytes(config_bytes)
    (destination / "regional_extraction_config.json").write_text(json.dumps({
        "pyradiomics_version": radiomics.__version__,
        "simpleitk_version": sitk.Version_VersionString(),
        "params_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "image_dirs": image_dirs, "habitat_dir": str(habitat_dir),
        "expected_n_clusters": EXPECTED_N_CLUSTERS, "voxelBased": False,
        "feature_settings": extractor.settings,
        "enabled_image_types": extractor.enabledImagetypes,
        "enabled_feature_classes": extractor.enabledFeatures,
    }, indent=2), encoding="utf-8")

    images_by_center, rows, quality = {}, [], []
    seen_k, regions = None, None
    for case_id, habitat_path in habitats.items():
        center, local_id = identities[case_id]
        identity = dict(zip(IDENTITY_COLUMNS, [case_id, center, local_id]))
        print(f"Extracting regional features: {case_id}", flush=True)
        row = identity.copy()
        try:
            if center not in images_by_center:
                images_by_center[center] = collect_images(image_dirs[center])
            image_path = images_by_center[center].get(local_id)
            if image_path is None:
                raise ValueError(f"No CT for patient {local_id} in {image_dirs[center]}")
            features, case_quality, k = extract_patient(image_path, habitat_path, extractor)
            if seen_k is None:
                seen_k = k
                print(f"Habitat layout: K={k} core habitats" +
                      (", plus Rim" if any(q["Region"] == "Rim" for q in case_quality) else ", no rim (mode none)"))
            elif k != seen_k:
                raise ValueError(f"K={k} differs from K={seen_k} of earlier cases; all maps must come from one model")
            case_regions = {q["HabitatLabel"]: q["Region"] for q in case_quality}
            if regions is None:
                regions = case_regions
            elif case_regions != regions:
                raise ValueError(f"Region layout {case_regions} differs from earlier cases {regions}")
            row.update(features)
            quality.extend({**identity, "CTPath": str(image_path), "HabitatPath": str(habitat_path), **qc}
                           for qc in case_quality)
            enabled = case_quality
            ok = sum(qc["Status"] == "ok" for qc in enabled)
            print(f"Finished: {case_id}; {ok}/{len(enabled)} enabled regions passed QC, {len(features)} scalar features")
        except (RuntimeError, ValueError, OSError) as exc:
            quality.append({**identity, "HabitatPath": str(habitat_path),
                            "Region": "all", "Status": "case_error", "Error": str(exc)})
            print(f"Failed: {case_id}: {exc}", flush=True)
        rows.append(row)

    if regions is None:
        qc_only = pd.DataFrame(quality)
        safe_write(lambda p: qc_only.to_csv(p, index=False, encoding="utf-8-sig"),
                   destination / "regional_features_qc.csv")
        raise RuntimeError("No case could be read; see regional_features_qc.csv")
    table, qc = save_tables(rows, quality, destination, WRITE_EXCEL, regions)
    problematic = qc.loc[~qc["Status"].isin(["ok", "disabled_none"])]
    print(f"Saved {len(table)} patient rows and {len(table.columns) - len(IDENTITY_COLUMNS)} feature columns to {destination}")
    if not problematic.empty:
        warnings.warn(
            f"{len(problematic)} QC rows need review; missing/failed features remain NaN. "
            "Check regional_features_qc.csv before feature selection.", RuntimeWarning,
        )
    hard_errors = qc["Status"].isin(["case_error", "extraction_error", "too_small", "nonfinite_features"])
    if hard_errors.any():
        raise RuntimeError("Some regions failed extraction/QC; partial tables and diagnostics were saved for review")


if __name__ == "__main__":
    main()
