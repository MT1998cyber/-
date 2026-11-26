"""Shared geometry, table schema and training-only clustering utilities."""

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RIM_THICKNESS_MM = 3.0 
EXPECTED_SPACING_MM = (0.5, 0.5, 5.0) 
# ---------------------------------------------------------------------------
VOXEL_FEATURE_SETS = {
    "hu_entropy": [
        "firstorder_Entropy", "firstorder_Median", "firstorder_MeanAbsoluteDeviation",
        "glcm_DifferenceAverage", "glcm_DifferenceEntropy", "glcm_DifferenceVariance",
        "glcm_Imc1", "glcm_Imc2", "glcm_InverseVariance", "glcm_JointEnergy",
        "glcm_JointEntropy", "glcm_SumEntropy",
        "glrlm_LongRunEmphasis", "glrlm_RunEntropy", "glrlm_RunVariance",
        "glszm_SizeZoneNonUniformityNormalized", "glszm_SmallAreaHighGrayLevelEmphasis",
        "ngtdm_Contrast", "ngtdm_Strength",
    ],
}
VOXEL_FEATURE_SET = "hu_entropy"  

VOXEL_FEATURES = VOXEL_FEATURE_SETS[VOXEL_FEATURE_SET]


def voxel_column(feature):
    """'HU' -> 'nCE_HU'; 'glcm_JointEntropy' -> 'nCE_JointEntropy' (same names as the v1 tables)."""
    return "nCE_" + feature.split("_", 1)[-1]


FEATURE_COLUMNS = [voxel_column(f) for f in VOXEL_FEATURES]
if len(set(FEATURE_COLUMNS)) != len(FEATURE_COLUMNS):
    raise ValueError("Voxel feature short names must be unique within a feature set")


def radiomic_voxel_features():
    """{'glcm': ['JointEntropy', ...], ...} for extractor.enableFeaturesByName (HU excluded)."""
    classes = {}
    for f in VOXEL_FEATURES:
        if f == "HU":
            continue
        cls, name = f.split("_", 1)
        classes.setdefault(cls, []).append(name)
    return classes


# Optional PCA between StandardScaler and K-means (e.g. 0.95 = keep 95% variance).
# Useful with the 19 strongly correlated texture features; None = no PCA (paper).
PCA_VARIANCE = None
GEOMETRY_COLUMNS = ["Direction", "Origin", "Spacing", "Dimension", "Coordinate"]
SCHEMA_VERSION = "single-ct-core-rim-v1"
# Steps 3/4: at most this many core voxels per TRAINING patient enter K-means
# (equalises large vs small hematomas, and speeds up fitting). None = all voxels.
MAX_TRAIN_VOXELS_PER_PATIENT = 5000
SUBSAMPLE_SEED = 0
from config_loader import apply_config as _apply_config
_apply_config("utils", globals())   # values from pipeline_config.py (if present)
VOXEL_FEATURES = VOXEL_FEATURE_SETS[VOXEL_FEATURE_SET]
FEATURE_COLUMNS = [voxel_column(f) for f in VOXEL_FEATURES]
if len(set(FEATURE_COLUMNS)) != len(FEATURE_COLUMNS):
    raise ValueError("Voxel feature short names must be unique within a feature set")

def require_1mm_grid(image):
    if image.GetDimension() != 3 or not np.allclose(
        image.GetSpacing(), (1.0, 1.0, 1.0), rtol=0, atol=1e-5
    ):
        raise ValueError(
            "This workflow requires a 1 mm isotropic grid; first run "
            "preprocess_resample_1mm.py and use its output image/mask directories"
        )


def require_expected_grid(image):
    if image.GetDimension() != 3:
        raise ValueError("A 3D image/mask is required")
    if EXPECTED_SPACING_MM is None:
        return
    target = np.asarray(EXPECTED_SPACING_MM, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError("EXPECTED_SPACING_MM must contain three positive finite values, or None")
    if not np.allclose(image.GetSpacing(), target, rtol=0, atol=1e-5):
        raise ValueError(
            f"Spacing {image.GetSpacing()} does not match EXPECTED_SPACING_MM={tuple(target)}. "
            "Choose the intended native or resampled input directories; this check does not resample."
        )


def make_case_id(center, local_id):
    """Globally unique output ID; leave original image/mask names unchanged."""
    for value in (center, local_id):
        if (not isinstance(value, str) or not value or value != value.strip()
                or value in (".", "..") or "__" in value
                or re.search(r'[<>:"/\\|?*\x00-\x1f]', value)
                or value.endswith(".")):
            raise ValueError("Center and PatientID must be nonempty filename-safe names without '__'")
    return f"{center}__{local_id}"


def image_stem(filename):
    """Remove the whole image suffix, including the compound .nii.gz suffix."""
    name = Path(filename).name
    for suffix in (".nii.gz", ".nrrd", ".nii"):
        if name.lower().endswith(suffix):
            stem = name[:-len(suffix)]
            return stem if stem else None
    return None


def patient_id(filename):
    stem = image_stem(filename)
    if stem is not None and re.fullmatch(r"\d+b", stem, re.I):
        return stem[:-1]
    return stem


def collect_images(directory):
    """Index a flat, non-contrast-only image directory; reject ambiguous IDs."""
    images = {}
    for path in sorted(Path(directory).iterdir()):
        case_id = patient_id(path.name)
        if not path.is_file() or case_id is None:
            continue
        if case_id in images:
            raise ValueError(f"Ambiguous images for patient {case_id} in {directory}")
        images[case_id] = path
    if not images:
        raise ValueError(f"No .nii.gz, .nii or .nrrd images found in {directory}")
    return images


def same_geometry(first, second):
    return (
        first.GetDimension() == second.GetDimension() == 3
        and first.GetSize() == second.GetSize()
        and np.allclose(first.GetSpacing(), second.GetSpacing(), rtol=0, atol=1e-6)
        and np.allclose(first.GetOrigin(), second.GetOrigin(), rtol=0, atol=1e-5)
        and np.allclose(first.GetDirection(), second.GetDirection(), rtol=0, atol=1e-6)
    )


def image_from_array(array, reference):
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    return image


def validate_partition_settings(mode, thickness):
    if mode not in {"inner", "outer", "none"}:
        raise ValueError("PartitionMode must be inner, outer or none")
    if not np.isfinite(thickness) or thickness < 0:
        raise ValueError("Invalid partition thickness")
    if mode == "none" and thickness != 0:
        raise ValueError("none mode requires zero rim thickness")
    if mode != "none" and thickness <= 0:
        raise ValueError("inner/outer mode requires positive rim thickness")


def read_partition_settings(*images):
    """Use step-0 metadata; metadata-free legacy partitions are inner mode."""
    modes = {image.GetMetaData("PartitionMode") for image in images
             if image.HasMetaDataKey("PartitionMode")}
    if len(modes) > 1:
        raise ValueError("Inconsistent PartitionMode between masks")
    mode = next(iter(modes), "inner")
    widths = [float(image.GetMetaData("RimThicknessMM")) for image in images
              if image.HasMetaDataKey("RimThicknessMM")]
    if not widths or not np.allclose(widths, widths[0], rtol=0, atol=1e-6):
        raise ValueError("Partition thickness is missing or inconsistent between masks")
    width = widths[0]
    validate_partition_settings(mode, width)
    return mode, width


def validate_partition_arrays(core, rim, analysis, mode):
    if not core.any():
        raise ValueError("Core must contain voxels")
    if mode == "none":
        if rim.any():
            raise ValueError("none mode requires an empty rim")
    elif not rim.any():
        raise ValueError("inner/outer mode requires a nonempty rim")
    if (core & rim).any() or not np.array_equal(core | rim, analysis):
        raise ValueError(
            "Core and rim must be disjoint and cover the complete analysis ROI; "
            "use the root-level mask saved in step 0 output_dir (especially for outer mode)"
        )


def table_partition_settings(data):
    if "PartitionMode" in data:
        if data["PartitionMode"].isna().any() or data["PartitionMode"].nunique() != 1:
            raise ValueError("Mixed/missing PartitionMode within a patient table")
        mode = str(data["PartitionMode"].iloc[0])
    else:
        mode = "inner"  # Compatibility with original inner-only tables.
    widths = pd.to_numeric(data["RimThicknessMM"], errors="raise").to_numpy()
    if not np.isfinite(widths).all() or not np.allclose(widths, widths[0], rtol=0, atol=1e-6):
        raise ValueError("Invalid or mixed rim thicknesses")
    width = float(widths[0])
    validate_partition_settings(mode, width)
    if not (data["Region"] == "core").any():
        raise ValueError("Patient table has no core voxels")
    if mode == "none" and (data["Region"] == "rim").any():
        raise ValueError("none table cannot contain rim voxels")
    if mode != "none" and not (data["Region"] == "rim").any():
        raise ValueError("inner/outer table must contain rim voxels")
    return mode, width


def partition_roi(mask_image, thickness_mm=RIM_THICKNESS_MM):
    """Core = ROI voxels > thickness_mm from the closest background center.

    Use physical distance with anisotropic spacing; outside-image voxels count
    as background. Core and intratumoral rim exactly partition the positive ROI.
    """
    if mask_image.GetDimension() != 3:
        raise ValueError("A 3D mask is required")
    if not np.isfinite(thickness_mm) or thickness_mm <= 0:
        raise ValueError("Rim thickness must be positive and finite")
    roi = sitk.GetArrayFromImage(mask_image) > 0
    if not roi.any():
        raise ValueError("Empty tumor ROI")
    distance = distance_transform_edt(
        np.pad(roi, 1), sampling=mask_image.GetSpacing()[::-1]
    )[1:-1, 1:-1, 1:-1]
    core = roi & (distance > thickness_mm)
    if not core.any():
        raise ValueError(
            f"No core remains after {thickness_mm:g} mm erosion; "
            "review this lesion or choose a smaller prespecified thickness"
        )
    rim = roi & ~core
    images = [image_from_array(a.astype(np.uint8), mask_image) for a in (core, rim)]
    for image in images:
        image.SetMetaData("RimThicknessMM", str(thickness_mm))
    return tuple(images)


def read_voxel_table(path):
    data = pd.read_excel(path, dtype={"Center": str, "PatientID": str, "CaseID": str})
    required = GEOMETRY_COLUMNS + FEATURE_COLUMNS + ["Region", "RimThicknessMM"]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}; rerun step 2")
    if data.empty or not data["Region"].isin(["core", "rim"]).all():
        raise ValueError(f"{path}: expected a nonempty core/rim voxel table")
    mode, _ = table_partition_settings(data)
    data["PartitionMode"] = mode
    identity_columns = {"Center", "PatientID", "CaseID"}
    if identity_columns.intersection(data.columns):
        if not identity_columns.issubset(data.columns):
            raise ValueError(f"{path}: incomplete patient identity columns")
        for column in identity_columns:
            if data[column].isna().any() or data[column].nunique() != 1:
                raise ValueError(f"{path}: inconsistent {column} within one patient")
        expected = make_case_id(data["Center"].iloc[0], data["PatientID"].iloc[0])
        if data["CaseID"].iloc[0] != expected or Path(path).stem != expected:
            raise ValueError(f"{path}: filename and Center/PatientID/CaseID do not match")
    return data


def core_features(data):
    values = data.loc[data["Region"] == "core", FEATURE_COLUMNS].astype(float)
    return values.loc[np.isfinite(values.to_numpy()).all(axis=1)]


def load_training_features(directory, max_voxels_per_patient=None, seed=None):
    """Directory must contain training patients only, one table per patient.

    max_voxels_per_patient caps each patient's contribution (random, seeded) so that
    large hematomas do not dominate the population-level K-means centroids.
    None keeps every voxel (original behaviour).
    """
    if max_voxels_per_patient is None:
        max_voxels_per_patient = MAX_TRAIN_VOXELS_PER_PATIENT
    seed = SUBSAMPLE_SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    paths = sorted(p for p in Path(directory).iterdir()
                   if p.suffix.lower() in (".xlsx", ".xls") and not p.name.startswith("~$"))
    if not paths:
        raise ValueError(f"No training voxel tables found in {directory}")
    frames, thicknesses, modes = [], [], []
    for path in paths:
        data = read_voxel_table(path)
        features = core_features(data)
        if features.empty:
            raise ValueError(f"{path}: no core voxel with all features finite")
        if max_voxels_per_patient and len(features) > max_voxels_per_patient:
            keep = rng.choice(len(features), size=int(max_voxels_per_patient), replace=False)
            features = features.iloc[np.sort(keep)]
        frames.append(features)
        thicknesses.append(float(data["RimThicknessMM"].iloc[0]))
        modes.append(data["PartitionMode"].iloc[0])
    if len(set(modes)) != 1:
        raise ValueError("Training tables use different partition modes; keep inner/outer/none separate")
    if not np.allclose(thicknesses, thicknesses[0], rtol=0, atol=1e-6):
        raise ValueError("Training tables use different rim thicknesses")
    features = pd.concat(frames, ignore_index=True)
    features.attrs["partition_mode"] = modes[0]
    return features, thicknesses[0]


def make_cluster_pipeline(n_clusters):
    steps = [("scale", StandardScaler())]
    if PCA_VARIANCE is not None:
        steps.append(("pca", PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=0)))
    steps.append(("kmeans", KMeans(n_clusters=n_clusters, random_state=0, n_init=10)))
    return Pipeline(steps)


def cluster_space(pipeline, data):
    """Data in the space K-means actually sees (scaled, and PCA-projected if enabled)."""
    return pipeline[:-1].transform(data)


def geometry_and_coordinates(data):
    """Decode named columns and reject inconsistent or out-of-image geometry."""
    geometry = {}
    for column, length in [("Direction", 9), ("Origin", 3), ("Spacing", 3), ("Dimension", 3)]:
        if data[column].nunique() != 1:
            raise ValueError(f"Inconsistent {column} within one patient")
        value = np.asarray(ast.literal_eval(str(data[column].iloc[0])), dtype=float)
        if value.shape != (length,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid {column}")
        geometry[column] = value
    size = geometry["Dimension"]
    if (size <= 0).any() or not np.equal(size, np.floor(size)).all():
        raise ValueError("Dimension must contain positive integers")
    if (geometry["Spacing"] <= 0).any():
        raise ValueError("Spacing must be positive")
    coordinates = np.asarray([
        ast.literal_eval(str(value)) for value in data["Coordinate"]
    ], dtype=float)
    if coordinates.shape != (len(data), 3) or not np.isfinite(coordinates).all():
        raise ValueError("Invalid voxel coordinates")
    if not np.equal(coordinates, np.floor(coordinates)).all():
        raise ValueError("Voxel coordinates must be integers")
    coordinates = coordinates.astype(int)
    if (coordinates < 0).any() or (coordinates >= size[::-1]).any():
        raise ValueError("Voxel coordinates outside image dimensions")
    if len(np.unique(coordinates, axis=0)) != len(coordinates):
        raise ValueError("Duplicate voxel coordinates")
    return geometry, coordinates
