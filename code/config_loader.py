import argparse
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_CTX = {}
_CFG = None

_CTX_KEYS = ("center", "split", "source")

def set_context(**kw):
    """Called by run_pipeline.py. Also exported as env vars so worker processes inherit it."""
    _CTX.clear()
    _CTX.update({k: v for k, v in kw.items() if v is not None})
    for k in _CTX_KEYS:
        if k in _CTX:
            os.environ[f"HE_CTX_{k.upper()}"] = str(_CTX[k])
        else:
            os.environ.pop(f"HE_CTX_{k.upper()}", None)


def _cli():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config")
    p.add_argument("--center")
    p.add_argument("--split")
    p.add_argument("--source")
    args, _ = p.parse_known_args(sys.argv[1:])
    return args


def load_config():
    global _CFG
    if _CFG is not None:
        return _CFG or None
    args = _cli()
    path = args.config or os.environ.get("HE_PIPELINE_CONFIG") or str(HERE / "pipeline_config.py")
    if not Path(path).exists():
        _CFG = False
        return None
    spec = importlib.util.spec_from_file_location("pipeline_config_loaded", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.__file_path__ = str(Path(path).resolve())
    _CFG = mod
    return mod


# ----------------------------------------------------------------------------- paths
def run_root(c):
    return Path(c.WORK_ROOT) / c.RUN_NAME


def center_dir(c, center, what):
    return str(run_root(c) / "centers" / center / what)


def split_dir(c, *parts):
    return str(run_root(c).joinpath(*parts))


def yaml_path(c, name):
    v = getattr(c, name)
    return str(v) if v else str(HERE / ("voxel_radiomics.yaml" if name == "VOXEL_YAML" else "regional_radiomics.yaml"))


def regional_csvs(c):
    return [str(run_root(c) / "regional_features" / g / "regional_features_all.csv") for g in c.SPLIT_VALUES]


def _need(ctx, key, section, choices):
    v = ctx.get(key)
    if v is None:
        raise SystemExit(f"[config] {section} runs once per {key}: pass --{key} <{'|'.join(map(str, choices))}> "
                         f"or use run_pipeline.py")
    if v not in choices:
        raise SystemExit(f"[config] unknown {key} {v!r}; configured: {list(choices)}")
    return v


# ----------------------------------------------------------------------------- sections
def build_section(c, section, ctx):
    R = run_root(c)
    common = {"CLINICAL_CSV": c.CLINICAL_CSV, "ID_COL": c.ID_COL, "SPLIT_COL": c.SPLIT_COL,
              "TRAIN_VALUES": list(c.TRAIN_VALUES), "LABEL_COL": c.LABEL_COL}
    if section == "utils":
        return {"RIM_THICKNESS_MM": c.RIM_THICKNESS_MM, "EXPECTED_SPACING_MM": c.EXPECTED_SPACING_MM,
                "VOXEL_FEATURE_SET": c.VOXEL_FEATURE_SET, "PCA_VARIANCE": c.PCA_VARIANCE,
                "MAX_TRAIN_VOXELS_PER_PATIENT": c.MAX_TRAIN_VOXELS_PER_PATIENT}
    if section in ("step0", "step1", "step2"):
        ctr = _need(ctx, "center", section, c.CENTERS)
        part = center_dir(c, ctr, "partition")
        if section == "step0":
            return {"mask_dir": c.CENTERS[ctr]["mask"], "image_dir": c.CENTERS[ctr]["image"],
                    "PARTITION_MODE": c.PARTITION_MODE, "output_dir": part,
                    "core_dir": os.path.join(part, "core"), "rim_dir": os.path.join(part, "rim"),
                    "RIM_HU_RANGE": c.RIM_HU_RANGE, "CORE_HU_RANGE": getattr(c, "CORE_HU_RANGE", None)}
        if section == "step1":
            return {"image_dir": c.CENTERS[ctr]["image"], "mask_dir": part,
                    "output_dir": center_dir(c, ctr, "feature_maps"), "params_file": yaml_path(c, "VOXEL_YAML"),
                    "MAX_CORE_NAN_FRACTION": c.MAX_CORE_NAN_FRACTION, "MAX_RIM_NAN_FRACTION": c.MAX_RIM_NAN_FRACTION,
                    "N_WORKERS": c.N_WORKERS, "SKIP_EXISTING": c.SKIP_EXISTING}
        return {"CENTER_NAME": ctr, "mask_dir": part, "feature_root": center_dir(c, ctr, "feature_maps"),
                "output_dir": center_dir(c, ctr, "voxel_tables")}
    if section == "sort":
        return {"CLINICAL_CSV": c.CLINICAL_CSV, "ID_COL": c.ID_COL, "SPLIT_COL": c.SPLIT_COL,
                "VOXEL_TABLE_DIRS": [center_dir(c, k, "voxel_tables") for k in c.CENTERS],
                "OUT_ROOT": split_dir(c, "splits"), "FIT_VALUES": list(c.TRAIN_VALUES),
                "FIT_DIR": split_dir(c, "splits", "_fit")}
    if section == "step3":
        return {"data_dir": split_dir(c, "splits", "_fit"), "output_dir": split_dir(c, "models"),
                "K_RANGE": list(c.K_RANGE), "SHOW_PLOTS": False}
    if section == "step4":
        return {"train_dir": split_dir(c, "splits", "_fit"), "model_dir": split_dir(c, "models"),
                "N_CLUSTERS": c.N_CLUSTERS}
    if section in ("step5", "step6"):
        g = _need(ctx, "split", section, c.SPLIT_VALUES)
        if section == "step5":
            return {"input_dir": split_dir(c, "splits", g),
                    "model_path": split_dir(c, "models", f"joint_kmeans_k{c.N_CLUSTERS}.pkl"),
                    "output_dir": split_dir(c, "habitat_maps", g)}
        return {"image_dirs": {k: v["image"] for k, v in c.CENTERS.items()},
                "habitat_dir": split_dir(c, "habitat_maps", g),
                "output_dir": split_dir(c, "regional_features", g),
                "params_file": yaml_path(c, "REGIONAL_YAML"), "WRITE_EXCEL": c.WRITE_EXCEL,
                "EXPECTED_N_CLUSTERS": c.N_CLUSTERS}
    if section == "step7":
        return {"image_dirs": {k: v["image"] for k, v in c.CENTERS.items()},
                "mask_dirs": {k: v["mask"] for k, v in c.CENTERS.items()},
                "output_dir": split_dir(c, "whole_roi_features"),
                "params_file": Path(yaml_path(c, "REGIONAL_YAML")), "WRITE_EXCEL": c.WRITE_EXCEL}
    if section == "step8":
        return {**common, "REGIONAL_FEATURE_CSVS": regional_csvs(c),
                "OUT_DIR": split_dir(c, "habitat_selected"), **c.SELECTION}
    if section == "ecosnet_data":       # clinical-table settings shared by e2 / e3
        return {**common, "HABITAT_CSV": split_dir(c, "habitat_selected", "habitat_features_selected.csv"),
                "DEEP_CSV": "", "OUT_DIR": split_dir(c, "analysis"), "AUX_COL": getattr(c, "AUX_COL", None),
                "CLINICAL_FEATURES": list(c.CLINICAL_FEATURES), "LOG_FEATURES": list(c.LOG_FEATURES)}
    if section == "e2":
        return {**c.E2, "REGIONAL_FEATURE_CSVS": regional_csvs(c),
                "OUT": Path(split_dir(c, "analysis", "e2_diagnostics"))}
    if section == "e3":
        return {**c.E3, "SIGNS": list(c.SIGNS), "OUT": Path(split_dir(c, "analysis", "e3_final"))}
    if section == "e4":
        return {**getattr(c, "E4", {}), "OUT": Path(split_dir(c, "analysis", "e4_shap_ml"))}
    raise KeyError(f"unknown config section {section!r}")


def apply_config(section, g):
    """Overwrite module globals g with the configured values of `section`."""
    c = load_config()
    if c is None:
        return
    args = _cli()
    ctx = {k: os.environ.get(f"HE_CTX_{k.upper()}") for k in _CTX_KEYS}       # inherited (workers)
    ctx.update({k: v for k, v in vars(args).items() if k in _CTX_KEYS and v})  # command line
    ctx.update(_CTX)                                                          # run_pipeline.py
    values = build_section(c, section, ctx)
    unknown = [k for k in values if k not in g]
    if unknown:
        print(f"[config] note: {section} sets new names {unknown}")
    g.update(values)


def check_inputs():
    """Light sanity check of the configured inputs (used by run_pipeline.py)."""
    c = load_config()
    if c is None:
        raise SystemExit("pipeline_config.py not found")
    problems = []
    for k, v in c.CENTERS.items():
        for kind in ("image", "mask"):
            if not Path(v[kind]).is_dir():
                problems.append(f"{k} {kind} folder not found: {v[kind]}")
    if not Path(c.CLINICAL_CSV).is_file():
        problems.append(f"clinical csv not found: {c.CLINICAL_CSV}")
    if not set(c.TRAIN_VALUES) <= set(c.SPLIT_VALUES):
        problems.append("TRAIN_VALUES must be a subset of SPLIT_VALUES")
    missing_signs = [s for s in c.SIGNS if s not in c.CLINICAL_FEATURES]
    if missing_signs:
        problems.append(f"SIGNS not in CLINICAL_FEATURES: {missing_signs}")
    if Path(c.CLINICAL_CSV).is_file():
        import pandas as pd
        cols = set(pd.read_csv(c.CLINICAL_CSV, nrows=1, encoding="utf-8-sig").columns)
        miss = [x for x in [c.ID_COL, c.SPLIT_COL, c.LABEL_COL] + list(c.CLINICAL_FEATURES) if x not in cols]
        if miss:
            problems.append(f"columns missing in clinical csv: {miss}")
    for y in ("VOXEL_YAML", "REGIONAL_YAML"):
        p = Path(yaml_path(c, y))
        if not p.is_file():
            problems.append(f"{y} not found: {p}")
        elif any(b > 127 for b in p.read_bytes()):
            problems.append(f"{p.name} contains non-ASCII characters (PyRadiomics reads it as GBK on Chinese Windows)")
    return c, problems
