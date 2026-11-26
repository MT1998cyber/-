import argparse
import base64
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# ---------------- settings ----------------
NNUNET_RESULTS = os.environ.get("HAB_NNUNET_RESULTS",
                               r"nnUNet_results_DIR\cv5_standard_v1\nnUNet_results")
NNUNET_DATASET = "501"                    # labels: background 0, Hematoma 1
NNUNET_CONFIG, NNUNET_TRAINER, NNUNET_PLANS = "3d_fullres", "nnUNetTrainer", "nnUNetPlans"
NNUNET_CHECKPOINT = "checkpoint_final.pth"
NNUNET_FOLDS = (0, 1, 2, 3, 4)            # ensemble of the five cross-validation models
NNUNET_PYTHON = os.environ.get("HAB_NNUNET_PYTHON")   # python.exe of the environment that has nnunetv2; None = this one
DEFAULT_BUNDLE = os.environ.get("HAB_BUNDLE")   # None = OUT_DIR/deploy/bundle_HE_First_nii/model.json of the study
KMEANS_PKL = os.environ.get("HAB_KMEANS_PKL")   # None = the model of the study configuration (joint_kmeans_k<K>.pkl)
INTERNAL_ID = "case"                      # file stem used inside the work folder (no "__")
CENTER_NAME = "DEPLOY"                    # identity prefix required by the step scripts
# mask plausibility (warnings only)
MIN_VOLUME_ML, MAX_VOLUME_ML = 1.0, 250.0
MEAN_HU_RANGE = (25.0, 90.0)
MAX_COMPONENTS = 3
FEATURE_RE = re.compile(r"^(Hab\d+)_(original|log-sigma-(\d+)-(\d+)-mm-3D)_firstorder_(\w+)$")
POS_COLOR, NEG_COLOR = "#b5505a", "#3f7b8c"
DISCLAIMER = ("Research use only. This tool is not a medical device and must not be used for diagnosis or "
              "treatment decisions. The result depends on the automatic segmentation, which you must check.")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- study modules
def study_modules():
    """Import the step scripts. Steps 0/1/2 and 5/6 normally run once per center / split, so their
    configuration asks for --center / --split; we only use their functions and settings (partition mode,
    YAML files, K-means path), none of which depends on the center or split, so any configured value works."""
    import config_loader as CL
    cfg = CL.load_config()
    if cfg is not None:
        CL.set_context(center=next(iter(cfg.CENTERS)), split=list(cfg.SPLIT_VALUES)[0])
    try:
        return {"C": import_module("single_ct_utils"),
                "S0": import_module("0 roi_partition"), "S1": import_module("1 feature_map_extraction"),
                "S2": import_module("2 voxel_feature_extraction"), "S5": import_module("5 habitat_map_generation"),
                "S6": import_module("6 regional_feature_extraction")}
    finally:
        CL.set_context()                      # clear the placeholder context again


def parse_features(features):
    """Bundle features -> (regions, LoG sigmas). Only first-order Original / LoG features are supported."""
    sig, regions = set(), set()
    for f in features:
        m = FEATURE_RE.match(f)
        if not m:
            raise SystemExit(f"feature {f!r} is not a first-order Original/LoG feature of a core habitat; "
                             "predict_case.py extracts only those")
        regions.add(m.group(1))
        if m.group(3) is not None:
            sig.add(float(f"{m.group(3)}.{m.group(4)}"))
    return sorted(regions), sorted(sig)


# ----------------------------------------------------------------------------- image
def load_ct(path):
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    if img.GetDimension() != 3:
        raise SystemExit("a 3D CT volume is required")
    if img.GetNumberOfComponentsPerPixel() != 1:
        raise SystemExit("CT must be a single-channel volume")
    arr = sitk.GetArrayViewFromImage(img)
    if not np.isfinite(arr).all():
        raise SystemExit("CT contains NaN / Inf")
    warns = []
    lo, hi = float(arr.min()), float(arr.max())
    if lo > -200 or hi < 100:
        warns.append(f"intensity range {lo:.0f}..{hi:.0f} does not look like Hounsfield units (expected about -1000..+3000)")
    if img.GetSize()[2] < 8:
        warns.append(f"only {img.GetSize()[2]} slices")
    for key in img.GetMetaDataKeys():                   # header text fields that can carry identifiers
        if key in ("descrip", "aux_file", "intent_name", "db_name") and img.GetMetaData(key).strip():
            warns.append(f"NIfTI header field {key!r} is not empty - make sure it holds no patient information")
    return img, warns


def to_grid(img, grid, strict):
    import SimpleITK as sitk
    sp = np.array(img.GetSpacing())
    if np.allclose(sp, grid, rtol=0, atol=1e-3):
        return img, False
    if strict:
        raise SystemExit(f"spacing {tuple(np.round(sp, 3))} mm differs from the study grid {tuple(grid)} mm (--no-resample)")
    size = [int(round(n * s / t)) for n, s, t in zip(img.GetSize(), sp, grid)]
    out = sitk.Resample(img, size, sitk.Transform(), sitk.sitkLinear, img.GetOrigin(), tuple(grid),
                        img.GetDirection(), -1024.0, sitk.sitkFloat32)
    return out, True


# ----------------------------------------------------------------------------- segmentation
def check_nnunet_python(py):
    """nnU-Net may live in another environment than PyRadiomics (different torch / numpy needs)."""
    r = subprocess.run([py, "-c", "import nnunetv2"], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(
            f"nnunetv2 is not installed in the Python used for the segmentation:\n    {py}\n"
            "Either point to the environment where you trained / ran nnU-Net:\n"
            "    --nnunet-python D:\\Anaconda3\\envs\\<your_nnunet_env>\\python.exe      (conda env list shows the folders)\n"
            "or install it here:  pip install nnunetv2     (can change the torch / numpy versions of this environment)\n"
            "or skip the segmentation by giving your own mask:  --mask <mask.nii.gz>")


def run_nnunet(ct, work, folds, device, tta, py=None):
    import SimpleITK as sitk
    py = py or NNUNET_PYTHON or sys.executable
    check_nnunet_python(py)
    results = Path(NNUNET_RESULTS)
    model_dir = next(iter(results.glob(f"Dataset{NNUNET_DATASET}_*")), None)
    if model_dir is None or not (model_dir / f"{NNUNET_TRAINER}__{NNUNET_PLANS}__{NNUNET_CONFIG}").is_dir():
        raise SystemExit(f"nnU-Net model not found under {results} (Dataset{NNUNET_DATASET}_*/"
                         f"{NNUNET_TRAINER}__{NNUNET_PLANS}__{NNUNET_CONFIG})")
    inp, outp = work / "nnunet_in", work / "nnunet_out"
    for d in (inp, outp):
        d.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(ct, str(inp / "case_0000.nii.gz"), True)
    env = os.environ.copy()
    env["nnUNet_results"] = str(results)
    env.setdefault("nnUNet_raw", str(work / "nn_raw"))
    env.setdefault("nnUNet_preprocessed", str(work / "nn_pre"))
    cmd = [py, "-c",
           "from nnunetv2.inference.predict_from_raw_data import predict_entry_point; predict_entry_point()",
           "-i", str(inp), "-o", str(outp), "-d", NNUNET_DATASET, "-c", NNUNET_CONFIG, "-tr", NNUNET_TRAINER,
           "-p", NNUNET_PLANS, "-f", *map(str, folds), "-chk", NNUNET_CHECKPOINT, "-device", device]
    if not tta:
        cmd.append("--disable_tta")
    log(f"nnU-Net: folds {list(folds)}, device {device}, TTA {'on' if tta else 'off'}")
    subprocess.run(cmd, env=env, check=True)
    return sitk.ReadImage(str(outp / "case.nii.gz"))


def align_mask(mask, ct_grid, ct_orig, C):
    import SimpleITK as sitk
    if C.same_geometry(mask, ct_grid):
        return mask
    if C.same_geometry(mask, ct_orig) or mask.GetSize() == ct_orig.GetSize():
        return sitk.Resample(mask, ct_grid, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    raise SystemExit("the mask does not share the geometry of the CT (neither the uploaded nor the resampled one)")


def mask_qc(mask_arr, ct_arr, spacing_xyz):
    from scipy import ndimage
    roi = mask_arr > 0
    n = int(roi.sum())
    if n == 0:
        raise SystemExit("the segmentation is empty: no hematoma found (check the CT, or give a mask with --mask)")
    _, comps = ndimage.label(roi)
    vol = n * float(np.prod(spacing_xyz)) / 1000.0
    hu = ct_arr[roi]
    touch = any(np.take(roi, 0, axis=a).any() or np.take(roi, -1, axis=a).any() for a in range(3))
    q = {"voxels": n, "volume_ml": vol, "components": int(comps), "mean_hu": float(hu.mean()),
         "median_hu": float(np.median(hu)), "p05_hu": float(np.percentile(hu, 5)),
         "p95_hu": float(np.percentile(hu, 95)), "touches_image_border": bool(touch), "warnings": []}
    w = q["warnings"]
    if vol < MIN_VOLUME_ML:
        w.append(f"very small segmentation ({vol:.2f} mL)")
    if vol > MAX_VOLUME_ML:
        w.append(f"very large segmentation ({vol:.0f} mL)")
    if comps > MAX_COMPONENTS:
        w.append(f"{comps} separate components - check for false-positive regions")
    if not MEAN_HU_RANGE[0] <= q["mean_hu"] <= MEAN_HU_RANGE[1]:
        w.append(f"mean HU {q['mean_hu']:.0f} outside {MEAN_HU_RANGE} - check the mask")
    if touch:
        w.append("segmentation touches the image border")
    return q


# ----------------------------------------------------------------------------- habitats + features
def build_habitat(ct_grid, mask, work, M):
    """Steps 0, 1, 2, 5 of the study on one case. Returns (habitat image, habitat voxel counts, K, paths)."""
    import SimpleITK as sitk
    import joblib
    C, S0, S1, S2, S5 = M["C"], M["S0"], M["S1"], M["S2"], M["S5"]
    roi_dir, img_dir, fmap_dir = work / "roi", work / "image", work / "feature_maps"
    for d in (roi_dir / "core", roi_dir / "rim", img_dir, fmap_dir):
        d.mkdir(parents=True, exist_ok=True)
    pid = INTERNAL_ID
    img_path = img_dir / f"{pid}.nii.gz"
    sitk.WriteImage(ct_grid, str(img_path), True)
    C.require_expected_grid(ct_grid)

    mode = S0.PARTITION_MODE.lower()
    use_hu = mode == "outer" and S0.RIM_HU_RANGE is not None
    core, rim, analysis = S0.partition_by_mode(mask, mode, C.RIM_THICKNESS_MM, ct_image=ct_grid,
                                               rim_hu_range=S0.RIM_HU_RANGE if use_hu else None,
                                               core_hu_range=S0.CORE_HU_RANGE)
    sitk.WriteImage(core, str(roi_dir / "core" / f"{pid}.nrrd"), True)
    sitk.WriteImage(rim, str(roi_dir / "rim" / f"{pid}.nrrd"), True)
    sitk.WriteImage(analysis, str(roi_dir / f"{pid}.nrrd"), True)

    log(f"voxel-wise features ({len(C.FEATURE_COLUMNS)} per voxel; this is the slow step)")
    S1.extract_case(img_path, roi_dir / f"{pid}.nrrd", roi_dir / "core" / f"{pid}.nrrd",
                    roi_dir / "rim" / f"{pid}.nrrd", fmap_dir, S1.build_extractor())
    data = S2.extract_voxels(roi_dir / f"{pid}.nrrd", fmap_dir, pid, CENTER_NAME)
    # The study writes this table to Excel and reads it back, which turns the float32 maps into float64 before
    # K-means. Do the same here, otherwise voxels close to a cluster boundary can fall on the other side.
    data[C.FEATURE_COLUMNS] = data[C.FEATURE_COLUMNS].astype("float64")

    kpath = KMEANS_PKL or S5.model_path
    log(f"habitats: K-means {Path(kpath).name}")
    model = joblib.load(kpath)
    data, hab = S5.generate_habitat(data, model)
    hab_path = work / f"{CENTER_NAME}__{pid}_habitat.nrrd"
    sitk.WriteImage(hab, str(hab_path), True)
    k = int(model["n_clusters"])
    arr = sitk.GetArrayViewFromImage(hab)
    counts = {f"Hab{i}": int((arr == i).sum()) for i in range(1, k + 1)}
    return hab, counts, k, img_path, hab_path


def regional_values(img_path, hab_path, features, M):
    """Step 6 restricted to the model's features (same extractor settings, same code)."""
    from radiomics import featureextractor
    S6 = M["S6"]
    _, sigmas = parse_features(features)
    ext = featureextractor.RadiomicsFeatureExtractor(str(S6.params_file))
    ext.addProvenance(False)
    ext.disableAllImageTypes()
    ext.enableImageTypeByName("Original")
    if sigmas:
        ext.enableImageTypeByName("LoG", customArgs={"sigma": sigmas})
    ext.disableAllFeatures()
    ext.enableFeatureClassByName("firstorder")
    feats, quality, _ = S6.extract_patient(img_path, hab_path, ext)
    values = {f: feats.get(f, float("nan")) for f in features}
    status = {q["Region"]: q["Status"] for q in quality}
    return values, status


# ----------------------------------------------------------------------------- figures
def shap_figure(res, label, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = sorted(res["shap"].items(), key=lambda kv: kv[1], reverse=True)
    vals = np.array([v for _, v in items])
    n = len(items)
    lo, hi = min(0.0, vals.min()), max(0.0, vals.max())
    span = (hi - lo) or 1.0
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * n + 1.0))
    ax.barh(np.arange(n), vals, height=0.6, color=[POS_COLOR if v >= 0 else NEG_COLOR for v in vals], zorder=2)
    ax.axvline(0, color="#4d4d4d", lw=1.2, ls=(0, (4, 3)), zorder=3)
    for y, v in enumerate(vals):
        ax.text(max(v, 0) + span * 0.012, y, f"{v:+.3f}".replace("-", "\u2212"), va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([label(f) for f, _ in items], fontsize=9, fontweight="bold")
    ax.set_ylim(n - 0.45, -0.75)
    ax.set_xlim(lo - 0.10 * span, hi + 0.22 * span)
    ax.tick_params(axis="y", length=0, pad=6)
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.set_xlabel("Contribution to predicted risk (SHAP value, log-odds)", fontsize=9, fontweight="bold")
    ax.text(0, 1.0, f"Prediction = {res['probability']:.3f}", transform=ax.get_xaxis_transform(),
            ha="left", va="bottom", fontsize=10, fontweight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def qc_overlay(ct_arr, mask_arr, hab_arr, path):
    """Three largest slices: CT | CT + segmentation outline | CT + habitats."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    area = (mask_arr > 0).sum(axis=(1, 2))
    zs = sorted(np.argsort(area)[::-1][:3])
    ys, xs = np.nonzero((mask_arr > 0).any(axis=0))
    pad = 40
    yb = slice(max(ys.min() - pad, 0), ys.max() + pad)
    xb = slice(max(xs.min() - pad, 0), xs.max() + pad)
    cols = ListedColormap(["#47b0c5", "#db5143", "#319c87", "#eb927e", "#9aa4b1", "#7a5aa6"])
    fig, axes = plt.subplots(len(zs), 3, figsize=(7.2, 2.3 * len(zs)), squeeze=False)
    for r, z in enumerate(zs):
        ct2, m2, h2 = ct_arr[z, yb, xb], (mask_arr[z, yb, xb] > 0), hab_arr[z, yb, xb]
        for c, ax in enumerate(axes[r]):
            ax.imshow(ct2, cmap="gray", vmin=0, vmax=80, aspect=1)
            ax.axis("off")
        axes[r, 1].contour(m2.astype(float), [0.5], colors="#ff3b30", linewidths=1.0)
        axes[r, 2].imshow(np.where(h2 > 0, h2, np.nan), cmap=cols, vmin=0.5, vmax=6.5, alpha=0.75, aspect=1)
        axes[r, 0].set_title(f"CT, slice {z}", fontsize=8, loc="left")
    axes[0, 1].set_title("Segmentation", fontsize=8)
    axes[0, 2].set_title("Habitats", fontsize=8)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_heatmaps(name, img_path, hab_path, features, out_dir):
    import habitat_heatmap_png as H
    H.CASES = [{"image": str(img_path), "habitat": str(hab_path), "name": name}]
    H.FEATURES = list(features)
    sys.argv = [sys.argv[0], "--out", str(out_dir)]
    H.main()
    return Path(out_dir) / H.HFM.safe(name)


def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def write_report(path, name, res, info, qc, counts, values, status, files, warnings_all, label):
    def row(f):
        return (f"<tr><td>{html.escape(label(f))}</td><td>{values[f]:.2f}</td><td>{res['z'][f]:+.2f}</td>"
                f"<td>{res['shap'][f]:+.3f}</td></tr>")
    imgs = "".join(f"<h2>{html.escape(t)}</h2><img src='data:image/png;base64,{b64(p)}'>"
                   for t, p in files if p and Path(p).exists())
    warn = "".join(f"<li>{html.escape(w)}</li>" for w in warnings_all) or "<li>none</li>"
    doc = f"""<!doctype html><meta charset="utf-8"><title>{html.escape(name)}</title>
<style>body{{font:14px/1.5 system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#222}}
img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{border-bottom:1px solid #ddd;padding:4px 10px;text-align:left}}
.box{{background:#fff4e5;border:1px solid #f0c36d;padding:8px 12px;border-radius:6px}}</style>
<h1>Habitat model result: {html.escape(name)}</h1>
<p class="box">{html.escape(DISCLAIMER)}</p>
<p><b>Predicted probability of hematoma expansion: {res['probability']:.3f}</b>
 (model {html.escape(str(info['model_name']))}; validation on the held-out center: AUC {info['validation'].get('auc', float('nan')):.2f})</p>
<h2>Warnings</h2><ul>{warn}</ul>
<h2>Segmentation</h2><p>{qc['volume_ml']:.1f} mL, {qc['voxels']} voxels, mean {qc['mean_hu']:.0f} HU;
habitat voxels: {html.escape(', '.join(f'{k} {v}' for k, v in counts.items()))}</p>
<h2>Features</h2><table><tr><th>Feature</th><th>Value</th><th>z</th><th>SHAP</th></tr>{''.join(row(f) for f in values)}</table>
{imgs}"""
    Path(path).write_text(doc, encoding="utf-8")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image", required=True, help="anonymized CT (.nii.gz / .nii)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE, help="model.json from export_deploy_bundle.py")
    ap.add_argument("--mask", help="own / corrected mask (label > 0 = hematoma + IVH); skips nnU-Net")
    ap.add_argument("--name", default="case", help="name shown in the report (no patient identifiers!)")
    ap.add_argument("--folds", nargs="+", type=int, default=list(NNUNET_FOLDS))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"], help="auto = cuda if available")
    ap.add_argument("--no-tta", action="store_true", help="faster nnU-Net inference (no mirroring)")
    ap.add_argument("--nnunet-python", default=None, help="python.exe of the environment that has nnunetv2 "
                    "(default: this environment, or env var HAB_NNUNET_PYTHON)")
    ap.add_argument("--no-resample", action="store_true", help="fail instead of resampling to the study grid")
    ap.add_argument("--no-heatmaps", action="store_true")
    ap.add_argument("--keep-work", action="store_true", help="keep intermediate files in OUT/work")
    ap.add_argument("--compare-study", metavar="CASEID", help="print the six features next to the step-6 table of the study")
    ap.add_argument("--compare-habitat", metavar="NRRD", help="habitat map of the same case made by the study (step 5): "
                    "count the voxels whose habitat differs")
    args = ap.parse_args()

    import SimpleITK as sitk
    import hab_model
    if args.device == "auto":
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    work = out / "work" if args.keep_work else Path(tempfile.mkdtemp(prefix="habwork_"))
    work.mkdir(parents=True, exist_ok=True)
    M = study_modules()
    C = M["C"]
    all_warn = []
    try:
        bundle = args.bundle
        if bundle is None:
            import ecosnet_data as D
            bundle = Path(D.OUT_DIR) / "deploy" / "bundle_HE_First_nii" / "model.json"
        model = hab_model.HabModel.load(bundle)
        features = model.features
        parse_features(features)

        log("reading CT")
        ct0, w = load_ct(args.image)
        all_warn += w
        grid = np.asarray(C.EXPECTED_SPACING_MM, float)
        ct, resampled = to_grid(ct0, grid, args.no_resample)
        if resampled:
            all_warn.append(f"CT resampled from {tuple(np.round(ct0.GetSpacing(), 3))} to {tuple(grid)} mm; "
                            "first-order features can shift slightly with resampling")
        sitk.WriteImage(ct, str(out / "image_on_study_grid.nii.gz"), True)

        if args.mask:
            mask = align_mask(sitk.ReadImage(args.mask), ct, ct0, C)
            seg_source = "user-provided mask"
        else:
            mask = align_mask(run_nnunet(ct, work, args.folds, args.device, not args.no_tta, args.nnunet_python), ct, ct0, C)
            seg_source = f"nnU-Net {NNUNET_CONFIG}, folds {args.folds}"
        mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
        sitk.WriteImage(mask, str(out / "segmentation.nii.gz"), True)
        ct_arr, mask_arr = sitk.GetArrayFromImage(ct).astype(float), sitk.GetArrayFromImage(mask)
        qc = mask_qc(mask_arr, ct_arr, ct.GetSpacing())
        all_warn += qc["warnings"]
        log(f"segmentation: {qc['volume_ml']:.1f} mL, mean {qc['mean_hu']:.0f} HU, {qc['components']} component(s)")

        hab, counts, k, img_path, hab_path = build_habitat(ct, mask, work, M)
        sitk.WriteImage(hab, str(out / "habitat_map.nrrd"), True)
        values, status = regional_values(img_path, hab_path, features, M)
        need = parse_features(features)[0]
        for r in need:
            if status.get(r) != "ok":
                all_warn.append(f"{r}: region status '{status.get(r, 'absent')}' - its features are missing and were replaced by the training median")
        res = model.predict(values)
        if res["imputed"]:
            all_warn.append("imputed features: " + ", ".join(res["imputed"]) + " (prediction less reliable)")
        if res["out_of_range"]:
            all_warn.append("features far outside the training distribution (|z| > 3): " + ", ".join(res["out_of_range"]))
        log(f"predicted probability = {res['probability']:.3f}")

        import habitat_heatmap_png as H
        label = H.label
        shap_png, qc_png = out / "figures" / "shap.png", out / "figures" / "qc_overlay.png"
        shap_figure(res, label, shap_png)
        qc_overlay(ct_arr, mask_arr, sitk.GetArrayFromImage(hab), qc_png)
        files = [("Segmentation and habitats (check this first)", qc_png), ("SHAP", shap_png)]
        if not args.no_heatmaps:
            log("feature heat maps")
            hm = make_heatmaps(args.name, img_path, hab_path, features, out / "heatmaps")
            files.append(("Feature heat maps (largest slice)", hm / f"{H.HFM.safe(args.name)}__all_features.png"))

        info = model.info()
        result = {"name": args.name, "probability": res["probability"], "logit": res["logit"],
                  "base_value": res["base_value"], "features": values, "z": res["z"], "shap": res["shap"],
                  "imputed": res["imputed"], "out_of_range": res["out_of_range"], "habitat_voxels": counts,
                  "segmentation": {k_: v for k_, v in qc.items() if k_ != "warnings"}, "segmentation_source": seg_source,
                  "warnings": all_warn, "model": {k_: info[k_] for k_ in ("model_name", "held_out_center", "created")},
                  "model_validation": info["validation"], "disclaimer": DISCLAIMER,
                  "seconds": round(time.time() - t0, 1)}
        (out / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        write_report(out / "report.html", args.name, res, info, qc, counts, values, status, files, all_warn, label)

        if args.compare_study:
            try:
                E2 = import_module("e2 habitat_diagnostics")
            except ImportError:
                raise SystemExit("--compare-study needs the full study folder (e2 habitat_diagnostics.py)")
            df, _ = E2.load()
            row = df[df.CaseID == args.compare_study]
            if row.empty:
                print(f"--compare-study: {args.compare_study} not in the study tables")
            else:
                ref = {f: float(row.iloc[0][f]) for f in features}
                print(f"\n{'feature':<58}{'this run':>12}{'study':>12}{'diff':>10}{'rel.':>9}")
                for f in features:
                    print(f"{f:<58}{values[f]:>12.4f}{ref[f]:>12.4f}{values[f] - ref[f]:>10.2e}"
                          f"{abs(values[f] - ref[f]) / (abs(ref[f]) + 1e-12):>9.2%}")
                p_ref = model.predict(ref)["probability"]
                print(f"probability: this run {res['probability']:.4f} | model applied to the study's feature values {p_ref:.4f}")
        if args.compare_habitat:
            ref_hab = sitk.GetArrayFromImage(sitk.ReadImage(args.compare_habitat))
            cur_hab = sitk.GetArrayFromImage(hab)
            if ref_hab.shape != cur_hab.shape:
                print(f"--compare-habitat: shapes differ {ref_hab.shape} vs {cur_hab.shape}")
            else:
                inside = (ref_hab > 0) | (cur_hab > 0)
                print(f"habitat map vs study: {int((ref_hab != cur_hab).sum())} of {int(inside.sum())} voxels differ; "
                      f"voxels per label  this run {dict(zip(*np.unique(cur_hab[cur_hab > 0], return_counts=True)))}  "
                      f"study {dict(zip(*np.unique(ref_hab[ref_hab > 0], return_counts=True)))}")
        log(f"done in {time.time() - t0:.0f} s -> {out / 'report.html'}")
        for wmsg in all_warn:
            print("WARNING:", wmsg)
    finally:
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()