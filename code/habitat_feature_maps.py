import argparse
import difflib
import glob
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml
from numpy.lib.stride_tricks import sliding_window_view

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# ============================================================================= settings
CASES = [
    {"case_id": "CASE_ID"},
    # {"image": r"IMAGE_DIR\ID.nii.gz",
    #  "habitat": r"HABITAT_DIR\ID.nrrd",
    #  "name": "S003"},
]
# Exact step-6 column names (regional_features_all.csv). A wrong name prints the closest matches.
FEATURES = [
    "Hab1_log-sigma-2-0-mm-3D_firstorder_10Percentile",
    "Hab2_log-sigma-3-0-mm-3D_firstorder_10Percentile",
    "Hab2_log-sigma-1-0-mm-3D_firstorder_90Percentile",
    "Hab1_log-sigma-1-0-mm-3D_firstorder_Minimum",
    "Hab1_log-sigma-1-0-mm-3D_firstorder_Skewness",
    "Hab1_log-sigma-1-0-mm-3D_firstorder_90Percentile",
    "Hab2_log-sigma-2-0-mm-3D_firstorder_90Percentile",
    "Hab1_log-sigma-1-0-mm-3D_firstorder_Maximum",
    "Hab1_log-sigma-1-0-mm-3D_firstorder_InterquartileRange",
    "Hab2_original_firstorder_90Percentile",
    "Hab1_original_firstorder_90Percentile",
    "Hab2_original_firstorder_Median",
    "Hab1_log-sigma-2-0-mm-3D_firstorder_InterquartileRange",
    "Hab2_original_firstorder_10Percentile",
]
OUTPUT_DIR = None              
REGIONAL_YAML = None           

MAPS = ["filtered", "local", "tail"]
LOCAL_EXTENT = "habitat"       # "habitat": map only inside the feature's habitat (= what the feature measures)
                               # "hematoma": same statistic over the whole hematoma (all core habitats),
                               #             to see where in the hematoma low / high values lie
KERNEL_RADIUS_VOX = (4, 4, 0)  # (x, y, z) voxels: 9 x 9 in-plane window = 4.5 x 4.5 mm on one 5 mm slice
                               # (z = 1 adds the slices above and below = 15 mm; usually too coarse)
MIN_KERNEL_VOXELS = 10         # fewer valid voxels in the window -> no value
BACKGROUND = np.nan            # value outside the map (np.nan or 0; NaN is transparent in most viewers)
FILTER_ENGINE = "auto"         # "auto" (PyRadiomics if installed) | "pyradiomics" | "simpleitk" (Original / LoG only)
SAVE_HABITAT_NII = True
SAVE_PNG = True
PNG_CMAP = "jet"
PNG_SLICE = "max_area"         # "max_area" (largest cross-section of the hematoma) or a slice index (int)
CT_WINDOW = (0, 80)            # HU, for the PNG
PNG_ZOOM_PAD_MM = 10
PNG_COLUMNS = 4
CHECK_AGAINST_STEP6 = True     # compare the recomputed region value with regional_features_all.csv
TEXTURE_KERNEL_RADIUS = 2      # PyRadiomics voxel-based (texture features only), isotropic in voxels
FIRSTORDER_SHIFT = 0           # voxelArrayShift (PyRadiomics default 0 when normalize is false)

FEATURE_RE = re.compile(r"^(Hab\d+|Rim)_(original|log-sigma-[\d-]+-mm-3D|wavelet-[HL]{3}|square|squareroot|"
                        r"logarithm|exponential|gradient|lbp-[23]D[\w-]*)_(firstorder|glcm|glrlm|glszm|gldm|ngtdm|shape)_(\w+)$")


# ============================================================================= config / inputs
def pipeline_config():
    try:
        from config_loader import load_config, run_root
        c = load_config()
        return c, (run_root(c) if c is not None else None)
    except Exception as exc:                       # noqa: BLE001 - config is optional here
        print(f"[config] not used: {exc}")
        return None, None


def regional_settings(c):
    path = REGIONAL_YAML or (getattr(c, "REGIONAL_YAML", None) if c is not None else None) or HERE / "regional_radiomics.yaml"
    cfg = yaml.safe_load(Path(path).read_text(encoding="ascii")) or {}
    s = cfg.get("setting", {}) or {}
    if s.get("normalize"):
        raise SystemExit("regional YAML uses normalize: true - this script reproduces raw-HU extraction only")
    return {"binWidth": float(s.get("binWidth", 25)), "padDistance": int(s.get("padDistance", 5)),
            "preCrop": bool(s.get("preCrop", False)), "force2D": bool(s.get("force2D", False)),
            "force2Ddimension": int(s.get("force2Ddimension", 0)), "imageType": cfg.get("imageType", {}) or {},
            "path": str(path), "raw": s}


def resolve_case(spec, c, root):
    if "image" in spec and "habitat" in spec:
        name = spec.get("name") or Path(spec["habitat"]).name.split(".")[0]
        return name, Path(spec["image"]), Path(spec["habitat"])
    cid = spec["case_id"]
    if c is None:
        raise SystemExit(f"{cid}: no pipeline_config.py found - give 'image' and 'habitat' paths instead")
    center, pid = cid.split("__", 1)
    if center not in c.CENTERS:
        raise SystemExit(f"{cid}: center {center} not in pipeline_config.CENTERS")
    from single_ct_utils import collect_images
    image = collect_images(c.CENTERS[center]["image"]).get(pid)
    if image is None:
        raise SystemExit(f"{cid}: no CT for {pid} in {c.CENTERS[center]['image']}")
    hits = sorted(glob.glob(str(root / "habitat_maps" / "*" / f"{cid}.*")))
    if not hits:
        raise SystemExit(f"{cid}: no habitat map under {root / 'habitat_maps'} - run step 5 or give the path")
    return cid, Path(image), Path(hits[0])


def habitat_labels(hab):
    k, rim = None, None
    if hab.HasMetaDataKey("HabitatLabels"):
        m = re.fullmatch(r"background=0; core=1\.\.(\d+); rim=(none|\d+)", hab.GetMetaData("HabitatLabels").strip())
        if m:
            k, rim = int(m.group(1)), (None if m.group(2) == "none" else int(m.group(2)))
    arr = sitk.GetArrayFromImage(hab).astype(int)
    if k is None:                                  # no metadata: every positive label is a core habitat
        k = int(arr.max())
        warnings.warn("habitat map has no HabitatLabels metadata; assuming labels 1..max are core habitats")
    return arr, k, rim


def parse_feature(name):
    m = FEATURE_RE.match(name)
    if not m:
        return None
    region, itype, cls, feat = m.groups()
    return {"region": region, "image_type": itype, "class": cls, "feature": feat}


def check_features(features, root):
    known = set()
    if root is not None:
        for p in glob.glob(str(root / "regional_features" / "*" / "regional_features_all.csv")):
            known |= set(pd.read_csv(p, nrows=0, encoding="utf-8-sig").columns)
    bad = []
    for f in features:
        info = parse_feature(f)
        if info is None or (known and f not in known):
            near = difflib.get_close_matches(f, sorted(known), n=5, cutoff=0.5) if known else []
            bad.append(f"  {f}" + (f"\n      closest: {near}" if near else ""))
        elif info["class"] == "shape":
            bad.append(f"  {f}: shape features describe the whole region and have no voxel map")
    if bad:
        raise SystemExit("Unknown / unsupported feature names (use step-6 column names):\n" + "\n".join(bad))


def step6_values(root, case_id):
    if root is None:
        return {}
    for p in glob.glob(str(root / "regional_features" / "*" / "regional_features_all.csv")):
        t = pd.read_csv(p, encoding="utf-8-sig")
        if "CaseID" in t and (t.CaseID == case_id).any():
            return t.loc[t.CaseID == case_id].iloc[0].to_dict()
    return {}


# ============================================================================= filtered images (as step 6)
def crop_box(mask, pad):
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(axis=0) - pad, 0)
    hi = np.minimum(idx.max(axis=0) + pad + 1, mask.shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def sub_image(ct, arr, box):
    img = sitk.GetImageFromArray(arr[box])
    img.SetSpacing(ct.GetSpacing())
    img.SetDirection(ct.GetDirection())
    img.SetOrigin(ct.TransformIndexToPhysicalPoint([int(box[2].start), int(box[1].start), int(box[0].start)]))
    return img


def filtered_crop(ct, ct_arr, region_mask, image_type, st, engine):
    """Filtered image on the cropped region exactly as PyRadiomics does in step 6 (numpy, z-y-x)."""
    box = crop_box(region_mask, st["padDistance"]) if st["preCrop"] else tuple(slice(0, n) for n in region_mask.shape)
    img = sub_image(ct, ct_arr, box)
    if image_type == "original":
        return sitk.GetArrayFromImage(img).astype(float), box
    use_pr = engine == "pyradiomics" or (engine == "auto" and _has_radiomics())
    if use_pr:
        from radiomics import imageoperations as io
        msk = sub_image(ct, region_mask.astype(np.uint8), box)
        base = image_type.split("-")[0]
        kw = dict(st["raw"])
        if base == "log":
            kw["sigma"] = [float(image_type.split("-")[2] + "." + image_type.split("-")[3])]
            gen = io.getLoGImage(img, msk, **kw)
        elif base == "wavelet":
            kw.update(st["imageType"].get("Wavelet", {}) or {})
            gen = io.getWaveletImage(img, msk, **kw)
        else:
            kw.update(st["imageType"].get(base.capitalize(), {}) or {})
            gen = getattr(io, f"get{base.capitalize()}Image")(img, msk, **kw)
        for out, name, _ in gen:
            if name == image_type:
                return sitk.GetArrayFromImage(out).astype(float), box
        raise RuntimeError(f"PyRadiomics did not produce image type {image_type} (region too small?)")
    if image_type.startswith("log-sigma"):
        sigma = float(image_type.split("-")[2] + "." + image_type.split("-")[3])
        size, spacing = np.array(img.GetSize()), np.array(img.GetSpacing())
        if not np.all(size >= np.ceil(sigma / spacing) + 1):
            raise RuntimeError(f"region too small for LoG sigma {sigma} mm")
        f = sitk.LaplacianRecursiveGaussianImageFilter()      # identical to radiomics.imageoperations.getLoGImage
        f.SetNormalizeAcrossScale(True)
        f.SetSigma(sigma)
        return sitk.GetArrayFromImage(f.Execute(img)).astype(float), box
    raise SystemExit(f"{image_type} needs PyRadiomics (pip install pyradiomics) or FILTER_ENGINE='pyradiomics'")


def _has_radiomics():
    try:
        import radiomics  # noqa: F401
        return True
    except ImportError:
        return False


# ============================================================================= first-order statistics
def bin_edges(values, bw):
    mn, mx = np.nanmin(values), np.nanmax(values)
    low = mn - (mn % bw)
    return np.arange(low, mx + 2 * bw, bw)          # = radiomics.imageoperations.getBinEdges (fixed bin width)


def stat_rows(W, name, edges=None, shift=0.0):
    """Statistic `name` of every row of W (NaN = not in the window); PyRadiomics definitions."""
    n = np.sum(np.isfinite(W), axis=1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if name == "Mean":
            return np.nanmean(W, 1)
        if name == "Median":
            return np.nanmedian(W, 1)
        if name.endswith("Percentile"):
            return np.nanpercentile(W, float(name[:-len("Percentile")]), axis=1)
        if name == "InterquartileRange":
            return np.nanpercentile(W, 75, axis=1) - np.nanpercentile(W, 25, axis=1)
        if name == "Minimum":
            return np.nanmin(W, 1)
        if name == "Maximum":
            return np.nanmax(W, 1)
        if name == "Range":
            return np.nanmax(W, 1) - np.nanmin(W, 1)
        if name == "MeanAbsoluteDeviation":
            return np.nanmean(np.abs(W - np.nanmean(W, 1, keepdims=True)), 1)
        if name == "RobustMeanAbsoluteDeviation":
            p10, p90 = np.nanpercentile(W, 10, axis=1, keepdims=True), np.nanpercentile(W, 90, axis=1, keepdims=True)
            R = np.where((W >= p10) & (W <= p90), W, np.nan)
            return np.nanmean(np.abs(R - np.nanmean(R, 1, keepdims=True)), 1)
        if name == "Variance":
            return np.nanvar(W, 1)
        if name == "StandardDeviation":
            return np.nanstd(W, 1)
        if name in ("Skewness", "Kurtosis"):
            d = W - np.nanmean(W, 1, keepdims=True)
            m2 = np.nanmean(d ** 2, 1)
            return np.nanmean(d ** 3, 1) / m2 ** 1.5 if name == "Skewness" else np.nanmean(d ** 4, 1) / m2 ** 2
        if name == "RootMeanSquared":
            return np.sqrt(np.nanmean((W + shift) ** 2, 1))
        if name in ("Energy", "TotalEnergy"):
            return np.nansum((W + shift) ** 2, 1)           # TotalEnergy: x voxel volume (applied by caller)
        if name in ("Entropy", "Uniformity"):
            B = np.where(np.isfinite(W), np.digitize(W, edges), -1)
            out = np.zeros(len(W))
            for b in np.unique(B[B >= 0]):
                p = (B == b).sum(1) / np.maximum(n, 1)
                out += -p * np.log2(p + np.spacing(1)) if name == "Entropy" else p ** 2
            return out
    raise SystemExit(f"first-order feature {name!r} is not implemented for local maps")


def local_firstorder(F, extent, name, radius_xyz, edges, chunk=20000):
    rx, ry, rz = radius_xyz
    A = np.where(extent, F, np.nan)
    A = np.pad(A, ((rz, rz), (ry, ry), (rx, rx)), constant_values=np.nan)
    V = sliding_window_view(A, (2 * rz + 1, 2 * ry + 1, 2 * rx + 1))
    pos = np.argwhere(extent)
    out = np.full(F.shape, np.nan)
    for i in range(0, len(pos), chunk):
        p = pos[i:i + chunk]
        W = V[p[:, 0], p[:, 1], p[:, 2]].reshape(len(p), -1)
        val = stat_rows(W, name, edges, FIRSTORDER_SHIFT)
        val[np.isfinite(W).sum(1) < MIN_KERNEL_VOXELS] = np.nan
        out[p[:, 0], p[:, 1], p[:, 2]] = val
    return out


def tail_map(F, region, name):
    v = F[region]
    out = np.full(F.shape, np.nan)
    if name.endswith("Percentile"):
        q = float(name[:-len("Percentile")])
        t = np.percentile(v, q)
        out[region] = (F[region] <= t) if q <= 50 else (F[region] >= t)
    elif name == "InterquartileRange":
        q1, q3 = np.percentile(v, [25, 75])
        out[region] = np.where(F[region] < q1, -1, np.where(F[region] > q3, 1, 0))
    elif name in ("Median", "Mean"):
        c = np.median(v) if name == "Median" else v.mean()
        out[region] = np.sign(F[region] - c)
    elif name in ("Minimum", "Maximum"):
        out[region] = F[region] == (v.min() if name == "Minimum" else v.max())
    else:
        return None
    return out


# ============================================================================= texture (PyRadiomics voxel-based)
def local_texture(ct, ct_arr, extent_full, info, st):
    if not _has_radiomics():
        raise SystemExit(f"{info['class']} maps need PyRadiomics")
    from radiomics import featureextractor
    s = {k: v for k, v in st["raw"].items() if v is not None}
    s.update({"kernelRadius": TEXTURE_KERNEL_RADIUS, "maskedKernel": True, "initValue": np.nan, "voxelBatch": 5000})
    ex = featureextractor.RadiomicsFeatureExtractor(**s)
    ex.disableAllImageTypes()
    base = info["image_type"].split("-")[0]
    args = {}
    if base == "log":
        args = {"sigma": [float(info["image_type"].split("-")[2] + "." + info["image_type"].split("-")[3])]}
    elif base == "wavelet":
        args = st["imageType"].get("Wavelet", {}) or {}
    ex.enableImageTypeByName({"original": "Original", "log": "LoG", "wavelet": "Wavelet"}.get(base, base.capitalize()),
                             customArgs=args)
    ex.disableAllFeatures()
    ex.enableFeaturesByName(**{info["class"]: [info["feature"]]})
    msk = sitk.GetImageFromArray(extent_full.astype(np.uint8))
    msk.CopyInformation(ct)
    res = ex.execute(ct, msk, label=1, voxelBased=True)
    key = f"{info['image_type']}_{info['class']}_{info['feature']}"
    if key not in res:
        raise RuntimeError(f"PyRadiomics returned no {key}")
    full = sitk.Resample(res[key], ct, sitk.Transform(), sitk.sitkNearestNeighbor, float("nan"), sitk.sitkFloat64)
    arr = sitk.GetArrayFromImage(full)
    return np.where(extent_full, arr, np.nan)


# ============================================================================= output
def write(arr, ct, path, dtype=np.float32):
    a = np.where(np.isfinite(arr), arr, BACKGROUND).astype(dtype)
    img = sitk.GetImageFromArray(a)
    img.CopyInformation(ct)
    sitk.WriteImage(img, str(path), True)
    print("  saved", path.name)


def overview_png(path, ct_arr, hab, k, local_maps, spacing, case):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    core = (hab >= 1) & (hab <= k)
    z = int(np.argmax(core.sum(axis=(1, 2)))) if PNG_SLICE == "max_area" else int(PNG_SLICE)
    ys, xs = np.nonzero(hab[z] > 0)
    pad_y, pad_x = int(PNG_ZOOM_PAD_MM / spacing[1]), int(PNG_ZOOM_PAD_MM / spacing[0])
    yb = slice(max(ys.min() - pad_y, 0), ys.max() + pad_y + 1)
    xb = slice(max(xs.min() - pad_x, 0), xs.max() + pad_x + 1)
    aspect = spacing[1] / spacing[0]
    n = 2 + len(local_maps)
    ncol = min(PNG_COLUMNS, n)
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.9 * nrow), constrained_layout=True, squeeze=False)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    axes = axs.ravel()[:n]
    ctw = ct_arr[z, yb, xb]
    for ax in axes:
        ax.imshow(ctw, cmap="gray", vmin=CT_WINDOW[0], vmax=CT_WINDOW[1], aspect=aspect, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_title(f"{case}\nCT, slice {z}", fontsize=8)
    try:
        sys.path.insert(0, str(HERE / "figures"))
        import plot_config as P
        cols = [P.HABITAT_COLORS[i % len(P.HABITAT_COLORS)] for i in range(k)] + ["#bdbdbd"]
    except Exception:                               # noqa: BLE001
        cols = list(plt.get_cmap("tab10").colors[:k]) + ["#bdbdbd"]
    h = np.where(hab[z, yb, xb] > 0, hab[z, yb, xb], np.nan)
    axes[1].imshow(h, cmap=ListedColormap(cols[:int(np.nanmax(h))]), vmin=0.5, vmax=np.nanmax(h) + 0.5,
                   aspect=aspect, interpolation="nearest", alpha=0.85)
    axes[1].set_title("Habitats " + " ".join(f"{i}" for i in range(1, k + 1)), fontsize=8)
    for ax, (name, m) in zip(axes[2:], local_maps.items()):
        im = ax.imshow(m[z, yb, xb], cmap=PNG_CMAP, aspect=aspect, interpolation="nearest", alpha=0.9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
        ax.set_title(short(name).replace(" | ", "\n"), fontsize=7)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("  saved", path.name)


def short(feature):
    i = parse_feature(feature)
    t = i["image_type"].replace("original", "").replace("log-sigma-", "LoG ").replace("-0-mm-3D", " mm").replace("-mm-3D", " mm")
    return f"{i['region'].replace('Hab', 'Habitat ')} | {t + ' ' if t else ''}{i['feature']}"


def safe(name):
    return re.sub(r"[^\w.-]+", "_", name)


# ============================================================================= main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", nargs="+", help="case IDs <Center>__<PatientID>")
    ap.add_argument("--image")
    ap.add_argument("--habitat")
    ap.add_argument("--features", nargs="+")
    ap.add_argument("--out")
    args, _ = ap.parse_known_args()
    cases = ([{"case_id": c} for c in args.case] if args.case else
             [{"image": args.image, "habitat": args.habitat}] if args.image and args.habitat else CASES)
    features = args.features or FEATURES
    c, root = pipeline_config()
    st = regional_settings(c)
    if st["force2D"]:
        print("NOTE: force2D is on in the YAML - texture maps are 3D here; first-order maps are unaffected")
    check_features(features, root)
    infos = {f: parse_feature(f) for f in features}

    qc = []
    for spec in cases:
        case, img_path, hab_path = resolve_case(spec, c, root)
        out = Path(args.out or OUTPUT_DIR or (root / "feature_maps" if root else HERE / "feature_maps")) / safe(case)
        out.mkdir(parents=True, exist_ok=True)
        print(f"{case}\n  CT      {img_path}\n  habitat {hab_path}")
        ct = sitk.ReadImage(str(img_path))
        hab_img = sitk.ReadImage(str(hab_path))
        if ct.GetSize() != hab_img.GetSize() or not np.allclose(ct.GetSpacing(), hab_img.GetSpacing(), atol=1e-6):
            raise SystemExit(f"{case}: CT and habitat map are on different grids - use the CT used in step 5")
        ct_arr = sitk.GetArrayFromImage(ct).astype(float)
        hab, k, rim = habitat_labels(hab_img)
        if SAVE_HABITAT_NII:
            write(hab, ct, out / f"{safe(case)}__habitats.nii.gz", np.uint8)
        ref = step6_values(root, case) if CHECK_AGAINST_STEP6 else {}
        spacing = ct.GetSpacing()
        local_maps = {}
        for f, info in infos.items():
            lab = rim if info["region"] == "Rim" else int(info["region"][3:])
            if lab is None or lab > (k + (1 if rim else 0)):
                print(f"  {f}: label not in this habitat map, skipped")
                continue
            region = hab == lab
            if region.sum() < 2:
                print(f"  {f}: habitat {lab} has {int(region.sum())} voxels in this case, skipped")
                qc.append({"case": case, "feature": f, "habitat_label": lab, "voxels": int(region.sum()),
                           "status": "habitat absent"})
                continue
            extent = region if LOCAL_EXTENT == "habitat" else (hab >= 1) & (hab <= k)
            # filtered image cropped around the habitat itself = exactly the step-6 input for this feature
            Fc, box = filtered_crop(ct, ct_arr, region, info["image_type"], st, FILTER_ENGINE)
            Fr = np.full(ct_arr.shape, np.nan)
            Fr[box] = Fc
            if LOCAL_EXTENT == "hematoma":           # filtered image of the whole hematoma for the wider map
                Fe_c, box_e = filtered_crop(ct, ct_arr, extent, info["image_type"], st, FILTER_ENGINE)
                Fe = np.full(ct_arr.shape, np.nan)
                Fe[box_e] = Fe_c
            else:
                Fe = Fr
            vals = Fr[region]
            edges = bin_edges(vals, st["binWidth"])
            rec = np.nan
            if info["class"] == "firstorder":
                rec = float(stat_rows(vals[None, :], info["feature"], edges, FIRSTORDER_SHIFT)[0])
                if info["feature"] == "TotalEnergy":
                    rec *= float(np.prod(spacing))
            s6 = ref.get(f, np.nan)
            qc.append({"case": case, "feature": f, "habitat_label": lab, "voxels": int(region.sum()),
                       "recomputed_value": rec, "step6_value": s6,
                       "abs_difference": abs(rec - s6) if np.isfinite(rec) and pd.notna(s6) else np.nan,
                       "filter_engine": "pyradiomics" if (FILTER_ENGINE == "pyradiomics" or
                                                          (FILTER_ENGINE == "auto" and _has_radiomics())) else "simpleitk",
                       "local_extent": LOCAL_EXTENT, "kernel_radius_xyz": str(KERNEL_RADIUS_VOX), "status": "ok"})
            stem = out / f"{safe(case)}__{safe(f)}"
            if "filtered" in MAPS:
                write(np.where(extent, Fe, np.nan), ct, Path(f"{stem}__filtered.nii.gz"))
            if "local" in MAPS:
                if info["class"] == "firstorder":
                    e_edges = bin_edges(Fe[extent], st["binWidth"]) if LOCAL_EXTENT == "hematoma" else edges
                    L = local_firstorder(Fe, extent, info["feature"], KERNEL_RADIUS_VOX, e_edges)
                    if info["feature"] == "TotalEnergy":
                        L *= float(np.prod(spacing))
                else:
                    L = local_texture(ct, ct_arr, extent, info, st)
                write(L, ct, Path(f"{stem}__local.nii.gz"))
                local_maps[f] = L
            if "tail" in MAPS and info["class"] == "firstorder":
                T = tail_map(Fr, region, info["feature"])
                if T is not None:
                    write(T, ct, Path(f"{stem}__tail.nii.gz"), np.float32)
        if SAVE_PNG and local_maps:
            overview_png(out / f"{safe(case)}__overview.png", ct_arr, hab, k, local_maps, spacing, case)

    q = pd.DataFrame(qc)
    qdir = Path(args.out or OUTPUT_DIR or (root / "feature_maps" if root else HERE / "feature_maps"))
    q.to_csv(qdir / "feature_map_qc.csv", index=False, encoding="utf-8-sig")
    print(f"saved {qdir / 'feature_map_qc.csv'}")
    if len(q) and "abs_difference" in q:
        big = q[q.abs_difference > 1e-3 * np.maximum(1, q.step6_value.abs())]
        if len(big):
            print("WARNING: recomputed region values differ from step 6 (check CT / YAML / filter engine):")
            print(big[["case", "feature", "recomputed_value", "step6_value"]].to_string(index=False))
        elif q.step6_value.notna().any():
            print("region values reproduce step 6 - the maps show exactly what the features measure")


if __name__ == "__main__":
    main()
