import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

import habitat_feature_maps as HFM

# ---------------- settings ----------------
CASES = [
    {"case_id": "CASE_ID"},
    # {"image": r"IMAGE_DIR\ID.nii.gz",
    #  "habitat": r"HABITAT_DIR\ID.nrrd",
]
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
FEATURE_LABELS = {}            # optional display names, e.g. {"Hab2_original_firstorder_Median": "Habitat 2: Median"}
OUTPUT_DIR = None              # None = WORK_ROOT/RUN_NAME/feature_heatmaps
MAP_TYPE = "local"             # "local" (feature computed in a sliding window) | "filtered" (filtered image)
HEAT_EXTENT = "habitat"        # "habitat" = only the feature's habitat; "hematoma" = whole hematoma
KERNEL_RADIUS_VOX = (4, 4, 0)  # local window (x, y, z) in voxels: 9 x 9 in-plane = 4.5 mm
MIN_KERNEL_VOXELS = 10

# slices
SLICES = "top3"                # "max_area" | "topN" (N largest habitat slices, e.g. "top3") | "all" | [8, 9, 10]
ORDER_SLICES = "position"      # "position" (cranio-caudal) or "area"
ZOOM_PAD_MM = 8                # margin around the hematoma
FLIP_UD, FLIP_LR = False, False    # display orientation (check against your viewer once)

# heat map look
CMAP = "jet"                   # "jet" (Grad-CAM look), "turbo", "inferno", "RdYlBu_r" ...
ALPHA = 0.55                   # heat map opacity
SMOOTH = True                  # bilinear upsampling + soft edges (Grad-CAM look); False = blocky voxels
UPSAMPLE = 4
COLOR_RANGE = "case"           # "case": 2-98th percentile of this case's map; "fixed": FIXED_RANGE[feature]
FIXED_RANGE = {}               # e.g. {"Hab2_original_firstorder_Median": (40, 75)} - same scale for every case
REVERSE_FOR = []               # features whose LOW values point to HE: colours flipped (red = low value)
SHOW_OUTLINES = True           # habitat outlines on the heat-map panel
OUTLINE_WIDTH = 0.9
HABITAT_COLORS = None          # None = figures/plot_config.HABITAT_COLORS
CT_WINDOW = (0, 80)
PANEL_INCH = 2.3
DPI = 300
FORMATS = ["png"]              # e.g. ["png", "tiff", "pdf"]


# ----------------------------------------------------------------------------- helpers
def habitat_colors(k):
    cols = HABITAT_COLORS
    if cols is None:
        try:
            import sys
            sys.path.insert(0, str(HFM.HERE / "figures"))
            import plot_config as P
            cols = P.HABITAT_COLORS
        except Exception:                          # noqa: BLE001
            cols = ["#47b0c5", "#db5143", "#319c87", "#eb927e", "#9aa4b1", "#7a5aa6"]
    return [cols[i % len(cols)] for i in range(k + 1)]


def label(f):
    return FEATURE_LABELS.get(f) or HFM.short(f).replace(" | ", ": ")


def pick_slices(region):
    area = region.sum(axis=(1, 2))
    present = np.nonzero(area)[0]
    if isinstance(SLICES, (list, tuple)):
        zs = [int(z) for z in SLICES]
    elif SLICES == "all":
        zs = list(present)
    elif SLICES == "max_area":
        zs = [int(np.argmax(area))]
    elif str(SLICES).startswith("top"):
        n = int(str(SLICES)[3:])
        zs = list(present[np.argsort(area[present])[::-1][:n]])
    else:
        raise SystemExit(f"unknown SLICES {SLICES!r}")
    return sorted(zs) if ORDER_SLICES == "position" else zs


def orient(a):
    if FLIP_UD:
        a = a[::-1]
    if FLIP_LR:
        a = a[:, ::-1]
    return a


def smooth_layer(values, mask, aspect):
    """Upsample a 2D map (NaN outside mask) for a soft Grad-CAM look; returns (values, alpha)."""
    v = values.copy()
    if mask.any() and (~mask).any():           # fill outside by nearest inside value, so edges don't fade to NaN
        idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
        v = v[tuple(idx)]
    v = np.where(np.isfinite(v), v, np.nanmedian(values) if np.isfinite(values).any() else 0)
    if not SMOOTH:
        return v, mask.astype(float)
    zf = (UPSAMPLE, UPSAMPLE)
    vz = ndimage.zoom(v, zf, order=1)
    az = np.clip(ndimage.zoom(mask.astype(float), zf, order=1), 0, 1)
    return vz, az


def draw_ct(ax, ct2, aspect):
    ax.imshow(ct2, cmap="gray", vmin=CT_WINDOW[0], vmax=CT_WINDOW[1], aspect=aspect,
              interpolation="bilinear" if SMOOTH else "nearest", extent=(0, ct2.shape[1], ct2.shape[0], 0))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_heat(ax, plt, heat2, mask2, aspect, vmin, vmax, cmap):
    v, a = smooth_layer(heat2, mask2, aspect)
    rgba = plt.get_cmap(cmap)(np.clip((v - vmin) / max(vmax - vmin, 1e-12), 0, 1))
    rgba[..., 3] = a * ALPHA
    ax.imshow(rgba, extent=(0, heat2.shape[1], heat2.shape[0], 0), aspect=aspect,
              interpolation="bilinear" if SMOOTH else "nearest")


def draw_outlines(ax, hab2, k, cols, lw):
    yy, xx = np.mgrid[0:hab2.shape[0], 0:hab2.shape[1]] + 0.5
    for lab in range(1, k + 2):
        m = (hab2 == lab).astype(float)
        if m.any() and lab - 1 < len(cols):
            if SMOOTH:
                m = ndimage.zoom(m, UPSAMPLE, order=1)
                yy2, xx2 = np.mgrid[0:m.shape[0], 0:m.shape[1]]
                ax.contour((xx2 + 0.5) / UPSAMPLE, (yy2 + 0.5) / UPSAMPLE, m, [0.5], colors=[cols[lab - 1]], linewidths=lw)
            else:
                ax.contour(xx, yy, m, [0.5], colors=[cols[lab - 1]], linewidths=lw)


def save(fig, path):
    for ext in FORMATS:
        fig.savefig(f"{path}.{ext}", dpi=DPI, bbox_inches="tight", facecolor="white")
    print("  saved", Path(f"{path}.png").name)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", nargs="+")
    ap.add_argument("--features", nargs="+")
    ap.add_argument("--out")
    args, _ = ap.parse_known_args()
    cases = [{"case_id": c} for c in args.case] if args.case else CASES
    features = args.features or FEATURES
    HFM.KERNEL_RADIUS_VOX, HFM.MIN_KERNEL_VOXELS = KERNEL_RADIUS_VOX, MIN_KERNEL_VOXELS

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize

    c, root = HFM.pipeline_config()
    st = HFM.regional_settings(c)
    HFM.check_features(features, root)
    infos = {f: HFM.parse_feature(f) for f in features}
    out_root = Path(args.out or OUTPUT_DIR or ((root / "feature_heatmaps") if root else HFM.HERE / "feature_heatmaps"))
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in cases:
        case, img_path, hab_path = HFM.resolve_case(spec, c, root)
        print(case)
        ct = sitk.ReadImage(str(img_path))
        hab_img = sitk.ReadImage(str(hab_path))
        if ct.GetSize() != hab_img.GetSize():
            raise SystemExit(f"{case}: CT and habitat map have different sizes")
        ct_arr = sitk.GetArrayFromImage(ct).astype(float)
        hab, k, rim = HFM.habitat_labels(hab_img)
        sx, sy, _ = ct.GetSpacing()
        aspect = sy / sx
        cols = habitat_colors(k)
        hem = (hab >= 1) & (hab <= k)
        ys, xs = np.nonzero(hem.any(axis=0))
        py, px = int(ZOOM_PAD_MM / sy), int(ZOOM_PAD_MM / sx)
        yb = slice(max(ys.min() - py, 0), ys.max() + py + 1)
        xb = slice(max(xs.min() - px, 0), xs.max() + px + 1)
        out = out_root / HFM.safe(case)
        out.mkdir(parents=True, exist_ok=True)
        summary = {}

        for f, info in infos.items():
            lab = rim if info["region"] == "Rim" else int(info["region"][3:])
            region = hab == lab if lab else np.zeros_like(hem)
            if region.sum() < 2:
                print(f"  {f}: habitat absent in this case, skipped")
                rows.append({"case": case, "feature": f, "status": "habitat absent"})
                continue
            extent = region if HEAT_EXTENT == "habitat" else hem
            Fc, box = HFM.filtered_crop(ct, ct_arr, extent, info["image_type"], st, HFM.FILTER_ENGINE)
            F = np.full(ct_arr.shape, np.nan)
            F[box] = Fc
            if HEAT_EXTENT == "habitat":
                Fr = F
            else:                                    # region value from the habitat crop (= step 6)
                Fr_c, box_r = HFM.filtered_crop(ct, ct_arr, region, info["image_type"], st, HFM.FILTER_ENGINE)
                Fr = np.full(ct_arr.shape, np.nan)
                Fr[box_r] = Fr_c
            edges = HFM.bin_edges(Fr[region], st["binWidth"])
            value = float(HFM.stat_rows(Fr[region][None, :], info["feature"], edges)[0]) \
                if info["class"] == "firstorder" else np.nan
            if MAP_TYPE == "filtered":
                M = np.where(extent, F, np.nan)
            elif info["class"] == "firstorder":
                M = HFM.local_firstorder(F, extent, info["feature"], KERNEL_RADIUS_VOX,
                                         HFM.bin_edges(F[extent], st["binWidth"]))
            else:
                M = HFM.local_texture(ct, ct_arr, extent, info, st)
            fin = M[np.isfinite(M)]
            if not len(fin):
                print(f"  {f}: no map values (habitat too small for the window?), skipped")
                continue
            vmin, vmax = FIXED_RANGE.get(f) if COLOR_RANGE == "fixed" and f in FIXED_RANGE else \
                tuple(np.percentile(fin, [2, 98]))
            if vmax <= vmin:
                vmax = vmin + 1e-6
            cmap = CMAP + "_r" if f in REVERSE_FOR and not CMAP.endswith("_r") else CMAP
            summary[f] = (M, vmin, vmax, cmap, extent)
            rows.append({"case": case, "feature": f, "habitat_label": lab, "voxels": int(region.sum()),
                         "region_value": value, "map": MAP_TYPE, "color_min": vmin, "color_max": vmax,
                         "reversed_colors": f in REVERSE_FOR, "status": "ok"})

            zs = pick_slices(region)
            fig, axes = plt.subplots(len(zs), 3, figsize=(PANEL_INCH * 3 + 0.9, PANEL_INCH * len(zs) * 0.95 + 0.5),
                                     squeeze=False, layout="constrained")
            for r, z in enumerate(zs):
                ct2 = orient(ct_arr[z, yb, xb])
                hab2 = orient(hab[z, yb, xb])
                m2 = orient(M[z, yb, xb])
                e2 = orient(extent[z, yb, xb]) & np.isfinite(m2)
                for ax in axes[r]:
                    draw_ct(ax, ct2, aspect)
                h = np.where(hab2 > 0, hab2, np.nan)
                axes[r, 1].imshow(h, cmap=ListedColormap(cols[:k + 1]), vmin=0.5, vmax=k + 1.5, alpha=0.8,
                                  aspect=aspect, interpolation="nearest", extent=(0, h.shape[1], h.shape[0], 0))
                draw_heat(axes[r, 2], plt, m2, e2, aspect, vmin, vmax, cmap)
                if SHOW_OUTLINES:
                    draw_outlines(axes[r, 2], hab2, k, cols, OUTLINE_WIDTH)
                axes[r, 0].set_ylabel(f"slice {z}", fontsize=8)
                if r == 0:
                    axes[r, 0].set_title("CT", fontsize=9)
                    axes[r, 1].set_title("Habitats " + "  ".join(f"H{i}" for i in range(1, k + 1)), fontsize=9)
                    axes[r, 2].set_title("Heat map" + (" (whole hematoma)" if HEAT_EXTENT == "hematoma" else ""),
                                         fontsize=9)
            sm = plt.cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap)
            cb = fig.colorbar(sm, ax=axes[:, 2], fraction=0.05, pad=0.02, shrink=0.8)
            cb.set_label(("local " if MAP_TYPE == "local" else "") + info["feature"], fontsize=8)
            cb.ax.tick_params(labelsize=7)
            fig.suptitle(f"{case}\n{label(f)}   (region value = {value:.2f})", fontsize=9)
            hp = [plt.Line2D([], [], color=cols[i], lw=2) for i in range(k)]
            fig.legend(hp, [f"Habitat {i + 1}" for i in range(k)], loc="outside lower center", ncol=k,
                       fontsize=7, frameon=False)
            save(fig, out / f"{HFM.safe(case)}__{HFM.safe(f)}")
            plt.close(fig)

        if summary:                                   # all features on the largest hematoma slice
            z = int(np.argmax(hem.sum(axis=(1, 2))))
            n = len(summary) + 1
            ncol = min(4, n)
            nrow = int(np.ceil(n / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(PANEL_INCH * ncol + 0.6, PANEL_INCH * nrow + 0.6),
                                     squeeze=False, layout="constrained")
            ct2, hab2 = orient(ct_arr[z, yb, xb]), orient(hab[z, yb, xb])
            for ax in axes.ravel():
                ax.axis("off")
            ax0 = axes.ravel()[0]
            ax0.axis("on")
            draw_ct(ax0, ct2, aspect)
            h = np.where(hab2 > 0, hab2, np.nan)
            ax0.imshow(h, cmap=ListedColormap(cols[:k + 1]), vmin=0.5, vmax=k + 1.5, alpha=0.8, aspect=aspect,
                       interpolation="nearest", extent=(0, h.shape[1], h.shape[0], 0))
            ax0.set_title(f"Habitats (slice {z})", fontsize=8)
            for ax, (f, (M, vmin, vmax, cmap, extent)) in zip(axes.ravel()[1:], summary.items()):
                ax.axis("on")
                m2 = orient(M[z, yb, xb])
                draw_ct(ax, ct2, aspect)
                draw_heat(ax, plt, m2, orient(extent[z, yb, xb]) & np.isfinite(m2), aspect, vmin, vmax, cmap)
                if SHOW_OUTLINES:
                    draw_outlines(ax, hab2, k, cols, OUTLINE_WIDTH * 0.8)
                cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap), ax=ax,
                                  fraction=0.046, pad=0.02)
                cb.ax.tick_params(labelsize=6)
                ax.set_title(label(f), fontsize=7)
            fig.suptitle(case, fontsize=9)
            save(fig, out / f"{HFM.safe(case)}__all_features")
            plt.close(fig)

    pd.DataFrame(rows).to_csv(out_root / "heatmap_values.csv", index=False, encoding="utf-8-sig")
    print(f"saved {out_root / 'heatmap_values.csv'}")


if __name__ == "__main__":
    main()