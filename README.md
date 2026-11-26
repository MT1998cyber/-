# Hematoma habitat model - local research demo

Upload an anonymized non-contrast head CT (`.nii.gz`). The tool

1. segments the hematoma (including intraventricular hemorrhage) with an nnU-Net ensemble,
2. divides it into habitats (K-means on voxel-wise radiomic features),
3. estimates the risk of hematoma expansion with a habitat-based logistic model, and
4. shows SHAP bars (what drives the estimate) and voxel-wise heat maps of the six model features.

Everything runs on your own computer; nothing is uploaded anywhere.

> **Research use only. This is not a medical device and must not be used for diagnosis or treatment decisions.** The result depends on an automatic segmentation that you must check, and the model was developed in a limited
> number of centers.

- Code: [GitHub URL] (this repository)
- Model files (`assets/`): [Zenodo URL / DOI]
- Paper: [citation]

---

## 1. What you need

|        |                                                                        |
| ------ | ---------------------------------------------------------------------- |
| Python | 3.9 (the version used in development; see `requirements.txt`)          |
| Disk   | a few GB (PyTorch, the five nnU-Net models)                            |
| RAM    | 16 GB recommended                                                      |
| GPU    | optional. Without a GPU the segmentation is slower (see "Speed" below) |

## 2. Get the files

```
git clone [GitHub URL]
cd [repository name]
```

Download `assets.zip` from Zenodo ([Zenodo URL / DOI]) and unzip it **into this folder**, so that you have

```
app.py
code/
requirements.txt
assets/
    model.json
    joint_kmeans.pkl
    nnUNet_results/Dataset501_*/nnUNetTrainer__nnUNetPlans__3d_fullres/{dataset.json, plans.json, fold_0 ... fold_4/checkpoint_final.pth}
    SHA256SUMS
```

Optional integrity check (Windows PowerShell, for one file as an example):

```
Get-FileHash -Algorithm SHA256 assets\model.json
```

Linux / macOS: `cd assets && sha256sum -c SHA256SUMS`.

## 3. Install

The tool needs PyRadiomics, SimpleITK, scikit-learn, Gradio and nnU-Net (which needs PyTorch). Use a fresh environment.

### Option A - one environment (simplest)

```
conda create -n he_habitat python=3.9
conda activate he_habitat
pip install torch
pip install -r requirements.txt
```

`pip install torch` installs the CPU build on Windows. For an NVIDIA GPU, install PyTorch with the command from
https://pytorch.org/get-started/locally/ first, then run the second command.

The versions in `requirements.txt` are those of the development environment. If `pip` reports a version conflict
between nnU-Net, PyTorch and the pinned packages, use Option B.

### Option B - two environments (the setup used and tested by the authors)

nnU-Net runs in its own environment and is called only for the segmentation step.

```
# environment 1: everything except nnU-Net
conda create -n he_habitat python=3.9
conda activate he_habitat
#   open requirements.txt, delete the line "nnunetv2", save
pip install -r requirements.txt

# environment 2: nnU-Net
conda create -n nnunet python=3.9
conda activate nnunet
pip install torch nnunetv2
```

Before starting the app, tell it where environment 2's Python is (see section 4).

## 4. Start the web app

In environment 1, in this folder:

```
python app.py
```

Then open **http://127.0.0.1:7860** in your browser.

With Option B, set the path of the nnU-Net environment's Python first, in the same terminal window:

```
# Windows PowerShell
$env:HAB_NNUNET_PYTHON="D:\Anaconda3\envs\nnunet\python.exe"
# Windows cmd
set HAB_NNUNET_PYTHON=D:\Anaconda3\envs\nnunet\python.exe
# Linux / macOS
export HAB_NNUNET_PYTHON=/path/to/envs/nnunet/bin/python
```

(`conda env list` shows where an environment is. The variable lasts only for that terminal window; set it again
in each new window.) Optional: `HAB_PORT=7861` changes the port.

## 5. Use it

1. Upload the CT (`.nii` / `.nii.gz`). It must be **anonymized, including the NIfTI header**.
2. Optional: upload your own mask (label > 0 = hematoma + IVH). The automatic segmentation is then skipped.
3. Choose the number of segmentation models:
   - one model without mirroring: faster;
   - "all 5 models": slower, more accurate. This is ticked by default only if a GPU is found.
4. Tick the confirmation box and press **Run**.
5. **Look at the segmentation image first.** If the mask is wrong, correct it in a viewer (e.g. ITK-SNAP, using `image_on_study_grid.nii.gz` from the download) and run again with that mask.

Results shown: predicted probability, warnings, segmentation / habitat overlay, SHAP bars, feature heat maps,
feature table, and a zip with everything.

| File in the zip              | Content                                                          |
| ---------------------------- | ---------------------------------------------------------------- |
| `report.html`                | self-contained report                                            |
| `results.json`               | probability, feature values, z-scores, SHAP values, QC, warnings |
| `segmentation.nii.gz`        | mask used (on the study grid)                                    |
| `image_on_study_grid.nii.gz` | the CT on the grid used for the analysis                         |
| `habitat_map.nrrd`           | habitat labels                                                   |
| `figures/`, `heatmaps/`      | images                                                           |

### Speed

On the authors' CPU-only workstation one case with one segmentation model and no mirroring took about 2 minutes
in total. Five models with mirroring are considerably slower on a CPU; a GPU is recommended for that setting.

## 6. Command line (optional)

`app.py` just calls `code/predict_case.py`. To use it directly, point it to the model files first
(PowerShell shown; use `export` on Linux / macOS):

```
$env:HAB_BUNDLE="assets\model.json"
$env:HAB_KMEANS_PKL="assets\joint_kmeans.pkl"
$env:HAB_NNUNET_RESULTS="assets\nnUNet_results"
cd code
python predict_case.py --image C:\path\ct.nii.gz --out C:\path\result --device cpu --folds 0 --no-tta
```

Useful options: `--mask file.nii.gz` (own mask), `--folds 0 1 2 3 4`, `--no-heatmaps`, `--device cpu|cuda`, `--nnunet-python <python.exe>`, `--no-resample`, `--keep-work`. `python predict_case.py --help` lists all.

## 7. Troubleshooting

| Message                                                             | What to do                                                                                                                      |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `nnunetv2 is not installed in the Python used for the segmentation` | Option A: `pip install nnunetv2` in the same environment. Option B: set `HAB_NNUNET_PYTHON` (section 4) and start the app again |
| `cannot import name 'HfFolder' from 'huggingface_hub'` (Python 3.9) | `pip install "huggingface_hub==0.25.2" "pydantic==2.10.6" "fastapi==0.115.6" "starlette==0.41.3"`                               |
| Segmentation is empty                                               | check the CT (non-contrast head CT in Hounsfield units) or provide your own mask                                                |
| Warning "does not look like Hounsfield units"                       | the file may be normalized or rescaled; use the original HU values                                                              |
| Warning "CT resampled"                                              | the CT spacing differs from the study grid (0.5 x 0.5 x 5 mm); features can shift slightly                                      |
| Warning "far outside the training distribution"                     | a feature value is unusual for the development data; interpret with great caution                                               |
| Port already in use                                                 | set `HAB_PORT` to another number                                                                                                |
| Out of memory                                                       | close other programs; use one segmentation model                                                                                |

## 8. Model and limits

- Model: logistic regression on six first-order habitat features (habitats 1 and 2; original and Laplacian-of-Gaussian
  filtered CT): `Hab2_original_firstorder_10Percentile`, `Hab2_original_firstorder_Median`, `Hab2_log-sigma-3-0-mm-3D_firstorder_10Percentile`, `Hab1_log-sigma-2-0-mm-3D_firstorder_InterquartileRange`, `Hab1_original_firstorder_90Percentile`, `Hab1_log-sigma-2-0-mm-3D_firstorder_10Percentile`.
  The model was fitted without one center, whose statistics are used to harmonize every uploaded case.
  Performance is reported in the paper.
- SHAP values are exact for this linear model and are given on the log-odds scale.
- The model was built on manual segmentations; the demo uses automatic ones.
- Scanners and protocols that differ from the development data can shift first-order (HU) features.
- Exploratory model; no prospective validation.

## 9. Privacy

The app runs locally and sends nothing to any server. Uploaded files are processed in a temporary folder of your
computer and removed after about one hour. By default the app listens on `127.0.0.1` only, so it is not reachable
from other computers; do not change this when working with patient images. Make sure images are anonymized.

The K-means model (`joint_kmeans.pkl`) is a Python pickle (joblib): only load files from sources you trust, and
compare them with `assets/SHA256SUMS`.

## 10. Citation and license

If you use this tool, please cite [paper citation] and the archived release [Zenodo DOI]. nnU-Net, PyRadiomics and the
other software have their own citation requirements; see their documentation. License: [license].
