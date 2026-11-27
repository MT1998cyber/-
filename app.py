import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE, ASSETS = HERE / "code", HERE / "assets"
RUNS = Path(tempfile.gettempdir()) / "hab_runs"
os.environ.setdefault("GRADIO_TEMP_DIR", str(Path(tempfile.gettempdir()) / "hab_gradio"))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
ENV = dict(os.environ, HAB_BUNDLE=str(ASSETS / "model.json"), HAB_KMEANS_PKL=str(ASSETS / "joint_kmeans.pkl"),
           HAB_NNUNET_RESULTS=str(ASSETS / "nnUNet_results"), PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
KEEP_RUNS_SECONDS = 3600
NAME = "uploaded_case"

DISCLAIMER = ("**Research use only.** This tool is not a medical device and must not be used for diagnosis or "
              "treatment decisions. The result depends on an automatic segmentation that you must check, and "
              "the model was developed in a limited number of centers; scanners or protocols that differ "
              "from them can shift the result.")


def has_gpu():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:                                             # noqa: BLE001
        return False


def clean_old_runs():
    RUNS.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for d in RUNS.iterdir():
        if d.is_dir() and now - d.stat().st_mtime > KEEP_RUNS_SECONDS:
            shutil.rmtree(d, ignore_errors=True)


def run(image, mask, all_folds, heatmaps, agreed):
    """Generator: streams the log while predict_case.py runs, then fills in the results."""
    nothing = gr.update()
    if not agreed:
        raise gr.Error("Please confirm that the image is anonymized and that you accept the notice above.")
    if image is None:
        raise gr.Error("Please upload a CT image (.nii.gz).")
    if not str(image).lower().endswith((".nii", ".nii.gz")):
        raise gr.Error("The CT must be a NIfTI file (.nii or .nii.gz).")
    clean_old_runs()
    work = Path(tempfile.mkdtemp(prefix="run_", dir=RUNS))
    out = work / "result"
    cmd = [sys.executable, str(CODE / "predict_case.py"), "--image", str(image), "--out", str(out), "--name", NAME]
    if mask:
        cmd += ["--mask", str(mask)]
    if not all_folds:
        cmd += ["--folds", "0", "--no-tta"]
    if not heatmaps:
        cmd.append("--no-heatmaps")
    log = ["Started. The segmentation is the slowest step, especially on a computer without GPU."]
    yield "\n".join(log), "Running ...", nothing, nothing, nothing, nothing, nothing
    proc = subprocess.Popen(cmd, cwd=str(CODE), env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log.append(line)
            yield "\n".join(log[-60:]), nothing, nothing, nothing, nothing, nothing, nothing
    code = proc.wait()
    if code != 0 or not (out / "results.json").exists():
        tail = "\n".join(log[-12:])
        yield "\n".join(log[-60:]), "Failed - see the log.", nothing, nothing, nothing, nothing, nothing
        raise gr.Error(f"The analysis stopped with an error:\n{tail}")

    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    seg = res["segmentation"]
    warn = "".join(f"\n- {w}" for w in res["warnings"]) or "\n- none"
    summary = (f"### Predicted probability of hematoma expansion: **{res['probability']:.3f}**\n\n"
               f"Segmentation: {seg['volume_ml']:.1f} mL, {seg['components']} component(s), mean {seg['mean_hu']:.0f} HU "
               f"({res['segmentation_source']}).\n\n**Warnings:**{warn}\n\n"
               f"**Check the segmentation image first.** If it is wrong, correct the mask in a viewer (e.g. ITK-SNAP, "
               f"using `image_on_study_grid.nii.gz` from the download) and upload it as 'own mask'.\n\n{DISCLAIMER}")
    table = pd.DataFrame({"Feature": list(res["features"]),
                          "Value": [round(v, 3) for v in res["features"].values()],
                          "z-score": [round(res["z"][f], 3) for f in res["features"]],
                          "SHAP (log-odds)": [round(res["shap"][f], 3) for f in res["features"]]})
    qc, shap = out / "figures" / "qc_overlay.png", out / "figures" / "shap.png"
    heat = next(iter((out / "heatmaps").glob("*/*__all_features.png")), None) if heatmaps else None
    zip_path = shutil.make_archive(str(work / "result"), "zip", root_dir=str(out))
    yield ("\n".join(log[-60:]), summary, str(qc) if qc.exists() else None, str(shap) if shap.exists() else None,
           str(heat) if heat else None, table, zip_path)


def build_ui():
    with gr.Blocks(title="Hematoma habitat model (research demo)") as demo:
        gr.Markdown("# HabiLens, Habitat-based hematoma expansion model\n"
                    "Upload an **anonymized** non-contrast head CT. The hematoma is segmented automatically, split into "
                    "habitats, and the habitat-based model estimates the risk of hematoma expansion; SHAP shows what drives "
                    "it. Everything runs on this computer.\n\n" + DISCLAIMER)
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.File(label="CT image (.nii.gz)", file_types=[".nii", ".gz"], type="filepath")
                mask = gr.File(label="Own mask (optional; skips the automatic segmentation)",
                               file_types=[".nii", ".gz"], type="filepath")
                all_folds = gr.Checkbox(value=has_gpu(), label="Use all 5 segmentation models (slower, more accurate)")
                heatmaps = gr.Checkbox(value=True, label="Draw feature heat maps")
                agreed = gr.Checkbox(value=False, label="The image is anonymized and I accept the research-use notice")
                go = gr.Button("Run", variant="primary")
                log = gr.Textbox(label="Progress", lines=10, max_lines=10, interactive=False)
            with gr.Column(scale=2):
                summary = gr.Markdown()
                qc = gr.Image(label="Segmentation and habitats (check first)", interactive=False)
                shap = gr.Image(label="SHAP", interactive=False)
                heat = gr.Image(label="Feature heat maps", interactive=False)
                table = gr.Dataframe(label="Features", interactive=False)
                download = gr.File(label="Download all results (report.html, results.json, masks, figures)")
        go.click(run, [image, mask, all_folds, heatmaps, agreed], [log, summary, qc, shap, heat, table, download])
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(max_size=4, default_concurrency_limit=1)
    demo.launch(server_name=os.environ.get("HAB_HOST", "127.0.0.1"), server_port=int(os.environ.get("HAB_PORT", "7860")),
                show_api=False, inbrowser=os.environ.get("HAB_HOST", "127.0.0.1") == "127.0.0.1")
