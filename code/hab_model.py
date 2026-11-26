import json
import math
from pathlib import Path

OOD_Z = 3.0      

class HabModel:
    def __init__(self, spec):
        self.spec = spec
        self.features = list(spec["features"])

    @classmethod
    def load(cls, path):
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        if spec.get("bundle_version") != 1:
            raise ValueError(f"unsupported bundle_version {spec.get('bundle_version')!r}")
        return cls(spec)

    # ------------------------------------------------------------------ core
    def predict(self, values):
        """values: {feature: float or None/NaN}. Returns a dict (all numbers are plain floats)."""
        s = self.spec
        pre, h = s["preprocessing"], s["preprocessing"]["harmonization"]
        imputed, z, harm = [], {}, {}
        for f in self.features:
            v = values.get(f)
            if v is None or not math.isfinite(float(v)):
                v = pre["impute_median"][f]
                imputed.append(f)
            v = float(v)
            hv = (v - h["center_mean"][f]) / h["center_sd"][f] * h["pooled_sd"][f] + h["pooled_mean"][f]
            harm[f] = hv
            z[f] = (hv - pre["zscore_mean"][f]) / pre["zscore_sd"][f]
        coef, bg = s["model"]["coef"], s["shap"]["background_mean_z"]
        shap = {f: coef[f] * (z[f] - bg[f]) for f in self.features}
        logit = s["model"]["intercept"] + sum(coef[f] * z[f] for f in self.features)
        base = s["shap"]["base_value"]
        ood = [f for f in self.features if f not in imputed and abs(z[f]) > OOD_Z]
        return {"probability": 1.0 / (1.0 + math.exp(-logit)), "logit": logit, "base_value": base,
                "shap": shap, "z": z, "harmonized": harm, "imputed": imputed, "out_of_range": ood,
                "shap_additivity_error": abs(base + sum(shap.values()) - logit)}

    def predict_table(self, rows):
        """rows: iterable of dicts -> list of results."""
        return [self.predict(r) for r in rows]

    def info(self):
        s = self.spec
        return {k: s.get(k) for k in ("model_name", "held_out_center", "created", "features", "validation", "training", "notes")}
