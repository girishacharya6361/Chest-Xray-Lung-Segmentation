"""Create downloadable PNGs / ZIP of the processed results."""
import io
import zipfile
import cv2
import numpy as np
import pandas as pd

from utils.pipeline import METHOD_LABELS


def png_bytes(img) -> bytes:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def results_table(res) -> pd.DataFrame:
    rows = []
    for k, m in res["methods"].items():
        if not m["ok"]:
            continue
        row = {"Method": METHOD_LABELS[k], "Time (ms)": round(m["ms"], 2)}
        if m.get("metrics"):
            row.update({a: round(b, 4) for a, b in m["metrics"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def build_zip(res, name="xray") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{name}_original.png", png_bytes(res["gray"]))
        z.writestr(f"{name}_clahe.png", png_bytes(res["enh"]["clahe"]))
        z.writestr(f"{name}_frequency_filtered.png", png_bytes(res["enh"]["filtered"]))
        for k, m in res["methods"].items():
            if m["ok"]:
                z.writestr(f"{name}_{k}_mask.png", png_bytes(m["mask"]))
                z.writestr(f"{name}_{k}_overlay.png", png_bytes(m["overlay"]))
        tbl = results_table(res)
        if len(tbl):
            z.writestr(f"{name}_results.csv", tbl.to_csv(index=False))
    return buf.getvalue()
