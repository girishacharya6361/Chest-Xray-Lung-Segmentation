import json
import subprocess
import sys
import time
import streamlit as st

import config
from preprocessing.image_loader import dataset_status
from utils import ui, plots, inference

ui.setup("Model Training", "🧠")
ui.sidebar_brand()
ui.hero("Model Training", "Train U-Net or the CNN-Transformer hybrid. Training runs in a separate process and saves a checkpoint.")

ds = dataset_status()
if not ds["ready"]:
    st.error(ds["message"]); st.stop()
if not inference.torch_available():
    st.error("PyTorch is not installed. Run `pip install torch torchvision` and restart the app."); st.stop()

kind = st.selectbox("Model", ["unet", "hybrid"], format_func=lambda k: config.MODELS[k]["label"])
d = config.default_train_cfg()
c = st.columns(4)
epochs = c[0].number_input("Epochs", 1, 500, d["epochs"])
bs = c[1].selectbox("Batch size", [2, 4, 8, 16, 32], index=2)
lr = c[2].select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3], value=1e-3, format_func=lambda v: f"{v:g}")
size = c[3].selectbox("Image size", [128, 192, 256, 320], index=2)
with st.expander("Advanced options"):
    a = st.columns(4)
    wd = a[0].slider("Dice loss weight", 0.0, 1.0, 0.5, 0.1)
    clip = a[1].slider("CLAHE clip limit", 1.0, 5.0, 2.0, 0.5)
    use_clahe = a[2].checkbox("Use CLAHE", True)
    cutoff = a[3].slider("Frequency low-pass cutoff (0 = off)", 0.0, 0.8, 0.0, 0.1)
    if kind == "hybrid":
        h = st.columns(3)
        backbone = h[0].selectbox("CNN backbone", ["resnet18", "resnet34", "simple"])
        depth = h[1].selectbox("Transformer depth", [1, 2, 4, 6], index=1)
        use_tr = h[2].checkbox("Use Transformer", True)

st.caption(f"Starting training overwrites `checkpoints/{kind}.pt`. Device: "
           f"{'GPU (CUDA)' if __import__('torch').cuda.is_available() else 'CPU - training will be slow'}.")
proc = st.session_state.get("train_proc")
running = proc is not None and proc.poll() is None

b = st.columns([1, 1, 4])
if b[0].button("Start training", type="primary", disabled=running):
    cmd = [sys.executable, "-m", f"training.train_{kind}", "--epochs", str(epochs), "--batch-size", str(bs),
           "--lr", str(lr), "--size", str(size), "--w-dice", str(wd), "--w-bce", str(round(1 - wd, 3)),
           "--clahe-clip", str(clip), "--freq-cutoff", str(cutoff)]
    if not use_clahe: cmd.append("--no-clahe")
    if kind == "hybrid":
        cmd += ["--backbone", backbone, "--depth", str(depth)]
        if not use_tr: cmd.append("--no-transformer")
    log = open(config.LOG_DIR / f"{kind}.log", "w")
    (config.RESULTS_DIR / f"history_{kind}.json").unlink(missing_ok=True)
    st.session_state["train_proc"] = subprocess.Popen(cmd, cwd=config.ROOT, stdout=log, stderr=subprocess.STDOUT)
    st.session_state["train_kind"] = kind
    st.rerun()
if b[1].button("Stop", disabled=not running):
    proc.terminate(); st.warning("Training stopped (the best checkpoint so far is kept).")

view = st.session_state.get("train_kind", kind)
hp = config.RESULTS_DIR / f"history_{view}.json"
lp = config.LOG_DIR / f"{view}.log"
if hp.exists():
    h = json.loads(hp.read_text())
    ep = h["epochs"]
    st.subheader(f"Training progress - {config.MODELS[view]['label']}")
    st.progress(min(1.0, len(ep) / h["epochs_total"]), text=f"Epoch {len(ep)} / {h['epochs_total']}  ·  status: {h['status']}  ·  device: {h['device']}")
    if ep:
        k = st.columns(4)
        with k[0]: ui.kpi("Best val Dice", f"{h['best_val_dice']:.4f}", f"epoch {h['best_epoch']}")
        with k[1]: ui.kpi("Last val Dice", f"{ep[-1]['val_dice']:.4f}")
        with k[2]: ui.kpi("Last train loss", f"{ep[-1]['train_loss']:.4f}")
        with k[3]: ui.kpi("Parameters", f"{h['n_params'] / 1e6:.2f} M", f"{h['n_train']} train / {h['n_val']} val images")
        g = st.columns(2)
        g[0].plotly_chart(plots.loss_curves(ep), use_container_width=True)
        g[1].plotly_chart(plots.dice_curve(ep), use_container_width=True)
elif not running:
    st.info("No training history yet for this model. Press *Start training*.")

if lp.exists():
    with st.expander("Training log"):
        st.code("\n".join(lp.read_text().splitlines()[-25:]) or "(empty)")
if running:
    st.caption("Training is running... this page refreshes every 3 seconds.")
    time.sleep(3)
    st.rerun()
elif proc is not None and proc.poll() not in (None, 0):
    st.error("The training process exited with an error - see the log above.")
