"""Dashboard - run with:  streamlit run app.py"""
import streamlit as st

import config
from preprocessing.image_loader import dataset_status
from utils import ui, inference

ui.setup("Dashboard")
ui.sidebar_brand()
ui.hero(config.PROJECT_TITLE,
        "Upload a chest X-ray, enhance it with CLAHE and Fourier filtering, and compare traditional, "
        "U-Net and CNN-Transformer lung segmentation.")

ds = dataset_status()
unet_ok, hyb_ok = (inference.model_status(k)["ok"] for k in ("unet", "hybrid"))
c = st.columns(4)
with c[0]: ui.kpi("Dataset pairs", ds["n_pairs"], "image + mask pairs found in dataset/")
with c[1]: ui.kpi("Segmentation methods", 3, "traditional · U-Net · hybrid")
with c[2]: ui.kpi("Trained models", f"{int(unet_ok) + int(hyb_ok)} / 2", "checkpoints available")
with c[3]: ui.kpi("Experiments", 6, "traditional · DL · hybrid · tuning · ablation · comparison")
st.write("")

if not ds["ready"]:
    st.warning(ds["message"])
if not (unet_ok and hyb_ok):
    st.info("Some trained checkpoints are missing. Train them with `python -m training.train_unet` and "
            "`python -m training.train_hybrid` (or use the **Model Training** page). Traditional segmentation works without training.")

l, r = st.columns([3, 2])
with l:
    ui.card("Project overview",
            "<p>Lung segmentation in chest radiographs is the first step of many computer-aided diagnosis systems. "
            "Low contrast and noise make it hard, so this project first <b>enhances</b> the X-ray (CLAHE + frequency-domain "
            "filtering) and then compares three ways of segmenting the lungs: a <b>classical image-processing pipeline</b>, a "
            "<b>CNN (U-Net)</b> and a <b>proposed CNN-Transformer hybrid</b> that adds global self-attention to a CNN encoder-decoder.</p>")
with r:
    ui.card("Dataset",
            f"<p><b>Montgomery County Chest X-ray Dataset</b> - frontal chest X-rays with manual lung masks.</p>"
            f"<p>Found locally: <b>{ds['n_images']}</b> images, <b>{ds['n_masks']}</b> masks, <b>{ds['n_pairs']}</b> matched pairs. "
            f"Split: 70 % train / 15 % validation / 15 % test (fixed seed).</p>")

st.subheader("Available models")
m = st.columns(3)
with m[0]:
    ui.card("Traditional", "<p>CLAHE · Otsu threshold · morphology · contours · connected components.</p>"
            + ui.badge(True, "No training needed"))
for col, kind in zip(m[1:], ("unet", "hybrid")):
    st_ = inference.model_status(kind)
    with col:
        ui.card(config.MODELS[kind]["label"],
                "<p>" + ("Trained weights loaded from <code>checkpoints/%s</code>." % config.MODELS[kind]["ckpt"].name
                         if st_["ok"] else st_["message"]) + "</p>" + ui.badge(st_["ok"], "Trained", "Not trained"))

st.subheader("Implemented methods")
ui.card("Pipeline components",
        "<ul><li><b>Enhancement:</b> CLAHE, Fourier transform, Butterworth low-pass / high-pass / high-boost filtering, inverse Fourier transform</li>"
        "<li><b>Traditional:</b> thresholding, morphological operations, contour detection, connected-component analysis</li>"
        "<li><b>Deep learning:</b> U-Net; ResNet/simple-CNN encoder + Transformer encoder (multi-head self-attention) + CNN decoder</li>"
        "<li><b>Evaluation:</b> PSNR, SSIM, Dice, IoU, precision, recall, F1, sensitivity, specificity, HD95, ms/image, model size</li></ul>")

st.subheader("Project workflow")
steps = ["Upload X-ray", "Pre-process", "CLAHE", "Fourier filtering", "Traditional segmentation",
         "U-Net segmentation", "Hybrid segmentation", "Compare & download"]
cols = st.columns(4)
for i, s in enumerate(steps):
    with cols[i % 4]:
        st.markdown(f'<div class="step"><b>{i + 1}.</b> {s}</div><div style="height:10px"></div>', unsafe_allow_html=True)
