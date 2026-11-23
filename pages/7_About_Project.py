import streamlit as st

import config
from utils import ui

ui.setup("About Project", "ℹ️")
ui.sidebar_brand()
ui.hero("About the Project", config.PROJECT_TITLE)

a, b = st.columns(2)
with a:
    ui.card("Objectives",
            "<ul><li>Improve the visibility of lung fields in chest X-rays with CLAHE and frequency-domain filtering.</li>"
            "<li>Segment the lungs with a traditional pipeline, a U-Net and a CNN-Transformer hybrid.</li>"
            "<li>Compare all methods with Dice, IoU, precision, recall, F1, sensitivity, specificity, HD95, speed and model size.</li>"
            "<li>Study the effect of hyper-parameters and of each component (ablation).</li></ul>")
with b:
    ui.card("Methodology",
            "<ol><li>Load and match images with masks, resize, normalise, split 70/15/15.</li>"
            "<li>Enhance with CLAHE, Fourier transform, Butterworth filtering, inverse transform.</li>"
            "<li>Traditional: Otsu threshold, morphology, contours, connected components.</li>"
            "<li>U-Net trained with Dice + BCE loss.</li>"
            "<li>Hybrid: CNN encoder -> Transformer self-attention -> feature fusion -> CNN decoder with skip connections.</li>"
            "<li>Evaluate every method on the same held-out test set.</li></ol>")
ui.card("Technologies",
        "<p>Python · PyTorch · torchvision · OpenCV · NumPy / SciPy · scikit-image · pandas · Plotly · Streamlit</p>")
st.subheader("Team")
cols = st.columns(len(config.TEAM))
for col, t in zip(cols, config.TEAM):
    with col:
        ui.card(t["name"], f"<p>{t['role']}<br><span style='color:#6b7a90'>{t['id']}</span></p>")
st.caption(f"Guide: {config.GUIDE} · {config.INSTITUTION}   (edit these details in config.py)")
