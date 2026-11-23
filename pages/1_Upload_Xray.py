import numpy as np
import streamlit as st

import config
from preprocessing.image_loader import decode_image
from utils import ui
from utils.pipeline import run_pipeline

ui.setup("Upload X-ray", "📤")
ui.sidebar_brand()
ui.hero("Upload X-ray", "Upload any frontal chest X-ray (it does not need to be part of the dataset).")

if "params" not in st.session_state:
    st.session_state["params"] = config.traditional_params()

left, right = st.columns([3, 2])
with left:
    up = st.file_uploader("Drag and drop a chest X-ray here", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    gt_up = st.file_uploader("Optional: ground-truth lung mask (enables Dice / IoU / HD95 for this image)",
                             type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

if up is not None:
    data = up.getvalue()
    key = (up.name, len(data))
    if st.session_state.get("_key") != key:                       # new image: reset previous results
        try:
            st.session_state["gray"] = decode_image(data)
        except Exception as e:
            st.error(f"Could not read this file as an image: {e}")
            st.stop()
        st.session_state.update(_key=key, filename=up.name, file_size=len(data), file_type=up.type)
        st.session_state.pop("results", None)
    if gt_up is not None:
        try:
            gt = decode_image(gt_up.getvalue())
            st.session_state["gt_mask"] = ((gt > 127) * 255).astype(np.uint8)
        except Exception as e:
            st.error(f"Could not read the mask: {e}")
    else:
        st.session_state.pop("gt_mask", None)

if "gray" in st.session_state:
    g = st.session_state["gray"]
    with left:
        st.image(g, caption="Preview", use_container_width=True, clamp=True)
    with right:
        ui.card("File information",
                f"<ul><li><b>Name:</b> {st.session_state['filename']}</li>"
                f"<li><b>Size:</b> {st.session_state['file_size'] / 1024:.1f} KB</li>"
                f"<li><b>Resolution:</b> {g.shape[1]} x {g.shape[0]} px</li>"
                f"<li><b>Intensity range:</b> {int(g.min())} - {int(g.max())}</li>"
                f"<li><b>Ground-truth mask:</b> {'provided' if 'gt_mask' in st.session_state else 'not provided'}</li></ul>")
        p = st.session_state["params"]
        st.caption(f"Processing parameters: CLAHE clip {p['clip']}, {p['ftype']} cutoff {p['cutoff']} "
                   "(change them on the *Image Enhancement* page).")
        if st.button("Process image", use_container_width=True, type="primary"):
            with st.spinner("Enhancing and segmenting..."):
                st.session_state["results"] = run_pipeline(g, p["clip"], p["tile"], p["cutoff"], p["ftype"],
                                                           st.session_state.get("gt_mask"))
            ms = st.session_state["results"]["methods"]
            st.success("Done. Open **Lung Segmentation** or **Model Comparison** in the sidebar.")
            for k, m in ms.items():
                if not m["ok"]:
                    st.warning(f"{k}: {m['message']}")
else:
    with right:
        st.info("Upload an image to begin.")
