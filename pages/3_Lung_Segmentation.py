import streamlit as st

from utils import ui
from utils.pipeline import run_pipeline, METHOD_LABELS
from utils.export import build_zip, png_bytes, results_table

ui.setup("Lung Segmentation", "🫁")
ui.sidebar_brand()
ui.hero("Lung Segmentation", "Traditional vs CNN (U-Net) vs proposed CNN-Transformer hybrid.")
gray = ui.need_image()

if "results" not in st.session_state:
    st.info("This image has not been processed yet.")
    if st.button("Run segmentation now", type="primary"):
        p = st.session_state["params"]
        with st.spinner("Segmenting..."):
            st.session_state["results"] = run_pipeline(gray, p["clip"], p["tile"], p["cutoff"], p["ftype"],
                                                       st.session_state.get("gt_mask"))
        st.rerun()
    st.stop()

res = st.session_state["results"]
cols = st.columns(3)
for col, (k, m) in zip(cols, res["methods"].items()):
    with col:
        st.markdown(f"### {METHOD_LABELS[k]}")
        if not m["ok"]:
            st.warning(m["message"]); continue
        st.image(m["overlay"], caption="Overlay", use_container_width=True)
        st.image(m["mask"], caption="Binary mask", use_container_width=True, clamp=True)
        ui.kpi("Processing time", f"{m['ms']:.1f} ms", m["params"])
        if m["metrics"]:
            mm = m["metrics"]
            st.caption(f"Dice {mm['Dice']:.3f} · IoU {mm['IoU']:.3f} · Precision {mm['Precision']:.3f} · Recall {mm['Recall']:.3f}")
        st.download_button("Download mask", png_bytes(m["mask"]), f"{k}_mask.png", "image/png", key=f"m{k}")

if res["gt"] is None:
    st.caption("No ground-truth mask was provided, so no accuracy metrics are shown for this image "
               "(metrics are only computed when ground truth is available).")
else:
    st.subheader("Metrics against your ground-truth mask")
    st.dataframe(results_table(res), use_container_width=True, hide_index=True)

if res["methods"]["traditional"]["ok"]:
    with st.expander("Traditional pipeline - intermediate steps"):
        steps = res["methods"]["traditional"]["steps"]
        sc = st.columns(3)
        for i, (n, img) in enumerate(steps.items()):
            sc[i % 3].image(img, caption=n, use_container_width=True, clamp=True)

name = st.session_state.get("filename", "xray").rsplit(".", 1)[0]
st.download_button("Download all results (ZIP)", build_zip(res, name), f"{name}_results.zip", "application/zip")
