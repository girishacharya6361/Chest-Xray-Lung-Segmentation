import streamlit as st

import config
from utils import ui, plots
from utils.pipeline import enhance
from utils.export import png_bytes

ui.setup("Image Enhancement", "✨")
ui.sidebar_brand()
ui.hero("Image Enhancement", "CLAHE contrast enhancement followed by Fourier-domain filtering.")
gray = ui.need_image()

p = st.session_state.setdefault("params", config.traditional_params())
with st.sidebar:
    st.subheader("Parameters")
    clip = st.slider("CLAHE clip limit", 0.5, 8.0, float(p["clip"]), 0.5)
    tile = st.select_slider("CLAHE tile grid", [4, 8, 12, 16], value=int(p["tile"]))
    ftype = st.selectbox("Frequency filter", ["lowpass", "highpass", "highboost"],
                         index=["lowpass", "highpass", "highboost"].index(p["ftype"]))
    cutoff = st.slider("Filter cutoff (0 = off)", 0.0, 1.0, float(p["cutoff"]), 0.05,
                       help="Fraction of the maximum frequency radius")
    boost = st.slider("High-boost gain", 0.0, 3.0, 1.0, 0.1) if ftype == "highboost" else 1.0

e = enhance(gray, clip, tile, cutoff, ftype, boost)
c = st.columns(3)
for col, img, t in zip(c, (gray, e["clahe"], e["filtered"]),
                       ("Original", f"CLAHE (clip {clip})", f"CLAHE + {ftype} filter")):
    with col:
        st.image(img, caption=t, use_container_width=True, clamp=True)
k = st.columns(4)
with k[0]: ui.kpi("PSNR: CLAHE", f"{e['psnr_clahe']:.2f} dB", "vs original")
with k[1]: ui.kpi("SSIM: CLAHE", f"{e['ssim_clahe']:.3f}", "vs original")
with k[2]: ui.kpi("PSNR: filtered", f"{e['psnr_filtered']:.2f} dB", "vs original")
with k[3]: ui.kpi("SSIM: filtered", f"{e['ssim_filtered']:.3f}", "vs original")
st.caption("PSNR/SSIM measure how far the enhanced image departs from the original; "
           "they are not accuracy scores (no 'perfect' enhanced reference exists).")

with st.expander("Fourier spectra and filter transfer function"):
    s = st.columns(3)
    s[0].plotly_chart(plots.spectrum(e["spectrum"], "Spectrum of CLAHE image"), use_container_width=True)
    s[1].plotly_chart(plots.spectrum(e["transfer"] * 255, f"{ftype} transfer function"), use_container_width=True)
    s[2].plotly_chart(plots.spectrum(e["spectrum_filtered"], "Filtered spectrum"), use_container_width=True)

b = st.columns(3)
b[0].download_button("Download original", png_bytes(gray), "original.png", "image/png")
b[1].download_button("Download CLAHE", png_bytes(e["clahe"]), "clahe.png", "image/png")
b[2].download_button("Download filtered", png_bytes(e["filtered"]), "filtered.png", "image/png")
if st.button("Use these parameters for segmentation"):
    st.session_state["params"] = dict(clip=clip, tile=tile, cutoff=cutoff, ftype=ftype)
    st.session_state.pop("results", None)
    st.success("Saved. Press *Process image* on the Upload page (or run segmentation) to apply them. "
               "The U-Net / hybrid models always use the settings they were trained with.")
