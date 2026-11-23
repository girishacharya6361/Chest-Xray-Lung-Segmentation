import json
from PIL import Image
import streamlit as st

import config
from preprocessing.image_loader import dataset_status, find_pairs, read_gray, read_mask, resize_image, _list_images
from training.dataset_loader import make_splits
from utils import ui
from utils.pipeline import overlay_mask

ui.setup("Dataset", "🗂️")
ui.sidebar_brand()
ui.hero("Dataset", "Montgomery County Chest X-ray Dataset - images with manual lung segmentation masks.")

ds = dataset_status()
c = st.columns(3)
with c[0]: ui.kpi("Images", ds["n_images"], "dataset/images")
with c[1]: ui.kpi("Masks", ds["n_masks"], "dataset/masks")
with c[2]: ui.kpi("Matched pairs", ds["n_pairs"], "used for training / evaluation")
if not ds["ready"]:
    st.error(ds["message"])
    st.code("python -m utils.prepare_dataset --src /path/to/MontgomerySet", language="bash")
    st.stop()

pairs = find_pairs()
stems = [p[0] for p in pairs]
sp = make_splits(stems)
st.subheader("Dataset split")
s = st.columns(3)
for col, k in zip(s, ("train", "val", "test")):
    with col: ui.kpi(k.capitalize(), len(sp[k]), f"{100 * len(sp[k]) / len(stems):.0f} % of the data")
st.caption(f"Random split with seed {sp['seed']} - stored in results/split.json so every model and experiment uses identical splits. "
           "The test split is never used for training or model selection.")

st.subheader("Image sizes (read from file headers)")
sizes = {}
for _, ip, _ in pairs:
    with Image.open(ip) as im:
        sizes[im.size] = sizes.get(im.size, 0) + 1
st.dataframe([{"Width x Height": f"{w} x {h}", "Images": n} for (w, h), n in sorted(sizes.items(), key=lambda x: -x[1])],
             use_container_width=True, hide_index=True)

st.subheader("Browse samples")
split = st.radio("Split", ["train", "val", "test"], horizontal=True)
sel = st.selectbox("Image", sp[split])
_, ip, mp = next(p for p in pairs if p[0] == sel)
img = resize_image(read_gray(ip), 512)
m = resize_image(read_mask(mp), 512, mask=True)
g = st.columns(3)
g[0].image(img, caption="X-ray", use_container_width=True, clamp=True)
g[1].image(m, caption="Ground-truth mask", use_container_width=True, clamp=True)
g[2].image(overlay_mask(img, m), caption="Overlay", use_container_width=True)

st.subheader("Sample gallery")
n = st.slider("Number of samples", 3, 12, 6)
cols = st.columns(3)
for i, stem in enumerate(sp[split][:n]):
    _, ip, mp = next(p for p in pairs if p[0] == stem)
    gi = resize_image(read_gray(ip), 384)
    cols[i % 3].image(overlay_mask(gi, resize_image(read_mask(mp), 384, mask=True)), caption=stem, use_container_width=True)
