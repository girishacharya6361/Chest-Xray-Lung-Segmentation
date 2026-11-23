import json
import pandas as pd
import streamlit as st

import config
from utils import ui, plots
from utils.pipeline import METHOD_LABELS
from utils.export import results_table, build_zip

ui.setup("Model Comparison", "📊")
ui.sidebar_brand()
ui.hero("Model Comparison", "Real measurements only - computed from your trained models and dataset.")
R = config.RESULTS_DIR


def read(name):
    p = R / name
    return pd.read_csv(p) if p.exists() else None


tab1, tab2, tab3, tab4 = st.tabs(["Test-set comparison", "Ablation study", "Hyper-parameter tuning", "Uploaded image"])

with tab1:
    if st.button("Run / refresh evaluation on the test split"):
        try:
            from evaluation.comparison import run_comparison
            with st.spinner("Evaluating all methods on the held-out test split..."):
                run_comparison(log=lambda *_: None)
            st.success("Evaluation finished.")
        except Exception as e:
            st.error(f"Could not run evaluation: {e}")
    summ, std, enh = read("comparison_summary.csv"), read("comparison_std.csv"), read("enhancement_metrics.csv")
    if summ is None:
        st.info("No evaluation results yet. Make sure the dataset is in place, train the models, then press the button above "
                "or run `python -m evaluation.comparison`.")
    else:
        meta = json.loads((R / "comparison_meta.json").read_text())
        st.caption(f"{meta['n_images']} held-out test images at {meta['eval_size']}x{meta['eval_size']} px - computed {meta['created']}")
        st.subheader("Segmentation metrics (mean over test images)")
        st.dataframe(summ.round(4), use_container_width=True, hide_index=True)
        with st.expander("Standard deviations"):
            st.dataframe(std.round(4), use_container_width=True, hide_index=True)
        st.plotly_chart(plots.grouped_bar(summ, "Method", ["Dice", "IoU", "Precision", "Recall", "F1"], "Overlap metrics"),
                        use_container_width=True)
        c = st.columns(3)
        c[0].plotly_chart(plots.simple_bar(summ, "Method", "Time_ms", "Execution time", "ms / image"), use_container_width=True)
        c[1].plotly_chart(plots.simple_bar(summ, "Method", "HD95", "HD95 (lower is better)", "pixels"), use_container_width=True)
        c[2].plotly_chart(plots.simple_bar(summ, "Method", "Model_MB", "Model size", "MB"), use_container_width=True)
        if enh is not None:
            st.subheader("Image enhancement quality (PSNR / SSIM vs original)")
            st.dataframe(enh.round(4), use_container_width=True, hide_index=True)
        st.download_button("Download comparison table (CSV)", summ.to_csv(index=False), "comparison_summary.csv", "text/csv")

with tab2:
    ab = read("ablation.csv")
    if ab is None:
        st.info("No ablation results yet. After training the hybrid model run `python -m evaluation.ablation`.")
    else:
        st.dataframe(ab.round(4), use_container_width=True, hide_index=True)
        st.plotly_chart(plots.grouped_bar(ab, "Variant", ["Dice", "IoU", "Precision", "Recall"], "Ablation study"),
                        use_container_width=True)

with tab3:
    tt, td = read("tuning_traditional.csv"), read("tuning_dl.csv")
    if tt is None and td is None:
        st.info("No tuning results yet. Run `python -m evaluation.tuning --stage traditional` and `--stage dl`.")
    if tt is not None:
        st.subheader("Traditional pipeline: CLAHE clip x filter cutoff (validation Dice)")
        piv = tt.pivot(index="clahe_clip", columns="freq_cutoff", values="val_dice")
        st.dataframe(piv.round(4), use_container_width=True)
    if td is not None:
        st.subheader("Deep model: one-parameter-at-a-time search (validation Dice)")
        for par, g in td.groupby("parameter", sort=False):
            with st.expander(par, expanded=False):
                st.dataframe(g.drop(columns="parameter").round(4), use_container_width=True, hide_index=True)
        st.plotly_chart(plots.grouped_bar(td.assign(label=td.parameter + "=" + td.value), "label", ["val_dice"],
                                          "Validation Dice for every tuning run", yrange=(0, 1.05)), use_container_width=True)

with tab4:
    res = st.session_state.get("results")
    if res is None:
        st.info("Upload and process an X-ray first (Upload X-ray page).")
    else:
        tbl = results_table(res)
        st.dataframe(tbl, use_container_width=True, hide_index=True)
        if res["gt"] is None:
            st.caption("Accuracy metrics need a ground-truth mask; upload one on the Upload page to see Dice / IoU / etc. here.")
        elif len(tbl):
            st.plotly_chart(plots.grouped_bar(tbl, "Method", ["Dice", "IoU", "Precision", "Recall", "F1"], "This image"),
                            use_container_width=True)
        e = res["enh"]
        st.write(f"**Enhancement:** CLAHE PSNR {e['psnr_clahe']:.2f} dB / SSIM {e['ssim_clahe']:.3f}; "
                 f"filtered PSNR {e['psnr_filtered']:.2f} dB / SSIM {e['ssim_filtered']:.3f}")
        st.download_button("Download results (ZIP)", build_zip(res, st.session_state.get("filename", "xray").rsplit(".", 1)[0]),
                           "results.zip", "application/zip")
