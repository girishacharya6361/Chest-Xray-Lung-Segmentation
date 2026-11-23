# Chest X-Ray Enhancement and Lung Segmentation Using CNNs and Transformers

Streamlit web application + PyTorch models. Upload a chest X-ray -> CLAHE + Fourier filtering -> three lung
segmentations (traditional, U-Net, proposed CNN-Transformer hybrid) -> comparison and downloads.

> **Honesty note.** The repository contains **no pre-computed results**. Every number shown in the app comes from
> models you train and evaluate on your own copy of the dataset (`results/*.csv`). Until you do, the app shows
> clear "not trained / no results yet" messages instead of placeholders.

## 1. Install (Python 3.10+)
```bash
python -m venv venv
venv\Scripts\activate            # Windows      (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```
For a CUDA GPU install the matching PyTorch build from https://pytorch.org first (training on CPU works but is slow).

## 2. Dataset (Montgomery County Chest X-ray Set)

The dataset is hosted separately by the U.S. National Library of Medicine (NLM/LHNCBC).
It is **not bundled inside this project ZIP** because the source data is a separate,
large dataset. Download it directly from the official NLM host:

- Dataset directory: https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/
- Dataset documentation: https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/NLM-MontgomeryCXRSet-ReadMe.pdf

Automated download (requires internet access):
```bash
python download_dataset.py
```

Then prepare paired images and masks:
```bash
python -m utils.prepare_dataset --src data/raw/MontgomerySet
```

The preparation script copies `CXR_png/*` into `dataset/images/` and merges the
left and right masks from `ManualMask/leftMask/` and `ManualMask/rightMask/` into
one binary mask per X-ray in `dataset/masks/`. Matching is by the original filename.
The NLM dataset documentation describes 138 frontal X-rays and manually generated
left/right lung annotations. Cite the NLM dataset documentation in your report.

## 3. Train, tune and evaluate (run in this order)
```bash
# (optional) tuning ------------------------------------------------------------------
python -m evaluation.tuning --stage traditional            # CLAHE clip x filter cutoff  -> results/best_traditional.json
python -m evaluation.tuning --stage dl --epochs 15         # 8 hyper-parameters (slow)   -> results/best_config.json

# training (saves checkpoints/unet.pt and checkpoints/hybrid.pt) -----------------------
python -m training.train_unet   --epochs 60
python -m training.train_hybrid --epochs 60                # add --use-best to use the tuned configuration

# experiments --------------------------------------------------------------------------
python -m evaluation.comparison                            # test-set metrics, tables, chart
python -m evaluation.ablation                              # full vs no-CLAHE vs no-Transformer
```
Useful options: `--size 256 --batch-size 8 --lr 1e-3 --w-dice 0.5 --w-bce 0.5 --clahe-clip 2 --freq-cutoff 0.3`,
hybrid: `--backbone resnet18|resnet34|simple --depth 2 --no-transformer`.
Training can also be started from the **Model Training** page of the app.

## 4. Launch the application
```bash
streamlit run app.py
```

## Experiments -> where they are
| # | Experiment | Code | Output |
|---|-----------|------|--------|
| 1 | Traditional image processing | `traditional/segmentation.py` | in app / `comparison_summary.csv` |
| 2 | ML/DL technique (U-Net) | `models/unet.py`, `training/train_unet.py` | `checkpoints/unet.pt` |
| 3 | Proposed hybrid | `models/hybrid_model.py`, `models/transformer.py` | `checkpoints/hybrid.pt` |
| 4 | Hyper-parameter tuning | `evaluation/tuning.py` | `tuning_traditional.csv`, `tuning_dl.csv`, `best_config.json` |
| 5 | Ablation | `evaluation/ablation.py` | `ablation.csv` |
| 6 | Performance comparison | `evaluation/comparison.py` | `comparison_summary.csv`, `comparison_chart.png` |

Tuned: CLAHE clip limit, frequency cutoff, image resolution, CNN backbone, Transformer depth, learning rate,
batch size, Dice/BCE weights. Tuning is one-parameter-at-a-time (coordinate descent) using **validation Dice only**.

## Design notes
* **Split:** 70/15/15 (train/val/test), fixed seed, saved in `results/split.json`; all methods share the same test set.
* **Hybrid architecture:** CNN encoder (ResNet-18/34 or plain CNN, 1-channel) -> 1x1 projection -> Transformer encoder
  (multi-head self-attention, 2-D sin-cos positional encoding) -> concat(CNN, Transformer) + 1x1 fusion ->
  CNN decoder with skip connections from every encoder stage -> mask logits.
* **Training/inference consistency:** models store their preprocessing config (size, CLAHE, filter) in the checkpoint,
  and `preprocess_for_model` is used for both training and prediction.
* **Metrics:** Dice, IoU, precision, recall, F1, sensitivity, specificity, HD95 (pixels, at 256x256), ms/image
  (includes pre-processing, after a warm-up run), model size (checkpoint file MB). PSNR/SSIM are computed
  original vs enhanced image (no enhanced "reference" exists, so they describe change, not accuracy).
  Segmentation metrics for an uploaded image appear only if you also upload its ground-truth mask.
* No horizontal-flip augmentation (heart position is anatomically meaningful).
* Images are resized to a square (aspect ratio is not preserved); predictions are resized back to the original size.
* Edit team details in `config.py` (`TEAM`, `GUIDE`, `INSTITUTION`).

## Structure
```
app.py  config.py  requirements.txt  README.md
dataset/{images,masks}   preprocessing/{image_loader,enhancement,frequency_filter}.py
traditional/segmentation.py     models/{unet,transformer,hybrid_model}.py
training/{dataset_loader,augmentation,trainer,train_unet,train_hybrid}.py
evaluation/{metrics,comparison,ablation,tuning}.py     utils/{inference,pipeline,export,plots,ui,prepare_dataset}.py
pages/1..7_*.py   checkpoints/   results/
```


## Dataset citation and limitations

Dataset source: U.S. National Library of Medicine, Montgomery County Chest X-ray Set.
The source provides 138 frontal chest radiographs and left/right lung masks.
The downloader retrieves the source files; it does not create synthetic samples.
The project ZIP contains code and setup tools, not the original medical images.

Before reporting results, run the training and evaluation commands on the downloaded
data. The test split should remain held out from training and tuning.
