# BBCH Phenological Stage Identification

Code and pretrained model weights for the paper on BBCH phenological stage identification based on
a dual-stream EfficientFormer (RGB + multispectral) fusion network.

> **Paper title**: Phenological Stages (placeholder — update after acceptance)

## Overview

This repository releases the training code, model checkpoint and evaluation results
used in the paper.

**Model** — Dual-stream fusion network:

- Backbone: two EfficientFormerV2-S0 branches (RGB + 4-channel MS: G / NIR / R / RE)
- Auxiliary vegetation-index (VI) branches with dual-gate modulation and security fusion
- Full CSoP (covariance-of-second-order-pooling style) similarity-modulated fusion
  with a CBAM difference-modulation weighting module
- Output: 15 BBCH phenological stages (`240716` … `241008`, i.e. dates from
  2024-07-16 to 2024-10-08)

**Performance** (checkpoint: `checkpoints/EfficientFormer_RGBMS_Aux_CSoP2/best_model.pth`):

| Metric | Value |
| ------ | ----- |
| Best validation accuracy | **98.82%** (epoch 66) |
| Test macro-F1 | see `test_results/classification_report.txt` |

## Repository structure

```
├── train.py                              # Training script (original name: train_rgbms_vi_csop_cbam251229.py)
├── data_matching.py                      # RGB–MS sample binding module (dependency of train.py)
├── infer.py                              # Inference script for the released checkpoint
├── models/
│   └── efficientformer_v2.py             # EfficientFormerV2-S0 backbone definition
├── checkpoints/EfficientFormer_RGBMS_Aux_CSoP2/
│   ├── best_model.pth                    # Released weights (80.9 MB)
│   ├── classification_report_rgbms_aux_full_csop.txt   # Validation classification report
│   ├── train_val_curve_rgbms_aux_full_csop.png         # Training/validation curves
│   ├── train_val_metrics_rgbms_aux_full_csop.csv       # Per-epoch metrics
│   └── test_results/                     # Test-set evaluation
│       ├── classification_report.txt
│       ├── confusion_matrix.png
│       ├── test_predictions.csv
│       └── test_samples.csv
└── requirements.txt
```

## Environment

Tested with:

- Python 3.10 / PyTorch 2.1.2+cu121
- timm 1.0.12, tifffile 2025.5.10, scikit-learn 1.4.2
- See `requirements.txt` for the full list

## Inference

```bash
python infer.py \
    --checkpoint checkpoints/EfficientFormer_RGBMS_Aux_CSoP2/best_model.pth \
    --rgb path/to/rgb.jpg \
    --g   path/to/G.tif  --nir path/to/NIR.tif \
    --r   path/to/R.tif  --re  path/to/RE.tif
```

Each MS channel is a single-band TIFF; preprocessing (min-max scaling per channel,
224×224 bilinear resize, mean/std normalization stored in the checkpoint) follows
the training pipeline exactly.

## Training

```bash
python train.py
```

`train.py` expects the dataset under `./data-RGBMS/fold5` with the layout produced
by `data_matching.py` (RGB images plus G/NIR/R/RE single-band TIFFs grouped by
phenological stage).

## Dataset

The full dataset (data-MS / data-RGB / data-RGBMS, ~73 GB of UAV imagery) is not
included in this repository due to its size. It is available from the corresponding
author upon reasonable request (smw0908@nuaa.edu.cn).

## Citation

If you use this code or model in your research, please cite the corresponding paper
(entry to be completed after publication).
