# STSR Task 2: 3D Dental Registration

[中文说明](./README_zh.md)

This repository contains our current pipeline for STSR Task 2: registering upper- and lower-jaw intraoral scans (STL) to a CBCT volume (`.nii.gz`) and predicting a 4 x 4 transformation matrix for each jaw.

The current model uses two PointNet++ encoders, bidirectional cross-attention, dual-softmax correspondence matching, and a weighted SVD/Procrustes registration head. Upper and lower jaws are trained independently with the same training script. The pipeline also supports optional contrastive pretraining, reusable point-cloud caches, internal validation, test-time augmentation (TTA), and automatic submission packaging.

## Repository structure

```text
.
├── configs/
│   ├── pretrain_config.yaml     # Optional contrastive pretraining
│   ├── train_config.yaml        # Supervised registration training
│   └── inference_config.yaml    # Batch inference and submission
├── data/                        # Dataset, preprocessing, and augmentation
├── losses/                      # Contrastive and registration losses
├── models/                      # PointNet++, cross-attention, and SVD heads
├── utils/                       # Transform and affine-domain utilities
├── main_pretrain.py             # Optional encoder pretraining
├── main_train.py                # Upper/lower jaw training
├── main_inference.py            # Inference and ZIP packaging
├── eval_local.py                # Local translation/rotation evaluation
└── requirement.txt
```

## Installation

Python 3.9 and a CUDA-capable GPU are recommended. Install PyTorch for your CUDA version first, then install the remaining dependencies:

```bash
conda create -n stsr-task2 python=3.9 -y
conda activate stsr-task2

# Install a CUDA-compatible PyTorch build first if it is not already installed:
# https://pytorch.org/get-started/locally/
pip install -r requirement.txt
```

`torch_geometric` relies on compiled operators such as `torch-cluster`. If your environment cannot resolve a compatible build automatically, install the matching PyG wheels by following the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Dataset layout

Labeled training data must contain `Images` and `Labels` (lowercase names are also accepted):

```text
Train-Labeled/
├── Images/
│   └── 001/
│       ├── CBCT.nii.gz
│       ├── upper.stl
│       └── lower.stl
└── Labels/
    └── 001/
        ├── upper_gt.npy
        └── lower_gt.npy
```

Each label is a `(4, 4)` NumPy transformation matrix. Unlabeled pretraining data uses the same `Images/<case_id>/...` layout without `Labels`. For inference, `inference_data_root` points directly to the directory containing case folders:

```text
Validation/Images/
└── 001/
    ├── CBCT.nii.gz
    ├── upper.stl
    └── lower.stl
```

## Running the pipeline

### 1. Configure paths

Update the dataset and output paths in:

- `configs/pretrain_config.yaml` for optional self-supervised pretraining;
- `configs/train_config.yaml` for supervised training;
- `configs/inference_config.yaml` for checkpoint loading, prediction output, and the submission ZIP.

Keep architecture and preprocessing options in the training and inference configurations consistent, especially point counts, `feature_dim`, `target_dense_points`, `registration_head`, CBCT threshold/ROI settings, and translation-residual settings.

### 2. Optional contrastive pretraining

Set `jaw_type` and `experiment_name` in `configs/pretrain_config.yaml`, then pretrain each jaw encoder separately:

```bash
python main_pretrain.py
```

The best encoder checkpoint is saved to:

```text
<output_dir>/<experiment_name>/checkpoints/best_model.pth
```

To use it, set `load_pretrained: true` and update `pretrained_checkpoints` in `configs/train_config.yaml`. Pretraining is optional; set `load_pretrained: false` to train from scratch.

### 3. Train upper and lower jaws

The command-line options override `jaw_type` and the experiment name in the YAML file:

```bash
python main_train.py --config configs/train_config.yaml --jaw upper
python main_train.py --config configs/train_config.yaml --jaw lower
```

By default, the corresponding names under `experiment_names.upper` and `experiment_names.lower` are used. A custom name can be supplied with `--experiment-name`. Training outputs are written to:

```text
<output_dir>/<experiment_name>/
├── checkpoints/
│   ├── best_model.pth
│   └── latest_model.pth
├── logs/
└── log.txt
```

When the configured validation set has no labels, the current configuration creates a deterministic, chirality-stratified validation split from the labeled training set. With point caching enabled, the first run builds reusable STL/CBCT archives under `point_cache_dir`; later runs reuse them.

Monitor both experiments with TensorBoard:

```bash
tensorboard --logdir ./experiments22
```

### 4. Run inference and create a submission

In `configs/inference_config.yaml`, set `inference_data_root`, `output_dir`, the upper/lower experiment names (or explicit checkpoint paths), `prediction_root`, and `submission_zip_path`. Then run:

```bash
python main_inference.py --config configs/inference_config.yaml
```

The script loads both jaw models, extracts thresholded CBCT surface points, applies the same normalization and jaw-aware ROI settings used during training, aggregates TTA predictions, and writes both a prediction folder and a submission ZIP:

```text
prediction_root/
└── 001/
    ├── upper_gt.npy
    └── lower_gt.npy
```

Optional ICP refinement, metric-specific checkpoints, and affine-domain corrections are available in the inference configuration but are disabled in the default competition setup.

### 5. Local evaluation

For a labeled dataset, evaluate mean translation error (mm) and rotation error (degrees) with:

```bash
python eval_local.py \
  --prediction_root ./prediction_output \
  --label_root /path/to/Validation/Labels \
  --jaw both
```

Use `--jaw upper` or `--jaw lower` to evaluate one jaw only.

## Notes

- Run all commands from the repository root so that relative configuration paths resolve correctly.
- Upper and lower checkpoints are independent even though they share the same architecture and training code.
- `main_inference.py` recreates `prediction_root` at startup. Do not point it to a directory containing files that must be preserved.
- `best_model.pth` is the inference checkpoint, while `latest_model.pth` stores the latest training state.
