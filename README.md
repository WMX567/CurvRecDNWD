# Curvature Diversity-Driven Deformation and Domain Alignment for Point Cloud

Official implementation for the paper "Curvature Diversity-Driven Deformation and Domain Alignment for Point Cloud" published in *Transactions on Machine Learning Research* (2025).

## Environment

The recorded environment uses Python 3.8.18, PyTorch 2.0.1 with CUDA 11.8, NumPy 1.24.4, and Open3D 0.18.0. Training also requires SciPy, scikit-learn, pandas, and h5py; the download scripts use gdown.

`requirements.txt` contains the full Linux environment export, including Conda packages and packages installed through pip. Use it as a version reference when setting up your environment; it is not a standard pip requirements file. Use a CUDA-enabled environment for training.

## Data preparation

### Download the datasets

Run each download script from its own directory so the data lands in the expected location:

```bash
(cd PointDA/data && python download.py)
(cd PointSegDA/data && python download.py)
```

The PointDA script downloads and extracts `PointDA_data.zip`. The PointSegDA script downloads `PointSegDAdataset.rar`; extract it manually:

```bash
cd PointSegDA/data
unrar x PointSegDAdataset.rar
cd ../..
```

The training scripts expect these directories:

```text
PointDA/data/PointDA_data/
├── modelnet/
├── shapenet/
└── scannet/

PointSegDA/data/PointSegDAdataset/
├── adobe/
├── faust/
├── mit/
└── scape/
```

ModelNet and ShapeNet store `.npy` files under `<domain>/<class>/<split>/`. ScanNet uses HDF5 files. PointSegDA stores `.npy` files under `<domain>/<split>/`, with `train`, `val`, and `test` splits.

### Compute curvature

Training requires a curvature value for each point, stored in the last column of the point array. `compute_norm_curv.py` estimates curvature from the covariance eigenvalues of a 32-point neighborhood and normalizes the values to `[-1, 1]` across the processed files.

The script currently reads PointSegDA `.npy` files with columns `[x, y, z, label]` and writes `[x, y, z, label, curvature]`. Set `root` and `save_root` in its main block to the input and output split directories, then run:

```bash
python compute_norm_curv.py
```

Repeat for the domains and splits you need. If you save to a separate directory, place the processed files in the dataset layout above before training. Setting `save_root` to `root` overwrites the input files.

For PointDA, adapt the script's input/output loop to the classification formats: ModelNet and ShapeNet use directory-based class labels, while ScanNet stores point arrays and labels in HDF5 files. Keep the XYZ coordinates in the first three columns and append curvature as the last column of each point array. The current script is not a direct preprocessing command for these formats.

## Training

Run the following commands from the repository root after preparing the data.

### PointDA classification

Example: ModelNet → ShapeNet, using the configuration from the original running instructions:

```bash
python trainer_ours.py \
  --dataroot PointDA/data \
  --src_dataset modelnet \
  --trgt_dataset shapenet \
  --seed 1 \
  --largest False \
  --nwd False \
  --lr 1e-3 \
  --model_type pointda
```

Available domains are `modelnet`, `shapenet`, and `scannet`. The default backbone is DGCNN; `--model pointnet` selects PointNet. This example disables NWD alignment with `--nwd False`; use `--nwd True` to enable it.

### PointSegDA segmentation

Example: FAUST → MIT:

```bash
python trainer_ours_seg.py \
  --dataroot PointSegDA/data/PointSegDAdataset \
  --src_dataset faust \
  --trgt_dataset mit \
  --seed 2 \
  --model_type ver11 \
  --DefRec_weight 0.05 \
  --target_weight 0.2 \
  --nwd_weight 1.0
```

Available domains are `adobe`, `faust`, `mit`, and `scape`. The segmentation model uses DGCNN, and NWD alignment is enabled by default.

Both scripts write logs and model checkpoints to `./experiments` by default. Use `--out_path` to change the output directory. Always provide `--model_type`: it is used as a suffix in log and checkpoint filenames. Training includes validation and target-domain test evaluation.

For all command-line options:

```bash
python trainer_ours.py --help
python trainer_ours_seg.py --help
```

## Repository structure

```text
CurvDefRec/             Curvature-based deformation and reconstruction
PointDA/               Classification models and data loaders
PointSegDA/            Segmentation models and data loaders
utils/                 Domain alignment, point cloud utilities, and logging
compute_norm_curv.py   Curvature preprocessing
trainer_ours.py        Classification training
trainer_ours_seg.py    Segmentation training
requirements.txt       Recorded environment
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{
wu2025curvature,
title={Curvature Diversity-Driven Deformation and Domain Alignment for Point Cloud},
author={Mengxi Wu and Hao Huang and Yi Fang and Mohammad Rostami},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=ePXWnH7rGk},
note={}
}
```
