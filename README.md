# HGA

Core code for **A globally assembled harmonic forward model for EEG source
imaging and motor decoding**.

Harmonic global assembly (HGA) represents layered
head boundaries with real spherical harmonics and jointly solves potential
continuity and normal-current constraints. A shared forward operator supports
EEG source imaging and source-derived features for motor decoding.

This repository contains numerical kernels, compact experiment adapters and
the final method configurations.

## Contents

| Location | Purpose |
| --- | --- |
| `hga/harmonics.py` | Real SH convention, quadrature and log-radial geometry |
| `hga/operator.py` | Coupled global interface system and its stable solve |
| `hga/source.py`, `hga/coordinates.py`, `hga/covariant_operator.py` | Exact source injection and final frame-covariant quadrature/fallback |
| `hga/forward.py`, `hga/inverse.py`, `hga/clinical.py` | Public forward API, inverse interfaces and spatial endpoints |
| `hga/decoding/` | ROI/sensor F1 power features, train-only PCA/LR, calibration and fusion |
| `hga/models/` | ATCNet, required TCFormer modules and MIRepNet interfaces |
| `hga/data/` | BCIC GDF and HGD EDF readers |
| `configs/` | Final forward, epilepsy, BCIC, MIRepNet and HGD settings |
| `experiments/` | Small data preparation, forward, clinical and decoding entry points |

The `hga` package provides harmonic boundary representations, global
interface assembly, source projection and inverse estimation. Experiment
adapters connect these kernels to clinical localization and motor decoding.

## Installation

Python 3.10 or newer is required. SciPy must provide `sph_harm_y` (1.15+).

```bash
python -m pip install -e .
# Add only the optional components needed for the experiment:
python -m pip install -e ".[clinical,deep]"
```

Deep training uses a CUDA-enabled PyTorch installation, float32, and the
registered fixed final epoch. It does not silently switch to CPU or change
batch size. Core numerical tests run on CPU without recordings or training.

```bash
python -m unittest discover -s tests -v
```

## Forward model and source features

Place anatomical geometry and recordings in `data/`. All forward
coordinates must be in **metres in a common frame**. Clinical spatial distances use native scanner-RAS millimetres.
Registration transforms must be established externally; the code does not
infer a registration from clinical outcomes.

```python
from hga import build_leadfield, minimum_norm_kernel

# coefficients: inner_skull, outer_skull and outer_skin SH coefficient arrays
gain, diagnostics = build_leadfield(
    coefficients, center_m, radius_scale_m, sources_m, electrodes_m,
    geometry_order=12, field_order=12,
)
kernel = minimum_norm_kernel(gain, lambda2=1/9)
```

The gain and input EEG both use common-average reference. The decoding gain
has 22 rows and 6,156 vertex-major XYZ columns over 2,052 template vertices.
Do not average signed XYZ components before computing power. Final source
features use eight anatomically defined bilateral parcels, four fixed EEG
power bands and three windows. They are 272-dimensional before training-only
PCA48; the sensor block is 264-dimensional. A source-enhanced LR combines
sensor48 with source48, compared with an equal-size sensor-extra control.

`experiments.build_forward` accepts a head-assets NPZ with `inner_skull`,
`outer_skull`, `outer_skin`, `center_m`, `radius_scale_m`, `sources_m` and
`electrodes_m`. Optional `scanner_to_surface` carries the coordinate transform.
For decoding, add eight integer vertex-index arrays `roi_0` through `roi_7`
ordered as four left parcels followed by their right homologues, as listed in
the configurations. Geometry and ROI assets derive from the selected
anatomy and atlas.

```bash
python -m experiments.build_forward --assets data/head_assets.npz --output outputs/source_assets.npz
```

## Experiment interfaces

**Epilepsy.** `configs/epilepsy.json` records matched BEM/Sphere/HGA inputs,
frozen imaging time, four classical inverses, and Hit@0/10/20 definitions.
Construct anatomy and covariance without postoperative outcome information;
freeze source maps before applying independent postoperative masks. The
clinical adapter accepts an existing matching MNE forward container and
covariance. Use `replace_mne_leadfield` to insert an HGA matrix into a matching
Cartesian MNE forward, then apply `hga.inverse.clinical_inverse`.

```bash
python -m experiments.epilepsy --evoked data/epilepsy/ied-ave.fif --forward data/epilepsy/hga-fwd.fif --covariance data/epilepsy/noise-cov.fif --output outputs/epilepsy/dspm
```

**BCIC IV-2a.** Training uses AxxT, six native runs, all 288 trials and seeds
11/22/33. AxxE labels are used only after prediction locking. ATCNet and
TCFormer use 22-channel CAR volts and a 1,125-sample input; MIRepNet uses a
separate acquisition-reference, filtered, 1,001-sample input, official batch
EA and 22-to-45-channel interpolation, with initialization from its official
pretrained weights. The three absolute accuracies are not a controlled
architecture ranking because their preprocessing/training differ.

**HGD.** Preserve official EDF train/test files for H01–H14. This experiment
uses actual movement execution/rest, not four-class motor imagery. The reader
selects the fixed 22 channels, resamples continuous 500-Hz signals to 250 Hz,
and creates cue-relative [−0.5,4) s epochs. Native run boundaries use cue gaps
greater than 10 s. Only ATCNet is configured. The frozen source representation
is reused; classifier heads and training transforms are refitted on HGD.
This is not zero-shot transfer of BCIC network weights.

```bash
python -m experiments.prepare_data --dataset hgd --input data/hgd --subject H01 --split train --output outputs/H01_train.npz
python -m experiments.prepare_data --dataset hgd --input data/hgd --subject H01 --split test --output outputs/H01_test_blind.npz

# These are substantial training jobs; they are not run during installation.
python -m experiments.motor train --dataset hgd --model ATCNet --input outputs/H01_train.npz --assets outputs/source_assets.npz --output outputs/H01_ATCNet
python -m experiments.motor predict --input outputs/H01_test_blind.npz --assets outputs/source_assets.npz --output outputs/H01_ATCNet
python -m experiments.motor score --labels data/H01_test_labels.npy --output outputs/H01_ATCNet
```

For BCIC, use `--dataset bcic` and the corresponding prepared GDF inputs.
MIRepNet additionally needs `--model MIRepNet --pretrained data/weights/MIRepNet.pth`
at training time. External evaluation labels must be zero-based integers in
the exact trial order; HGD class order is left hand/right hand/feet/rest and
BCIC ends with tongue. The `predict` stage rejects NPZ files containing labels,
and `score` records the sole access attempt before opening a label file.

The compact runner operates on one participant/output directory. For a cohort,
complete and lock every participant's predictions before performing cohort
scoring. Numerical reproducibility depends on library, hardware and
data-reader versions.

Average matched seeds within participant before paired statistics; do not
treat trials or seeds as independent participants. BCIC and HGD retain their
own sample sizes and correction families. The study protocol documents
development-stage exposure to evaluation data.

## Data and upstream software

- [BCI Competition IV, dataset 2a](https://www.bbci.de/competition/iv/)
- [High Gamma Dataset](https://github.com/robintibor/high-gamma-dataset)
- Anatomy/templates and EEG must be obtained from their original providers.
- See [third-party notices](THIRD_PARTY_NOTICES.md) and the preserved model
  licenses for classifier attribution.

## 中文说明

本仓库提供 HGA 的核心计算代码及癫痫、BCIC、HGD 的方法配置。
HGA 通过球谐边界表示与全局矩阵组装，支持 EEG 源成像和源特征辅助分类。
HGD 仅包含 ATCNet 验证接口，任务为实际动作与静息。三个分类骨干属于分类模型，
源重建使用独立的前向/逆解模块。
