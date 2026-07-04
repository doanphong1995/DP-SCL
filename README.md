# DP-SCL

DP-SCL is a model for predicting student dropout in MOOC courses using
Supervised Contrastive Learning. Each learner-course record is represented as a
temporal activity sequence; the model learns a behavior representation and then
classifies the dropout probability.

## Model

DP-SCL has three main components:

- **Temporal encoder**: reshapes the activity tensor
  `(week_count, days_per_week, activity_num)` into a temporal sequence, then
  encodes it with LSTM and Multi-Head Attention.
- **Projection head**: creates embeddings for Supervised Contrastive Loss.
  During training, each sample is augmented into two views using time masking,
  feature masking, and noise.
- **Classifier**: receives the encoder representation and predicts dropout with
  `BCEWithLogitsLoss`.

Training loss:

```text
loss = BCEWithLogitsLoss(logits, label)
     + lambda_con * SupConLoss([z1, z2], label)
```

The repository supports three datasets: `xuetangx`, `oulad`, and `snap`.

## Reference Sources

This implementation was written for the DP-SCL project and uses the following
public references for architecture compatibility and loss implementation:

| Component | Reference |
| --- | --- |
| CA-TFHN baseline/code structure reference | https://github.com/codeds27/CA-TFHN |
| CA-TFHN XuetangX data download script | https://github.com/codeds27/CA-TFHN/blob/main/dump_data.sh |
| CA-TFHN feature extraction script | https://github.com/codeds27/CA-TFHN/blob/main/feat_extract.sh |
| CA-TFHN data preprocessing package | https://github.com/codeds27/CA-TFHN/tree/main/src/dataprocess |
| CA-TFHN `MySelfAttention` reference | https://github.com/codeds27/CA-TFHN/blob/main/src/models.py |
| SupContrast PyTorch implementation | https://github.com/HobbitLong/SupContrast |
| Supervised Contrastive Learning paper | https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html |

## Main Structure

```text
.
|-- train.py                   # CLI entry point
|-- src/
|   `-- dpscl/                 # DP-SCL package
|       |-- cli.py             # argument parsing
|       |-- experiment.py      # multi-seed orchestration and output layout
|       |-- trainer.py         # single-seed train/eval loop
|       |-- data.py            # NPZ loading, seeding, DataLoader creation
|       |-- splits.py          # stratified split helpers
|       |-- metrics.py         # threshold selection, metrics, summary stats
|       |-- reporting.py       # CSV and text report writers
|       |-- datasets.py        # dataset metadata
|       |-- modes.py           # mode constants
|       |-- config.py          # shared constants and model params
|       `-- model/             # DP-SCL model, layers, losses
|-- tests/                     # smoke tests
|-- datastore/                 # .npz data files
|-- deleted/                   # parked files not used by the clean runtime
|-- requirements.txt
`-- README.md
```

The training flow is intentionally linear:

```text
train.py -> src/dpscl/cli.py
         -> src/dpscl/experiment.py
         -> src/dpscl/trainer.py
         -> src/dpscl/model/
```

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

For GPU training, install a PyTorch build that matches the CUDA version on your
machine before running experiments.

## Data

Place the `.npz` files in the `datastore/` directory:

| Dataset | File |
| --- | --- |
| `xuetangx` | `datastore/all_data_std.npz` |
| `oulad` | `datastore/oulad_data_std.npz` |
| `snap` | `datastore/snap_data_std.npz` |

Each file must contain:

```text
t_data, t_label
v_data, v_label
```

The training script combines the original train/test arrays, then creates
stratified splits for each seed using the default ratio
`60% train / 10% validation / 30% test`.

## Usage

Quick run on XuetangX:

```bash
python train.py -indir . -outdir . --dataset xuetangx --max-epochs 200
```

By default, this runs the five reported seeds:

```text
1 11 111 1111 11111
```

You can also pass them explicitly:

```bash
python train.py -indir . -outdir . --dataset xuetangx --seeds 1 11 111 1111 11111 --max-epochs 200
```

Run on another dataset:

```bash
python train.py -indir . -outdir . --dataset oulad --max-epochs 200
python train.py -indir . -outdir . --dataset snap --max-epochs 200
```

Common arguments:

```text
--seeds           random seeds, default 1 11 111 1111 11111
--batch-size      batch size, default 256
--lr              learning rate, default 1e-4
--hidden-size     encoder hidden size, default 128
--lambda-con      SupCon loss weight, default 0.1
--temperature     SupCon temperature, default 0.07
--max-epochs      maximum number of epochs
--patience        early stopping patience
```

## Outputs

Each run creates a directory:

```text
results/dp_scl_<timestamp>/
```

Main files:

```text
config.json
checkpoints/
splits/
per_seed_results.csv
epoch_history.csv
summary_results.csv
report.txt
```

Main metrics include AUC, accuracy, precision, recall, and F1. The classification
threshold is selected on the validation split by maximizing F1, then applied to
the test split.

## Quick Test

```bash
python -m pytest tests/test_dp_scl.py
```

## Developer Notes

When changing the training pipeline, prefer editing the focused modules under
`src/dpscl/`:

| Task | File |
| --- | --- |
| Add or change CLI flags | `src/dpscl/cli.py` |
| Change run directory layout or multi-seed flow | `src/dpscl/experiment.py` |
| Change training objective or early stopping | `src/dpscl/trainer.py` |
| Change data loading or tensor conversion | `src/dpscl/data.py` |
| Change train/val/test split logic | `src/dpscl/splits.py` |
| Change metrics or threshold selection | `src/dpscl/metrics.py` |
| Change result files or report text | `src/dpscl/reporting.py` |
| Change neural network architecture | `src/dpscl/model/` |

Files that are not part of the clean training runtime were moved to
`deleted/` for review before permanent deletion. This includes old launchers,
plot outputs, reports, exploratory scripts, local docs, and preprocessing
helpers.
