# Phase 11B Colab execution apparatus

## Frozen scientific decision

The detector is Ultralytics YOLO11n from `ultralytics==8.3.150`. It is a compact
one-stage model (2,624,080 parameters and 6.6 GFLOPs in the official YOLO11
configuration) and is appropriate for the limited 720-image dataset and the
declared Colab GPU budget. The repository owner authorized licensing the entire
repository as `AGPL-3.0-only`; the root `LICENSE` and package metadata record
that decision. Ultralytics code and pretrained weights remain under their
published AGPL-3.0 terms.

The frozen matrix is **three total class-agnostic localization runs**, using
seeds 17, 29, and 43. The secondary six-class feasibility question does not add
another model family or another three runs. The primary selection metric is
validation `mAP@0.50:0.95`.

## Commit-before-acquisition rule

1. Commit and push this apparatus before opening the Colab dependency cell.
2. In the notebook, replace `REPLACE_WITH_APPARATUS_COMMIT` with that full
   40-character commit and check out exactly that commit.
3. Only then install `requirements-detection-colab.txt`.
4. Install the checked-out repository through its declared `src`-layout package
   configuration, without resolving dependencies a second time:

   ```bash
   python -m pip install --no-deps --editable /content/Bade-defect-recognition
   ```

5. Only after both installations pass may the apparatus checks run and
   `yolo11n.pt` be acquired.
6. `acquire-weight` downloads only the configured official URL and atomically
   writes the Drive weight-acquisition record before training. Training refuses
   missing, altered, or inconsistent weight records.

No package or model weight is vendored in the apparatus commit.

## Colab procedure

Open `notebooks/phase11b_train_validate.ipynb` in Google Colab, select a GPU
runtime with at least 8 GiB VRAM, set the apparatus commit, and run the cells in
order. Place the frozen raw archive at:

`MyDrive/windblade_phase11b/inputs/WT blade defect dataset.zip`

The notebook checks the archive size (78,958,553 bytes) and SHA-256
`466452f2a0cfc9ef6ba63ea2a3bbc7ea4262057dd07e4fc9e00eedf5bba305b4`,
materializes only train/validation data locally, and executes each declared seed
sequentially. Re-running a seed resumes from its Drive `last.pt` only when its
seed, apparatus configuration, initial weight, and materialization identities
match. No automatic batch sizing or optimizer selection is used.

The editable installation uses the repository's `pyproject.toml` and exposes
the `src/windblade` package to `scripts/run_phase11b.py`. `--no-deps` prevents
that step from resolving, upgrading, or replacing the already-pinned Colab GPU
dependencies.

The notebook selects one checkpoint per seed by maximum validation
`mAP@0.50:0.95` (earliest epoch on a tie), pools validation predictions from the
three selected checkpoints, and selects the confidence threshold with maximum
class-agnostic F1 (lower threshold on a tie). NMS remains locked at confidence
floor 0.001, IoU 0.70, at most 300 detections, and class-agnostic NMS.

## Persistent Drive layout

```text
MyDrive/windblade_phase11b/
├── inputs/
│   └── WT blade defect dataset.zip
├── provenance/
│   ├── environment_preflight.json
│   ├── phase11b_weight_acquisition.json
│   └── yolo11n.pt
├── runs/
│   ├── seed_17/
│   ├── seed_29/
│   └── seed_43/
├── selection/
│   ├── validation_checkpoint_candidates.json
│   └── validation_threshold_candidates.json
└── bundles/
    ├── phase11b_training_bundle.zip
    ├── phase11b_training_bundle_manifest.json
    └── phase11b_training_bundle_record.json
```

The local `/content/windblade_phase11b_data` directory is reproducible scratch
space. Checkpoints, logs, provenance, selection artifacts, and bundles persist
in Drive.

## Test firewall and later human action

The train/validation notebook never materializes the held-out images and has no
final-evaluation command. After training, inspect the generated repository file
`provenance/phase11b_selection_receipt.json`, then commit and push it. The
receipt freezes:

- the committed apparatus configuration SHA-256;
- all three selected checkpoint paths, sizes, and SHA-256 values;
- the validation-selected threshold and its candidate-artifact hash;
- the complete NMS configuration and its canonical hash.

Only from a clean checkout containing that committed receipt may a human run:

```bash
python scripts/run_phase11b.py \
  --drive-root /content/drive/MyDrive/windblade_phase11b \
  --data-root /content/windblade_phase11b_data \
  final-test
```

The command checks Git tracking/cleanliness and every frozen artifact hash
before it imports Ultralytics or materializes the held-out split. It refuses a
second evaluation if a final result already exists. Configuration, checkpoint,
threshold, and NMS changes after the first evaluation are prohibited.

All Ultralytics settings for Hub synchronization, W&B, MLflow, ClearML, Comet,
DVC, Neptune, Ray Tune, and TensorBoard integrations are disabled where the
pinned package exposes them. Corresponding environment-level telemetry and
tracker controls are also set before detector import.
