# Reproducing the public release

Tested locally on Linux x86_64 using Python 3.12 and the frozen scientific environment. Other platforms may differ numerically. Commands below run from the repository root.

## 1. Verify saved evidence without dependencies

```bash
python tools/verify_release.py
```

This requires only Python's standard library. It verifies the release manifest, metric arithmetic in published aggregate CSVs, the 16 primary cells, and the primary point estimate. It does not regenerate model predictions or independently establish source-data correctness.

## 2. Regenerate primary figures without ECG data

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-figures.txt
python research/scripts/plot_final_experiment.py
```

Five PNG/SVG pairs are written under `research/outputs/final_experiment/`. The input bootstrap NPZ contains only 2,000 aggregate effect replicates and four per-severity effect values per replicate, not participant-level arrays. The figures are a visualization of saved evidence, not a rerun of the bootstrap.

## 3. Run original scientific checks

Use Python 3.12. Install CPU PyTorch first, then the scientific dependencies:

```bash
python -m pip install -r research/requirements-torch-cpu.txt
python -m pip install -r requirements-science.txt
python -m unittest discover -s research/tests -v
```

The full original environment snapshot is also retained in `research/requirements-core-lock.txt`. The smaller requirements files specify the direct dependencies of the public commands. Local verification reused the frozen installed environment; a clean online dependency installation was not repeated during release preparation. CI's dependency-free evidence check provides a separate fresh-checkout check.

## 4. Recompute fixed-model primary test results

Obtain the original provider datasets as described in [data access](docs/DATA_ACCESS.md). You need the PTB-XL 1.0.3 metadata files, its official SHA256 inventory, the selected `records100/` files, and NSTDB 1.0.0 `em.hea`/`em.dat`. Using the provider's full 100 Hz dataset is the simplest way to ensure the selected records are present. The adapter refuses missing or modified inputs and silently drops no records.

```bash
python tools/reproduce_fixed_test.py \
  --ptbxl-root /path/to/ptb-xl/1.0.3 \
  --ptbxl-checksums /path/to/ptb-xl/1.0.3/SHA256SUMS.txt \
  --nstdb-root /path/to/nstdb/1.0.0 \
  --output reproduced/primary \
  --bootstrap
```

If metadata lives elsewhere, add `--metadata-dir /path/to/metadata`. The output directory must not exist. The command writes 140 matched-coverage rows, 20 fixed-policy rows, and `comparison.json`, comparing results with the original evidence. It exits unsuccessfully on numerical disagreement. The optional `--bootstrap` runs the same 2,000 paired participant resamples and checks the interval. Omit it for point estimates only.

This uses the original fixed CNN, rankers, class threshold, noise assignments and reserved seconds 1200–1780 of the withheld `em` source. It does not retrain, retune or change the study's claims. Identifiers and per-record arrays exist only in memory while replaying; outputs contain aggregates.

The public adapter does not rerun the original cross-role waveform-identity audit because the development arrays are not included. It checks official participant/fold integrity from provider metadata, exact publisher hashes for each evaluated waveform/header and noise file, cohort counts, signal structure and injected SNR. The original cross-role audit is not represented as newly replicated.

## What is outside the turnkey replay?

The entire original training, external-source acquisition, feature-ablation and seed-sensitivity code is available for inspection. Rebuilding those historical stages requires reconstructing their omitted working artifacts and handling historical hash locks deliberately. They are not advertised as verified fresh-checkout entry points. This release's tested data-dependent adapter covers the fixed primary model on clean and four withheld-noise conditions; it does not reproduce the exploratory ablation or all external/seed analyses from raw data.

Keep generated outputs separate from committed evidence. Do not change saved results to make a rerun pass. Inspect discrepancies, environment differences and provider input versions first.
