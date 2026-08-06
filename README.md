# Federated multimodal AI identifies high-risk liver cancer populations from electronic health records across diverse healthcare networks

EquityFedHCC combines site-heterogeneous clinical records, abdominal CT, pathology, molecular profiles, and population biomarkers through modality-masked late fusion. The training objective joins binary risk prediction, stratum-conditional cross-modal calibration, and demographic-parity regularization. Server updates use outer q-FFL site weighting and inner target-population prevalence weighting. Evaluation reports discrimination, calibration, subgroup equity, and leave-one-site-out concordance.

## Environment

The reported environment uses Python 3.10, PyTorch 2.2.0, Flower 1.6.0, and CUDA 12.4. Training used one NVIDIA A100 80 GB GPU and 1 TB CPU RAM. A five-seed primary grid consumed approximately 240 A100-hours.

Install with pip:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Install with conda:

```bash
conda env create -f environment.yml
conda activate equityfedhcc
pip install -e .
```

Build the container:

```bash
docker build -t equityfedhcc:1.0 .
```

## Data access

All dataset URLs are also collected in `dataset_links.txt`.

- eICU-CRD v2.0: credentialed access under the PhysioNet Credentialed Health Data License 1.5.0. Complete the required human-subjects training and data use agreement at https://physionet.org/content/eicu-crd/2.0/.
- TCGA-LIHC: clinical and molecular data from GDC harmonized release 36 or later under TCGA data-use terms at https://portal.gdc.cancer.gov/projects/TCGA-LIHC. Imaging is TCIA collection version 5 under the TCIA Data Usage Policy at https://www.cancerimagingarchive.net/collection/tcga-lihc/.
- NHANES: cycles 2013–2014, 2015–2016, 2017–March 2020, and 2021–2023 are United States Government public-domain releases at https://www.cdc.gov/nchs/nhanes/.
- Medical Segmentation Decathlon Task03 Liver: 201 3D portal-venous CT volumes under CC BY-SA 4.0 at https://medicaldecathlon.com/.

The code expects de-identified study identifiers. Do not place names, contact details, medical record numbers, full dates, or raw credentials in manifests or run directories.

Prepare an eICU cirrhosis table:

```bash
python -m equityfedhcc.commands.prepare --cohort eicu --input data/eicu/diagnosis.csv --output data/derived/eicu_cirrhosis.parquet
```

The cohort rules are ICD-10 K70–K77 for eICU, pathology-confirmed primary HCC for TCGA-LIHC, complete liver-biomarker panels with temporal cycle separation for NHANES, and primary-HCC lesion filtering for LiTS/MSD. Patient identifiers must not overlap among training, validation, and evaluation partitions.

## Configuration

`configs/main.yaml` contains the primary setting: batch size 32 per virtual site, 5 local epochs per round, 19 rounds, AdamW learning rate 1e-4, weight decay 1e-2, q-FFL exponent 2, calibration weight 0.5, equity weight 1.0, and five seeds 1234, 2345, 3456, 4567, and 5678. All four virtual sites participate in every round. Early stopping uses patience 3.

`configs/sites.yaml` records modality availability and partition keys. `configs/ablations.yaml` records the atomic removals and sensitivity grids.

## Model components

The package is under `code/equityfedhcc`.

- `data` defines de-identified records, cohort selection, leakage-safe splits, normalization, volume resampling, and the temporal clinical feature registry.
- `models` defines structured-EHR, volume, pathology-pooling, omics, biomarker, and modality-masked cross-attention modules.
- `objectives.py` defines binary task loss, stratum-conditional Brier-difference calibration, and smoothed demographic-parity loss.
- `federation` defines client update contracts, target-prevalence ratios, q-FFL site weights, dual-axis aggregation, and optional Gaussian update perturbation.
- `training` defines local optimization, server coordination, validation, early stopping, and atomic state persistence.
- `metrics` defines AUROC, AUPRC, Brier score, ECE, calibration slope and intercept, DPD, EOD, subgroup AUROC, bootstrap intervals, and multiplicity correction.
- `audit` defines leave-one-virtual-site-out per-stratum concordance analysis.

## Evaluation

Prediction tables require `label`, `score`, and `stratum` columns.

```bash
python -m equityfedhcc.commands.evaluate --predictions artifacts/predictions.csv --output artifacts/metrics.json --resamples 2000 --seed 1234
```

Primary evaluation uses AUROC with 2,000 stratified bootstrap resamples. Secondary outcomes are AUPRC, Brier score, 10-bin ECE, DPD, EOD, and maximum cross-site AUROC gap. Cross-site gap intervals use 1,000 resamples. Cells with fewer than 30 records are marked exploratory. Twelve primary comparisons use Holm–Bonferroni correction at alpha 0.05; exploratory comparisons use Benjamini–Hochberg correction at q 0.10.

Expected primary AUROC values are 0.842 for eICU, 0.881 for TCGA-LIHC, 0.838 for NHANES, and 0.879 for LiTS/MSD. The corresponding FedAvg values are 0.794, 0.823, 0.795, and 0.847. These values are acceptance targets for complete five-seed runs on the specified cohort construction, not guarantees for altered preprocessing or partitions.

## Privacy and scope

The federation layer is a single-node computational simulation. It does not establish network transport, authentication, clinical deployment, or a claim that data moved among institutions. Raw cohort records remain outside source control. Only aggregate parameters and de-identified evaluation outputs belong in run artifacts.

