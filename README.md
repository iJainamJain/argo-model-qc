# Model-based quality control of Argo salinity and oxygen

Code, trained models and result tables for the paper *Model-based quality control of Argo salinity and oxygen: detecting persistent sensor errors that pass real-time tests*
(Jain, Jaiswal and Mahajan, submitted).

Deep ensembles predict oxygen and salinity from the physical state, position and season of
each Argo observation. Their standardized residuals are scored per level, per profile and over
each float's life, and an alarm is raised where a score exceeds a threshold calibrated on floats
that are never tested. The repository benchmarks this against a reimplementation of the Argo
real-time quality-control tests on 45 planted error scenarios and on documented real sensor
failures in the Indian Ocean.

## Contents

```
paper_qc/
  lib/            level-table builder and loader, EOS-80 and oxygen solubility, the networks
  argofile.py     fast reads of Argo synthetic-profile files
  greylist.py     the Argo exclusion list against the level table
  adjustments.py  size of delayed-mode adjustments across all profiles
  split.py        float split: training, validation, clean test, known-bad test
  profiles.py     full-resolution and real-time views of every profile
  model.py        the oxygen and salinity ensembles
  score.py        level, profile and float scores; calibration helpers; detector choice
  crossfit.py     cross-fitted (out-of-fold) scores and thresholds
  rtqc.py         the Argo real-time tests (CTD manual v3.9, oxygen manual v2.2)
  inject.py       the planted error scenarios
  bench.py        the benchmark: both methods on every scenario, five repeats
  ci.py           float-bootstrap intervals, detection and distance figures
  realcases.py    documented real failures, scored in the real-time view
  attribution.py  which residual component raises an alarm
  figures_extra.py, make_arch_thumb.py   map, real-failure and gain figures
  make_numbers.py every value and table quoted in the paper, from the result tables
  eos.py          self-test of the equation of state against published check values
  tables/         result tables, including the benchmark records
  figures/        the figures of the paper
runs_qc/          trained ensembles: final models (doxy/, psal/) and the five cross-fitting folds (cv/)
```

`score.py`, `baseline_rtqc.py` and `compare.py` also hold a first single-repeat pass of the
benchmark (`tables/score_benchmark.csv`, `baseline_rtqc.csv`, `comparison.csv`); the paper uses
the five-repeat results of `bench.py` and `ci.py`.

## Data

The Argo data are not redistributed here. To rebuild everything from scratch:

1. Request all floats reporting `DOXY` in 20-120 E, 60 S-30 N, 2006-2026, from the Euro-Argo
   selection service (<https://dataselection.euro-argo.eu>), in *Argo original NetCDF* format,
   and extract them to `data/argo_synthetic/` (`<dac>/<wmo>/profiles/S*.nc`).
2. Download the exclusion list `ar_greylist.txt` from an Argo GDAC
   (<https://data-argo.ifremer.fr/ar_greylist.txt>) to `data/argo_greylist/`.

The trained models, the benchmark records and every result table are included, so the
intervals and the detection and distance figures can be regenerated without the raw data or
retraining: `python -m paper_qc.ci`. `make_numbers.py`, `realcases.py` and `figures_extra.py`
also read the level-wise tables in `data/` (built by the steps below; together about 350 MB,
too large for this repository) and the exclusion list.

## Reproducing, in order (from the repository root)

| step | command | time (CPU) | output |
|---|---|---|---|
| level table | `python -m paper_qc.lib.build --raw-dir data/argo_synthetic --out data/qc_levels.nc` | 20-60 min | `data/qc_levels.nc` |
| exclusion-list overlap | `python paper_qc/greylist.py` | 1 min | `paper_qc/tables/greylist_overlap.csv` |
| adjustment sizes | `python -m paper_qc.adjustments` | 20 min | `paper_qc/tables/adjustments.csv.gz` |
| split | `python -m paper_qc.split` | 1 min | `paper_qc/tables/split.json` |
| full-resolution profiles | `python -m paper_qc.profiles` | 30 min | `data/qc_profiles.parquet` |
| real-time view | `python -m paper_qc.profiles --raw` | 6 min | `data/qc_profiles_raw.parquet` |
| final models | `python -m paper_qc.model --target DOXY --out runs_qc/doxy` (and `PSAL`, `runs_qc/psal`) | 20 min each | `runs_qc/` |
| detector choice | `python -m paper_qc.score --select` | 10 min | `paper_qc/tables/float_method_selection.csv` |
| cross-fitting | `python -m paper_qc.crossfit train`, then `score`, then `calibrate` | 4 h | `paper_qc/tables/score_thresholds_cv.json` |
| benchmark | `python -m paper_qc.bench rtqc --reps 5`, then `bench model --reps 5` | 2.5 h | `paper_qc/tables/records_*.parquet` |
| intervals, figures | `python -m paper_qc.ci` | 3 min | `paper_qc/tables/ci_*.csv`, `paper_qc/figures/` |
| real failures | `python -m paper_qc.realcases --cal paper_qc/tables/score_thresholds_cv.json` | 5 min | `paper_qc/tables/real_*` |
| further figures | `python -m paper_qc.figures_extra`, `python -m paper_qc.make_arch_thumb` | 4 min | `paper_qc/figures/` |
| quoted values | `python -m paper_qc.make_numbers` | 3 min | `paper_qc/manuscript/numbers.tex`, `paper_qc/manuscript/tables/` |

Self-tests: `python -m paper_qc.eos`, `python -m paper_qc.rtqc`, `python -m paper_qc.inject`.
Times are for a laptop CPU; no GPU is needed.

## Design rules the code keeps

* The real-time tests run on full-resolution profiles, never on the thinned training table.
* Thresholds come from out-of-fold scores of training and validation floats; test floats never
  set anything. Scales are robust (median and median absolute deviation), and evidently failed
  floats are excluded from calibration and listed.
* The real-time tests are credited only with flags the planted error caused.

## Requirements

Python 3.10 or later and the packages in `requirements.txt`.

## Argo acknowledgement

These data were collected and made freely available by the International Argo Program and the
national programs that contribute to it (<https://argo.ucsd.edu>, <https://www.ocean-ops.org>).
The Argo Program is part of the Global Ocean Observing System.

## Citation and licence

See `CITATION.cff`. The code is released under the MIT licence (`LICENSE`).
