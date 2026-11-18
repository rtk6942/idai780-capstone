# Sequence-Aware Collision Risk Forecasting: Checkpoint 3

Ryan Kenward, IDAI 780 Capstone, RIT

Predicts the final collision risk of satellite conjunction events from the ESA Collision Avoidance Challenge, using a dual-head GRU, and compares it with three baselines: persistence, recent average and LightGBM. The method, figures and results are in the Checkpoint 3 report.

## Files

- `prepare_data.py`: downloads the ESA data if missing and builds the train/test splits. Split 42 is the Checkpoint 2 split.
- `eval_utils.py`: ESA score, censored loss and the LightGBM baseline.
- `train_gru.py`: trains and tests the GRU and compares it with the baselines.
- `results/`: output of the full experiment (5 splits x 3 training runs).

## Setup

Python 3.13. A GPU is used if available but not required.

```
pip install -r requirements.txt
```

The data (ESA Kelvins `train_data.csv`) downloads automatically and is checked against a SHA-256 hash. To use your own copy, put it in `data/raw/`.

## Run

| What | Command | Time |
| Baselines only, Checkpoint 2 split | `python train_gru.py --baselines-only --splits 42 --out demo` | seconds |
| Quick GRU run | `python train_gru.py --splits 42 --seeds 0 --folds 2 --epochs 10 --no-lgbm --out demo` | about 1-3 minutes |
| Full experiment | `python train_gru.py` | about 20 minutes with nVidia GPU |

Results are saved to `results/cp3_results.csv` - one row per method, split and run 
and `results/cp3_cost_sweep.csv` - alert cutoffs for cost ratios 1 to 100, updated after every split.
