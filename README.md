# Predictive Quality Control

A predictive quality control pipeline for semiconductor manufacturing, built on the
UCI SECOM dataset. The goal is to predict which units will fail end-of-line testing
from in-process sensor readings, and to identify which sensors are most associated
with failures.

The dataset is small and hard to predict from, so much of the work here is about what
the data can and can't support rather than the final score.

## The dataset

SECOM is 1567 production units, each with 590 sensor/process measurements, a
pass/fail label, and a timestamp. It spans about three months (July to October 2008).

Main difficulty of this dataset is the class imbalance and messy data.
- Only 6.6% of units fail (104 of 1567), which leaves about 83 failures in the training fold to learn from.
- 4.54% of values are missing, 28 columns are more than half empty, and 116 columns (about 20%) are constant and carry no information.

The raw data is not committed. Download it from
[UCI](https://archive.ics.uci.edu/dataset/179/secom) and place the two files in
`data/raw/` (see `data/README.md`).

## Structure

```
src/            reusable logic (loading, preprocessing, metrics, stability)
notebooks/      the analysis, one notebook per stage
data/           dataset lives here (gitignored)
```

- `01_eda.ipynb` — class imbalance, missingness, dead columns, and failure rate over time
- `02_preprocessing.ipynb` — the leakage-safe preprocessing pipeline, step by step
- `03_modeling.ipynb` — baselines, model comparison, and threshold analysis
- `04_stability.ipynb` — which sensors are *robustly* important, not just important once

## Key decisions

**PR-AUC as the main metric:** Accuracy is useless for this dataset because a model that predicts "pass" for every unit scores 93.4% accuracy while catching zero defects. The primary
metric is PR-AUC (average precision), which focuses on the rare failure class and
summarises performance across all decision thresholds. Recall, precision, and MCC are
reported alongside it.

**Stratified split instead of time-based:** The data is chronological and the failure rate
drifts over the three months (higher early, lower later). A time-based split would
train and test on different base rates and leave the ~100 failures badly balanced, so
a stratified split is used to keep the 6.6% ratio in both folds. The cost is that this
doesn't test how the model handles the drift over time, which is an accepted trade-off.

**Dropping high-missing columns, checked first:** Before dropping the 28 mostly-empty
columns, I checked whether their missingness related to the label. The largest gap in
failure rate between "missing" and "present" was 0.012 against a 6.6% base rate, so the
missingness carries no signal, and dropping is safe rather than imputing values that are
mostly absent anyway.

**Using StandardScaler after RobustScaler broke:** RobustScaler was the first choice,
since it ignores outliers when measuring spread. It failed because several sensors are
almost constant with rare excursions, so their IQR is near zero, and dividing by it
produced scaled values up to ~155,000. That stopped logistic regression from converging.
StandardScaler divides by the standard deviation instead, which outliers inflate rather
than shrink, so nothing blows up.

## Results

Cross-validated on the training fold:

| model               | PR-AUC | ROC-AUC |
|---------------------|--------|---------|
| all-pass baseline   | 0.066  | 0.500   |
| logistic regression | 0.119  | 0.613   |
| random forest       | 0.168  | 0.709   |
| hist gradient boost | 0.157  | 0.713   |

The random forest is the best model, and gradient boosting gets almost the same PR-AUC.
Class weighting barely changed PR-AUC either. Two different model families landing in
the same place suggests the limit is the data itself, not the model. Base rate for PR-AUC is 0.066, so 
all the models land above that but not by a lot.

**Decision threshold:** At the default 0.5 cutoff the
random forest catches zero defects, because it rarely pushes a probability past 0.5.
Its PR-AUC is still the best, meaning that it ranks failures well but the cutoff just sits in
the wrong place. Sweeping the threshold, MCC peaks at 0.25:

- At threshold 0.25, the model flags 137 of 1253 units (about 11%) and catches 30 of
  the 83 failures (about 36%, 3.3× improvement over inspecting a random 11%)

The right threshold depends on the real cost of a missed defect versus a false alarm,
which the dataset doesn't include, so 0.25 is the MCC-optimal point for now.

**On the held-out test set (untouched until the end):** The random forest gets PR-AUC 0.252 and MCC 0.347, a little above the cross-validated estimate. At that threshold it catches 10 of 21 failures while flagging 30 of 314 units, so inspecting about 10% of production catches roughly half the defects.
The test set only has 21 failures, so there's real uncertainty around the numbers but
the model clearly held up on the test data.

**Sensors that are associated with failures:** A single feature-importance
ranking from 83 failures would be noisy (refitting on different resamples would reshuffles the ranking). So importance is measured across 200 bootstrap resamples, recording how often each sensor is in the
top 15. Only three sensors (103, 59, 33) show up in the top 15 most of the time, in
87–92% of resamples, and everything else drops off sharply below 50%. So the same few
sensors matter across resamples, rather than the fifteen a single ranking would list.

However, the data is observational and the sensors are
anonymised, so there's no basis to claim these sensors cause defects.


## Running it

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
