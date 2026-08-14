"""Feature-importance stability analysis.

Problem: a single feature-importance ranking from ~83 failures
is noisy, so reporting one ranking and calling those "the important sensors" 
overstates what the data supports.

Solution approach: resample the training data many times (bootstrap), refit the model
on each resample, and record feature importances every time. Then report, per
feature, how often it lands in the top-k and how much its importance varies.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone


def _importances_from_fitted(pipeline, feature_names):
    # pull feature importances out of a fitted preprocessor+classifier pipeline
    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]

    # names of the columns that survived preprocessing, in classifier order
    out_names = pre.get_feature_names_out()
    importances = clf.feature_importances_

    return pd.Series(importances, index=out_names)


def bootstrap_importances(pipeline, X, y, n_resamples=200, top_k=15,
                          random_state=42):
    """resample the data n ways, refit each time, and count how often each sensor stays important

    returns a DataFrame indexed by feature, with:
      - mean_importance : average importance across resamples (0 when the feature
                          was dropped by preprocessing in a given resample)
      - std_importance  : spread of importance across resamples
      - selection_freq  : fraction of resamples where the feature landed in top_k
    """
    rng = np.random.default_rng(random_state)
    n = len(X)

    # reset index so positional bootstrap indices line up cleanly
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    all_importances = []   # list of pd.Series, one per resample
    topk_hits = {}         # feature -> count of resamples where it was top_k

    for _ in range(n_resamples):
        # bootstrap: sample n rows with replacement
        idx = rng.integers(0, n, size=n)
        X_bs = X.iloc[idx]
        y_bs = y.iloc[idx]

        # a bootstrap sample can (rarely) contain only one class; skip if so
        if y_bs.nunique() < 2:
            continue

        model = clone(pipeline)
        model.fit(X_bs, y_bs)

        imp = _importances_from_fitted(model, X.columns)
        all_importances.append(imp)

        # which features were in the top_k this resample
        top_features = imp.sort_values(ascending=False).head(top_k).index
        for f in top_features:
            topk_hits[f] = topk_hits.get(f, 0) + 1

    n_valid = len(all_importances)

    # align all importance Series into one frame (missing = feature dropped that
    # resample -> treat as 0 importance)
    imp_frame = pd.DataFrame(all_importances).fillna(0.0)

    summary = pd.DataFrame({
        "mean_importance": imp_frame.mean(axis=0),
        "std_importance": imp_frame.std(axis=0),
    })
    summary["selection_freq"] = (
        pd.Series(topk_hits).reindex(summary.index).fillna(0) / n_valid
    )

    summary = summary.sort_values("selection_freq", ascending=False)
    summary.attrs["n_valid_resamples"] = n_valid
    return summary
