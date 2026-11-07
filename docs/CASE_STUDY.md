# Technical case study

The central engineering problem was not fitting a network; it was making evaluation resemble deployment. A first diagnostic run selected models at the last cycle of run-to-failure validation engines. Every target was zero, so a boosted tree appeared exceptional on validation and failed on truncated test engines. The pipeline now fixes evaluation horizons before training and scores 11 causal cutoffs per held-out engine. It also removed `cycle / eventual_lifetime`, a subtle future-data leak. Under the corrected protocol the LSTM won validation and generalized to 16.02 uncapped official-test RMSE.

The platform treats RUL as one decision signal. It adds calibrated 15/30/60-cycle risks, healthy-state anomaly scores, conformal intervals with observed coverage, horizon error tables, and held-out permutation evidence. Promotion records explicitly assert that the official test was not used for selection.

Operationally, artifacts are immutable evidence bundles, MLflow uses a local SQLite backend by default, and a promotion manifest points the API to a champion. FastAPI provides versioned typed contracts and dry-run-only retraining; PostgreSQL and MLflow are composed for production-style local operation. The UI distinguishes simulator measurements, predictions, intervals, and labels rather than presenting a decorative dashboard.
