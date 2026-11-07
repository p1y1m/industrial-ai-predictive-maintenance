from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from industrial_pm.data.split import select_engines, split_development_engines
from industrial_pm.modeling.anomaly import HealthyIsolationForest
from industrial_pm.modeling.features import (
    FeatureBuilder,
    final_cycle_mask,
    pseudo_test_endpoint_mask,
)
from industrial_pm.modeling.metrics import json_safe, regression_metrics
from industrial_pm.modeling.risk import CalibratedHorizonRisk
from industrial_pm.modeling.temporal import (
    TemporalLSTM,
    build_sequences,
    predict_temporal,
    subset_sequences,
    train_temporal_model,
)
from industrial_pm.modeling.uncertainty import ConformalInterval


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Training configuration must be a mapping")
    return data


def _selected_arrays(
    frame: pd.DataFrame, features: pd.DataFrame, mask: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = mask.to_numpy()
    return (
        features.loc[selected].to_numpy(),
        frame.loc[selected, "rul"].to_numpy(dtype=float),
        frame.loc[selected, "unit_id"].to_numpy(dtype=int),
    )


def _fit_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> tuple[Pipeline, dict[str, Any], dict[str, float]]:
    best: tuple[float, Pipeline, dict[str, Any], dict[str, float]] | None = None
    for alpha in config["alpha_grid"]:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=float(alpha), solver="lsqr", tol=1e-4)),
            ]
        )
        model.fit(train_x, train_y)
        metrics = regression_metrics(validation_y, np.clip(model.predict(validation_x), 0, 125))
        item = (metrics["rmse"], model, {"alpha": alpha}, metrics)
        if best is None or item[0] < best[0]:
            best = item
    if best is None:
        raise RuntimeError("Ridge search produced no candidate")
    return best[1], best[2], best[3]


def _fit_tree(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> tuple[HistGradientBoostingRegressor, dict[str, Any], dict[str, float]]:
    best: tuple[float, HistGradientBoostingRegressor, dict[str, Any], dict[str, float]] | None = None
    for learning_rate, leaves in product(
        config["learning_rate_grid"], config["max_leaf_nodes_grid"]
    ):
        model = HistGradientBoostingRegressor(
            learning_rate=float(learning_rate),
            max_leaf_nodes=int(leaves),
            max_iter=int(config["max_iter"]),
            min_samples_leaf=20,
            l2_regularization=0.1,
            early_stopping=False,
            random_state=seed,
        )
        model.fit(train_x, train_y)
        metrics = regression_metrics(validation_y, np.clip(model.predict(validation_x), 0, 125))
        params = {"learning_rate": learning_rate, "max_leaf_nodes": leaves}
        item = (metrics["rmse"], model, params, metrics)
        if best is None or item[0] < best[0]:
            best = item
    if best is None:
        raise RuntimeError("Tree search produced no candidate")
    return best[1], best[2], best[3]


def _temporal_checkpoint(model: TemporalLSTM, input_size: int, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_dict": model.state_dict(),
        "input_size": input_size,
        "hidden_size": int(config["hidden_size"]),
        "num_layers": int(config["num_layers"]),
        "dropout": float(config["dropout"]),
    }


def _log_mlflow(
    config: dict[str, Any], metrics: dict[str, Any], artifact_dir: Path, run_name: str
) -> str | None:
    try:
        import mlflow
    except ImportError:
        return None
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///./mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "cmapss-fd001"))
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "subset": config["experiment"]["subset"],
                "seed": config["experiment"]["seed"],
                "rul_cap": config["experiment"]["rul_cap"],
                "sequence_length": config["features"]["sequence_length"],
                "selected_model": metrics["selection"]["selected_model"],
            }
        )
        flattened: dict[str, float] = {}
        for model_name, values in metrics["test"].items():
            if isinstance(values, dict):
                for metric_name, value in values.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        flattened[f"test_{model_name}_{metric_name}"] = float(value)
        mlflow.log_metrics(flattened)
        mlflow.log_artifacts(str(artifact_dir))
        return cast(str, run.info.run_id)


def train_benchmark(
    config_path: Path,
    data_root: Path = Path("data"),
    artifact_root: Path = Path("artifacts"),
    registry_root: Path = Path("model_registry"),
) -> dict[str, Any]:
    """Run the bounded leakage-free FD001 benchmark and persist its full evidence bundle."""

    config = _load_config(config_path)
    experiment = config["experiment"]
    seed = int(experiment["seed"])
    subset = str(experiment["subset"]).upper()
    processed = data_root / "processed" / subset.lower()
    train_frame = pd.read_csv(processed / "train.csv.gz")
    test_frame = pd.read_csv(processed / "test.csv.gz")
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    print("[train] loaded official processed dataset", flush=True)

    # 70/15/15 engine-level development train, tuning validation, calibration split.
    outer = split_development_engines(train_frame, validation_fraction=0.30, seed=seed)
    development = select_engines(train_frame, outer.train_engine_ids)
    holdout = select_engines(train_frame, outer.validation_engine_ids)
    inner = split_development_engines(holdout, validation_fraction=0.50, seed=seed + 1)
    validation = select_engines(holdout, inner.train_engine_ids)
    calibration = select_engines(holdout, inner.validation_engine_ids)
    all_sets = [
        set(development["unit_id"]),
        set(validation["unit_id"]),
        set(calibration["unit_id"]),
        set(test_frame["unit_id"]),
    ]
    if all_sets[0] & all_sets[1] or all_sets[0] & all_sets[2] or all_sets[1] & all_sets[2]:
        raise AssertionError("Development engine partitions overlap")

    feature_builder = FeatureBuilder(tuple(config["features"]["rolling_windows"]))
    development_x_df = feature_builder.fit_transform(development)
    validation_x_df = feature_builder.transform(validation)
    calibration_x_df = feature_builder.transform(calibration)
    test_x_df = feature_builder.transform(test_frame)
    development_x = development_x_df.to_numpy()
    development_y = development["rul"].to_numpy(dtype=float)
    validation_endpoint_mask = pseudo_test_endpoint_mask(validation)
    calibration_endpoint_mask = pseudo_test_endpoint_mask(calibration)
    official_test_mask = final_cycle_mask(test_frame)
    validation_eval_x, validation_eval_y, _ = _selected_arrays(
        validation, validation_x_df, validation_endpoint_mask
    )
    calibration_eval_x, calibration_eval_y, _ = _selected_arrays(
        calibration, calibration_x_df, calibration_endpoint_mask
    )
    test_final_x, test_final_y, test_final_ids = _selected_arrays(
        test_frame, test_x_df, official_test_mask
    )
    test_final_uncapped = test_frame.loc[
        official_test_mask, "rul_uncapped"
    ].to_numpy(dtype=float)
    print("[train] built causal features and engine-level partitions", flush=True)

    naive_value = float(np.median(development_y))
    validation_metrics: dict[str, dict[str, float]] = {
        "naive_median": regression_metrics(
            validation_eval_y, np.full_like(validation_eval_y, naive_value)
        )
    }
    ridge, ridge_params, validation_metrics["ridge_regression"] = _fit_ridge(
        development_x,
        development_y,
        validation_eval_x,
        validation_eval_y,
        config["models"]["ridge"],
        seed,
    )
    print("[train] completed Ridge regression tuning", flush=True)
    tree, tree_params, validation_metrics["hist_gradient_boosting"] = _fit_tree(
        development_x,
        development_y,
        validation_eval_x,
        validation_eval_y,
        config["models"]["tree"],
        seed,
    )
    print("[train] completed histogram gradient boosting tuning", flush=True)

    temporal_columns = feature_builder.base_columns_
    temporal_scaler = StandardScaler().fit(development[temporal_columns])
    development_scaled = temporal_scaler.transform(development[temporal_columns]).astype(np.float32)
    validation_scaled = temporal_scaler.transform(validation[temporal_columns]).astype(np.float32)
    calibration_scaled = temporal_scaler.transform(calibration[temporal_columns]).astype(np.float32)
    test_scaled = temporal_scaler.transform(test_frame[temporal_columns]).astype(np.float32)
    sequence_length = int(config["features"]["sequence_length"])
    temporal_train = build_sequences(
        development,
        development_scaled,
        sequence_length,
        stride=int(config["features"]["train_stride"]),
    )
    temporal_validation = build_sequences(
        validation, validation_scaled, sequence_length, stride=2
    )
    temporal_validation_all = build_sequences(
        validation, validation_scaled, sequence_length, stride=1
    )
    temporal_validation_eval = subset_sequences(
        temporal_validation_all, validation_endpoint_mask.to_numpy()
    )
    temporal_calibration_all = build_sequences(
        calibration, calibration_scaled, sequence_length, stride=1
    )
    temporal_calibration_eval = subset_sequences(
        temporal_calibration_all, calibration_endpoint_mask.to_numpy()
    )
    temporal_test_final = build_sequences(
        test_frame, test_scaled, sequence_length, final_only=True
    )
    lstm_config = config["models"]["lstm"]
    temporal_result = train_temporal_model(
        temporal_train,
        temporal_validation,
        hidden_size=int(lstm_config["hidden_size"]),
        num_layers=int(lstm_config["num_layers"]),
        dropout=float(lstm_config["dropout"]),
        batch_size=int(lstm_config["batch_size"]),
        learning_rate=float(lstm_config["learning_rate"]),
        max_epochs=int(lstm_config["max_epochs"]),
        patience=int(lstm_config["patience"]),
        seed=seed,
    )
    print("[train] completed temporal LSTM training", flush=True)
    validation_metrics["lstm"] = regression_metrics(
        temporal_validation_eval.targets,
        predict_temporal(temporal_result.model, temporal_validation_eval),
    )

    candidates = ["ridge_regression", "hist_gradient_boosting", "lstm"]
    selected_name = min(candidates, key=lambda name: validation_metrics[name]["rmse"])
    calibration_predictions: dict[str, np.ndarray] = {
        "naive_median": np.full_like(calibration_eval_y, naive_value),
        "ridge_regression": np.clip(ridge.predict(calibration_eval_x), 0, 125),
        "hist_gradient_boosting": np.clip(tree.predict(calibration_eval_x), 0, 125),
        "lstm": predict_temporal(temporal_result.model, temporal_calibration_eval),
    }
    test_predictions: dict[str, np.ndarray] = {
        "naive_median": np.full_like(test_final_y, naive_value),
        "ridge_regression": np.clip(ridge.predict(test_final_x), 0, 125),
        "hist_gradient_boosting": np.clip(tree.predict(test_final_x), 0, 125),
        "lstm": predict_temporal(temporal_result.model, temporal_test_final),
    }
    test_metrics: dict[str, dict[str, float]] = {}
    for name, predictions_for_model in test_predictions.items():
        capped = regression_metrics(test_final_y, predictions_for_model)
        uncapped = regression_metrics(test_final_uncapped, predictions_for_model)
        capped.update({f"{key}_uncapped": value for key, value in uncapped.items()})
        test_metrics[name] = capped

    conformal = ConformalInterval.fit(
        calibration_eval_y,
        calibration_predictions[selected_name],
        coverage=float(config["uncertainty"]["coverage"]),
    )
    uncertainty_metrics = conformal.evaluate(test_final_y, test_predictions[selected_name])
    test_lower, test_upper = conformal.predict(test_predictions[selected_name])

    risk_models: dict[int, CalibratedHorizonRisk] = {}
    risk_metrics: dict[str, dict[str, float]] = {}
    risk_test_probabilities: dict[int, np.ndarray] = {}
    calibration_x = calibration_x_df.loc[calibration_endpoint_mask].to_numpy()
    calibration_rul_uncapped = calibration.loc[
        calibration_endpoint_mask, "rul_uncapped"
    ].to_numpy()
    for horizon in config["risk"]["horizons"]:
        horizon = int(horizon)
        risk = CalibratedHorizonRisk(horizon, seed=seed).fit(
            development_x,
            development["rul_uncapped"].to_numpy(),
            calibration_x,
            calibration_rul_uncapped,
        )
        risk_models[horizon] = risk
        risk_metrics[str(horizon)] = risk.evaluate(
            test_final_x, test_final_uncapped
        )
        risk_test_probabilities[horizon] = risk.predict_proba(test_final_x)
    print("[train] completed risk calibration", flush=True)

    anomaly_scaler = StandardScaler().fit(development_x)
    healthy_mask = development["rul_uncapped"].to_numpy() > 100
    anomaly = HealthyIsolationForest(seed=seed).fit(
        anomaly_scaler.transform(development_x[healthy_mask])
    )
    anomaly_score = anomaly.score(anomaly_scaler.transform(test_final_x))
    anomaly_flag = anomaly.is_anomaly(anomaly_scaler.transform(test_final_x))

    # Model-agnostic global explanations on the tuned tree, evaluated on validation engines.
    importance = permutation_importance(
        tree,
        validation_eval_x,
        validation_eval_y,
        n_repeats=5,
        random_state=seed,
        scoring="neg_root_mean_squared_error",
        # The validation set is intentionally small; process-spawn overhead dominates here.
        n_jobs=1,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": feature_builder.feature_columns_,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
            "explained_model": "hist_gradient_boosting",
            "method": "permutation_importance_validation_engines",
        }
    ).sort_values("importance_mean", ascending=False)
    print("[train] completed anomaly detection and explanation analysis", flush=True)

    run_id = f"{subset.lower()}-{manifest['archive_sha256'][:8]}-seed{seed}"
    artifact_dir = artifact_root / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_builder, artifact_dir / "feature_builder.joblib")
    joblib.dump(ridge, artifact_dir / "ridge_regression.joblib")
    joblib.dump(tree, artifact_dir / "hist_gradient_boosting.joblib")
    joblib.dump(temporal_scaler, artifact_dir / "temporal_scaler.joblib")
    torch.save(
        _temporal_checkpoint(temporal_result.model, len(temporal_columns), lstm_config),
        artifact_dir / "lstm.pt",
    )
    joblib.dump(risk_models, artifact_dir / "risk_models.joblib")
    joblib.dump(anomaly_scaler, artifact_dir / "anomaly_scaler.joblib")
    joblib.dump(anomaly, artifact_dir / "anomaly_detector.joblib")
    joblib.dump(conformal, artifact_dir / "conformal_interval.joblib")
    importance_frame.to_csv(artifact_dir / "feature_importance.csv", index=False)
    (artifact_dir / "lstm_history.json").write_text(
        json.dumps(temporal_result.history, indent=2) + "\n", encoding="utf-8"
    )

    predictions = pd.DataFrame(
        {
            "unit_id": test_final_ids,
            "actual_rul": test_final_y,
            "actual_rul_uncapped": test_final_uncapped,
            **{f"prediction_{name}": value for name, value in test_predictions.items()},
            "prediction_selected": test_predictions[selected_name],
            "interval_lower": test_lower,
            "interval_upper": test_upper,
            "anomaly_score": anomaly_score,
            "is_anomaly": anomaly_flag,
            **{f"risk_{horizon}_cycles": value for horizon, value in risk_test_probabilities.items()},
        }
    )
    predictions["absolute_error"] = np.abs(
        predictions["prediction_selected"] - predictions["actual_rul"]
    )
    predictions.to_csv(artifact_dir / "test_predictions.csv", index=False)

    horizon_rows: list[dict[str, Any]] = []
    bins = [0, 15, 30, 60, 100, np.inf]
    labels = ["0-15", "16-30", "31-60", "61-100", "101+"]
    groups = pd.cut(test_final_uncapped, bins=bins, labels=labels, include_lowest=True)
    for label in labels:
        mask = np.asarray(groups == label)
        if mask.any():
            horizon_rows.append(
                {
                    "rul_horizon": label,
                    "engines": int(mask.sum()),
                    **regression_metrics(
                        test_final_uncapped[mask], test_predictions[selected_name][mask]
                    ),
                }
            )
    pd.DataFrame(horizon_rows).to_csv(artifact_dir / "horizon_analysis.csv", index=False)

    metrics: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "subset": subset,
            "source_type": manifest["source_type"],
            "archive_sha256": manifest["archive_sha256"],
            "train_engines": manifest["train_engines"],
            "test_engines": manifest["test_engines"],
        },
        "split": {
            "development_engines": len(development["unit_id"].unique()),
            "validation_engines": len(validation["unit_id"].unique()),
            "calibration_engines": len(calibration["unit_id"].unique()),
            "validation_pseudo_endpoints": int(validation_endpoint_mask.sum()),
            "calibration_pseudo_endpoints": int(calibration_endpoint_mask.sum()),
            "official_test_engines": len(test_frame["unit_id"].unique()),
            "seed": seed,
        },
        "selection": {
            "selected_model": selected_name,
            "criterion": "lowest multi-cutoff pseudo-test validation RMSE",
            "test_used_for_selection": False,
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "uncertainty": uncertainty_metrics,
        "risk": risk_metrics,
        "tuning": {"ridge_regression": ridge_params, "hist_gradient_boosting": tree_params},
        "lstm": {
            "best_epoch": temporal_result.best_epoch,
            "best_all_sequence_validation_rmse": temporal_result.best_validation_rmse,
            "epochs_completed": len(temporal_result.history),
        },
        "explainability": {
            "method": "permutation importance on held-out validation engines",
            "explained_model": "hist_gradient_boosting",
            "top_features": importance_frame.head(10)["feature"].tolist(),
        },
    }
    metrics = json_safe(metrics)
    (artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (artifact_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    registry_root.mkdir(parents=True, exist_ok=True)
    promotion_gates: dict[str, bool] = {
        "beats_naive_validation_rmse": validation_metrics[selected_name]["rmse"]
        < validation_metrics["naive_median"]["rmse"],
        "interval_test_coverage_report_only": uncertainty_metrics["empirical_coverage"]
        >= float(config["promotion"]["min_interval_coverage"]),
        "test_not_used_for_selection": True,
    }
    promotion: dict[str, Any] = {
        "stage": "champion",
        "run_id": run_id,
        "selected_model": selected_name,
        "artifact_path": str(artifact_dir),
        "promoted_at": datetime.now(UTC).isoformat(),
        "gates": promotion_gates,
    }
    if not promotion_gates["beats_naive_validation_rmse"]:
        raise RuntimeError("Promotion blocked: selected model does not beat naive validation baseline")
    (registry_root / "champion.json").write_text(
        json.dumps(promotion, indent=2) + "\n", encoding="utf-8"
    )
    mlflow_run_id = _log_mlflow(config, metrics, artifact_dir, run_id)
    metrics["mlflow_run_id"] = mlflow_run_id
    (artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("[train] persisted benchmark, registry, and MLflow artifacts", flush=True)
    return metrics
