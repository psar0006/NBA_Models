"""Retrain NBA points, rebounds and assists on *_train.parquet outputs.

Chronological expanding-window OOF predictions calibrate NB2 dispersion;
final LightGBM mean models are fitted on every eligible training row.

Examples:
  python retrain_nba_models.py
  python retrain_nba_models.py --data-dir processed_nba_props --props points assists
  python retrain_nba_models.py --points-config models/nba_points/v1/model_config.json
  python retrain_nba_models.py --rebounds-config models/nba_rebounds/v1_probabilistic/probability_config.json

Points and rebounds always include is_starter and pregame_available.
Default feature mode is all for every prop; current preprocessing feature lists take priority.
No hyperparameter search or held-out test evaluation is performed here.
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

EPS = 1e-8
ALPHA_MIN, ALPHA_MAX = 1e-4, 5.0
TARGETS = {"points": "points", "rebounds": "reboundsTotal", "assists": "assists"}

# Force these two pregame context columns into the requested mean models,
# even when an older model configuration supplies a stale feature list.
REQUIRED_CONTEXT_FEATURES = {
    "points": ("is_starter", "pregame_available"),
    "rebounds": ("is_starter", "pregame_available"),
    "assists": ("is_starter", "pregame_available"),
}

# Points Optuna winner supplied by user; tree count is configurable because
# original early-stopping best_iteration was not supplied.
POINTS_PARAMS = dict(
    learning_rate=0.03521358805467869, num_leaves=35, max_depth=-1,
    min_child_samples=94, reg_alpha=0.4424625102595975,
    reg_lambda=1.959828624191452, colsample_bytree=0.6180909155642152,
    subsample=0.7301321323053057, subsample_freq=1, max_bin=127,
    n_estimators=650,
)
# Rebounds/assists initial LightGBM configuration from their notebooks.
OTHER_PARAMS = dict(
    n_estimators=650, learning_rate=0.04, num_leaves=31, max_depth=-1,
    min_child_samples=75, colsample_bytree=0.85, subsample=0.85,
    subsample_freq=1, reg_lambda=1.0,
)
BUCKET_EDGES = {
    "points": [0, 10, 20, 30, 40, float("inf")],
    "rebounds": [0, 3, 6, 9, 13, float("inf")],
    "assists": [0, 2, 4, 6, 9, float("inf")],
}


def read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data file: {path}")


def nb_nll(y, mu, alpha):
    mu = np.maximum(np.asarray(mu, dtype=float), EPS)
    alpha = np.maximum(np.asarray(alpha, dtype=float), ALPHA_MIN)
    r = 1.0 / alpha
    return -(gammaln(y + r) - gammaln(r) - gammaln(y + 1)
             + r * (np.log(r) - np.log(r + mu))
             + y * (np.log(mu) - np.log(r + mu)))


def fit_alpha(y, mu):
    result = minimize_scalar(
        lambda log_a: float(np.mean(nb_nll(y, mu, np.exp(log_a)))),
        bounds=(np.log(ALPHA_MIN), np.log(ALPHA_MAX)), method="bounded",
    )
    if not result.success:
        raise RuntimeError(f"NB dispersion optimization failed: {result.message}")
    return float(np.exp(result.x))


def fit_buckets(y, mu, global_alpha, edges, min_n=250, prior_n=500):
    ids = np.digitize(mu, edges[1:-1])
    alphas, details = [], []
    for k in range(len(edges) - 1):
        mask = ids == k
        n = int(mask.sum())
        local = fit_alpha(y[mask], mu[mask]) if n >= min_n else global_alpha
        w = n / (n + prior_n) if n >= min_n else 0.0
        shrink = float(np.exp(w * np.log(local) + (1 - w) * np.log(global_alpha)))
        alphas.append(shrink)
        details.append({"bucket": k, "lower": edges[k],
                        "upper": None if np.isinf(edges[k+1]) else edges[k+1],
                        "observations": n, "raw_alpha": local, "alpha": shrink})
    return alphas, details


def load_config(path):
    if not path:
        return {}
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


REBOUND_PERCENTAGE_FEATURES = [
    f"{metric}_{suffix}"
    for metric in ("reboundPercentage", "offensiveReboundPercentage", "defensiveReboundPercentage")
    for suffix in ("ewma", "avg10")
]


def select_features(prop, df, feature_file, cfg, mode):
    # The preprocessing feature manifest is authoritative: a saved model config
    # describes an OLD model and can omit newly engineered predictors.
    if feature_file.exists():
        features = [x.strip() for x in feature_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        if cfg.get("mean_features") or cfg.get("features"):
            print(f"  {prop}: using current preprocessing feature manifest instead of saved model config", flush=True)
    elif cfg.get("mean_features"):
        features = list(cfg["mean_features"])
    elif cfg.get("features"):
        features = list(cfg["features"])
    else:
        raise FileNotFoundError(f"No feature list for {prop}: {feature_file}")

    if mode == "player" and prop in ("rebounds", "assists"):
        features = [x for x in features if not x.startswith(("team_", "opponent_", "opp_"))]

    # Require the new rebound-percentage predictors even with a stale config.
    if prop == "rebounds":
        features.extend(REBOUND_PERCENTAGE_FEATURES)
    features = list(dict.fromkeys([*features, *REQUIRED_CONTEXT_FEATURES.get(prop, ())]))
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"{prop}: missing selected features ({len(missing)}): {missing[:25]}. "
                         "Regenerate the training dataset using the updated preprocessing script.")
    if not features:
        raise ValueError(f"{prop}: no features selected")
    bad = [c for c in features if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    if bad:
        raise TypeError(f"{prop}: non-numeric features: {bad}. Encode explicitly before training.")
    return features


def make_oof(X, y, dates, params, n_folds, warmup_fraction):
    day = pd.to_datetime(dates).dt.normalize()
    unique = np.sort(day.unique())
    warmup = max(20, int(len(unique) * warmup_fraction))
    if len(unique) - warmup < n_folds:
        raise ValueError("Not enough distinct dates for chronological OOF folds")
    edges = np.linspace(warmup, len(unique), n_folds + 1, dtype=int)
    out = np.full(len(X), np.nan)
    for k in range(n_folds):
        tr = (day < unique[edges[k]]).to_numpy()
        end = unique[edges[k+1]] if edges[k+1] < len(unique) else None
        va = ((day >= unique[edges[k]]) & ((day < end) if end is not None else True)).to_numpy()
        if not tr.any() or not va.any():
            raise ValueError(f"Empty OOF fold {k+1}")
        model = lgb.LGBMRegressor(**params)
        model.fit(X.loc[tr], y[tr])
        out[va] = np.maximum(model.predict(X.loc[va]), EPS)
        print(f"  OOF {k+1}/{n_folds}: train={tr.sum():,}, validation={va.sum():,}", flush=True)
    return out


def retrain(prop, args):
    target = TARGETS[prop]
    path = args.data_dir / f"nba_{prop}_train.parquet"
    if not path.exists():
        csv = path.with_suffix(".csv")
        if not csv.exists():
            raise FileNotFoundError(f"Missing {path} (or {csv})")
        path = csv
    df = read_frame(path)
    cfg = load_config(getattr(args, f"{prop}_config"))
    feature_file = args.data_dir / f"nba_{prop}_features.txt"
    mode = args.feature_mode if args.feature_mode != "auto" else "all"
    features = select_features(prop, df, feature_file, cfg, mode)
    print(f"  Context features selected: {[f for f in REQUIRED_CONTEXT_FEATURES.get(prop, ()) if f in features]}", flush=True)
    if target not in df or "gameDateTimeEst" not in df:
        raise ValueError(f"{path} must contain {target} and gameDateTimeEst")
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"], errors="raise")
    df[target] = pd.to_numeric(df[target], errors="coerce")
    mask = df[target].notna() & (df[target] >= 0)
    if "numMinutes" in df:
        mask &= pd.to_numeric(df["numMinutes"], errors="coerce").gt(0)
    if "is_prediction_row" in df:
        mask &= ~df["is_prediction_row"].fillna(False).astype(bool)
    df = df.loc[mask].sort_values(
        [c for c in ("gameDateTimeEst", "gameId", "personId") if c in df.columns],
        kind="stable",
    ).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{prop}: no eligible completed player-games")
    y = df[target].to_numpy(dtype=np.float64)
    if not np.allclose(y, np.round(y)):
        raise ValueError(f"{prop}: NB2 requires integer count targets")
    X = df[features].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    if X.notna().sum().eq(0).any():
        empty = X.columns[X.notna().sum().eq(0)].tolist()
        warnings.warn(f"{prop}: entirely null features: {empty[:30]}")

    params = dict(POINTS_PARAMS if prop == "points" else OTHER_PARAMS)
    params.update(objective="regression", random_state=42, n_jobs=args.jobs, verbosity=-1)
    if args.n_estimators:
        params["n_estimators"] = args.n_estimators
    if args.params_dir:
        p = args.params_dir / f"{prop}_params.json"
        if p.exists():
            params.update(load_config(p))
    # Mean required for NB2; median (regression_l1) is not an NB mean.
    if params.get("objective") != "regression":
        raise ValueError(f"{prop}: objective must be regression for NB2 calibration")
    print(f"\n[{prop.upper()}] rows={len(df):,}, features={len(features)}, "
          f"dates={df.gameDateTimeEst.min().date()}..{df.gameDateTimeEst.max().date()}", flush=True)
    print(f"  Mean model trees: {params['n_estimators']}", flush=True)
    oof = make_oof(X, y, df["gameDateTimeEst"], params, args.folds, args.warmup_fraction)
    valid = np.isfinite(oof)
    alpha = fit_alpha(y[valid], oof[valid])
    bucket_alphas, diagnostics = fit_buckets(y[valid], oof[valid], alpha, BUCKET_EDGES[prop])
    print(f"  OOF coverage: {valid.sum():,}; global NB2 alpha={alpha:.5f}", flush=True)

    print("  Fitting final mean model on ALL eligible historical observations ...", flush=True)
    final = lgb.LGBMRegressor(**params)
    final.fit(X, y)
    output = args.output_dir / prop
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, output / f"{prop}_mean_lightgbm.joblib")
    config = {
        "prop": prop, "target": target, "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(path), "train_rows": int(len(df)),
        "training_start": str(df.gameDateTimeEst.min().date()),
        "training_end": str(df.gameDateTimeEst.max().date()),
        "mean_model": f"{prop}_mean_lightgbm.joblib", "mean_features": features,
        "required_context_features": list(REQUIRED_CONTEXT_FEATURES.get(prop, ())),
        "lightgbm_params": params, "distribution": "negative_binomial_NB2",
        "nb2_global_alpha": alpha,
        "nb2_bucket_edges": [None if np.isinf(x) else float(x) for x in BUCKET_EDGES[prop]],
        "nb2_bucket_alphas": bucket_alphas,
        "oof_rows": int(valid.sum()), "oof_folds": args.folds,
        "oof_warmup_fraction": args.warmup_fraction,
        "notes": "NB dispersion calibrated on chronological OOF means; mean model refit on full training data. Conditional on participation; no DNP probability model.",
    }
    (output / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    pd.DataFrame(diagnostics).to_csv(output / "dispersion_buckets.csv", index=False)
    pd.DataFrame({"feature": features, "gain_importance": final.booster_.feature_importance(importance_type="gain")})\
        .sort_values("gain_importance", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    print(f"  Saved {output.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("processed_nba_props"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/nba_props_full"))
    parser.add_argument("--props", nargs="+", choices=list(TARGETS), default=list(TARGETS))
    parser.add_argument("--points-config", type=Path, help="Prior points model_config.json with mean_features")
    parser.add_argument("--rebounds-config", type=Path, help="Prior rebound probability_config.json")
    parser.add_argument("--assists-config", type=Path, help="Prior assist probability_config.json")
    parser.add_argument("--params-dir", type=Path, help="Optional directory with {prop}_params.json overrides")
    parser.add_argument("--feature-mode", choices=["auto", "player", "all"], default="auto",
                        help="auto: all engineered features for every prop; player: player-only for rebounds/assists")
    parser.add_argument("--n-estimators", type=int, default=None, help="Override tree count for all models")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--warmup-fraction", type=float, default=0.40)
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()
    if args.folds < 2 or not 0 < args.warmup_fraction < 1:
        parser.error("--folds must be >=2 and --warmup-fraction between 0 and 1")
    for prop in args.props:
        retrain(prop, args)
    print("\nAll requested models retrained.", flush=True)


if __name__ == "__main__":
    main()
