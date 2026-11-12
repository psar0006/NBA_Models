"""Predict NBA player props from retrain_nba_models.py artifacts.

Reads nba_{prop}_new.parquet, model_config.json and {prop}_mean_lightgbm.joblib.
Outputs exact conditional mean forecasts and fair NB2 odds for integer 'N+' lines.

Examples:
 python predict_nba_props.py
 python predict_nba_props.py --data-dir processed_nba_props --models-dir models/nba_props_full
 python predict_nba_props.py --props points --output-dir predictions

NOTE: Fair odds are conditional on playing, not market odds. The lineup roster
feed may include inactive players and unconfirmed starters; those are marked.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import nbinom

LINES = {"points": range(5, 31), "rebounds": range(3, 17), "assists": range(2, 21)}
ID_COLS = ["gameDateTimeEst", "gameId", "personId", "player_name", "playerName", "playerteamId",
           "opponentteamId", "playerteamName", "opponentteamName", "home",
           "rosterStatus", "lineupStatus", "position", "is_starter", "pregame_available"]


def load_frame(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    csv = path.with_suffix('.csv')
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing {path} (or {csv})")


def validate_config(config: dict, prop: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    if config.get("prop") != prop:
        raise ValueError(f"{prop}: model_config.json prop mismatch: {config.get('prop')}")
    if config.get("distribution") != "negative_binomial_NB2":
        raise ValueError(f"{prop}: expected negative_binomial_NB2 distribution")
    features = config.get("mean_features")
    if not isinstance(features, list) or not features:
        raise ValueError(f"{prop}: no mean_features saved in model config")
    edges = config.get("nb2_bucket_edges")
    alphas = config.get("nb2_bucket_alphas")
    if not isinstance(edges, list) or not isinstance(alphas, list) or len(edges) != len(alphas) + 1:
        raise ValueError(f"{prop}: invalid NB2 bucket configuration")
    # The retraining script serializes infinity as null in its final edge.
    edges = np.array([float('inf') if x is None else float(x) for x in edges], dtype=float)
    alphas = np.array(alphas, dtype=float)
    if not np.all(np.diff(edges) > 0) or not np.all(np.isfinite(alphas) & (alphas > 0)):
        raise ValueError(f"{prop}: invalid NB2 bucket edges/alpha values")
    return features, edges, alphas


def load_player_names(data_dir: Path, lineup_file: Path | None = None) -> pd.DataFrame:
    """Use NBA lineup CSV written by preprocessing to recover player names."""
    paths = [lineup_file] if lineup_file else sorted(
        data_dir.glob("nba_lineups_*.csv"), reverse=True
    )
    for path in paths:
        if path is None or not path.exists():
            continue
        names = pd.read_csv(path, dtype={"personId": "string", "gameId": "string"})
        name_col = "playerName" if "playerName" in names else "player_name"
        if {"personId", name_col}.issubset(names.columns):
            names = names.rename(columns={name_col: "player_name"})
            names["player_name"] = names["player_name"].astype("string").str.strip()
            names = names.dropna(subset=["personId", "player_name"])
            names = names.loc[names["player_name"].ne("")]
            if not names.empty:
                print(f"Player names loaded from {path}", flush=True)
                return names[[c for c in ["personId", "gameId", "player_name"] if c in names]].drop_duplicates()
    warnings.warn("No lineup CSV with player names found; names may be missing")
    return pd.DataFrame(columns=["personId", "gameId", "player_name"])


def attach_player_names(frame: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    """Prefer names already present, then match by game/player and finally player ID."""
    frame = frame.copy()
    if "player_name" not in frame:
        frame["player_name"] = frame["playerName"] if "playerName" in frame else pd.NA
    if "personId" not in frame or names.empty:
        return frame
    frame["personId"] = frame["personId"].astype("string")
    names = names.copy()
    names["personId"] = names["personId"].astype("string")
    if "gameId" in frame and "gameId" in names:
        frame["gameId"] = frame["gameId"].astype("string")
        exact = names.drop_duplicates(["gameId", "personId"])
        frame = frame.merge(exact, on=["gameId", "personId"], how="left", suffixes=("", "_lookup"), validate="many_to_one")
        frame["player_name"] = frame["player_name"].fillna(frame.pop("player_name_lookup"))
    fallback = names.drop_duplicates("personId")[["personId", "player_name"]]
    frame = frame.merge(fallback, on="personId", how="left", suffixes=("", "_lookup"), validate="many_to_one")
    frame["player_name"] = frame["player_name"].fillna(frame.pop("player_name_lookup"))
    return frame


def score_prop(prop: str, data_dir: Path, models_dir: Path, output_dir: Path,
               include_inactive: bool = False, player_names: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = load_frame(data_dir / f"nba_{prop}_new.parquet")
    if player_names is not None:
        frame = attach_player_names(frame, player_names)
    artifact_dir = models_dir / prop
    config_path = artifact_dir / "model_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    features, edges, alphas = validate_config(config, prop)
    model = joblib.load(artifact_dir / config["mean_model"])

    if "is_prediction_row" in frame.columns:
        # A new dataset can also contain already-completed rows after the cutoff.
        prediction = frame["is_prediction_row"].fillna(False).astype(bool)
        frame = frame.loc[prediction].copy()
    if not include_inactive and "rosterStatus" in frame.columns:
        # Only explicit inactive designations are excluded; unknown statuses remain.
        inactive = frame["rosterStatus"].astype("string").str.lower().isin(
            ["inactive", "out", "injured", "suspended"]
        )
        frame = frame.loc[~inactive].copy()
    if frame.empty:
        warnings.warn(f"{prop}: no eligible prediction rows; skipping")
        return pd.DataFrame(), pd.DataFrame()
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise ValueError(f"{prop}: missing {len(missing)} model features: {missing[:30]}")
    X = frame[features].copy()
    for c in features:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.astype(np.float32).replace([np.inf, -np.inf], np.nan)
    empty = X.columns[X.isna().all()].tolist()
    if empty:
        warnings.warn(f"{prop}: entirely missing prediction features: {empty[:20]}")
    mu = np.maximum(np.asarray(model.predict(X), dtype=float), 1e-8)
    if not np.isfinite(mu).all():
        raise ValueError(f"{prop}: model returned non-finite means")
    bucket = np.searchsorted(edges[1:-1], mu, side="right")
    alpha = alphas[bucket]
    size = 1.0 / alpha
    p = size / (size + mu)

    metadata = frame[[c for c in ID_COLS if c in frame.columns]].reset_index(drop=True)
    metadata.insert(0, "stat", prop)
    metadata["predicted_mean"] = mu
    metadata["nb2_alpha"] = alpha
    metadata["dispersion_bucket"] = bucket
    if "rosterStatus" in metadata.columns:
        status = metadata["rosterStatus"].astype("string").str.lower()
        metadata["availability_warning"] = (~status.isin(["active"])).fillna(True)
    else:
        metadata["availability_warning"] = True

    rows = []
    for threshold in LINES[prop]:
        # N+ means P(X >= N); equivalent to Over (N - 0.5).
        over = nbinom.sf(threshold - 1, size, p)
        under = nbinom.cdf(threshold - 1, size, p)
        # Avoid infinite/undefined odds at probability zero, retain true probabilities.
        over_odds = np.divide(1.0, over, out=np.full_like(over, np.inf), where=over > 0)
        under_odds = np.divide(1.0, under, out=np.full_like(under, np.inf), where=under > 0)
        temp = metadata.copy()
        temp["line_plus"] = threshold
        temp["handicap"] = threshold - 0.5
        temp["over_probability"] = over
        temp["under_probability"] = under
        temp["over_fair_decimal_odds"] = over_odds
        temp["under_fair_decimal_odds"] = under_odds
        rows.append(temp)
    long = pd.concat(rows, ignore_index=True)
    long = long.sort_values([c for c in ["gameDateTimeEst", "gameId", "personId", "line_plus"]
                             if c in long.columns], kind="stable")
    output_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(output_dir / f"nba_{prop}_line_odds.csv", index=False)
    metadata.to_csv(output_dir / f"nba_{prop}_predictions.csv", index=False)
    print(f"{prop.upper()}: {len(metadata):,} players/game rows; "
          f"{len(long):,} line odds -> {output_dir.resolve()}", flush=True)
    return metadata, long


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("processed_nba_props"))
    ap.add_argument("--models-dir", type=Path, default=Path("models/nba_props_full"))
    ap.add_argument("--output-dir", type=Path, default=Path("predictions"))
    ap.add_argument("--props", nargs="+", choices=list(LINES), default=list(LINES))
    ap.add_argument("--include-inactive", action="store_true", help="Also predict explicitly inactive players")
    ap.add_argument("--lineup-file", type=Path, help="Optional explicit nba_lineups_YYYYMMDD.csv for player names")
    args = ap.parse_args()
    player_names = load_player_names(args.data_dir, args.lineup_file)
    all_means = []
    all_lines = []
    for prop in args.props:
        means, lines = score_prop(prop, args.data_dir, args.models_dir, args.output_dir,
                                  args.include_inactive, player_names)
        if not means.empty:
            all_means.append(means)
            all_lines.append(lines)
    if all_means:
        pd.concat(all_means, ignore_index=True).to_csv(args.output_dir / "nba_all_predictions.csv", index=False)
        pd.concat(all_lines, ignore_index=True).to_csv(args.output_dir / "nba_all_line_odds.csv", index=False)
    print("Finished.")

if __name__ == "__main__":
    main()
