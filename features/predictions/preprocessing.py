from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

def preprocess_player_data(df):
    """Preprocess the player data for modeling"""
    
    # 1. Sort and filter
    df = df.sort_values(["player_id", "date"])
    df = df[     # Keep rows where team_id matches t1 or t2 OR where both t1 and t2 are missing
        (df["team_id"] == df["t1"]) |
        (df["team_id"] == df["t2"]) |
        (df["t1"].isna() & df["t2"].isna())
    ]

    # Convert date columns to datetime
    df["date"] = pd.to_datetime(df["date"])
    df["md"] = pd.to_datetime(df["md"])

    # 2. Date and matchday calculations 
    df["next_day"] = df.groupby("player_id")["date"].shift(-1) 
    df["next_md"] = df.groupby("player_id")["md"].transform(
        lambda x: x.shift(-1).where(x.shift(-1) != x).bfill()
    )
    df["days_to_next"] = (df["next_md"] - df["date"]).dt.days

    # 3. Next day market value
    df["mv_next_day"] = df.groupby("player_id")["mv"].shift(-1)
    df["mv_target"] = df["mv_next_day"] - df["mv"]
    df = df[df["mv"] != 0.0]

    # 4. Feature engineering 
    # Market value trend 1d
    df["mv_change_1d"] = df["mv"] - df.groupby("player_id")["mv"].shift(1)
    df["mv_trend_1d"] = df.groupby("player_id")["mv"].pct_change(fill_method=None)
    df["mv_trend_1d"] = df["mv_trend_1d"].replace([np.inf, -np.inf], 0).fillna(0)

    # Market value trend 3d
    df["mv_change_3d"] = df["mv"] - df.groupby("player_id")["mv"].shift(3)
    df["mv_vol_3d"] = df.groupby("player_id")["mv"].rolling(3).std().reset_index(0,drop=True)

    # Market value trend 7d
    df["mv_trend_7d"] = df.groupby("player_id")["mv"].pct_change(periods=7, fill_method=None)
    df["mv_trend_7d"] = df["mv_trend_7d"].replace([np.inf, -np.inf], 0).fillna(0)

    ## League-wide market context
    df["market_divergence"] = (df["mv"] / df.groupby("md")["mv"].transform("mean")).rolling(3).mean()

    # Enhanced features for better predictions
    # Player form indicators
    df["points_ma_3"] = df.groupby("player_id")["p"].rolling(3).mean().reset_index(0,drop=True)
    df["points_ma_5"] = df.groupby("player_id")["p"].rolling(5).mean().reset_index(0,drop=True)
    df["points_trend"] = df.groupby("player_id")["p"].pct_change(periods=3, fill_method=None).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Minutes played consistency
    df["mp_ma_3"] = df.groupby("player_id")["mp"].rolling(3).mean().reset_index(0,drop=True)
    df["mp_consistency"] = 1 - (df.groupby("player_id")["mp"].rolling(3).std().reset_index(0,drop=True) / (df["mp_ma_3"] + 1e-8))
    
    # Points per minute efficiency
    df["ppm_ma_3"] = df.groupby("player_id")["ppm"].rolling(3).mean().reset_index(0,drop=True)
    df["ppm_trend"] = df.groupby("player_id")["ppm"].pct_change(periods=3, fill_method=None).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Match outcome influence
    df["win_rate_3"] = df.groupby("player_id")["won"].rolling(3).mean().reset_index(0,drop=True)
    df["recent_form"] = (df["points_ma_3"] * 0.4 + df["mp_consistency"] * 0.3 + df["win_rate_3"] * 0.3).fillna(0)
    
    # Position-based features
    position_stats = df.groupby("position").agg({
        "p": ["mean", "std"],
        "mv": ["mean", "std"],
        "ppm": ["mean", "std"]
    }).round(2)
    position_stats.columns = ["pos_p_mean", "pos_p_std", "pos_mv_mean", "pos_mv_std", "pos_ppm_mean", "pos_ppm_std"]
    df = df.merge(position_stats, left_on="position", right_index=True, how="left")
    
    # Player performance relative to position
    df["p_vs_position"] = (df["p"] - df["pos_p_mean"]) / (df["pos_p_std"] + 1e-8)
    df["mv_vs_position"] = (df["mv"] - df["pos_mv_mean"]) / (df["pos_mv_std"] + 1e-8)
    df["ppm_vs_position"] = (df["ppm"] - df["pos_ppm_mean"]) / (df["pos_ppm_std"] + 1e-8)
    
    # Market value momentum indicators
    df["mv_momentum_short"] = df.groupby("player_id")["mv"].pct_change(periods=2, fill_method=None).replace([np.inf, -np.inf], 0).fillna(0)
    df["mv_momentum_long"] = df.groupby("player_id")["mv"].pct_change(periods=5, fill_method=None).replace([np.inf, -np.inf], 0).fillna(0)
    df["mv_acceleration"] = df["mv_momentum_short"] - df["mv_momentum_long"]

    # 5. Clip outliers in mv_target
    Q1 = df["mv_target"].quantile(0.25)
    Q3 = df["mv_target"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2.5 * IQR
    upper_bound = Q3 + 2.5 * IQR

    df["mv_target_clipped"] = df["mv_target"].clip(lower_bound, upper_bound)

    # 6. Fill missing values
    df = df.fillna({
        "market_divergence": 1,
        "mv_change_3d": 0,
        "mv_vol_3d": 0,
        "p": 0,
        "ppm": 0,
        "mp": 0,
        "won": -1,
        "points_ma_3": 0,
        "points_ma_5": 0,
        "points_trend": 0,
        "mp_ma_3": 0,
        "mp_consistency": 0,
        "ppm_ma_3": 0,
        "ppm_trend": 0,
        "win_rate_3": 0,
        "recent_form": 0,
        "p_vs_position": 0,
        "mv_vs_position": 0,
        "ppm_vs_position": 0,
        "mv_momentum_short": 0,
        "mv_momentum_long": 0,
        "mv_acceleration": 0
    })

    # 7. Cutout todays values and store them
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    cutoff_time = now.replace(hour=22, minute=15, second=0, microsecond=0)
    max_date = (now - timedelta(days=1)) if now <= cutoff_time else now
    max_date = max_date.date()

    today_df = df[df["date"].dt.date >= max_date]

    # Drop those values from today from df
    df = df[df["date"].dt.date < max_date]

    # 8. Drop rows with NaN in critical columns
    df = df.dropna(subset=["mv_change_1d", "next_day", "next_md", "days_to_next", "mv_next_day", "mv_target", "mv_target_clipped"])

    return df, today_df


def split_data(df, features, target):
    """Split the data into training and testing sets based on date to avoid data leakage"""

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    split_idx = int(len(df) * 0.75)
    split_date = df["date"].iloc[split_idx]

    # Split by time, to avoid data leakage
    train = df[df["date"] < split_date]
    test = df[(df["date"] >= split_date)]

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]

    return X_train, X_test, y_train, y_test