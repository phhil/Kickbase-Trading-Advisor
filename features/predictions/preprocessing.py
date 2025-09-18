from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
from numba import jit
import warnings

@jit(nopython=True)
def fast_rolling_mean(values, window):
    """Fast rolling mean calculation using numba"""
    result = np.full_like(values, np.nan, dtype=np.float64)
    for i in range(window-1, len(values)):
        result[i] = np.mean(values[i-window+1:i+1])
    return result

@jit(nopython=True)
def fast_rolling_std(values, window):
    """Fast rolling std calculation using numba"""
    result = np.full_like(values, np.nan, dtype=np.float64)
    for i in range(window-1, len(values)):
        result[i] = np.std(values[i-window+1:i+1])
    return result

def preprocess_player_data(df, fast_mode=False):
    """Preprocess the player data for modeling with enhanced features for small changes"""
    
    print(f"Starting {'fast' if fast_mode else 'enhanced'} data preprocessing...")
    
    # 1. Sort and filter (optimized)
    df = df.sort_values(["player_id", "date"])
    df = df[     # Keep rows where team_id matches t1 or t2 OR where both t1 and t2 are missing
        (df["team_id"] == df["t1"]) |
        (df["team_id"] == df["t2"]) |
        (df["t1"].isna() & df["t2"].isna())
    ]

    # Convert date columns to datetime with better performance
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    if not pd.api.types.is_datetime64_any_dtype(df["md"]):
        df["md"] = pd.to_datetime(df["md"])

    # 2. Optimized date and matchday calculations 
    df["next_day"] = df.groupby("player_id")["date"].shift(-1) 
    df["next_md"] = df.groupby("player_id")["md"].transform(
        lambda x: x.shift(-1).where(x.shift(-1) != x).bfill()
    )
    df["days_to_next"] = (df["next_md"] - df["date"]).dt.days

    # 3. Next day market value
    df["mv_next_day"] = df.groupby("player_id")["mv"].shift(-1)
    df["mv_target"] = df["mv_next_day"] - df["mv"]
    df = df[df["mv"] != 0.0]

    # 4. Enhanced feature engineering with focus on small changes
    print("Creating enhanced features for small value changes...")
    
    # Basic momentum features
    df["mv_change_1d"] = df["mv"] - df.groupby("player_id")["mv"].shift(1)
    df["mv_trend_1d"] = df.groupby("player_id")["mv"].pct_change(fill_method=None)
    df["mv_trend_1d"] = df["mv_trend_1d"].replace([np.inf, -np.inf], 0).fillna(0)

    # Multi-period market value features  
    df["mv_change_3d"] = df["mv"] - df.groupby("player_id")["mv"].shift(3)
    df["mv_change_7d"] = df["mv"] - df.groupby("player_id")["mv"].shift(7)
    
    # Volatility measures (important for small changes)
    df["mv_vol_3d"] = df.groupby("player_id")["mv"].rolling(3).std().reset_index(0,drop=True)
    if not fast_mode:
        df["mv_vol_7d"] = df.groupby("player_id")["mv"].rolling(7).std().reset_index(0,drop=True)
    else:
        df["mv_vol_7d"] = df["mv_vol_3d"]  # Use 3d as proxy for speed
    
    # Market value trend analysis
    df["mv_trend_7d"] = df.groupby("player_id")["mv"].pct_change(periods=7, fill_method=None)
    df["mv_trend_7d"] = df["mv_trend_7d"].replace([np.inf, -np.inf], 0).fillna(0)

    # Enhanced league-wide market context
    df["market_divergence"] = (df["mv"] / df.groupby("md")["mv"].transform("mean")).rolling(3).mean()
    
    # Small change specific features
    df["mv_micro_trend"] = df.groupby("player_id")["mv"].pct_change(periods=2, fill_method=None).replace([np.inf, -np.inf], 0).fillna(0)
    df["mv_stability"] = 1 / (df["mv_vol_3d"] + 1)  # Higher for more stable players
    df["mv_recent_direction"] = np.sign(df["mv_change_1d"])  # Recent direction
    
    # Price level indicators (important for small changes)
    df["mv_percentile"] = df.groupby("position")["mv"].transform(lambda x: x.rank(pct=True))
    df["mv_zscore"] = df.groupby("position")["mv"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))

    # Enhanced player form indicators
    df["points_ma_3"] = df.groupby("player_id")["p"].rolling(3).mean().reset_index(0,drop=True)
    df["points_ma_5"] = df.groupby("player_id")["p"].rolling(5).mean().reset_index(0,drop=True)
    df["points_trend"] = df.groupby("player_id")["p"].pct_change(periods=3, fill_method=None).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Performance consistency (key for small changes)
    df["points_consistency"] = 1 - (df.groupby("player_id")["p"].rolling(5).std().reset_index(0,drop=True) / (df["points_ma_5"] + 1e-8))
    df["points_consistency"] = df["points_consistency"].clip(0, 2)
    
    # Minutes played features
    df["mp_ma_3"] = df.groupby("player_id")["mp"].rolling(3).mean().reset_index(0,drop=True)
    df["mp_consistency"] = 1 - (df.groupby("player_id")["mp"].rolling(3).std().reset_index(0,drop=True) / (df["mp_ma_3"] + 1e-8))
    df["mp_trend"] = df.groupby("player_id")["mp"].pct_change(periods=3, fill_method=None).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Points per minute efficiency
    df["ppm_ma_3"] = df.groupby("player_id")["ppm"].rolling(3).mean().reset_index(0,drop=True)
    df["ppm_trend"] = df.groupby("player_id")["ppm"].pct_change(periods=3, fill_method=None).replace([np.inf, -np.inf], 0).fillna(0)
    df["ppm_volatility"] = df.groupby("player_id")["ppm"].rolling(3).std().reset_index(0,drop=True)
    
    # Match outcome influence
    df["win_rate_3"] = df.groupby("player_id")["won"].rolling(3).mean().reset_index(0,drop=True)
    df["win_rate_5"] = df.groupby("player_id")["won"].rolling(5).mean().reset_index(0,drop=True)
    
    # Enhanced recent form calculation
    df["recent_form"] = (df["points_ma_3"] * 0.3 + df["mp_consistency"] * 0.25 + 
                        df["win_rate_3"] * 0.25 + df["points_consistency"] * 0.2).fillna(0)
    
    # Position-based features (vectorized for performance)
    print("Computing position-based features...")
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
    
    # Market value momentum indicators (enhanced)
    df["mv_momentum_short"] = df.groupby("player_id")["mv"].pct_change(periods=2, fill_method=None).replace([np.inf, -np.inf], 0).fillna(0)
    df["mv_momentum_long"] = df.groupby("player_id")["mv"].pct_change(periods=5, fill_method=None).replace([np.inf, -np.inf], 0).fillna(0)
    
    if not fast_mode:
        df["mv_momentum_very_long"] = df.groupby("player_id")["mv"].pct_change(periods=10, fill_method=None).replace([np.inf, -np.inf], 0).fillna(0)
    else:
        df["mv_momentum_very_long"] = df["mv_momentum_long"]  # Use long as proxy for speed
        
    df["mv_acceleration"] = df["mv_momentum_short"] - df["mv_momentum_long"]
    
    # Advanced features for small change prediction
    df["mv_relative_change"] = df["mv_change_1d"] / (df["mv"] + 1e-8)  # Relative to current value
    df["mv_price_pressure"] = df.groupby("player_id")["mv_change_1d"].rolling(3).mean().reset_index(0,drop=True)  # Recent pressure
    df["form_mv_interaction"] = df["recent_form"] * df["mv_trend_1d"]  # Form-price interaction
    
    # Team performance indicators (simplified in fast mode)
    if not fast_mode:
        team_stats = df.groupby(["team_name", "md"]).agg({
            "won": "mean",
            "p": "mean"
        }).reset_index()
        team_stats.columns = ["team_name", "md", "team_win_rate", "team_avg_points"]
        df = df.merge(team_stats, on=["team_name", "md"], how="left")
        
        df["team_form"] = df.groupby("team_name")["team_win_rate"].rolling(3).mean().reset_index(0,drop=True)
    else:
        # Use simplified team features for speed
        df["team_win_rate"] = df.groupby("team_name")["won"].transform("mean")
        df["team_avg_points"] = df.groupby("team_name")["p"].transform("mean")
        df["team_form"] = df["team_win_rate"]

    # 5. Enhanced outlier clipping for better small change handling
    print("Applying enhanced outlier treatment...")
    Q1 = df["mv_target"].quantile(0.15)  # More aggressive clipping
    Q3 = df["mv_target"].quantile(0.85)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2.0 * IQR  # Less aggressive bounds to preserve small changes
    upper_bound = Q3 + 2.0 * IQR

    df["mv_target_clipped"] = df["mv_target"].clip(lower_bound, upper_bound)

    # 6. Comprehensive missing value handling
    print("Handling missing values...")
    fill_values = {
        "market_divergence": 1,
        "mv_change_3d": 0, "mv_change_7d": 0,
        "mv_vol_3d": 0, "mv_vol_7d": 0,
        "mv_micro_trend": 0, "mv_stability": 1, "mv_recent_direction": 0,
        "mv_percentile": 0.5, "mv_zscore": 0,
        "p": 0, "ppm": 0, "mp": 0, "won": -1,
        "points_ma_3": 0, "points_ma_5": 0, "points_trend": 0, "points_consistency": 0,
        "mp_ma_3": 0, "mp_consistency": 0, "mp_trend": 0,
        "ppm_ma_3": 0, "ppm_trend": 0, "ppm_volatility": 0,
        "win_rate_3": 0, "win_rate_5": 0, "recent_form": 0,
        "p_vs_position": 0, "mv_vs_position": 0, "ppm_vs_position": 0,
        "mv_momentum_short": 0, "mv_momentum_long": 0, "mv_momentum_very_long": 0, "mv_acceleration": 0,
        "mv_relative_change": 0, "mv_price_pressure": 0, "form_mv_interaction": 0,
        "team_win_rate": 0.5, "team_avg_points": 0, "team_form": 0.5
    }
    df = df.fillna(fill_values)

    # 7. Cutout todays values and store them (optimized)
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    cutoff_time = now.replace(hour=22, minute=15, second=0, microsecond=0)
    max_date = (now - timedelta(days=1)) if now <= cutoff_time else now
    max_date = max_date.date()

    today_df = df[df["date"].dt.date >= max_date].copy()
    df = df[df["date"].dt.date < max_date].copy()

    # 8. Drop rows with NaN in critical columns
    critical_columns = ["mv_change_1d", "next_day", "next_md", "days_to_next", "mv_next_day", "mv_target", "mv_target_clipped"]
    df = df.dropna(subset=critical_columns)

    print(f"Preprocessing complete! Training data: {len(df):,} rows, Today's data: {len(today_df):,} rows")
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