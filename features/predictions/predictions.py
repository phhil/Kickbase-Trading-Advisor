from kickbase_api.league import get_league_players_on_market
from kickbase_api.user import get_players_in_squad
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

def live_data_predictions(today_df, model, features):
    """Make live data predictions for today_df using the trained model"""

    # Set features and copy df
    today_df_features = today_df[features]
    today_df_results = today_df.copy()

    # Predict mv_target
    today_df_results["predicted_mv_target"] = np.round(model.predict(today_df_features), 2)

    # Sort by predicted_mv_target descending
    today_df_results = today_df_results.sort_values("predicted_mv_target", ascending=False)

    # Filter date to today or yesterday if before 22:15, because mv is updated around 22:15
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    cutoff_time = now.replace(hour=22, minute=15, second=0, microsecond=0)
    date = (now - timedelta(days=1)) if now <= cutoff_time else now
    date = date.date()

    # Drop rows where NaN mv
    today_df_results = today_df_results.dropna(subset=["mv"])

    # Keep only relevant columns
    today_df_results = today_df_results[["player_id", "first_name", "last_name", "position", "team_name", "date", "mv_change_1d", "mv_trend_1d", "mv", "predicted_mv_target"]]

    return today_df_results


def multi_horizon_predictions(today_df, model, features, horizons=[1, 3, 7]):
    """Make multi-horizon predictions (1-day, 3-day, 7-day)"""
    
    today_df_features = today_df[features]
    results = today_df.copy()
    
    # Base prediction (1-day)
    base_prediction = model.predict(today_df_features)
    results["predicted_mv_1d"] = np.round(base_prediction, 2)
    
    # Multi-horizon predictions using different approaches
    for horizon in horizons:
        if horizon == 1:
            continue
            
        # For longer horizons, apply scaling based on historical volatility patterns
        # This is a simplified approach - in practice, you'd want separate models
        if horizon == 3:
            # 3-day prediction: scale by 1.5x with some dampening
            scaling_factor = 1.4
            volatility_factor = np.random.normal(1, 0.1, len(base_prediction))  # Add some uncertainty
        elif horizon == 7:
            # 7-day prediction: scale by 2x with more dampening
            scaling_factor = 1.8
            volatility_factor = np.random.normal(1, 0.15, len(base_prediction))  # More uncertainty
        else:
            scaling_factor = 1.0
            volatility_factor = 1.0
            
        prediction = base_prediction * scaling_factor * volatility_factor
        results[f"predicted_mv_{horizon}d"] = np.round(prediction, 2)
    
    # Calculate prediction confidence/risk
    if hasattr(model, 'models') and 'rf' in model.models:
        from features.predictions.modeling import get_prediction_confidence
        pred_std, conf_lower, conf_upper = get_prediction_confidence(model, today_df_features)
        results["prediction_confidence"] = np.round(1 / (pred_std + 1e-6), 2)  # Higher = more confident
        results["risk_score"] = np.round(pred_std / (np.abs(base_prediction) + 1e-6), 3)  # Higher = more risky
    
    # Sort by 1-day prediction
    results = results.sort_values("predicted_mv_1d", ascending=False)
    
    # Drop rows where NaN mv
    results = results.dropna(subset=["mv"])
    
    return results


def join_current_squad(token, league_id, today_df_results):
    squad_players = get_players_in_squad(token, league_id)

    squad_df = pd.DataFrame(squad_players["it"])

    # Join squad_df ("i") with today_df ("player_id")
    squad_df = (
        pd.merge(today_df_results, squad_df, left_on="player_id", right_on="i")
        .drop(columns=["i"])
    )

    # Rename prob to s_11_prob for better understanding
    if "prob" not in squad_df.columns:
        squad_df["prob"] = np.nan  # Placeholder for non-pro users
    squad_df = squad_df.rename(columns={"prob": "s_11_prob"})

    # Rename mv_change_1d to mv_change_yesterday for better understanding
    squad_df = squad_df.rename(columns={"mv_change_1d": "mv_change_yesterday"})

    # Rename "mv_x" to "mv" for better understanding
    squad_df = squad_df.rename(columns={"mv_x": "mv"})

    # Keep only relevant columns
    squad_df = squad_df[["last_name", "team_name", "mv", "mv_change_yesterday", "predicted_mv_target", "s_11_prob"]]

    return squad_df 


# TODO Add fail-safe check before player expires if the prob (starting 11) is still high, so no injuries or anything. if it dropped. dont bid / reccommend
def join_current_market(token, league_id, today_df_results):
    """Join the live predictions with the current market data to get bid recommendations"""

    players_on_market = get_league_players_on_market(token, league_id)

    # players_on_market to DataFrame
    market_df = pd.DataFrame(players_on_market)

    # Join market_df ("id") with today_df ("player_id")
    bid_df = (
        pd.merge(today_df_results, market_df, left_on="player_id", right_on="id")
        .drop(columns=["id"])
    )

    # exp contains seconds until expiration
    bid_df["hours_to_exp"] = np.round((bid_df["exp"] / 3600), 2)

    # check if current sysdate + hours_to_exp is after the next 22:00
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    next_22 = now.replace(hour=22, minute=0, second=0, microsecond=0)
    diff = np.round((next_22 - now).total_seconds() / 3600, 2)

    # If hours_to_exp < diff then it expires today
    bid_df["expiring_today"] = bid_df["hours_to_exp"] < diff

    # Drop rows where predicted_mv_target is less than 5000
    bid_df = bid_df[bid_df["predicted_mv_target"] > 5000]

    # Sort by predicted_mv_target descending
    bid_df = bid_df.sort_values("predicted_mv_target", ascending=False)

    # Rename prob to s_11_prob for better understanding
    if "prob" not in bid_df.columns:
        bid_df["prob"] = np.nan  # Placeholder for non-pro users
    bid_df = bid_df.rename(columns={"prob": "s_11_prob"})

    # Rename mv_change_1d to mv_change_yesterday for better understanding
    bid_df = bid_df.rename(columns={"mv_change_1d": "mv_change_yesterday"})

    # Enhanced market analysis with volatility assessment
    if "risk_score" in today_df_results.columns:
        bid_df["risk_score"] = bid_df["player_id"].map(
            today_df_results.set_index("player_id")["risk_score"]
        ).fillna(0.5)
    
    # Add investment recommendation based on prediction and risk
    def get_investment_grade(row):
        pred = row.get("predicted_mv_target", 0)
        risk = row.get("risk_score", 0.5)
        
        if pred > 75000 and risk < 0.3:
            return "🟢 Strong Buy"
        elif pred > 50000 and risk < 0.4:
            return "🔵 Buy"
        elif pred > 25000:
            return "🟡 Hold/Watch"
        elif pred < -25000:
            return "🔴 Avoid"
        else:
            return "⚪ Neutral"
    
    bid_df["investment_grade"] = bid_df.apply(get_investment_grade, axis=1)

    # Keep only relevant columns including new analysis
    columns_to_keep = ["last_name", "team_name", "mv", "mv_change_yesterday", "predicted_mv_target", "s_11_prob", "hours_to_exp", "expiring_today"]
    if "risk_score" in bid_df.columns:
        columns_to_keep.append("risk_score")
    if "investment_grade" in bid_df.columns:
        columns_to_keep.append("investment_grade")
    
    bid_df = bid_df[columns_to_keep]

    return bid_df