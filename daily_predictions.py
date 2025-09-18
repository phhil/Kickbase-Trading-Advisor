from features.predictions.predictions import live_data_predictions, join_current_market, join_current_squad, multi_horizon_predictions
from features.predictions.preprocessing import preprocess_player_data, split_data
from features.predictions.modeling import train_model, evaluate_model, cross_validate_model
from features.predictions.simulation import ThresholdStrategy, backtest_strategy, run_strategy_comparison
from kickbase_api.league import get_league_id
from kickbase_api.user import login
from features.notifier import send_mail
from features.predictions.data_handler import (
    create_player_data_table,
    check_if_data_reload_needed,
    save_player_data_to_db,
    load_player_data_from_db,
)
from features.budgets import calc_manager_budgets
from features.console_formatter import (
    print_header, print_success, print_info, print_warning, 
    display_dataframe, print_model_evaluation, print_separator, print_feature_importance
)
from IPython.display import display
from dotenv import load_dotenv
import os, pandas as pd

# Load environment variables from .env file
load_dotenv() 

# ----------------- Notes & TODOs -----------------

# TODO Fix the UTC timezone problems in the github actions scheduling
# TODO Add prediction of 3, 7 days, to give more context
# TODO Based upon the overpay of the other users, calculate a max price to pay for a player
# TODO Add features like starting 11 probability, injuries, ...
# TODO Improve budget calculation, weird bug that for me the budgets is 513929 off, idk why, checked everything

# ----------------- SYSTEM PARAMETERS -----------------
# Should be left unchanged unless you know what you're doing

last_mv_values = 365    # in days, max 365
last_pfm_values = 50    # in matchdays, max idk

# which features to use for training and prediction
features = [
    # Original features
    "p", "mv", "days_to_next", 
    "mv_change_1d", "mv_trend_1d", 
    "mv_change_3d", "mv_vol_3d",
    "mv_trend_7d", "market_divergence",
    # Enhanced features for better predictions
    "points_ma_3", "points_ma_5", "points_trend",
    "mp_ma_3", "mp_consistency",
    "ppm_ma_3", "ppm_trend",
    "win_rate_3", "recent_form",
    "p_vs_position", "mv_vs_position", "ppm_vs_position",
    "mv_momentum_short", "mv_momentum_long", "mv_acceleration"
]

# what column to learn and predict on
target = "mv_target_clipped"

# Set dot as thousands separator for better readability
pd.options.display.float_format = lambda x: '{:,.0f}'.format(x).replace(',', '.')

# Show all columns when displaying dataframes
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)

# ----------------- USER SETTINGS -----------------
# Adjust these settings to your preferences

competition_ids = [1]                   # 1 = Bundesliga, 2 = 2. Bundesliga, 3 = La Liga
league_name = "Cafefull 2.0"  # Name of your league, must be exact match, can be done via env or hardcoded
start_budget = 50_000_000               # Starting budget of your league, used to calculate current budgets of other managers
league_start_date = "2025-08-10"        # Start date of your league, used to filter activities, format: YYYY-MM-DD
email = os.getenv("EMAIL_USER")         # Email to send recommendations to, can be the same as EMAIL_USER or different

# ---------------------------------------------------

# Load environment variables and login to kickbase
USERNAME = os.getenv("KICK_USER") # DO NOT CHANGE THIS, YOU MUST SET THOSE IN GITHUB SECRETS OR A .env FILE
PASSWORD = os.getenv("KICK_PASS") # DO NOT CHANGE THIS, YOU MUST SET THOSE IN GITHUB SECRETS OR A .env FILE

print_header("🏈 Kickbase Trading Advisor", "Analyzing market opportunities and team performance")
print_separator()

token = login(USERNAME, PASSWORD)
print_success("Successfully logged in to Kickbase")

# Get league ID
league_id = get_league_id(token, league_name)

# Calculate (estimated) budgets of all managers in the league
manager_budgets_df = calc_manager_budgets(token, league_id, league_start_date, start_budget)
display_dataframe(manager_budgets_df, "💰 Manager Budgets", max_rows=20)

print_separator()

# Data handling
create_player_data_table()
reload_data = check_if_data_reload_needed()
save_player_data_to_db(token, competition_ids, last_mv_values, last_pfm_values, reload_data)
player_df = load_player_data_from_db()
print_success("Data loaded from database")

# Preprocess the data and spit the data
proc_player_df, today_df = preprocess_player_data(player_df)
X_train, X_test, y_train, y_test = split_data(proc_player_df, features, target)
print_success("Data preprocessed successfully")

# Train and evaluate the model
model = train_model(X_train, y_train)
results = evaluate_model(model, X_test, y_test)

# Handle both old and new evaluation function signatures
if len(results) == 8:
    signs_percent, rmse, mae, r2, mape, small_acc, medium_acc, large_acc = results
    print_model_evaluation(signs_percent, rmse, mae, r2, mape, small_acc, medium_acc, large_acc)
else:
    signs_percent, rmse, mae, r2 = results
    print_model_evaluation(signs_percent, rmse, mae, r2)

# Display feature importance if available
if hasattr(model, 'get_feature_importance'):
    feature_importance = model.get_feature_importance()
    print_separator()
    print_feature_importance(feature_importance)

# Cross-validation for better model assessment
print_separator()
print_info("Running cross-validation...")
try:
    cv_rmse_mean, cv_rmse_std = cross_validate_model(X_train, y_train)
    print_success(f"Cross-validation RMSE: {cv_rmse_mean:.2f} ± {cv_rmse_std:.2f}")
except Exception as e:
    print_warning(f"Cross-validation failed: {e}")

# Backtesting simulation
print_separator()
print_info("Running backtesting simulation...")
try:
    # Define strategies to test
    strategies = [
        ThresholdStrategy(buy_threshold=30000, sell_threshold=-20000, max_hold_days=21),
        ThresholdStrategy(buy_threshold=50000, sell_threshold=-30000, max_hold_days=14),
        ThresholdStrategy(buy_threshold=75000, sell_threshold=-40000, max_hold_days=10)
    ]
    
    # Run comparison on recent data (last 90 days)
    recent_date = proc_player_df['date'].max() - pd.Timedelta(days=90)
    strategy_results = run_strategy_comparison(
        proc_player_df[proc_player_df['date'] >= recent_date], 
        model, features, strategies
    )
    
    if not strategy_results.empty:
        display_dataframe(strategy_results[['strategy', 'total_return_pct', 'win_rate_pct', 'max_drawdown_pct', 'sharpe_ratio', 'total_trades']], 
                         "📊 Strategy Backtesting Results")
except Exception as e:
    print_warning(f"Backtesting failed: {e}")

print_separator()

# Make live data predictions
live_predictions_df = live_data_predictions(today_df, model, features)

# Make multi-horizon predictions (1-day, 3-day, 7-day)
multi_horizon_df = multi_horizon_predictions(today_df, model, features)
display_dataframe(multi_horizon_df[["last_name", "team_name", "mv", "predicted_mv_1d", "predicted_mv_3d", "predicted_mv_7d", "prediction_confidence", "risk_score"]].head(10), 
                 "🔮 Multi-Horizon Predictions", max_rows=10)

print_separator()

# Join with current available players on the market
market_recommendations_df = join_current_market(token, league_id, live_predictions_df)
display_dataframe(market_recommendations_df, "📈 Market Recommendations", max_rows=15)

print_separator()

# Join with current players on the team
squad_recommendations_df = join_current_squad(token, league_id, live_predictions_df)
display_dataframe(squad_recommendations_df, "⚽ Squad Analysis", max_rows=15)

# Send email with recommendations
print_separator()
send_mail(manager_budgets_df, market_recommendations_df, squad_recommendations_df, email)
print_info("Analysis complete! 🎉")
