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
    print_header, print_success, print_info, print_warning, print_error,
    display_dataframe, print_model_evaluation, print_separator, print_feature_importance,
    print_step, operation_timer, suppress_sklearn_warnings, print_model_warning, print_network_error,
    print_trading_tips, print_market_summary, print_prediction_methodology, print_data_freshness_info
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
    "mv_change_3d", "mv_change_7d", "mv_vol_3d", "mv_vol_7d",
    "mv_trend_7d", "market_divergence",
    # Enhanced features for better predictions
    "points_ma_3", "points_ma_5", "points_trend", "points_consistency",
    "mp_ma_3", "mp_consistency", "mp_trend",
    "ppm_ma_3", "ppm_trend", "ppm_volatility",
    "win_rate_3", "win_rate_5", "recent_form",
    "p_vs_position", "mv_vs_position", "ppm_vs_position",
    "mv_momentum_short", "mv_momentum_long", "mv_momentum_very_long", "mv_acceleration",
    # NEW: Small change specific features
    "mv_micro_trend", "mv_stability", "mv_recent_direction",
    "mv_percentile", "mv_zscore",
    "mv_relative_change", "mv_price_pressure", "form_mv_interaction",
    "team_win_rate", "team_avg_points", "team_form"
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

# Performance mode: set FAST_MODE=1 for faster training (useful for development/testing)
FAST_MODE = os.getenv("FAST_MODE", "0") == "1"

print_header("🏈 Kickbase Trading Advisor", "Analyzing market opportunities and team performance")
print_separator()

# Suppress sklearn warnings to clean up console output
suppress_sklearn_warnings()

print_step("Authentication", "Logging into Kickbase platform")
try:
    with operation_timer("Login"):
        token = login(USERNAME, PASSWORD)
    print_success("Successfully logged in to Kickbase")
except Exception as e:
    if "ConnectionError" in str(type(e)) or "NameResolutionError" in str(e):
        print_network_error(str(e))
    else:
        print_error(f"Login failed: {e}")
    raise

# Get league ID
print_step("League Analysis", "Fetching league information and manager budgets")
with operation_timer("League data retrieval"):
    league_id = get_league_id(token, league_name)

# Calculate (estimated) budgets of all managers in the league
with operation_timer("Budget calculation"):
    manager_budgets_df = calc_manager_budgets(token, league_id, league_start_date, start_budget)
display_dataframe(manager_budgets_df, "💰 Manager Budgets", max_rows=20)

print_separator()

# Data handling
print_step("Data Management", "Setting up database and loading player data")
create_player_data_table()
reload_data = check_if_data_reload_needed()

with operation_timer("Data loading and processing"):
    save_player_data_to_db(token, competition_ids, last_mv_values, last_pfm_values, reload_data)
    player_df = load_player_data_from_db()
print_success("Data loaded from database")

# Preprocess the data and split the data
print_step("Data Preprocessing", f"Cleaning and preparing data for machine learning ({'Fast' if FAST_MODE else 'Enhanced'} mode)")
with operation_timer("Data preprocessing"):
    proc_player_df, today_df = preprocess_player_data(player_df, fast_mode=FAST_MODE)
    X_train, X_test, y_train, y_test = split_data(proc_player_df, features, target)

# Data quality checks
print_info(f"Training data: {len(X_train):,} samples with {len(features)} features")
print_info(f"Test data: {len(X_test):,} samples")
if len(today_df) > 0:
    print_info(f"Current market data: {len(today_df):,} players available for prediction")
else:
    print_warning("No current market data available - predictions may be limited")

print_success("Data preprocessed successfully")

# Train and evaluate the model
print_step("Model Training", f"Training ensemble machine learning model ({'Fast' if FAST_MODE else 'Full'} mode)")
with operation_timer("Model training"):
    model = train_model(X_train, y_train, fast_mode=FAST_MODE)

print_step("Model Evaluation", "Testing model performance on unseen data")
with operation_timer("Model evaluation"):
    results = evaluate_model(model, X_test, y_test)

# Handle both old and new evaluation function signatures
if len(results) == 8:
    signs_percent, rmse, mae, r2, mape, small_acc, medium_acc, large_acc = results
    print_model_evaluation(signs_percent, rmse, mae, r2, mape, small_acc, medium_acc, large_acc)
else:
    signs_percent, rmse, mae, r2 = results
    print_model_evaluation(signs_percent, rmse, mae, r2)

# Check for mathematical warnings and provide context
if results[3] < 0.1:  # Low R² score
    print_model_warning("low_r2", f"R² = {results[3]:.3f} indicates model may need improvement")

# Display feature importance if available
if hasattr(model, 'get_feature_importance'):
    feature_importance = model.get_feature_importance()
    print_separator()
    print_feature_importance(feature_importance)

# Cross-validation for better model assessment
print_separator()
print_step("Model Validation", "Running cross-validation to assess model stability")
try:
    with operation_timer("Cross-validation"):
        cv_rmse_mean, cv_rmse_std = cross_validate_model(X_train, y_train)
    print_success(f"Cross-validation RMSE: {cv_rmse_mean:.2f} ± {cv_rmse_std:.2f}")
except Exception as e:
    print_warning(f"Cross-validation failed: {e}")

# Backtesting simulation
print_separator()
print_step("Strategy Backtesting", "Testing trading strategies on historical data")
try:
    # Define strategies to test
    strategies = [
        ThresholdStrategy(buy_threshold=30000, sell_threshold=-20000, max_hold_days=21),
        ThresholdStrategy(buy_threshold=50000, sell_threshold=-30000, max_hold_days=14),
        ThresholdStrategy(buy_threshold=75000, sell_threshold=-40000, max_hold_days=10)
    ]
    
    print_info("Testing 3 different trading strategies on last 90 days of data")
    
    # Run comparison on recent data (last 90 days)
    recent_date = proc_player_df['date'].max() - pd.Timedelta(days=90)
    
    with operation_timer("Strategy backtesting"):
        strategy_results = run_strategy_comparison(
            proc_player_df[proc_player_df['date'] >= recent_date], 
            model, features, strategies
        )
    
    if not strategy_results.empty:
        display_dataframe(strategy_results[['strategy', 'total_return_pct', 'win_rate_pct', 'max_drawdown_pct', 'sharpe_ratio', 'total_trades']], 
                         "📊 Strategy Backtesting Results")
    else:
        print_warning("No backtesting results generated")
except Exception as e:
    print_error(f"Backtesting failed: {e}")

print_separator()

# Make live data predictions
print_step("Live Predictions", "Generating predictions for current player values")
with operation_timer("Live prediction generation"):
    live_predictions_df = live_data_predictions(today_df, model, features)

# Make multi-horizon predictions (1-day, 3-day, 7-day)
print_step("Multi-Horizon Analysis", "Creating predictions for multiple time horizons (1, 3, 7 days)")
with operation_timer("Multi-horizon predictions"):
    multi_horizon_df = multi_horizon_predictions(today_df, model, features)

# Enhanced display with custom column descriptions
column_descriptions = {
    'last_name': 'Player\nName',
    'team_name': 'Team',
    'mv': 'Current\nValue (€)',
    'predicted_mv_1d': 'Predicted\n1-Day (€)',
    'predicted_mv_3d': 'Predicted\n3-Day (€)', 
    'predicted_mv_7d': 'Predicted\n7-Day (€)',
    'prediction_confidence': 'Confidence\nScore',
    'risk_score': 'Risk Level\n(0-1)'
}

display_dataframe(
    multi_horizon_df[["last_name", "team_name", "mv", "predicted_mv_1d", "predicted_mv_3d", "predicted_mv_7d", "prediction_confidence", "risk_score"]].head(15), 
    "🔮 Multi-Horizon Market Value Predictions", 
    max_rows=15,
    show_insights=True,
    column_descriptions=column_descriptions
)

print_separator()

# Join with current available players on the market
print_step("Market Analysis", "Analyzing current market opportunities and generating buy recommendations")
with operation_timer("Market analysis"):
    market_recommendations_df = join_current_market(token, league_id, live_predictions_df)

# Enhanced market display with better descriptions
market_column_descriptions = {
    'last_name': 'Player\nName',
    'team_name': 'Team',
    'mv': 'Current\nValue (€)',
    'mv_change_yesterday': 'Yesterday\nChange (€)',
    'predicted_mv_target': 'Predicted\nGain (€)',
    's_11_prob': 'Start XI\nProb (%)',
    'hours_to_exp': 'Time Left\n(Hours)',
    'expiring_today': 'Expires\nToday?',
    'risk_score': 'Risk Level\n(0-1)',
    'investment_grade': 'Investment\nRecommendation'
}

display_dataframe(
    market_recommendations_df, 
    "📈 Transfer Market Opportunities", 
    max_rows=20,
    show_insights=True,
    column_descriptions=market_column_descriptions
)

print_separator()

# Join with current players on the team
print_step("Squad Analysis", "Evaluating current squad performance and identifying sell opportunities")
with operation_timer("Squad analysis"):
    squad_recommendations_df = join_current_squad(token, league_id, live_predictions_df)

# Enhanced squad display
squad_column_descriptions = {
    'last_name': 'Player\nName',
    'team_name': 'Team',
    'mv': 'Current\nValue (€)',
    'mv_change_yesterday': 'Yesterday\nChange (€)',
    'predicted_mv_target': 'Predicted\nChange (€)',
    's_11_prob': 'Start XI\nProb (%)'
}

display_dataframe(
    squad_recommendations_df, 
    "⚽ Your Squad Performance Analysis", 
    max_rows=20,
    show_insights=True,
    column_descriptions=squad_column_descriptions
)

# Display comprehensive market summary
print_separator()
print_market_summary(market_recommendations_df, squad_recommendations_df, manager_budgets_df)

# Display trading methodology and tips
print_separator()
print_prediction_methodology()

print_separator()
print_trading_tips()

print_separator()
print_data_freshness_info()

# Send email with recommendations
print_separator()
print_step("Report Generation", "Sending comprehensive analysis report via email")
with operation_timer("Email report generation"):
    send_mail(manager_budgets_df, market_recommendations_df, squad_recommendations_df, email)

print_success("Analysis complete! All predictions and recommendations have been generated. 🎉")
print_info("Check your email for the detailed report with actionable recommendations.")
