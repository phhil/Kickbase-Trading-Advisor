"""
Kickbase Budget Analyzer - Standalone Script

This script provides a focused analysis of manager budgets within your Kickbase league,
without running any machine learning predictions or trading simulations.

Features:
- Fetches and calculates estimated budgets for all league managers
- Analyzes team values, bonuses, and trading activities
- Displays comprehensive budget breakdown
- Much faster execution compared to the full daily_predictions.py script

Usage:
- Run locally: python budgets_only.py
- Run via GitHub Actions: Use the "Budget Analysis Only" workflow
- Configure via environment variables or modify the USER SETTINGS section below
"""

from kickbase_api.league import get_league_id
from kickbase_api.user import login
from features.budgets import calc_manager_budgets
from features.console_formatter import (
    print_header, print_success, print_info, print_error,
    display_dataframe, print_separator, print_step, operation_timer, print_network_error
)
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv() 

# ----------------- USER SETTINGS -----------------
# Adjust these settings to your preferences

league_name = os.getenv("LEAGUE_NAME", "Cafefull 2.0")  # Name of your league, must be exact match, can be done via env or hardcoded
start_budget = int(os.getenv("START_BUDGET", "50000000"))  # Starting budget of your league, used to calculate current budgets of other managers
league_start_date = os.getenv("LEAGUE_START_DATE", "2025-08-10")  # Start date of your league, used to filter activities, format: YYYY-MM-DD

# ---------------------------------------------------

# Load environment variables and login to kickbase
USERNAME = os.getenv("KICK_USER") # DO NOT CHANGE THIS, YOU MUST SET THOSE IN GITHUB SECRETS OR A .env FILE
PASSWORD = os.getenv("KICK_PASS") # DO NOT CHANGE THIS, YOU MUST SET THOSE IN GITHUB SECRETS OR A .env FILE

print_header("💰 Kickbase Budget Analyzer", "Fetching and analyzing manager budgets")
print_separator()

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
print_success("Budget analysis complete! 🎉")
print_info("Manager budgets have been calculated and displayed above.")