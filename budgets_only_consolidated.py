#!/usr/bin/env python3
"""
Kickbase Budget Analyzer - Consolidated Single File Script

This script provides a focused analysis of manager budgets within your Kickbase league,
without running any machine learning predictions or trading simulations.

Features:
- Fetches and calculates estimated budgets for all league managers
- Analyzes team values, bonuses, and trading activities
- Displays comprehensive budget breakdown
- Much faster execution compared to the full daily_predictions.py script
- All dependencies consolidated into a single file for easy copy/paste usage

Usage:
- Run locally: python budgets_only_consolidated.py
- Configure the USER SETTINGS section below with your league details
- Set KICK_USER and KICK_PASS environment variables or modify credentials section

Requirements:
- pandas, requests, rich, python-dotenv (install with: pip install pandas requests rich python-dotenv)

What's included in this consolidated script:
- Complete Kickbase API client (login, league data, manager info, etc.)
- Budget calculation engine with trading activity analysis
- Achievement and bonus calculation logic
- Rich console formatting for beautiful output
- Error handling and network connectivity checks
- All functions from: kickbase_api/, features/budgets.py, features/console_formatter.py
"""

# ===============================================================================
# IMPORTS
# ===============================================================================
import requests
import pandas as pd
import os
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.align import Align
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv() 

# ===============================================================================
# CONFIGURATION & CONSTANTS
# ===============================================================================

# Kickbase API base URL
BASE_URL = "https://api.kickbase.com/v4"

# Initialize Rich console for formatting
console = Console()

# ===============================================================================
# USER SETTINGS - MODIFY THESE
# ===============================================================================
# Adjust these settings to your preferences

league_name = os.getenv("LEAGUE_NAME", "Cafefull 2.0")  # Name of your league, must be exact match
start_budget = int(os.getenv("START_BUDGET", "50000000"))  # Starting budget of your league
league_start_date = os.getenv("LEAGUE_START_DATE", "2025-08-10")  # Start date of your league, format: YYYY-MM-DD

# Load environment variables for login - DO NOT CHANGE THESE VARIABLE NAMES
# You must set these in environment variables or a .env file
USERNAME = os.getenv("KICK_USER")  # Your Kickbase email/username
PASSWORD = os.getenv("KICK_PASS")  # Your Kickbase password

# ===============================================================================
# KICKBASE API FUNCTIONS
# ===============================================================================

def get_json_with_token(url, token):
    """Fetch JSON data from a given URL using token for authorization."""
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()

def login(username, password):
    """Logs in to Kickbase and returns the authentication token."""
    url = f"{BASE_URL}/user/login"
    payload = {
        "em": username,
        "pass": password,
        "loy": False,
        "rep": {}
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()
    token = data.get("tkn")
    return token

def get_username(token):
    """Gets the username of the logged-in user."""
    url = f"{BASE_URL}/user/settings"
    data = get_json_with_token(url, token)
    username = data["u"]["unm"]
    return username

def get_budget(token, league_id):
    """Gets the user's budget for a given league."""
    url = f"{BASE_URL}/leagues/{league_id}/me/budget"
    data = get_json_with_token(url, token)
    data = data["b"]
    return data

def get_league_id(token, league_name):
    """Get the league ID based on the league name."""
    league_infos = get_leagues_infos(token)

    if not league_infos:
        print("Warning: You are not part of any league.")
        return None

    # Try to find leagues matching the given name
    selected_league = [league for league in league_infos if league["name"] == league_name]

    # If no exact match found, fall back to the first available league
    if not selected_league:
        fallback_league = league_infos[0]
        print(
            f"Warning: No league found with name '{league_name}'. "
            f"Falling back to the first available league: '{fallback_league['name']}'"
        )
        return fallback_league["id"]

    return selected_league[0]["id"]

def get_leagues_infos(token):
    """Get information about all leagues the user is part of."""
    url = f"{BASE_URL}/leagues/selection"
    data = get_json_with_token(url, token)

    result = []
    for item in data.get("it", []):
        result.append({
            "id": item.get("i"),
            "name": item.get("n")
        })

    return result

def get_league_activities(token, league_id, league_start_date):
    """Get league activities such as trades, logins, and achievements since the league start date."""
    # TODO magic number with 5000, have to find a better solution
    url = f"{BASE_URL}/leagues/{league_id}/activitiesFeed?max=5000"
    data = get_json_with_token(url, token)

    # Filter out entries prior to reset_Date
    filtered_activities = []
    for entry in data["af"]:
        entry_date = entry.get("dt", "")
        if entry_date >= league_start_date:
            filtered_activities.append(entry)

    login = [entry for entry in filtered_activities if entry.get("t") == 22]
    achievements = [entry for entry in filtered_activities if entry.get("t") == 26]
    trade = [entry for entry in filtered_activities if entry.get("t") == 15]
    trading = [
        {k: entry["data"].get(k) for k in ["byr", "slr", "pi", "pn", "tid", "trp"]}
        for entry in trade
        if entry.get("t") == 15
    ]

    return trading, login, achievements

def get_league_ranking(token, league_id):
    """Get the overall league ranking."""
    url = f"{BASE_URL}/leagues/{league_id}/ranking"
    data = get_json_with_token(url, token)

    players = [(user["n"], user["sp"]) for user in data["us"]]

    # Sort by score (descending)
    ranked = sorted(players, key=lambda x: x[1], reverse=True)

    return ranked

def get_managers(token, league_id):
    """Get a list of all managers in the league with their IDs and names."""
    url = f"{BASE_URL}/leagues/{league_id}/ranking"
    data = get_json_with_token(url, token)

    user_info = [(user["n"], user["i"]) for user in data["us"]]

    return user_info

def get_manager_info(token, league_id, manager_id):
    """Get detailed information about a specific manager in the league."""
    url = f"{BASE_URL}/leagues/{league_id}/managers/{manager_id}/dashboard"
    data = get_json_with_token(url, token)

    return data

def get_manager_performance(token, league_id, manager_id, manager_name):
    """Get performance data for a specific manager in the league."""
    url = f"{BASE_URL}/leagues/{league_id}/managers/{manager_id}/performance"
    data = get_json_with_token(url, token)
    
    # Look for season ID "34" (current season 2025/2026)
    tp_value = 0
    for season in data["it"]:
        if season["sid"] == "34":
            tp_value = season["tp"]
            break
    else:
        # Fallback to first season if sid "34" not found
        tp_value = data["it"][0]["tp"]
        print(f"Warning: Season ID '34' not found for {manager_name}, using first season")

    return {
        "name": manager_name,
        "tp": tp_value
    }

def get_achievement_reward(token, league_id, achievement_id):
    """Get the reward and how often this was achieved by the user for a specific achievement in a league."""
    url = f"{BASE_URL}/leagues/{league_id}/user/achievements/{achievement_id}"
    data = get_json_with_token(url, token)

    amount = data.get("ac", 0) or 0  # Ensure we get 0 instead of None
    reward = data.get("er", 0) or 0  # Ensure we get 0 instead of None

    return amount, reward

# ===============================================================================
# CONSOLE FORMATTING FUNCTIONS
# ===============================================================================

def print_header(title, subtitle=None):
    """Print a formatted section header"""
    if subtitle:
        header_text = f"[bold cyan]{title}[/bold cyan]\n[dim]{subtitle}[/dim]"
    else:
        header_text = f"[bold cyan]{title}[/bold cyan]"
    
    panel = Panel(
        Align.center(header_text), 
        style="blue", 
        padding=(0, 1)
    )
    console.print(panel)

def print_success(message):
    """Print a success message in green"""
    console.print(f"✅ [bold green]{message}[/bold green]")

def print_info(message):
    """Print an info message in blue"""
    console.print(f"ℹ️  [bold blue]{message}[/bold blue]")

def print_error(message):
    """Print an error message in red"""
    console.print(f"❌ [bold red]{message}[/bold red]")

def print_warning(message):
    """Print a warning message in yellow"""
    console.print(f"⚠️  [bold yellow]{message}[/bold yellow]")

def print_separator():
    """Print a visual separator"""
    console.print("\n" + "─" * 80 + "\n")

def print_step(step_name, description=None):
    """Print a step indicator with optional description"""
    if description:
        console.print(f"[bold cyan]🔄 {step_name}[/bold cyan]")
        console.print(f"[dim]   {description}[/dim]")
    else:
        console.print(f"[bold cyan]🔄 {step_name}[/bold cyan]")

def print_timing_info(operation_name, duration_seconds):
    """Print timing information for operations"""
    if duration_seconds < 60:
        time_str = f"{duration_seconds:.1f} seconds"
    else:
        minutes = int(duration_seconds // 60)
        seconds = duration_seconds % 60
        time_str = f"{minutes}m {seconds:.1f}s"
    
    console.print(f"[dim]⏱️  {operation_name} completed in {time_str}[/dim]")

@contextmanager
def operation_timer(operation_name, show_progress=True):
    """Context manager for timing operations with optional progress indicator"""
    start_time = time.time()
    
    if show_progress:
        with console.status(f"[bold green]{operation_name}...") as status:
            try:
                yield
            finally:
                duration = time.time() - start_time
                print_timing_info(operation_name, duration)
    else:
        try:
            yield
        finally:
            duration = time.time() - start_time
            print_timing_info(operation_name, duration)

def print_network_error(error_details):
    """Handle network-related errors with helpful context"""
    print_error("Network connection failed")
    console.print(f"[dim]   Error details: {error_details}[/dim]")
    console.print("[dim]   This usually means:[/dim]")
    console.print("[dim]   • No internet connection available[/dim]")
    console.print("[dim]   • Kickbase API is temporarily unavailable[/dim]")
    console.print("[dim]   • Firewall blocking the connection[/dim]")
    console.print("[dim]   Suggestion: Check your internet connection and try again[/dim]")

def format_currency(value):
    """Format currency values with colors"""
    if pd.isna(value):
        return "N/A"
    
    # Format with dots as thousand separators
    formatted = f"{value:,.0f}".replace(',', '.')
    
    if value > 0:
        return f"[green]{formatted}[/green]"
    elif value < 0:
        return f"[red]{formatted}[/red]"
    else:
        return formatted

def _format_cell_value(col, value):
    """Format individual cell values based on column type"""
    if pd.isna(value):
        return "[dim]N/A[/dim]"
    
    col_lower = col.lower()
    
    # Currency formatting
    if col_lower in ['budget', 'team value', 'available budget', 'max negative']:
        return format_currency(value)
    
    # Default formatting
    else:
        return str(value)

def display_dataframe(df, title=None, max_rows=None):
    """Display a pandas DataFrame as a formatted table with enhanced formatting"""
    if df.empty:
        print_warning("No data to display")
        return
    
    # Limit rows if specified
    display_df = df.head(max_rows) if max_rows else df
    
    # Create rich table with better formatting
    table = Table(
        show_header=True, 
        header_style="bold magenta",
        border_style="bright_blue",
        title_style="bold cyan",
        show_lines=True
    )
    
    # Add columns with appropriate styling
    for col in display_df.columns:
        # Apply different styles based on column type
        if col.lower() in ['budget', 'available budget']:
            style = "bold yellow"
        elif col.lower() in ['team value']:
            style = "bold green"
        else:
            style = "white"
            
        table.add_column(col, style=style, no_wrap=False)
    
    # Add rows with enhanced formatting
    for _, row in display_df.iterrows():
        formatted_row = []
        for col in display_df.columns:
            value = row[col]
            formatted_value = _format_cell_value(col, value)
            formatted_row.append(formatted_value)
        
        table.add_row(*formatted_row)
    
    # Display title
    if title:
        print_header(title, f"Budget analysis for {len(df)} managers • Data refreshed automatically")
    
    console.print(table)
    
    # Row count info
    total_rows = len(df)
    displayed_rows = len(display_df)
    if displayed_rows < total_rows:
        print_info(f"📊 Showing top {displayed_rows} of {total_rows} rows")
    else:
        print_info(f"📊 Total: {total_rows} managers analyzed")

# ===============================================================================
# BUDGET CALCULATION FUNCTIONS
# ===============================================================================

def calc_achievement_bonus_by_points(token, league_id, username, anchor_achievement_bonus):
    """Estimate achievement bonus for a user based on their total points compared to anchor user."""

    ranking = get_league_ranking(token, league_id)
    ranking_df = pd.DataFrame(ranking, columns=["Name", "Total Points"])

    # Total number of users
    num_users = len(ranking_df)
    if num_users == 0:
        return 0

    # Get anchor user's name and points
    anchor_user = get_username(token)
    anchor_row = ranking_df[ranking_df["Name"] == anchor_user]
    if anchor_row.empty:
        return 0
    anchor_points = anchor_row["Total Points"].values[0]

    # If the user is the anchor, return exactly the anchor achievement bonus
    if username == anchor_user:
        return anchor_achievement_bonus

    # Get target user's points
    user_row = ranking_df[ranking_df["Name"] == username]
    if user_row.empty:
        return 0
    user_points = user_row["Total Points"].values[0]

    # Calculate bonus scaling based on points ratio
    if anchor_points == 0:
        scale = 1.0
    else:
        scale = user_points / anchor_points

    estimated_bonus = anchor_achievement_bonus * scale
    return estimated_bonus

def calc_manager_budgets(token, league_id, league_start_date, start_budget):
    """Calculate manager budgets based on activities, bonuses, and team performance."""

    try:
        activities, login_bonus, achievement_bonus = get_league_activities(token, league_id, league_start_date)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch activities: {e}")

    activities_df = pd.DataFrame(activities)

    # Bonuses
    total_login_bonus = sum(entry.get("data", {}).get("bn", 0) for entry in login_bonus)

    total_achievement_bonus = 0
    for item in achievement_bonus:
        try:
            a_id = item.get("data", {}).get("t")
            if a_id is None:
                continue
            amount, reward = get_achievement_reward(token, league_id, a_id)
            total_achievement_bonus += amount * reward
        except Exception as e:
            print(f"Warning: Failed to process achievement bonus {item}: {e}")

    # Manager performances
    try:
        managers = get_managers(token, league_id)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch managers: {e}")

    performances = []
    for manager in managers:
        try:
            manager_name, manager_id = manager
            info = get_manager_info(token, league_id, manager_id)
            team_value = info.get("tv", 0)

            perf = get_manager_performance(token, league_id, manager_id, manager_name)
            perf["Team Value"] = team_value
            performances.append(perf)
        except Exception as e:
            print(f"Warning: Skipping manager {manager}: {e}")

    perf_df = pd.DataFrame(performances)
    if not perf_df.empty:
        perf_df["point_bonus"] = perf_df["tp"].fillna(0) * 1000
    else:
        perf_df["name"] = []
        perf_df["point_bonus"] = []
        perf_df["Team Value"] = []

    # Initial budgets from activities
    budgets = {user: start_budget for user in set(activities_df["byr"].dropna().unique())
                                          .union(set(activities_df["slr"].dropna().unique()))}

    for _, row in activities_df.iterrows():
        byr, slr, trp = row.get("byr"), row.get("slr"), row.get("trp", 0)
        try:
            if pd.isna(byr) and pd.notna(slr):
                budgets[slr] += trp
            elif pd.isna(slr) and pd.notna(byr):
                budgets[byr] -= trp
            elif pd.notna(byr) and pd.notna(slr):
                budgets[byr] -= trp
                budgets[slr] += trp
        except KeyError as e:
            print(f"Warning: Skipping invalid activity row {row}: {e}")

    budget_df = pd.DataFrame(list(budgets.items()), columns=["User", "Budget"])

    # Merge performance bonuses
    budget_df = budget_df.merge(
        perf_df[["name", "point_bonus", "Team Value"]],
        left_on="User",
        right_on="name",
        how="left"
    ).drop(columns=["name"], errors="ignore")

    budget_df["Budget"] = budget_df["Budget"] + budget_df["point_bonus"].fillna(0)
    budget_df.drop(columns=["point_bonus"], inplace=True, errors="ignore")

    # add total login bonus equally to everyone (100% estimation, if the user logged in every day)
    budget_df["Budget"] += total_login_bonus

    # Ensure consistent float format
    budget_df["Budget"] = budget_df["Budget"].astype(float)

    # add total achievement bonus based on anchor value and current ranking (estimation approach)
    for user in budget_df["User"]:
        achievement_bonus = calc_achievement_bonus_by_points(token, league_id, user, total_achievement_bonus)
        budget_df.loc[budget_df["User"] == user, "Budget"] += achievement_bonus

    # Sync with own actual budget
    try:
        own_budget = get_budget(token, league_id)
        own_username = get_username(token)
        mask = budget_df["User"] == own_username
        if not budget_df.loc[mask, "Budget"].eq(own_budget).all():
            budget_df.loc[mask, "Budget"] = own_budget
    except Exception as e:
        print(f"Warning: Could not sync own budget: {e}")

    # TODO check if this also applies if the user has positiv budget, currently only tested with negative budget
    budget_df["Max Negative"] = (budget_df["Team Value"].fillna(0) + budget_df["Budget"]) * -0.33

    # Calculate available budget
    budget_df["Available Budget"] = (budget_df["Max Negative"].fillna(0) - budget_df["Budget"]) * -1

    # Sort by available budget ascending
    budget_df.sort_values("Available Budget", ascending=False, inplace=True, ignore_index=True)

    return budget_df

# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

def main():
    """Main execution function"""
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
    
    # Display helpful insights
    print_separator()
    print_info("💡 Budget Analysis Insights:")
    if not manager_budgets_df.empty and 'Available Budget' in manager_budgets_df.columns:
        avg_budget = manager_budgets_df['Available Budget'].mean()
        max_budget = manager_budgets_df['Available Budget'].max()
        min_budget = manager_budgets_df['Available Budget'].min()
        
        print_info(f"• Average available budget: {format_currency(avg_budget)}")
        print_info(f"• Highest available budget: {format_currency(max_budget)}")
        print_info(f"• Lowest available budget: {format_currency(min_budget)}")
        print_info(f"• Competition level: {'High' if max_budget > avg_budget * 1.5 else 'Moderate'}")
    
    print_info("💡 Use budget information to:")
    print_info("  • Assess competition for high-value players")
    print_info("  • Time your bids strategically")
    print_info("  • Identify managers who might overpay")

if __name__ == "__main__":
    # Check if credentials are provided
    if not USERNAME or not PASSWORD:
        print_error("Missing credentials!")
        print_info("Please set KICK_USER and KICK_PASS environment variables")
        print_info("Or create a .env file with:")
        print_info("KICK_USER=your_email@example.com")
        print_info("KICK_PASS=your_password")
        print_separator()
        print_info("💡 This script requires your Kickbase login credentials to access:")
        print_info("  • League information and manager data")
        print_info("  • Trading activities and budget calculations")
        print_info("  • Team values and performance bonuses")
        print_info("🔒 Your credentials are only used to authenticate with Kickbase API")
        exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print_warning("\n\nScript interrupted by user")
        exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        print_info("💡 If this error persists:")
        print_info("  • Check your internet connection")
        print_info("  • Verify your Kickbase credentials")
        print_info("  • Ensure you're a member of the specified league")
        exit(1)