"""
Console formatting utilities for enhanced output display
"""

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.align import Align
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.live import Live
from contextlib import contextmanager
import pandas as pd
import numpy as np
import time
import warnings

console = Console()

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

def print_warning(message):
    """Print a warning message in yellow"""
    console.print(f"⚠️  [bold yellow]{message}[/bold yellow]")

def print_error(message):
    """Print an error message in red"""
    console.print(f"❌ [bold red]{message}[/bold red]")

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

def format_percentage(value):
    """Format percentage values with colors"""
    if pd.isna(value):
        return "N/A"
    
    formatted = f"{value:.2f}%"
    
    if value > 0:
        return f"[green]{formatted}[/green]"
    elif value < 0:
        return f"[red]{formatted}[/red]"
    else:
        return formatted

def _get_optimal_column_widths(df):
    """Calculate optimal column widths based on content"""
    widths = {}
    for col in df.columns:
        # Base width on column name and content
        header_len = len(col)
        if not df.empty:
            max_content_len = df[col].astype(str).str.len().max()
            optimal_width = min(max(header_len + 2, max_content_len + 2), 25)  # Cap at 25 chars
        else:
            optimal_width = header_len + 2
        
        # Special handling for specific columns
        if col.lower() in ['last_name', 'first_name']:
            widths[col] = min(optimal_width, 15)
        elif col.lower() in ['user']:
            widths[col] = 12
        elif col.lower() in ['team_name']:
            widths[col] = min(optimal_width, 12)
        elif col.lower() in ['mv', 'predicted_mv_target', 'predicted_mv_1d', 'predicted_mv_3d', 'predicted_mv_7d']:
            widths[col] = 12
        elif col.lower() in ['budget', 'team value', 'max negative', 'available budget']:
            widths[col] = 14
        elif col.lower() in ['position']:
            widths[col] = 6
        elif col.lower() in ['investment_grade']:
            widths[col] = 15
        elif 'prob' in col.lower() or 'risk' in col.lower():
            widths[col] = 8
        else:
            widths[col] = min(optimal_width, 20)
    
    return widths

def _get_enhanced_column_header(col, column_descriptions=None):
    """Get enhanced column header with descriptions"""
    # Use custom description if provided
    if column_descriptions and col in column_descriptions:
        return column_descriptions[col]
    
    # Default enhanced headers
    headers = {
        'last_name': 'Player',
        'first_name': 'First Name',
        'team_name': 'Team',
        'position': 'Pos',
        'mv': 'Market Value\n(Current)',
        'predicted_mv_target': 'Predicted Δ\n(1-day)',
        'predicted_mv_1d': 'Pred. Value\n(1-day)',
        'predicted_mv_3d': 'Pred. Value\n(3-day)',
        'predicted_mv_7d': 'Pred. Value\n(7-day)',
        'mv_change_yesterday': 'Yesterday\nChange',
        'mv_change_1d': 'Daily\nChange',
        'mv_trend_1d': 'Trend\n(%)',
        's_11_prob': 'Start XI\nProb (%)',
        'hours_to_exp': 'Hours to\nExpiry',
        'expiring_today': 'Expires\nToday',
        'risk_score': 'Risk\nScore',
        'prediction_confidence': 'Confidence\nScore',
        'investment_grade': 'Investment\nRecommendation',
        'budget': 'Budget\n(Estimated)',
        'team value': 'Team Value\n(Total)',
        'max negative': 'Max Negative\nBudget',
        'available budget': 'Available\nBudget'
    }
    
    return headers.get(col.lower(), col.replace('_', ' ').title())

def _format_cell_value(col, value):
    """Format individual cell values based on column type"""
    if pd.isna(value):
        return "[dim]N/A[/dim]"
    
    col_lower = col.lower()
    
    # Currency formatting
    if col_lower in ['budget', 'team value', 'mv', 'predicted_mv_target', 'predicted_mv_1d', 
                     'predicted_mv_3d', 'predicted_mv_7d', 'mv_change_yesterday', 'mv_change_1d',
                     'max negative', 'available budget']:
        return format_currency(value)
    
    # Percentage formatting
    elif col_lower in ['mv_trend_1d', 's_11_prob'] or 'percent' in col_lower:
        if 'prob' in col_lower and value <= 1:
            # Convert probability to percentage
            return format_percentage(value * 100)
        else:
            return format_percentage(value)
    
    # Risk score formatting
    elif 'risk' in col_lower:
        if value <= 0.3:
            return f"[green]{value:.3f}[/green]"
        elif value <= 0.5:
            return f"[yellow]{value:.3f}[/yellow]"
        else:
            return f"[red]{value:.3f}[/red]"
    
    # Confidence formatting
    elif 'confidence' in col_lower:
        if value >= 2.0:
            return f"[green]{value:.2f}[/green]"
        elif value >= 1.0:
            return f"[yellow]{value:.2f}[/yellow]"
        else:
            return f"[red]{value:.2f}[/red]"
    
    # Boolean formatting
    elif isinstance(value, bool):
        return "[green]Yes[/green]" if value else "[dim]No[/dim]"
    
    # Investment grade (already has emoji formatting)
    elif 'investment_grade' in col_lower:
        return str(value)
    
    # Hours formatting
    elif 'hours' in col_lower:
        if value < 6:
            return f"[red]{value:.1f}h[/red]"  # Urgent
        elif value < 24:
            return f"[yellow]{value:.1f}h[/yellow]"  # Soon
        else:
            return f"[green]{value:.1f}h[/green]"  # Plenty of time
    
    # Position formatting
    elif col_lower == 'position':
        position_colors = {
            'GK': 'yellow', 'TW': 'yellow',
            'LB': 'blue', 'CB': 'blue', 'RB': 'blue', 'LV': 'blue', 'IV': 'blue', 'RV': 'blue',
            'CDM': 'cyan', 'CM': 'cyan', 'LM': 'cyan', 'RM': 'cyan', 'CAM': 'cyan', 'ZM': 'cyan', 'LM': 'cyan', 'RM': 'cyan', 'OM': 'cyan',
            'LW': 'magenta', 'RW': 'magenta', 'ST': 'magenta', 'CF': 'magenta', 'MS': 'magenta'
        }
        color = position_colors.get(str(value).upper(), 'white')
        return f"[{color}]{value}[/{color}]"
    
    # Default formatting
    else:
        return str(value)

def _get_table_context_description(title, df):
    """Get context description for different table types"""
    if not title:
        return None
        
    title_lower = title.lower()
    
    if "multi-horizon" in title_lower:
        return "Predictions across different time horizons • Higher predicted values indicate better investment opportunities"
    elif "market" in title_lower:
        return f"Players available on transfer market • {len(df)} opportunities found • Look for high predictions with low risk"
    elif "squad" in title_lower:
        return f"Your current squad analysis • {len(df)} players evaluated • Consider selling players with negative predictions"
    elif "budget" in title_lower:
        return "Estimated budgets based on league activity • Use this to gauge competition for players"
    elif "strategy" in title_lower:
        return "Historical performance of different trading strategies • Higher returns indicate better strategies"
    elif "feature" in title_lower:
        return "Model feature importance • Higher values indicate more influential factors in predictions"
    else:
        return f"{len(df)} items analyzed • Data refreshed automatically"

def _display_table_insights(df, title):
    """Display insights and summary statistics for the table"""
    if df.empty or not title:
        return
    
    title_lower = title.lower()
    
    try:
        # Multi-horizon predictions insights
        if "multi-horizon" in title_lower:
            _display_multi_horizon_insights(df)
        
        # Market recommendations insights
        elif "market" in title_lower:
            _display_market_insights(df)
        
        # Squad analysis insights
        elif "squad" in title_lower:
            _display_squad_insights(df)
        
        # Budget insights
        elif "budget" in title_lower:
            _display_budget_insights(df)
            
    except Exception as e:
        # Don't break the flow if insights fail
        print_info(f"💡 Additional insights temporarily unavailable")

def _display_multi_horizon_insights(df):
    """Display insights for multi-horizon predictions"""
    if df.empty:
        return
        
    print_separator()
    print_info("🔮 Multi-Horizon Analysis Insights:")
    
    # Calculate statistics for different horizons
    if 'predicted_mv_1d' in df.columns:
        avg_1d = df['predicted_mv_1d'].mean()
        print_info(f"• Average 1-day prediction: {format_currency(avg_1d)}")
    
    if 'predicted_mv_3d' in df.columns:
        avg_3d = df['predicted_mv_3d'].mean()
        print_info(f"• Average 3-day prediction: {format_currency(avg_3d)}")
    
    if 'predicted_mv_7d' in df.columns:
        avg_7d = df['predicted_mv_7d'].mean()
        print_info(f"• Average 7-day prediction: {format_currency(avg_7d)}")
    
    # Risk analysis
    if 'risk_score' in df.columns:
        low_risk_count = len(df[df['risk_score'] <= 0.3])
        high_risk_count = len(df[df['risk_score'] > 0.5])
        print_info(f"• Low risk opportunities: {low_risk_count} players")
        if high_risk_count > 0:
            print_warning(f"• High risk players: {high_risk_count} (proceed with caution)")
    
    # Top opportunities
    if 'predicted_mv_1d' in df.columns:
        top_gains = df.nlargest(3, 'predicted_mv_1d')
        if not top_gains.empty:
            print_success("💎 Top 3 predicted gainers:")
            for _, row in top_gains.iterrows():
                name = row.get('last_name', 'Unknown')
                team = row.get('team_name', 'Unknown')
                pred = row.get('predicted_mv_1d', 0)
                print_info(f"  • {name} ({team}): {format_currency(pred)}")

def _display_market_insights(df):
    """Display insights for market recommendations"""
    if df.empty:
        return
        
    print_separator()
    print_info("📈 Market Analysis Insights:")
    
    # Investment grade distribution
    if 'investment_grade' in df.columns:
        grade_counts = df['investment_grade'].value_counts()
        for grade, count in grade_counts.items():
            print_info(f"• {grade}: {count} players")
    
    # Expiry urgency
    if 'expiring_today' in df.columns:
        expiring_today = len(df[df['expiring_today'] == True])
        if expiring_today > 0:
            print_warning(f"⏰ {expiring_today} players expiring today - act fast!")
    
    if 'hours_to_exp' in df.columns:
        urgent_players = len(df[df['hours_to_exp'] < 6])
        if urgent_players > 0:
            print_warning(f"🚨 {urgent_players} players expire in less than 6 hours")
    
    # Budget consideration
    if 'predicted_mv_target' in df.columns:
        avg_prediction = df['predicted_mv_target'].mean()
        high_value_count = len(df[df['predicted_mv_target'] > 50000])
        print_info(f"• Average predicted gain: {format_currency(avg_prediction)}")
        if high_value_count > 0:
            print_success(f"💰 {high_value_count} high-value opportunities (>50k predicted gain)")
    
    # Risk recommendations
    if 'risk_score' in df.columns:
        low_risk_high_reward = df[(df['risk_score'] <= 0.3) & (df['predicted_mv_target'] > 30000)]
        if not low_risk_high_reward.empty:
            print_success(f"🎯 {len(low_risk_high_reward)} low-risk, high-reward opportunities found!")

def _display_squad_insights(df):
    """Display insights for squad analysis"""
    if df.empty:
        return
        
    print_separator()
    print_info("⚽ Squad Analysis Insights:")
    
    # Performance analysis
    if 'predicted_mv_target' in df.columns:
        positive_outlook = len(df[df['predicted_mv_target'] > 0])
        negative_outlook = len(df[df['predicted_mv_target'] < -25000])
        
        print_info(f"• Players with positive outlook: {positive_outlook}")
        if negative_outlook > 0:
            print_warning(f"• Players to consider selling: {negative_outlook} (predicted loss >25k)")
    
    # Starting XI probability
    if 's_11_prob' in df.columns and not df['s_11_prob'].isna().all():
        high_prob_starters = len(df[df['s_11_prob'] > 0.7])
        if high_prob_starters > 0:
            print_success(f"🌟 {high_prob_starters} players likely to start (>70% probability)")
    
    # Squad value insights
    if 'mv' in df.columns:
        total_value = df['mv'].sum()
        avg_value = df['mv'].mean()
        print_info(f"• Total squad value: {format_currency(total_value)}")
        print_info(f"• Average player value: {format_currency(avg_value)}")

def _display_budget_insights(df):
    """Display insights for budget analysis"""
    if df.empty:
        return
        
    print_separator()
    print_info("💰 Budget Analysis Insights:")
    
    if 'budget' in df.columns:
        avg_budget = df['budget'].mean()
        max_budget = df['budget'].max()
        min_budget = df['budget'].min()
        
        print_info(f"• Average manager budget: {format_currency(avg_budget)}")
        print_info(f"• Highest budget: {format_currency(max_budget)}")
        print_info(f"• Competition level: {'High' if max_budget > avg_budget * 1.5 else 'Moderate'}")
    
    print_info("💡 Use budget information to:")
    print_info("  • Assess competition for high-value players")
    print_info("  • Time your bids strategically")
    print_info("  • Identify managers who might overpay")


def display_dataframe(df, title=None, max_rows=None, show_insights=True, column_descriptions=None):
    """Display a pandas DataFrame as a formatted table with enhanced formatting and insights"""
    if df.empty:
        print_warning("No data to display")
        if title and "market" in title.lower():
            print_info("💡 This could mean no players are currently available on the transfer market.")
        elif title and "squad" in title.lower():
            print_info("💡 This could mean no players in your squad have prediction data available.")
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
    
    # Enhanced column headers with descriptions and better widths
    column_widths = _get_optimal_column_widths(display_df)
    
    for col in display_df.columns:
        header_text = _get_enhanced_column_header(col, column_descriptions)
        width = column_widths.get(col, None)
        
        # Apply different styles based on column type
        if col.lower() in ['predicted_mv_target', 'predicted_mv_1d', 'predicted_mv_3d', 'predicted_mv_7d']:
            style = "bold green"
        elif col.lower() in ['mv', 'budget', 'team value']:
            style = "bold yellow"
        elif col.lower() in ['risk_score', 'mv_change_yesterday', 'mv_change_1d']:
            style = "cyan"
        elif 'investment_grade' in col.lower():
            style = "bold"
        else:
            style = "white"
            
        table.add_column(header_text, style=style, width=width, no_wrap=False)
    
    # Add rows with enhanced formatting
    for _, row in display_df.iterrows():
        formatted_row = []
        for col in display_df.columns:
            value = row[col]
            formatted_value = _format_cell_value(col, value)
            formatted_row.append(formatted_value)
        
        table.add_row(*formatted_row)
    
    # Display title with context
    if title:
        print_header(title, _get_table_context_description(title, df))
    
    console.print(table)
    
    # Enhanced row count info with insights
    total_rows = len(df)
    displayed_rows = len(display_df)
    if displayed_rows < total_rows:
        print_info(f"📊 Showing top {displayed_rows} of {total_rows} rows")
        print_info(f"💡 Use filters or increase max_rows to see more data")
    else:
        print_info(f"📊 Total: {total_rows} rows displayed")
    
    # Show insights and summary statistics if enabled
    if show_insights:
        _display_table_insights(df, title)

def print_model_evaluation(signs_percent, rmse, mae, r2, mape=None, small_acc=None, medium_acc=None, large_acc=None):
    """Print enhanced model evaluation metrics in a formatted way"""
    print_header("🤖 Model Performance")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Value", style="white", width=15)
    table.add_column("Status", style="white", width=15)
    
    # Signs correct
    signs_color = "green" if signs_percent > 55 else "yellow" if signs_percent > 50 else "red"
    signs_status = "Excellent" if signs_percent > 60 else "Good" if signs_percent > 55 else "Fair" if signs_percent > 50 else "Needs improvement"
    table.add_row(
        "Overall Signs Correct", 
        f"[{signs_color}]{signs_percent:.2f}%[/{signs_color}]", 
        f"[{signs_color}]{signs_status}[/{signs_color}]"
    )
    
    # Accuracy by change magnitude
    if small_acc is not None:
        small_color = "green" if small_acc > 50 else "red"
        table.add_row(
            "Small Changes (<50k)", 
            f"[{small_color}]{small_acc:.1f}%[/{small_color}]", 
            "[dim]Direction accuracy[/dim]"
        )
    
    if medium_acc is not None:
        medium_color = "green" if medium_acc > 50 else "red"
        table.add_row(
            "Medium Changes (50-200k)", 
            f"[{medium_color}]{medium_acc:.1f}%[/{medium_color}]", 
            "[dim]Direction accuracy[/dim]"
        )
    
    if large_acc is not None:
        large_color = "green" if large_acc > 50 else "red"
        table.add_row(
            "Large Changes (>200k)", 
            f"[{large_color}]{large_acc:.1f}%[/{large_color}]", 
            "[dim]Direction accuracy[/dim]"
        )
    
    # RMSE
    table.add_row("RMSE", f"{rmse:.2f}", "[dim]Lower is better[/dim]")
    
    # MAE
    table.add_row("MAE", f"{mae:.2f}", "[dim]Lower is better[/dim]")
    
    # MAPE
    if mape is not None:
        # Handle extremely large MAPE values
        if mape > 1000000:
            mape_display = f"{mape:.2e}"
            mape_color = "red"
            mape_status = "Very high (potential data issue)"
        elif mape > 100:
            mape_display = f"{mape:.1f}%"
            mape_color = "red"
            mape_status = "High"
        elif mape > 50:
            mape_display = f"{mape:.1f}%"
            mape_color = "yellow"
            mape_status = "Moderate"
        else:
            mape_display = f"{mape:.1f}%"
            mape_color = "green"
            mape_status = "Good"
        
        table.add_row("MAPE", f"[{mape_color}]{mape_display}[/{mape_color}]", f"[{mape_color}]{mape_status}[/{mape_color}]")
    
    # R²
    r2_color = "green" if r2 > 0.3 else "yellow" if r2 > 0.1 else "red"
    r2_status = "Good" if r2 > 0.3 else "Fair" if r2 > 0.1 else "Poor"
    table.add_row(
        "R² Score", 
        f"[{r2_color}]{r2:.3f}[/{r2_color}]", 
        f"[{r2_color}]{r2_status}[/{r2_color}]"
    )
    
    console.print(table)


def print_feature_importance(importance_df, top_n=15):
    """Print feature importance in a formatted table"""
    print_header("🎯 Feature Importance (Top Features)")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Feature", style="white", width=25)
    table.add_column("Importance", style="green", width=12)
    table.add_column("Bar", style="green", width=20)
    
    top_features = importance_df.head(top_n)
    max_importance = top_features['importance'].max()
    
    for idx, (_, row) in enumerate(top_features.iterrows(), 1):
        # Create a simple bar visualization
        bar_length = int((row['importance'] / max_importance) * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        table.add_row(
            str(idx),
            row['feature'],
            f"{row['importance']:.4f}",
            bar
        )
    
    console.print(table)

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

def suppress_sklearn_warnings():
    """Suppress known sklearn warnings that are not critical"""
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')

def print_warning_with_context(message, context=None, solution=None):
    """Print warning with additional context and potential solution"""
    console.print(f"⚠️  [bold yellow]{message}[/bold yellow]")
    if context:
        console.print(f"[dim]   Context: {context}[/dim]")
    if solution:
        console.print(f"[dim]   Suggestion: {solution}[/dim]")

def print_model_warning(warning_type, details=None):
    """Handle specific model-related warnings with context"""
    if warning_type == "feature_names":
        print_warning_with_context(
            "Model feature name warnings detected",
            "This is a known sklearn compatibility issue and doesn't affect predictions",
            "These warnings can be safely ignored for this application"
        )
    elif warning_type == "ill_conditioned":
        print_warning_with_context(
            "Matrix conditioning warning detected",
            "Some features may be highly correlated, but model should still work",
            "This may slightly affect numerical precision but not overall results"
        )
    elif warning_type == "low_r2":
        print_warning_with_context(
            "Low model performance detected",
            f"R² score is low: {details}",
            "Consider feature engineering or data quality improvements"
        )
    else:
        print_warning(f"Model warning: {warning_type}" + (f" - {details}" if details else ""))

def print_network_error(error_details):
    """Handle network-related errors with helpful context"""
    print_error("Network connection failed")
    console.print(f"[dim]   Error details: {error_details}[/dim]")
    console.print("[dim]   This usually means:[/dim]")
    console.print("[dim]   • No internet connection available[/dim]")
    console.print("[dim]   • Kickbase API is temporarily unavailable[/dim]")
    console.print("[dim]   • Firewall blocking the connection[/dim]")
    console.print("[dim]   Suggestion: Check your internet connection and try again[/dim]")

def print_processing_summary(total_items, success_count, error_count):
    """Print a summary of batch processing results"""
    console.print(f"\n[bold cyan]📊 Processing Summary[/bold cyan]")
    console.print(f"[green]✓ Successful: {success_count}/{total_items}[/green]")
    if error_count > 0:
        console.print(f"[red]✗ Failed: {error_count}/{total_items}[/red]")
    else:
        console.print(f"[green]✓ All items processed successfully![/green]")
    
    success_rate = (success_count / total_items * 100) if total_items > 0 else 0
    if success_rate == 100:
        console.print(f"[green]Success rate: {success_rate:.1f}%[/green]")
    elif success_rate >= 90:
        console.print(f"[yellow]Success rate: {success_rate:.1f}%[/yellow]")
    else:
        console.print(f"[red]Success rate: {success_rate:.1f}%[/red]")

def print_trading_tips():
    """Display helpful trading tips and strategies"""
    print_header("💡 Trading Tips & Best Practices")
    
    tips = [
        "🎯 Focus on players with high predicted gains and low risk scores",
        "⏰ Monitor expiry times - bid on players expiring soon for better deals",
        "📊 Use Starting XI probability to assess player value stability",
        "💰 Consider your budget relative to other managers for competitive bidding",
        "📈 Look for players with consistent positive trends over multiple days",
        "🔄 Diversify your portfolio across different positions and teams",
        "⚡ Act quickly on strong buy recommendations before they expire",
        "📉 Consider selling players with predicted losses >25k",
        "🎲 Higher risk can mean higher reward, but balance your portfolio",
        "📅 Check prediction confidence - higher confidence = more reliable predictions"
    ]
    
    for tip in tips:
        print_info(f"  {tip}")

def print_market_summary(market_df, squad_df=None, budget_df=None):
    """Print a comprehensive market summary with actionable insights"""
    print_header("📊 Market Intelligence Summary")
    
    if not market_df.empty:
        # Market opportunities
        strong_buys = len(market_df[market_df.get('investment_grade', '').str.contains('Strong Buy', na=False)])
        buys = len(market_df[market_df.get('investment_grade', '').str.contains('🔵 Buy', na=False)])
        
        if strong_buys > 0:
            print_success(f"🎯 {strong_buys} STRONG BUY opportunities identified")
        if buys > 0:
            print_info(f"📈 {buys} BUY opportunities available")
        
        # Urgency alerts
        if 'expiring_today' in market_df.columns:
            urgent = len(market_df[market_df['expiring_today'] == True])
            if urgent > 0:
                print_warning(f"⏰ {urgent} players expire TODAY - immediate action recommended")
        
        # Value analysis
        if 'predicted_mv_target' in market_df.columns:
            high_value = market_df[market_df['predicted_mv_target'] > 75000]
            if not high_value.empty:
                print_success(f"💎 {len(high_value)} high-value targets (>75k predicted gain)")
                top_target = high_value.iloc[0]
                if 'last_name' in top_target:
                    print_info(f"🥇 Top target: {top_target['last_name']} ({top_target.get('team_name', 'Unknown')})")
    
    # Squad analysis
    if squad_df is not None and not squad_df.empty:
        if 'predicted_mv_target' in squad_df.columns:
            underperformers = squad_df[squad_df['predicted_mv_target'] < -25000]
            if not underperformers.empty:
                print_warning(f"📉 Consider selling {len(underperformers)} underperforming players")
    
    # Budget context
    if budget_df is not None and not budget_df.empty and 'budget' in budget_df.columns:
        avg_budget = budget_df['budget'].mean()
        your_budget = avg_budget  # Placeholder - would need actual user budget
        if avg_budget > 10000000:  # 10M threshold
            print_info(f"💰 High competition environment (avg budget: {format_currency(avg_budget)})")
        else:
            print_info(f"💰 Moderate competition (avg budget: {format_currency(avg_budget)})")

def print_prediction_methodology():
    """Explain the prediction methodology to users"""
    print_header("🤖 Prediction Methodology", "Understanding how predictions are generated")
    
    console.print("🔬 [bold]Model Features:[/bold]")
    print_info("  • Historical market value changes and trends")
    print_info("  • Player performance metrics (points, minutes played)")
    print_info("  • Team performance and form indicators") 
    print_info("  • Market volatility and momentum indicators")
    print_info("  • Positional performance comparisons")
    
    console.print("\n📊 [bold]Prediction Types:[/bold]")
    print_info("  • 1-day: Most accurate, immediate market changes")
    print_info("  • 3-day: Medium-term trends, includes match effects")
    print_info("  • 7-day: Long-term outlook, considers multiple matches")
    
    console.print("\n⚖️ [bold]Risk Assessment:[/bold]")
    print_info("  • Low risk (≤0.3): High confidence predictions")
    print_info("  • Medium risk (0.3-0.5): Moderate uncertainty")
    print_info("  • High risk (>0.5): Higher volatility expected")
    
    console.print("\n🎯 [bold]Investment Grades:[/bold]")
    print_info("  • 🟢 Strong Buy: High predicted gain + Low risk")
    print_info("  • 🔵 Buy: Good predicted gain + Acceptable risk")
    print_info("  • 🟡 Hold/Watch: Moderate predicted gain")
    print_info("  • ⚪ Neutral: Minimal predicted change")
    print_info("  • 🔴 Avoid: Predicted significant loss")

def format_large_number(value):
    """Format large numbers with appropriate suffixes"""
    if pd.isna(value):
        return "N/A"
    
    abs_value = abs(value)
    if abs_value >= 1000000:
        formatted = f"{value/1000000:.1f}M"
    elif abs_value >= 1000:
        formatted = f"{value/1000:.0f}k"
    else:
        formatted = f"{value:.0f}"
    
    return formatted

def print_data_freshness_info():
    """Display information about data freshness and update cycles"""
    from datetime import datetime
    
    now = datetime.now()
    print_info("📅 Data Freshness Information:")
    print_info(f"  • Analysis generated: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print_info("  • Market values update: Daily around 22:15 CET")
    print_info("  • Starting XI probabilities: Updated after team news")
    print_info("  • Player performance data: Updated after each match")
    print_info("💡 For best results, run analysis after 22:15 CET daily")

def display_enhanced_summary_table(data_dict, title="Summary Statistics"):
    """Display a summary table with key statistics"""
    if not data_dict:
        return
    
    print_header(title)
    
    table = Table(show_header=True, header_style="bold magenta", border_style="bright_blue")
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Value", style="white", width=15)
    table.add_column("Description", style="dim", width=35)
    
    for metric, (value, description) in data_dict.items():
        # Format value based on type
        if isinstance(value, (int, float)):
            if abs(value) > 1000000:
                formatted_value = format_currency(value)
            elif 'percentage' in metric.lower() or 'rate' in metric.lower():
                formatted_value = format_percentage(value)
            else:
                formatted_value = str(value)
        else:
            formatted_value = str(value)
        
        table.add_row(metric, formatted_value, description)
    
    console.print(table)