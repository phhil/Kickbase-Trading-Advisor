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

def display_dataframe(df, title=None, max_rows=None):
    """Display a pandas DataFrame as a formatted table"""
    if df.empty:
        print_warning("No data to display")
        return
    
    # Limit rows if specified
    display_df = df.head(max_rows) if max_rows else df
    
    # Create rich table
    table = Table(show_header=True, header_style="bold magenta")
    
    # Add columns
    for col in display_df.columns:
        table.add_column(col, style="cyan", no_wrap=False)
    
    # Add rows
    for _, row in display_df.iterrows():
        formatted_row = []
        for col in display_df.columns:
            value = row[col]
            
            # Apply special formatting for certain column types
            if col.lower() in ['budget', 'team value', 'mv', 'predicted_mv_target', 'mv_change_1d']:
                formatted_row.append(format_currency(value))
            elif col.lower() in ['mv_trend_1d'] or 'percent' in col.lower():
                formatted_row.append(format_percentage(value * 100 if abs(value) < 1 else value))
            elif pd.isna(value):
                formatted_row.append("[dim]N/A[/dim]")
            else:
                formatted_row.append(str(value))
        
        table.add_row(*formatted_row)
    
    if title:
        print_header(title)
    
    console.print(table)
    
    # Show row count info
    total_rows = len(df)
    displayed_rows = len(display_df)
    if displayed_rows < total_rows:
        print_info(f"Showing {displayed_rows} of {total_rows} rows")
    else:
        print_info(f"Total: {total_rows} rows")

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
    else:
        print_warning(f"Model warning: {warning_type}" + (f" - {details}" if details else ""))