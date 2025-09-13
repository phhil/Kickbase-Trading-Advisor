"""
Console formatting utilities for enhanced output display
"""

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.align import Align
import pandas as pd
import numpy as np

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

def print_model_evaluation(signs_percent, rmse, mae, r2):
    """Print model evaluation metrics in a formatted way"""
    print_header("🤖 Model Performance")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="white", width=15)
    table.add_column("Status", style="white", width=15)
    
    # Signs correct
    signs_color = "green" if signs_percent > 50 else "red"
    signs_status = "Good" if signs_percent > 50 else "Needs improvement"
    table.add_row(
        "Signs Correct", 
        f"[{signs_color}]{signs_percent:.2f}%[/{signs_color}]", 
        f"[{signs_color}]{signs_status}[/{signs_color}]"
    )
    
    # RMSE
    table.add_row("RMSE", f"{rmse:.2f}", "[dim]Lower is better[/dim]")
    
    # MAE
    table.add_row("MAE", f"{mae:.2f}", "[dim]Lower is better[/dim]")
    
    # R²
    r2_color = "green" if r2 > 0.3 else "yellow" if r2 > 0.1 else "red"
    r2_status = "Good" if r2 > 0.3 else "Fair" if r2 > 0.1 else "Poor"
    table.add_row(
        "R² Score", 
        f"[{r2_color}]{r2:.3f}[/{r2_color}]", 
        f"[{r2_color}]{r2_status}[/{r2_color}]"
    )
    
    console.print(table)

def print_separator():
    """Print a visual separator"""
    console.print("\n" + "─" * 80 + "\n")