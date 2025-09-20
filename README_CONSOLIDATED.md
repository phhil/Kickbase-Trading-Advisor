# Kickbase Budget Analyzer - Consolidated Script

This repository now includes `budgets_only_consolidated.py` - a single-file version of the budget analyzer that contains all dependencies in one script for easy copying and usage.

## 🚀 Quick Start

1. **Copy the script**: Copy the entire content of `budgets_only_consolidated.py`
2. **Install dependencies**: `pip install pandas requests rich python-dotenv`
3. **Configure credentials**: Set environment variables or create a `.env` file:
   ```
   KICK_USER=your_email@example.com
   KICK_PASS=your_password
   ```
4. **Customize settings** (optional): Edit the USER SETTINGS section in the script:
   ```python
   league_name = "Your League Name"
   start_budget = 50000000
   league_start_date = "2025-08-10"
   ```
5. **Run the script**: `python budgets_only_consolidated.py`

## 📋 What's Included

The consolidated script contains all functionality from the original modular version:

- **Complete Kickbase API client** (login, league data, manager info, etc.)
- **Budget calculation engine** with trading activity analysis  
- **Achievement and bonus calculation logic**
- **Rich console formatting** for beautiful output
- **Error handling** and network connectivity checks

## 🔧 Original vs Consolidated

| Feature | Original | Consolidated |
|---------|----------|--------------|
| Files | Multiple modules | Single file |
| Size | ~10 files | ~24KB |
| Dependencies | Internal imports | All inlined |
| Functionality | ✅ Full | ✅ Identical |
| Copy/Paste | ❌ No | ✅ Yes |

## 📊 Output Example

The script provides beautifully formatted output showing:

- Manager names and current budgets
- Available spending capacity
- Team values and performance bonuses
- Competition analysis and insights

```
💰 Manager Budgets
┌─────────────┬──────────────┬──────────────┬──────────────────┐
│ User        │ Budget       │ Team Value   │ Available Budget │
├─────────────┼──────────────┼──────────────┼──────────────────┤
│ Manager1    │ 15.234.567   │ 45.123.456   │ 12.345.678       │
│ Manager2    │ -2.345.678   │ 52.987.654   │ 8.765.432        │
└─────────────┴──────────────┴──────────────┴──────────────────┘
```

## ⚡ Performance

- **Fast execution**: Focused only on budget analysis
- **Network efficient**: Minimal API calls required
- **Rich output**: Beautiful console formatting with colors and insights
- **Error handling**: Comprehensive error handling for network issues

This consolidated script gives you the exact same functionality as the original modular version, but in a single file that you can easily copy, modify, and run anywhere!