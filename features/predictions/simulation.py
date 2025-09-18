"""
Enhanced simulation and backtesting framework for the Kickbase Trading Advisor
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

class TradingSimulator:
    """Backtesting and simulation framework for trading strategies"""
    
    def __init__(self, initial_budget: float = 50_000_000, max_players: int = 15):
        self.initial_budget = initial_budget
        self.max_players = max_players
        self.reset()
    
    def reset(self):
        """Reset simulator to initial state"""
        self.current_budget = self.initial_budget
        self.portfolio = {}  # player_id -> {purchase_price, quantity, purchase_date}
        self.transaction_history = []
        self.portfolio_values = []
        self.dates = []
    
    def buy_player(self, player_id: int, price: float, date: datetime, player_name: str = None) -> bool:
        """Buy a player if budget allows and portfolio isn't full"""
        
        if len(self.portfolio) >= self.max_players:
            return False
        
        if self.current_budget < price:
            return False
        
        self.current_budget -= price
        self.portfolio[player_id] = {
            'purchase_price': price,
            'purchase_date': date,
            'player_name': player_name or f"Player_{player_id}"
        }
        
        self.transaction_history.append({
            'date': date,
            'action': 'BUY',
            'player_id': player_id,
            'player_name': player_name,
            'price': price,
            'budget_after': self.current_budget
        })
        
        return True
    
    def sell_player(self, player_id: int, price: float, date: datetime) -> bool:
        """Sell a player if owned"""
        
        if player_id not in self.portfolio:
            return False
        
        player_info = self.portfolio[player_id]
        self.current_budget += price
        del self.portfolio[player_id]
        
        # Calculate profit/loss
        profit = price - player_info['purchase_price']
        hold_days = (date - player_info['purchase_date']).days
        
        self.transaction_history.append({
            'date': date,
            'action': 'SELL',
            'player_id': player_id,
            'player_name': player_info['player_name'],
            'price': price,
            'purchase_price': player_info['purchase_price'],
            'profit': profit,
            'hold_days': hold_days,
            'budget_after': self.current_budget
        })
        
        return True
    
    def update_portfolio_value(self, date: datetime, market_values: Dict[int, float]):
        """Update portfolio value based on current market values"""
        
        portfolio_value = 0
        for player_id, player_info in self.portfolio.items():
            if player_id in market_values:
                portfolio_value += market_values[player_id]
            else:
                # If market value not available, use purchase price
                portfolio_value += player_info['purchase_price']
        
        total_value = self.current_budget + portfolio_value
        
        self.portfolio_values.append(total_value)
        self.dates.append(date)
        
        return total_value
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        # If no portfolio tracking has been done, return basic metrics
        if len(self.portfolio_values) < 1:
            # Calculate based on transaction history only
            total_trades = len([t for t in self.transaction_history if t['action'] == 'SELL'])
            profitable_trades = [t for t in self.transaction_history if t['action'] == 'SELL' and t.get('profit', 0) > 0]
            win_rate = len(profitable_trades) / max(total_trades, 1) * 100
            avg_profit = np.mean([t.get('profit', 0) for t in self.transaction_history if t['action'] == 'SELL']) if total_trades > 0 else 0
            avg_hold_days = np.mean([t.get('hold_days', 0) for t in self.transaction_history if t['action'] == 'SELL']) if total_trades > 0 else 0
            
            total_return = (self.current_budget - self.initial_budget) / self.initial_budget * 100
            
            return {
                'total_return_pct': total_return,
                'total_return_abs': self.current_budget - self.initial_budget,
                'final_value': self.current_budget,
                'volatility_pct': 0.0,
                'max_drawdown_pct': 0.0,
                'sharpe_ratio': 0.0,
                'total_trades': total_trades,
                'win_rate_pct': win_rate,
                'avg_profit': avg_profit,
                'avg_hold_days': avg_hold_days,
                'current_budget': self.current_budget,
                'portfolio_size': len(self.portfolio)
            }
        
        values = np.array(self.portfolio_values)
        
        if len(values) < 2:
            return {}
        
        returns = np.diff(values) / values[:-1]
        
        # Basic metrics
        total_return = (values[-1] - self.initial_budget) / self.initial_budget
        total_return_pct = total_return * 100
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        max_drawdown = self._calculate_max_drawdown(values)
        
        # Performance metrics
        avg_daily_return = np.mean(returns)
        sharpe_ratio = avg_daily_return / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        # Trading metrics
        profitable_trades = [t for t in self.transaction_history if t['action'] == 'SELL' and t.get('profit', 0) > 0]
        total_trades = len([t for t in self.transaction_history if t['action'] == 'SELL'])
        win_rate = len(profitable_trades) / max(total_trades, 1) * 100
        
        avg_profit = np.mean([t.get('profit', 0) for t in self.transaction_history if t['action'] == 'SELL']) if total_trades > 0 else 0
        avg_hold_days = np.mean([t.get('hold_days', 0) for t in self.transaction_history if t['action'] == 'SELL']) if total_trades > 0 else 0

        return {
            'total_return_pct': total_return_pct,
            'total_return_abs': values[-1] - self.initial_budget,
            'final_value': values[-1],
            'volatility_pct': volatility * 100,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': total_trades,
            'win_rate_pct': win_rate,
            'avg_profit': avg_profit,
            'avg_hold_days': avg_hold_days,
            'current_budget': self.current_budget,
            'portfolio_size': len(self.portfolio)
        }
    
    def _calculate_max_drawdown(self, values: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        return abs(np.min(drawdown))


class PredictionStrategy:
    """Base class for prediction-based trading strategies"""
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
    
    def should_buy(self, prediction: float, current_mv: float, confidence: float = 1.0, **kwargs) -> bool:
        """Decide whether to buy a player based on predictions and other factors"""
        raise NotImplementedError
    
    def should_sell(self, prediction: float, current_mv: float, purchase_price: float, 
                   hold_days: int, confidence: float = 1.0, **kwargs) -> bool:
        """Decide whether to sell a player based on predictions and other factors"""
        raise NotImplementedError


class ThresholdStrategy(PredictionStrategy):
    """Simple threshold-based strategy"""
    
    def __init__(self, buy_threshold: float = 50000, sell_threshold: float = -25000, 
                 min_confidence: float = 0.5, max_hold_days: int = 30):
        super().__init__(f"Threshold(buy={buy_threshold}, sell={sell_threshold})")
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_confidence = min_confidence
        self.max_hold_days = max_hold_days
    
    def should_buy(self, prediction: float, current_mv: float, confidence: float = 1.0, **kwargs) -> bool:
        """Buy if predicted gain exceeds threshold and we have confidence"""
        return (prediction > self.buy_threshold and 
                confidence >= self.min_confidence and
                current_mv >= 500000)  # Minimum player value
    
    def should_sell(self, prediction: float, current_mv: float, purchase_price: float, 
                   hold_days: int, confidence: float = 1.0, **kwargs) -> bool:
        """Sell if predicted loss exceeds threshold or held too long"""
        current_profit = current_mv - purchase_price
        
        # Sell if prediction is very negative
        if prediction < self.sell_threshold:
            return True
        
        # Sell if held too long
        if hold_days >= self.max_hold_days:
            return True
        
        # Sell if we have good profits and prediction turns negative
        if current_profit > 100000 and prediction < 0:
            return True
        
        return False


def backtest_strategy(data_df: pd.DataFrame, model, features: List[str], 
                     strategy: PredictionStrategy, 
                     start_date: str = None, end_date: str = None,
                     initial_budget: float = 50_000_000) -> Tuple[TradingSimulator, pd.DataFrame]:
    """
    Backtest a trading strategy using historical data
    
    Args:
        data_df: Historical data with market values and features
        model: Trained prediction model
        features: List of feature columns to use
        strategy: Trading strategy to test
        start_date: Start date for backtesting (YYYY-MM-DD)
        end_date: End date for backtesting (YYYY-MM-DD)
        initial_budget: Starting budget
    
    Returns:
        simulator: TradingSimulator object with results
        daily_performance: DataFrame with daily performance metrics
    """
    
    simulator = TradingSimulator(initial_budget)
    
    # Filter data by date range - make a copy to avoid SettingWithCopyWarning
    data_df = data_df.copy()
    data_df['date'] = pd.to_datetime(data_df['date'])
    if start_date:
        data_df = data_df[data_df['date'] >= start_date]
    if end_date:
        data_df = data_df[data_df['date'] <= end_date]
    
    # Sort by date
    data_df = data_df.sort_values(['date', 'player_id'])
    
    daily_performance = []
    
    # Group by date for day-by-day simulation
    for date, day_data in data_df.groupby('date'):
        # Make predictions for this day
        try:
            day_features = day_data[features].fillna(0)
            predictions = model.predict(day_features)
            day_data = day_data.copy()
            day_data['prediction'] = predictions
            
            # Get market values for portfolio update
            market_values = dict(zip(day_data['player_id'], day_data['mv']))
            
            # Update portfolio value
            total_value = simulator.update_portfolio_value(date, market_values)
            
            # Check sell signals for current portfolio
            to_sell = []
            for player_id, player_info in simulator.portfolio.items():
                if player_id in market_values:
                    current_mv = market_values[player_id]
                    purchase_price = player_info['purchase_price']
                    hold_days = (date - player_info['purchase_date']).days
                    
                    # Get prediction for this player
                    player_prediction = 0
                    if player_id in day_data['player_id'].values:
                        player_prediction = day_data[day_data['player_id'] == player_id]['prediction'].iloc[0]
                    
                    if strategy.should_sell(player_prediction, current_mv, purchase_price, hold_days):
                        to_sell.append((player_id, current_mv))
            
            # Execute sells
            for player_id, price in to_sell:
                simulator.sell_player(player_id, price, date)
            
            # Check buy signals
            buy_candidates = day_data[day_data['mv'] > 0].copy()
            buy_candidates = buy_candidates[~buy_candidates['player_id'].isin(simulator.portfolio.keys())]
            
            # Sort by prediction descending
            buy_candidates = buy_candidates.sort_values('prediction', ascending=False)
            
            for _, row in buy_candidates.iterrows():
                if len(simulator.portfolio) >= simulator.max_players:
                    break
                
                player_id = row['player_id']
                current_mv = row['mv']
                prediction = row['prediction']
                confidence = row.get('prediction_confidence', 1.0)
                
                if strategy.should_buy(prediction, current_mv, confidence):
                    success = simulator.buy_player(player_id, current_mv, date, 
                                                 f"{row.get('first_name', '')} {row.get('last_name', '')}")
                    if not success:
                        break  # No more budget
            
            # Record daily performance
            daily_performance.append({
                'date': date,
                'total_value': total_value,
                'budget': simulator.current_budget,
                'portfolio_size': len(simulator.portfolio),
                'daily_return': (total_value - simulator.initial_budget) / simulator.initial_budget * 100
            })
            
        except Exception as e:
            print(f"Error processing date {date}: {e}")
            continue
    
    return simulator, pd.DataFrame(daily_performance)


def run_strategy_comparison(data_df: pd.DataFrame, model, features: List[str], 
                           strategies: List[PredictionStrategy],
                           start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Compare multiple trading strategies"""
    
    results = []
    
    for strategy in strategies:
        print(f"Running backtest for strategy: {strategy.name}")
        simulator, daily_perf = backtest_strategy(data_df, model, features, strategy, 
                                                start_date, end_date)
        
        metrics = simulator.get_performance_metrics()
        metrics['strategy'] = strategy.name
        results.append(metrics)
    
    return pd.DataFrame(results)