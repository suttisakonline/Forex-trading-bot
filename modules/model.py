"""
Model module for the Forex Trading Bot
- Handles reinforcement learning and hyperparameter optimization
"""

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pickle
import os
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Tuple, Optional, Union, List, Any
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import logging
import optuna
from optuna.trial import Trial
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from shimmy import GymV21CompatibilityV0
from stable_baselines3 import PPO
import torch
from data_fetcher import cov_to_corr
from config import (
    MIN_POSITION_SIZE, 
    MAX_POSITION_SIZE, 
    MODELS_DIRECTORY, 
    RL_TRAINING_STEPS, 
    RL_WINDOW_SIZE, 
    GENETIC_POPULATION_SIZE, 
    GENETIC_GENERATIONS,
    MAX_TRADES_PER_WEEK,
    INITIAL_BALANCE,
    EPISODE_LENGTH,
    SPREAD,
    COMMISSION,
    SLIPPAGE,
    REWARD_BALANCE_CHANGE_WEIGHT,
    REWARD_POSITION_HOLDING_PROFIT,
    REWARD_POSITION_HOLDING_LOSS,
    REWARD_TRADE_PROFIT_WEIGHT,
    REWARD_TRADE_LOSS_WEIGHT,
    REWARD_TRADE_FREQUENCY,
    PORTFOLIO_EFFICIENT_FRONTIER_POINTS,
    PORTFOLIO_PLOT_HEIGHT,
    PORTFOLIO_PLOT_ASPECT,
    PORTFOLIO_PLOT_STYLE,
    PORTFOLIO_CORRELATION_CMAP,
    PORTFOLIO_CORRELATION_CENTER,
    PORTFOLIO_CORRELATION_FMT,
    PORTFOLIO_PLOT_TITLE_FONTSIZE,
    PORTFOLIO_AXIS_FONTSIZE,
    PPO_LEARNING_RATE,
    PPO_N_STEPS,
    PPO_BATCH_SIZE,
    PPO_N_EPOCHS,
    PPO_GAMMA,
    PPO_GAE_LAMBDA,
    PPO_CLIP_RANGE,
    PPO_ENT_COEF,
    PPO_VF_COEF,
    PPO_MAX_GRAD_NORM,
    PPO_TARGET_KL,
    PPO_USE_SDE,
    PPO_SDE_SAMPLE_FREQ,
    PPO_VERBOSE,
    HP_LEARNING_RATE_MIN,
    HP_LEARNING_RATE_MAX,
    HP_N_STEPS_MIN,
    HP_N_STEPS_MAX,
    HP_BATCH_SIZE_MIN,
    HP_BATCH_SIZE_MAX,
    HP_N_EPOCHS_MIN,
    HP_N_EPOCHS_MAX,
    HP_GAMMA_MIN,
    HP_GAMMA_MAX,
    HP_ENT_COEF_MIN,
    HP_ENT_COEF_MAX
)

# Configure logging
logger = logging.getLogger(__name__)

class DebugLogger:
    """Simple debug logger for tracking trading metrics"""
    def __init__(self) -> None:
        self.log_file: str = "trading_journal.log"
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def log_error(self, error: Exception, context: Optional[str] = None) -> None:
        """Log error with optional context"""
        error_msg: str = f"Error: {type(error).__name__} - {str(error)}"
        if context:
            error_msg += f" | Context: {context}"
        logger.error(error_msg)

# Constants
RL_TRAINING_STEPS = RL_TRAINING_STEPS
RL_WINDOW_SIZE = RL_WINDOW_SIZE
MODELS_DIRECTORY = MODELS_DIRECTORY
GENETIC_POPULATION_SIZE = GENETIC_POPULATION_SIZE
GENETIC_GENERATIONS = GENETIC_GENERATIONS

# PPO Parameters
PPO_PARAMS = {
    'learning_rate': PPO_LEARNING_RATE,
    'n_steps': PPO_N_STEPS,
    'batch_size': PPO_BATCH_SIZE,
    'n_epochs': PPO_N_EPOCHS,
    'gamma': PPO_GAMMA,
    'gae_lambda': PPO_GAE_LAMBDA,
    'clip_range': PPO_CLIP_RANGE,
    'ent_coef': PPO_ENT_COEF,
    'vf_coef': PPO_VF_COEF,
    'max_grad_norm': PPO_MAX_GRAD_NORM,
    'target_kl': PPO_TARGET_KL,
    'use_sde': PPO_USE_SDE,
    'sde_sample_freq': PPO_SDE_SAMPLE_FREQ,
    'verbose': PPO_VERBOSE
}

class TradingEnvironment(gym.Env):
    """
    Trading environment for reinforcement learning
    """
    def __init__(self, df: pd.DataFrame, initial_balance: float = INITIAL_BALANCE) -> None:
        super().__init__()
        
        # Store data as numpy arrays for memory efficiency
        self.data: Dict[str, np.ndarray] = {
            'open': df['open'].values,
            'high': df['high'].values,
            'low': df['low'].values,
            'close': df['close'].values,
            'volume': df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
        }
        
        self.current_step: int = 0
        self.initial_balance: float = initial_balance
        self.balance: float = initial_balance
        self.position: int = 0
        self.position_size: float = MIN_POSITION_SIZE
        self.trade_history: List[Dict[str, Any]] = []
        
        # Trading costs
        self.spread: float = SPREAD      # Cost of the spread (difference between bid and ask)
        self.commission: float = COMMISSION  # Broker's fee per trade (0.01% of trade value)
        self.slippage: float = SLIPPAGE  # Price movement during order execution
        
        # Episode tracking
        self.episode: int = 0
        self.episode_profit: float = 0
        self.episode_loss: float = 0
        self.episode_winning_trades: int = 0
        self.episode_losing_trades: int = 0
        self.episode_largest_win: float = 0
        self.episode_largest_loss: float = 0
        self.episode_trades: int = 0
        self.episode_start_balance: float = initial_balance
        
        # Weekly trade tracking
        self.weekly_trades: int = 0
        self.last_week: Optional[int] = None
        
        # Track profit factors for averaging
        self.profit_factors: List[float] = []
        
        # Define action and observation spaces
        self.action_space: spaces.Discrete = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        
        # Calculate observation space size
        n_features: int = len(self.data.keys())
        self.observation_space: spaces.Box = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )
        
        # Initialize trade tracking
        self.current_trade: Optional[Dict[str, Any]] = None
        self.weekly_trades = 0
        self.last_week = None
        
        # Initialize debug logging
        self.debug_logger: DebugLogger = DebugLogger()
        logger.info(f"Trading Environment initialized with {len(df)} data points")
        logger.info(f"Initial balance: ${initial_balance:,.2f}")
        logger.info(f"Position sizing is dynamic. Initial size: {self.position_size} lots")
        logger.info(f"Trading costs - Spread: {self.spread*10000} pips, Commission: {self.commission*100}%")
        
        self.df_index = df.index if hasattr(df, 'index') else None
        
    def _calculate_metrics(self) -> Dict[str, Union[float, int]]:
        """Calculate episode trading metrics"""
        if self.episode_trades == 0:
            return {
                'profit_factor': 0.0,
                'avg_profit_factor': 0.0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'net_profit': 0.0,
                'return_pct': 0.0
            }
            
        win_rate: float = self.episode_winning_trades / self.episode_trades
        avg_profit: float = self.episode_profit / self.episode_winning_trades if self.episode_winning_trades > 0 else 0
        avg_loss: float = self.episode_loss / self.episode_losing_trades if self.episode_losing_trades > 0 else 0
        
        # Calculate current episode profit factor
        profit_factor: float = float('inf') if self.episode_loss == 0 and self.episode_profit > 0 else (
            abs(self.episode_profit / self.episode_loss) if self.episode_loss != 0 else 0
        )
            
        # Add to profit factors list
        self.profit_factors.append(profit_factor)
        
        # Calculate average profit factor
        avg_profit_factor: float = sum(self.profit_factors) / len(self.profit_factors)
        
        net_profit: float = self.episode_profit - self.episode_loss
        return_pct: float = (net_profit / self.episode_start_balance) * 100
        
        return {
            'profit_factor': profit_factor,
            'avg_profit_factor': avg_profit_factor,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'largest_win': self.episode_largest_win,
            'largest_loss': self.episode_largest_loss,
            'net_profit': net_profit,
            'return_pct': return_pct
        }
        
    def _log_trading_metrics(self):
        """Log current episode trading metrics"""
        metrics = self._calculate_metrics()
        # Use EPISODE_LENGTH for the label and self.episode for the episode number
        logger.info(f"\n=== {EPISODE_LENGTH}-Day Trading Metrics (Episode {self.episode}) ===")
        logger.info(f"Starting Balance: ${self.episode_start_balance:,.2f}")
        logger.info(f"Current Balance: ${self.balance:,.2f}")
        logger.info(f"Net Profit: ${metrics['net_profit']:,.2f} ({metrics['return_pct']:.2f}%)")
        logger.info(f"Total Trades: {self.episode_trades}")
        logger.info(f"Win Rate: {metrics['win_rate']*100:.2f}%")
        logger.info(f"Average Profit Factor: {metrics['avg_profit_factor']:.2f}")
        logger.info(f"Average Profit: ${metrics['avg_profit']:,.2f}")
        logger.info(f"Average Loss: ${metrics['avg_loss']:,.2f}")
        logger.info(f"Largest Win: ${metrics['largest_win']:,.2f}")
        logger.info(f"Largest Loss: ${metrics['largest_loss']:,.2f}")
        logger.info("=============================\n")
        
    def _update_metrics(self, pnl):
        """Update episode trading metrics with new trade result"""
        self.episode_trades += 1
        if pnl > 0:
            self.episode_winning_trades += 1
            self.episode_profit += pnl
            self.episode_largest_win = max(self.episode_largest_win, pnl)
        else:
            self.episode_losing_trades += 1
            self.episode_loss += abs(pnl)
            self.episode_largest_loss = min(self.episode_largest_loss, pnl)
            
    def _get_observation(self):
        """Get current market state observation"""
        obs = []
        for key in self.data.keys():
            obs.append(self.data[key][self.current_step])
        return np.array(obs, dtype=np.float32)
    
    def _calculate_position_size(self) -> float:
        """Calculate position size based on win rate"""
        if not self.trade_history:
            return MIN_POSITION_SIZE
        
        # Calculate win rate
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for trade in self.trade_history if trade['reward'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate position size based on win rate
        # Linear scaling from MIN_POSITION_SIZE to MAX_POSITION_SIZE based on win rate
        position_size = MIN_POSITION_SIZE + (MAX_POSITION_SIZE - MIN_POSITION_SIZE) * win_rate
        
        # Ensure position size stays within bounds
        return np.clip(position_size, MIN_POSITION_SIZE, MAX_POSITION_SIZE)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment"""
        self.current_step += 1
        done: bool = self.current_step % EPISODE_LENGTH == 0  # End episode after configured number of days
        
        # Get current price
        current_price: float = self.data['close'][self.current_step]
        
        # Reset weekly trades if it's a new week
        current_week: int = self.current_step // 7
        if self.last_week is not None and current_week > self.last_week:
            self.weekly_trades = 0
            self.last_week = current_week
        elif self.last_week is None:
            self.last_week = current_week
        
        # Update position size based on win rate
        self.position_size = self._calculate_position_size()
        
        # Execute trade
        reward: float = self._execute_trade(action, current_price)
        
        # Get observation
        obs: np.ndarray = self._get_observation()
        
        # Log trading metrics every 10 episodes
        if done and self.episode % 10 == 0:
            self._log_trading_metrics()
        
        # Additional info
        info: Dict[str, Any] = {
            'balance': self.balance,
            'position': self.position,
            'position_size': self.position_size,
            'current_price': current_price,
            'weekly_trades': self.weekly_trades,
            'current_trade': self.current_trade,
            'metrics': self._calculate_metrics(),
            'episode_length': EPISODE_LENGTH,
            'current_month': (self.current_step // 30) + 1
        }
        
        return obs, reward, done, False, info
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.current_trade = None
        self.weekly_trades = 0
        self.last_week = None
        self.trade_history = []  # Reset trade history for the new episode
        # Reset episode metrics
        self.episode_profit = 0
        self.episode_loss = 0
        self.episode_winning_trades = 0
        self.episode_losing_trades = 0
        self.episode_largest_win = 0
        self.episode_largest_loss = 0
        self.episode_trades = 0
        self.episode_start_balance = self.initial_balance
        self.episode += 1
        return self._get_observation(), {}
    
    def _execute_trade(self, action: int, current_price: float) -> float:
        """Execute trade based on action"""
        reward: float = 0.0
        
        # Close position if it's been open for 5 days
        if self.current_trade and (self.current_step - self.current_trade['entry_step']) >= 5:
            pnl: float = self._close_position(current_price)
            reward = pnl / self.initial_balance  # Normalize reward
            self.current_trade = None
        
        # Execute new trade if we haven't reached weekly limit
        if self.weekly_trades < MAX_TRADES_PER_WEEK:  # Use imported constant
            if action == 1 and self.position == 0:  # Buy
                # Apply slippage to entry price
                entry_price: float = current_price * (1 + self.slippage)
                self._open_position(entry_price, 'LONG')
                self.weekly_trades += 1
                reward = 0.01  # Small positive reward for taking action
            elif action == 2 and self.position == 0:  # Sell
                # Apply slippage to entry price
                entry_price: float = current_price * (1 - self.slippage)
                self._open_position(entry_price, 'SHORT')
                self.weekly_trades += 1
                reward = 0.01  # Small positive reward for taking action
            elif action == 0 and self.position != 0:  # Close position
                # Apply slippage to exit price
                exit_price: float = current_price * (1 + self.slippage if self.position == -1 else 1 - self.slippage)
                pnl: float = self._close_position(exit_price)
                reward = pnl / self.initial_balance  # Normalize reward
                self.current_trade = None
        
        # Penalize if no trades were made this week
        if self.weekly_trades == 0 and self.current_step > 0:
            reward = -0.1  # Smaller penalty for not trading
        
        return reward

    def _calculate_trading_cost(self, price: float, direction: str) -> float:
        """Calculate trading costs for a position with dynamic adjustment"""
        # Calculate dynamic trading costs based on market volatility
        dynamic_costs = self._calculate_dynamic_trading_costs()
        
        # Spread cost (adjusted by volatility)
        spread_cost = dynamic_costs['spread'] * self.position_size * 100000
        
        # Commission cost
        commission_cost = price * self.position_size * 100000 * self.commission
        
        # Slippage cost (adjusted by volatility)
        slippage_cost = dynamic_costs['slippage'] * self.position_size * 100000
        
        return spread_cost + commission_cost + slippage_cost
    
    def _calculate_dynamic_trading_costs(self) -> Dict[str, float]:
        """Calculate dynamic trading costs based on market volatility"""
        try:
            # Calculate volatility from recent price data
            if self.current_step < 20:
                return {'spread': self.spread, 'slippage': self.slippage}
            
            # Get recent price data for volatility calculation
            recent_prices = self.data['close'][max(0, self.current_step-20):self.current_step]
            if len(recent_prices) < 10:
                return {'spread': self.spread, 'slippage': self.slippage}
            
            # Calculate price volatility
            price_changes = np.diff(recent_prices) / recent_prices[:-1]
            volatility = np.std(price_changes)
            
            # Adjust trading costs based on volatility
            # Higher volatility = higher trading costs
            volatility_multiplier = 1 + (volatility * 100)  # Scale volatility
            
            dynamic_spread = self.spread * volatility_multiplier
            dynamic_slippage = self.slippage * volatility_multiplier
            
            # Ensure costs don't become unreasonable
            dynamic_spread = min(dynamic_spread, self.spread * 3)  # Max 3x base spread
            dynamic_slippage = min(dynamic_slippage, self.slippage * 3)  # Max 3x base slippage
            
            return {
            'spread': dynamic_spread,
            'slippage': dynamic_slippage
}
            
        except Exception as e:
            # Fallback to base costs if calculation fails
            return {'spread': self.spread, 'slippage': self.slippage}

    @property
    def current_datetime(self):
        # Get the datetime for the current step from the data index
        # Assumes the original DataFrame index is datetime and matches the data arrays
        if hasattr(self, 'df_index'):
            return self.df_index[self.current_step]
        return None

    def _open_position(self, price: float, direction: str):
        """Open a new position"""
        self.position = 1 if direction == 'LONG' else -1
        # Calculate trading costs
        trading_cost = self._calculate_trading_cost(price, direction)
        self.balance -= trading_cost
        self.current_trade = {
            'direction': direction,
            'entry_price': price,
            'entry_step': self.current_step,
            'trading_cost': trading_cost,
            'timestamp': self.current_datetime  # Add timestamp for trade open
        }
        # Append trade to trade_history for per-episode stats
        self.trade_history.append({
            'direction': direction,
            'entry_price': price,
            'entry_step': self.current_step,
            'trading_cost': trading_cost,
            'reward': 0.0,  # Will be updated on close
            'timestamp': self.current_datetime,  # Add timestamp for trade open
            'position_size': self.position_size  # Ensure position_size is always present
        })

    def _close_position(self, price: float) -> float:
        """Close current position and calculate P&L"""
        if not self.current_trade:
            return 0
        entry_price: float = self.current_trade['entry_price']
        # Calculate trading costs
        trading_cost = self._calculate_trading_cost(price, self.current_trade['direction'])
        self.balance -= trading_cost
        # Calculate P&L including trading costs
        pnl: float = (price - entry_price) * self.position * self.position_size * 100000
        pnl -= trading_cost  # Subtract trading costs from P&L
        self.balance += pnl
        # Update metrics
        self._update_metrics(pnl)
        # Update reward and close_timestamp in trade_history for the last trade
        if self.trade_history:
            self.trade_history[-1]['reward'] = pnl
            self.trade_history[-1]['close_timestamp'] = self.current_datetime  # Add close timestamp
            # Ensure position_size is present in the last trade (for legacy/truncated records)
            if 'position_size' not in self.trade_history[-1]:
                self.trade_history[-1]['position_size'] = self.position_size
        self.position = 0
        return pnl

class TradingModel(pl.LightningModule):
    """PyTorch Lightning module for trading model"""
    def __init__(self, learning_rate=3e-4, env=None):
        super().__init__()
        self.save_hyperparameters()
        self.env = env or TradingEnvironment()
        self.model = None
        self.current_reward = 0
        self.scaler = StandardScaler()
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for model training"""
        # Create feature set
        feature_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Filter only available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        
        # Scale features
        df[available_cols] = self.scaler.fit_transform(df[available_cols])
        
        return df
        
    def setup(self, stage=None):
        if stage == 'fit':
            self.model = PPO('MlpPolicy', self.env)
            
    def training_step(self, batch, batch_idx):
        # Training logic with progress bar
        with tqdm(total=1000, desc=f"Training step {batch_idx}") as pbar:
            self.model.learn(
                total_timesteps=1000,
                progress_bar=True,
                callback=lambda locals, globals: pbar.update(1)
            )
        self.log('train_reward', self.current_reward)
        return self.current_reward
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    def save_model(self, filepath: str = None):
        """Save the trained model to disk"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if filepath is None:
            # Create models directory if it doesn't exist
            if not os.path.exists(MODELS_DIRECTORY):
                os.makedirs(MODELS_DIRECTORY)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(MODELS_DIRECTORY, f"ppo_model_{timestamp}")
        
        # Save model
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
        
        return filepath
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load a saved model from disk"""
        model = cls()
        model.model = PPO.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model

class PortfolioOptimizer:
    """Portfolio optimizer for currency pairs"""
    def __init__(self, returns_data: pd.DataFrame):
        self.returns_data = returns_data
        self.mean_returns = None
        self.cov_matrix = None
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess returns data"""
        self.mean_returns = self.returns_data.mean()
        self.cov_matrix = self.returns_data.cov()
        
    def plot_efficient_frontier(self, points=PORTFOLIO_EFFICIENT_FRONTIER_POINTS):
        """Plot the efficient frontier using seaborn"""
        # Calculate efficient frontier points
        frontier_data = self.efficient_frontier(points)
        
        # Create DataFrame for plotting
        frontier_df = pd.DataFrame({
            'Volatility': frontier_data['volatility'],
            'Return': frontier_data['return'],
            'Sharpe Ratio': frontier_data['sharpe_ratio']
        })
        
        # Create the plot
        sns.set_style(PORTFOLIO_PLOT_STYLE)
        plot = sns.relplot(
            data=frontier_df,
            x='Volatility',
            y='Return',
            hue='Sharpe Ratio',
            palette='viridis',
            height=PORTFOLIO_PLOT_HEIGHT,
            aspect=PORTFOLIO_PLOT_ASPECT
        )
        
        # Customize the plot
        plot.fig.suptitle('Efficient Frontier', fontsize=PORTFOLIO_PLOT_TITLE_FONTSIZE, y=1.02)
        plot.ax.set_xlabel('Portfolio Volatility', fontsize=PORTFOLIO_AXIS_FONTSIZE)
        plot.ax.set_ylabel('Expected Return', fontsize=PORTFOLIO_AXIS_FONTSIZE)
        
        return plot

    def plot_correlation_heatmap(self):
        """Plot correlation heatmap using seaborn"""
        # Calculate correlation matrix using imported function
        corr_matrix = cov_to_corr(self.cov_matrix)
        
        # Create the plot
        sns.set_style(PORTFOLIO_PLOT_STYLE)
        plot = sns.heatmap(
            corr_matrix,
            annot=True,
            cmap=PORTFOLIO_CORRELATION_CMAP,
            center=PORTFOLIO_CORRELATION_CENTER,
            square=True,
            fmt=PORTFOLIO_CORRELATION_FMT,
            cbar_kws={'label': 'Correlation'}
        )
        
        # Customize the plot
        plot.set_title('Asset Correlation Matrix', fontsize=PORTFOLIO_PLOT_TITLE_FONTSIZE, pad=20)
        
        return plot

    def plot_returns_distribution(self):
        """Plot returns distribution using seaborn"""
        # Create the plot
        sns.set_style(PORTFOLIO_PLOT_STYLE)
        plot = sns.displot(
            data=self.returns_data,
            kind='kde',
            height=PORTFOLIO_PLOT_HEIGHT,
            aspect=PORTFOLIO_PLOT_ASPECT
        )
        
        # Customize the plot
        plot.fig.suptitle('Returns Distribution', fontsize=PORTFOLIO_PLOT_TITLE_FONTSIZE, y=1.02)
        plot.ax.set_xlabel('Return', fontsize=PORTFOLIO_AXIS_FONTSIZE)
        plot.ax.set_ylabel('Density', fontsize=PORTFOLIO_AXIS_FONTSIZE)
        
        return plot

class CustomRewardWrapper(gym.Wrapper):
    """Custom reward wrapper for PPO model"""
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.last_balance = env.initial_balance
        self.last_position = 0
        self.last_trade_count = 0
        self.last_trade_pnl = 0.0
        self.last_trade_time = None
        self.trade_history = []
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        self.initial_balance = env.initial_balance
        self.consecutive_no_trade_steps = 0
        
    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        
        # Get current state
        current_balance = info['balance']
        current_position = info['position']
        current_trade_count = info['trades']
        current_trade_pnl = info.get('trade_pnl', 0.0)
        trade_executed = info.get('trade_executed', False)
        
        # Calculate reward components
        reward = 0.0
        
        # 1. Balance change reward (primary component)
        balance_change = current_balance - self.last_balance
        if balance_change != 0:
            reward += balance_change / self.last_balance * REWARD_BALANCE_CHANGE_WEIGHT
        
        # 2. Position holding reward/penalty
        if current_position != 0:
            if balance_change > 0:
                reward += REWARD_POSITION_HOLDING_PROFIT * abs(current_position)
            elif balance_change < 0:
                reward -= REWARD_POSITION_HOLDING_LOSS * abs(current_position)
        
        # 3. Trade execution reward/penalty
        if trade_executed:
            self.consecutive_no_trade_steps = 0
            if current_trade_pnl > 0:
                reward += REWARD_TRADE_PROFIT_WEIGHT * (current_trade_pnl / self.last_balance)
            else:
                reward -= REWARD_TRADE_LOSS_WEIGHT * abs(current_trade_pnl / self.last_balance)
        
        # 4. Trade frequency reward
        if current_trade_count > self.last_trade_count:
            reward += REWARD_TRADE_FREQUENCY
        
        # Update last state
        self.last_balance = current_balance
        self.last_position = current_position
        self.last_trade_count = current_trade_count
        self.last_trade_pnl = current_trade_pnl
        
        # Track episode reward
        self.current_episode_reward += reward
        
        # If episode is done, store the episode reward
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
            self.consecutive_no_trade_steps = 0
        
        return observation, reward, done, truncated, info
    
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.last_balance = self.initial_balance
        self.last_position = 0
        self.last_trade_count = 0
        self.last_trade_pnl = 0.0
        self.last_trade_time = None
        self.current_episode_reward = 0.0
        self.consecutive_no_trade_steps = 0
        return observation, info

def optimize_hyperparameters(n_trials=50):
    """Optimize PPO hyperparameters using Optuna"""
    def objective(trial: Trial):
        # Define hyperparameter search space
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'n_steps': trial.suggest_int('n_steps', 512, 2048),
            'batch_size': trial.suggest_int('batch_size', 32, 256),
            'n_epochs': trial.suggest_int('n_epochs', 5, 20),
            'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
            'ent_coef': trial.suggest_float('ent_coef', 0.01, 0.1)
        }
        
        # Create and train model with these parameters
        env = TradingEnvironment()
        model = PPO('MlpPolicy', env, **params)
        model.learn(total_timesteps=10000)
        
        # Evaluate model
        mean_reward = evaluate_model(model)
        return mean_reward

    # Create study and optimize with progress bar
    study = optuna.create_study(direction='maximize')
    with tqdm(total=n_trials, desc="Optimizing hyperparameters") as pbar:
        study.optimize(
            objective, 
            n_trials=n_trials,
            callbacks=[lambda study, trial: pbar.update(1)]
        )
    
    return study.best_params

def create_compatible_env(env):
    """Create a compatible environment using shimmy"""
    return GymV21CompatibilityV0(env)

def evaluate_model(model, n_episodes=10):
    """Evaluate model performance over multiple episodes"""
    env = TradingEnvironment()
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()[0]
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    return np.mean(rewards)

def train_ppo_model(df: pd.DataFrame, n_trials: int = 50, max_epochs: int = 100) -> Dict[str, Any]:
    """
    Train PPO reinforcement learning model with hyperparameter optimization
    
    Parameters
    ----------
    df : pd.DataFrame
        Training data
    n_trials : int
        Number of Optuna trials for hyperparameter optimization
    max_epochs : int
        Maximum number of training epochs
        
    Returns
    -------
    Dict[str, Any]
        Training results and model information
    """
    # Create environment
    env = TradingEnvironment(df)
    compatible_env = create_compatible_env(env)
    
    # Create studies directory if it doesn't exist
    studies_dir = os.path.join(MODELS_DIRECTORY, 'studies')
    if not os.path.exists(studies_dir):
        os.makedirs(studies_dir)
    
    def objective(trial: Trial):
        # Define hyperparameter search space using imported constants
        params = {
            'learning_rate': trial.suggest_float('learning_rate', HP_LEARNING_RATE_MIN, HP_LEARNING_RATE_MAX, log=True),
            'n_steps': trial.suggest_int('n_steps', HP_N_STEPS_MIN, HP_N_STEPS_MAX),
            'batch_size': trial.suggest_int('batch_size', HP_BATCH_SIZE_MIN, HP_BATCH_SIZE_MAX),
            'n_epochs': trial.suggest_int('n_epochs', HP_N_EPOCHS_MIN, HP_N_EPOCHS_MAX),
            'gamma': trial.suggest_float('gamma', HP_GAMMA_MIN, HP_GAMMA_MAX),
            'ent_coef': trial.suggest_float('ent_coef', HP_ENT_COEF_MIN, HP_ENT_COEF_MAX)
        }
        
        # Create model with trial parameters
        model = TradingModel(
            learning_rate=params['learning_rate'],
            env=compatible_env
        )
        
        # Create trainer with trial-specific callbacks
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[
                ModelCheckpoint(
                    monitor='train_reward',
                    mode='max',
                    filename=f'trial_{trial.number}'
                ),
                EarlyStopping(
                    monitor='train_reward',
                    patience=5,
                    mode='max'
                )
            ],
            enable_progress_bar=True
        )
        
        # Train model
        trainer.fit(model)
        
        # Evaluate model
        mean_reward = evaluate_model(model.model)
        
        # Report trial result
        trial.report(mean_reward, epoch=trainer.current_epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
            
        return mean_reward
    
    # Create study and optimize
    study = optuna.create_study(direction='maximize')
    with tqdm(total=n_trials, desc="Optimizing hyperparameters") as pbar:
        study.optimize(
            objective, 
            n_trials=n_trials,
            callbacks=[lambda study, trial: pbar.update(1)]
        )
    
    # Get best trial parameters
    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")
    
    # Save study results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_path = os.path.join(studies_dir, f"study_{timestamp}")
    
    # Save study object
    with open(f"{study_path}.pkl", "wb") as f:
        pickle.dump(study, f)
    
    # Save study visualization
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(f"{study_path}_history.html")
        
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(f"{study_path}_importances.html")
        
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(f"{study_path}_parallel.html")
    except Exception as e:
        logger.warning(f"Failed to create study visualizations: {e}")
    
    # Train final model with best parameters
    final_model = TradingModel(
        learning_rate=best_params['learning_rate'],
        env=compatible_env
    )
    
    final_trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(monitor='train_reward', mode='max'),
            EarlyStopping(monitor='train_reward', patience=5, mode='max')
        ],
        enable_progress_bar=True
    )
    
    final_trainer.fit(final_model)
    
    # Evaluate final model
    mean_reward = evaluate_model(final_model.model)
    logger.info(f"Final mean reward: {mean_reward:.2f}")
    
    # Save model
    model_path = os.path.join(MODELS_DIRECTORY, f"ppo_model_{timestamp}")
    final_model.model.save(model_path)
    
    return {
        'model_path': model_path,
        'mean_reward': mean_reward,
        'hyperparameters': best_params,
        'study_path': study_path
    }