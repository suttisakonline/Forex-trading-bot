from dotenv import load_dotenv
import os


load_dotenv()

"""
Configuration module for the Forex Trading Bot
- Contains shared constants and configuration values
- Each parameter is commented to explain its purpose and effect
"""

import os
from pathlib import Path
from datetime import datetime, timedelta

# === 1. Data Download/Source (Dukascopy) ===

# --- การแก้ไขที่ถูกต้องตาม Library ที่ติดตั้งไว้ ---
# Import ตัวแปร Timeframe ที่ถูกต้องจาก Library โดยตรง
from dukascopy_python import (
    INTERVAL_MIN_1,
    INTERVAL_MIN_5,
    INTERVAL_HOUR_1,
    OFFER_SIDE_BID
)

# ใช้ชื่อย่อของ Symbol เป็น String (ข้อความ)
SYMBOL_MAP = {
    'EURUSD',
    'GBPUSD',
    'USDJPY',
    'XAUUSD',
}

# เรียกใช้ตัวแปร Timeframe ที่ import มาอย่างถูกต้อง
TIMEFRAME_MAP = {
    INTERVAL_MIN_1,   # สำหรับ M1
    INTERVAL_MIN_5,   # สำหรับ M5
    INTERVAL_HOUR_1,  # สำหรับ H1
}

YEARS = 5  # Number of years of historical data to download

# === 2. Directory Paths ===
from pathlib import Path

# === 2. Directory Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent  # Project root directory
DATA_DIRECTORY = BASE_DIR / "data"
MODELS_DIRECTORY = BASE_DIR / "models"
REPORTS_DIRECTORY = BASE_DIR / "reports"
LOGS_DIRECTORY = BASE_DIR / "logs"
LOG_FILE_PATH = LOGS_DIRECTORY / "swing_trading.log"

# Create directories if they don't exist
for directory in [DATA_DIRECTORY, MODELS_DIRECTORY, REPORTS_DIRECTORY, LOGS_DIRECTORY]:
    directory.mkdir(parents=True, exist_ok=True)

    

# === Device/GPU Settings ===
DEVICE_TYPE = "cpu"  # Device for training/inference ("cpu" or "directml"(gpu))
OPENCL_COMPILER_OUTPUT = '0'  # OpenCL compiler output setting


# === 3. Trading Parameters ===
SYMBOL = 'GBPUSD'  # Default trading symbol
TIMEFRAME = INTERVAL_HOUR_1  # Default trading timeframe
INITIAL_BALANCE = 10000  # Starting account balance in USD
EPISODE_LENGTH = 90  # Number of days per training episode
SPREAD = 0.00005  # Spread in price (2 pips)
COMMISSION = 0.0001  # Commission per trade (0.01%)
SLIPPAGE = 0.00005  # Slippage in price (0.5 pips)
OFFER_SIDE = OFFER_SIDE_BID 

# === 4. Position Sizing ===
MIN_POSITION_SIZE = 0.1  # Minimum position size in lots
MAX_POSITION_SIZE = 0.5  # Maximum position size in lots
POSITION_SIZE_INCREMENT = 0.1  # Increment for position sizing

# === 5. Risk Management ===
MAX_TRADES_PER_WEEK = 100  # Max trades allowed per week
MAX_DAILY_TRADES = 100  # Max trades allowed per day
MAX_WEEKLY_TRADES = 200  # Max trades allowed per week (hard cap)
PROFIT_TARGET = 0.02  # Profit target per trade (2%)
STOP_LOSS = 0.01  # Stop loss per trade (1%)

# === 6. Training/Backtesting Parameters ===
MAX_EPISODES = 50000  # Maximum number of training episodes
MAX_TIMESTEPS = 5000000  # Maximum number of training timesteps
MIN_EPISODES = 1000  # Minimum number of training episodes
TARGET_WEEKLY_PROFIT = 1000.0  # Target profit per week in USD
BATCH_SIZE = 128  # Training batch size
N_STEPS = 1024  # Number of steps per PPO update
GRADIENT_ACCUMULATION_STEPS = 1  # Gradient accumulation steps

# === 7. Reward Shaping ===
REWARD_SHAPING = {
    'profit_multiplier': 1.5,  # Multiplier for profit rewards
    'loss_penalty': 1.0,  # Penalty for losses
    'win_streak_bonus': 0.2,  # Bonus for consecutive wins
    'pattern_recognition_bonus': 0.05,  # Bonus for recognizing patterns
    'position_size_bonus': 0.05,  # Bonus for optimal position size
    'trade_frequency_bonus': -0.1,  # Penalty for overtrading
    'weekly_profit_bonus': 0.0,  # Bonus for weekly profit
    'drawdown_penalty': 0.5,  # Penalty for drawdown
    'weekly_trade_bonus': 0.0,  # Penalty for not trading weekly
    'profit_target_bonus': 0.5  # Bonus for hitting profit target
}
REWARD_BALANCE_CHANGE_WEIGHT = 1.0  # Weight for balance change reward
REWARD_POSITION_HOLDING_PROFIT = 0.0002  # Reward for holding profitable position
REWARD_POSITION_HOLDING_LOSS = 0.0004  # Penalty for holding losing position
REWARD_TRADE_PROFIT_WEIGHT = 0.0  # Weight for profitable trade reward
REWARD_TRADE_LOSS_WEIGHT = 0.0  # Weight for losing trade penalty
REWARD_TRADE_FREQUENCY = 0.0  # Small reward for executing trades

# === 8. RL/PPO/Model Hyperparameters ===
RL_TRAINING_STEPS = 5000000  # Total RL training steps
RL_WINDOW_SIZE = 48  # Window size for RL state
PPO_LEARNING_RATE = 0.0003  # PPO learning rate
PPO_N_STEPS = 2048  # PPO steps per update
PPO_BATCH_SIZE = 128  # PPO batch size
PPO_N_EPOCHS = 8  # PPO epochs per update
PPO_GAMMA = 0.985  # PPO discount factor
PPO_GAE_LAMBDA = 0.95  # PPO GAE lambda
PPO_CLIP_RANGE = 0.2  # PPO clip range
PPO_ENT_COEF = 0.015  # PPO entropy coefficient
PPO_VF_COEF = 0.5  # PPO value function coefficient
PPO_MAX_GRAD_NORM = 0.5  # PPO max gradient norm
PPO_TARGET_KL = 0.02  # PPO target KL divergence
PPO_USE_SDE = False  # PPO state-dependent exploration
PPO_SDE_SAMPLE_FREQ = -1  # PPO SDE sample frequency
PPO_VERBOSE = 1  # PPO verbosity level
PPO_PARAMS = {
    'n_epochs': 8,  # PPO epochs per update
    'gamma': 0.985,  # PPO discount factor
    'gae_lambda': 0.95,  # PPO GAE lambda
    'clip_range': 0.2,  # PPO clip range
    'ent_coef': 0.015,  # PPO entropy coefficient
    'vf_coef': 0.5,  # PPO value function coefficient
    'max_grad_norm': 0.5,  # PPO max gradient norm
    'use_sde': False,  # PPO state-dependent exploration
    'sde_sample_freq': -1,  # PPO SDE sample frequency
    'target_kl': 0.02,  # PPO target KL divergence
}

# === 9. MetaTrader 5/Live Trading Config ===
MT5_CONFIG = {
    'MT5LOGIN': os.getenv('MT5LOGIN'),  # MT5 account number
    'MT5PASSWORD': os.getenv('MT5PASSWORD'),  # MT5 password
    'MT5SERVER': os.getenv('MT5SERVER'),  # MT5 broker server
    'MT5SYMBOL': "EURUSD",  # MT5 trading symbol
    'MT5TIMEFRAME': "M15",  # MT5 trading timeframe
    'MODEL_PATH': str(Path(__file__).parent.parent / "models" / "forex_model_final.zip"),  # Path to trained model
    'BASE_DEVIATION': 20,  # Base deviation in points
    'MAX_DEVIATION': 50,  # Max allowed deviation in points
    'MIN_DEVIATION': 10,  # Min allowed deviation in points
    'MAGIC_BASE': 234000,  # Base magic number for orders
    'COMMENT': "python script"  # Order comment for MT5
}

# === 10. Optimization/Hyperparameter Search ===
MODEL_PARAMS = {
    'n_estimators': 100,  # Number of trees in ensemble models
    'learning_rate': 0.05,  # Learning rate for boosting
    'max_depth': 4,  # Maximum tree depth
    'min_samples_split': 20,  # Minimum samples to split a node
    'min_samples_leaf': 10,  # Minimum samples at a leaf node
    'subsample': 0.8,  # Fraction of samples for fitting each tree
    'random_state': 42  # Random seed for reproducibility
}
TRAINING_PARAMS = {
    'test_size': 0.2,  # Fraction of data for testing
    'random_state': 42,  # Random seed
    'cv_folds': 5  # Number of cross-validation folds
}
RL_PARAMS = {
    'learning_rate': 0.0003,  # PPO learning rate
    'n_steps': 2048,  # PPO steps per update
    'batch_size': 64,  # PPO batch size
    'n_epochs': 10,  # PPO epochs per update
    'gamma': 0.99,  # PPO discount factor
    'gae_lambda': 0.95,  # PPO GAE lambda
    'clip_range': 0.2,  # PPO clip range
    'ent_coef': 0.01,  # PPO entropy coefficient
    'vf_coef': 0.5,  # PPO value function coefficient
    'max_grad_norm': 0.5  # PPO max gradient norm
}
HP_LEARNING_RATE_MIN = 1e-5  # Min learning rate for search
HP_LEARNING_RATE_MAX = 1e-3  # Max learning rate for search
HP_N_STEPS_MIN = 512  # Min steps for search
HP_N_STEPS_MAX = 2048  # Max steps for search
HP_BATCH_SIZE_MIN = 32  # Min batch size for search
HP_BATCH_SIZE_MAX = 256  # Max batch size for search
HP_N_EPOCHS_MIN = 5  # Min epochs for search
HP_N_EPOCHS_MAX = 20  # Max epochs for search
HP_GAMMA_MIN = 0.9  # Min gamma for search
HP_GAMMA_MAX = 0.9999  # Max gamma for search
HP_ENT_COEF_MIN = 0.01  # Min entropy coef for search
HP_ENT_COEF_MAX = 0.1  # Max entropy coef for search
BAYESIAN_N_CALLS = 30  # Number of Bayesian optimization calls
GRID_SEARCH_PARAMS = {
    'learning_rate': [0.0001, 0.0003, 0.001],  # Learning rates for grid search
    'n_steps': [1024, 2048, 4096],  # Steps for grid search
    'batch_size': [32, 64, 128],  # Batch sizes for grid search
    'n_epochs': [5, 10, 20],  # Epochs for grid search
    'gamma': [0.95, 0.99, 0.995],  # Discount factors for grid search
    'gae_lambda': [0.9, 0.95, 0.98],  # GAE lambdas for grid search
    'clip_range': [0.1, 0.2, 0.3],  # Clip ranges for grid search
    'ent_coef': [0.005, 0.01, 0.02],  # Entropy coefs for grid search
    'vf_coef': [0.3, 0.5, 0.7]  # Value function coefs for grid search
}

# === 11. Analytics/Visualization/Portfolio ===
PORTFOLIO_EFFICIENT_FRONTIER_POINTS = 20  # Points for efficient frontier plot
PORTFOLIO_PLOT_HEIGHT = 8  # Height of portfolio plots
PORTFOLIO_PLOT_ASPECT = 1.5  # Aspect ratio of portfolio plots
PORTFOLIO_PLOT_STYLE = "whitegrid"  # Style for portfolio plots
PORTFOLIO_CORRELATION_CMAP = 'coolwarm'  # Colormap for correlation heatmap
PORTFOLIO_CORRELATION_CENTER = 0  # Center value for correlation heatmap
PORTFOLIO_CORRELATION_FMT = '.2f'  # Format for correlation values
PORTFOLIO_PLOT_TITLE_FONTSIZE = 16  # Font size for plot titles
PORTFOLIO_AXIS_FONTSIZE = 12  # Font size for axis labels


# === 12. Other/Advanced ===
LOG_FILE_PATH = 'logs/swing_trading.log'  # Main log file path
DATA_CSV_PATH = 'data/{SYMBOL}_{{start}_{{end}.csv'  # Path to historical data CSV # type: ignore
INITIAL_LR = 3e-4  # Initial learning rate for schedule
FINAL_LR = 1e-4  # Final learning rate for schedule
MIN_LR = 5e-5  # Minimum learning rate for schedule
WARMUP_STEPS = 0.1  # Fraction of training for learning rate warmup
GENETIC_POPULATION_SIZE = 100  # Population size for genetic algorithm
GENETIC_GENERATIONS = 50  # Number of generations for genetic algorithm

# --- PPO_PARAMS vs GRID_SEARCH_PARAMS ---
# PPO_PARAMS contains the actual hyperparameter values used for a single PPO training run.
# GRID_SEARCH_PARAMS contains lists/ranges of possible values for each hyperparameter, used for hyperparameter search (e.g., grid search, random search, Bayesian optimization).
#
# Best practice: Do NOT reference GRID_SEARCH_PARAMS directly in PPO_PARAMS.
# Instead, your search/experiment code should dynamically create or update PPO_PARAMS for each trial using values from GRID_SEARCH_PARAMS.
# Example:
#   for n_epochs in GRID_SEARCH_PARAMS['n_epochs']:
#       params = PPO_PARAMS.copy()
#       params['n_epochs'] = n_epochs
#       model = PPO(..., **params)
#
# This keeps your config clean, flexible, and easy to maintain.


