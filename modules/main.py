"""
Main training script for the Forex Trading Bot (Refactored & Corrected)
"""
import logging
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import wandb

# --- Import from local modules ---
from model import TradingEnvironment
from debug import DebugLogger
from data_fetcher import DataFetcher
import config  # Import config module directly

# ===================================================================
# 1. Device Configuration
# ===================================================================

# Determine the computation device based on config, creating the device object correctly.
if config.DEVICE_TYPE.lower() == "directml":
    try:
        import torch_directml
        DEVICE = torch_directml.device()
        logging.info(f"Successfully configured DirectML device: {DEVICE}")
    except ImportError:
        logging.warning("torch_directml not found. Falling back to CPU.")
        DEVICE = "cpu"
    except Exception as e:
        logging.error(f"Error configuring DirectML: {e}. Falling back to CPU.")
        DEVICE = "cpu"
else:
    DEVICE = "cpu"
logging.info(f"Using device: {DEVICE}")


# ===================================================================
# 2. Custom Classes (Features Extractor & Callbacks)
# ===================================================================

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """Custom neural network for extracting features from observations."""
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input = int(np.prod(observation_space.shape))
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(n_input),
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(256),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if torch.isnan(observations).any():
            observations = torch.nan_to_num(observations, nan=0.0)
        return self.network(observations)

class ProgressCallback(BaseCallback):
    """A custom callback that logs to W&B and saves the best model."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        # Log mean reward and episode length to wandb, called at each rollout
        if len(self.model.ep_info_buffer) > 0:
            ep_rew_mean = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            ep_len_mean = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
            
            wandb.log({
                'rollout/ep_rew_mean': ep_rew_mean,
                'rollout/ep_len_mean': ep_len_mean,
            }, step=self.num_timesteps)

            # Save the model if it's the best one we've seen so far
            if ep_rew_mean > self.best_mean_reward:
                self.best_mean_reward = ep_rew_mean
                self.model.save("models/best_model.zip")
        return True

# ===================================================================
# 3. Helper Functions
# ===================================================================

def linear_schedule(initial_value: float):
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def make_env(df, rank=0, seed=0):
    """Utility function for creating the environment."""
    def _init():
        env = TradingEnvironment(df.copy())
        env.reset(seed=seed + rank)
        return Monitor(env)
    set_random_seed(seed)
    return _init

# ===================================================================
# 4. Main Training Function
# ===================================================================

def train():
    """The main function to orchestrate the training process."""
    run = wandb.init(
        project="forex-trading-bot",
        config=config.PPO_PARAMS,
        sync_tensorboard=True
    )
    
    # --- Step 1: Data Fetching ---
    logger.info("--- Starting Data Fetching ---")
    fetcher = DataFetcher()
    df = fetcher.fetch_historical_data(symbol=config.SYMBOL, timeframe=config.TIMEFRAME)
    if df is None or df.empty:
        raise ValueError("FATAL: No historical data loaded. Halting training.")
    
    # --- Step 2: Environment Creation ---
    logger.info("--- Creating Training Environment ---")
    env = DummyVecEnv([make_env(df, rank=0)])

    # --- Step 3: PPO Model Creation ---
    logger.info("--- Creating PPO Model ---")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(config.PPO_LEARNING_RATE),
        device=DEVICE,  # Use the correctly configured device
        policy_kwargs=dict(
            features_extractor_class=CustomFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128)
        ),
        verbose=config.PPO_VERBOSE,
        tensorboard_log=f"runs/{run.id}",
        **config.PPO_PARAMS
    )
    
    # --- Step 4: Model Training ---
    logger.info(f"--- Starting Model Training on device: {DEVICE} ---")
    progress_callback = ProgressCallback()
    
    model.learn(
        total_timesteps=config.MAX_TIMESTEPS,
        callback=progress_callback
    )

    # --- Step 5: Saving Final Model ---
    logger.info("--- Training Finished. Saving Final Model ---")
    model.save("models/final_model.zip")
    wandb.save("models/final_model.zip")
    
    run.finish()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    train()