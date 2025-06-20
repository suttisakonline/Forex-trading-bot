"""
Live trading environment for MetaTrader 5 integration
"""

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime
import time
import logging
import pickle
from typing import Dict, List, Tuple, Optional
from config import MT5_CONFIG, MIN_POSITION_SIZE, MAX_POSITION_SIZE
                    

logger = logging.getLogger(__name__)

class LiveTradingEnvironment:
    def __init__(self, symbol: str = MT5_CONFIG['MT5SYMBOL'], timeframe: str = MT5_CONFIG['MT5TIMEFRAME'], model_path: str = MT5_CONFIG['MODEL_PATH']):
        self.symbol = symbol
        self.timeframe = timeframe
        self.position = 0  # 0: No position, 1: Long, -1: Short
        self.entry_price = 0
        self.last_tick_time = None
        self.trade_history = []
        self.position_size = MIN_POSITION_SIZE  # Start with minimum position size
        
        # Generate unique magic number for this bot instance
        import random
        self.magic = MT5_CONFIG['MAGIC_BASE'] + random.randint(1, 9999)
        
        # Initialize MT5 connection
        if not mt5.initialize():
            raise RuntimeError("Failed to initialize MT5")
        
        # Get symbol info
        self.symbol_info = mt5.symbol_info(symbol)
        if self.symbol_info is None:
            raise ValueError(f"Symbol {symbol} not found")
        
        # Enable symbol for trading
        if not self.symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                raise RuntimeError(f"Failed to select {symbol}")
        
        # Load trained model using generic loader
        self.model = self._load_model(model_path)
        
        logger.info(f"Initialized live trading for {symbol} using model from {model_path}")
        logger.info(f"Bot instance magic number: {self.magic}")
    
    def _load_model(self, model_path: str):
        """Load model using generic approach"""
        try:
            # Try loading as stable-baselines3 model first
            from stable_baselines3 import PPO
            return PPO.load(model_path)
        except Exception as e:
            logger.warning(f"Failed to load as stable-baselines3 model: {e}")
            try:
                # Try loading as pickle file
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e2:
                logger.error(f"Failed to load model from {model_path}: {e2}")
                raise RuntimeError(f"Cannot load model from {model_path}")
    
    def _get_mt5_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get market data directly from MT5"""
        try:
            # Convert timeframe string to MT5 constant
            timeframe_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            
            mt5_timeframe = timeframe_map.get(self.timeframe, mt5.TIMEFRAME_M15)
            
            # Get rates from MT5 using the symbol
            rates = mt5.copy_rates_range(self.symbol, mt5_timeframe, start_time, end_time)
            
            if rates is None or len(rates) == 0:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get MT5 data for {self.symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_dynamic_deviation(self) -> int:
        """Calculate deviation based on current market volatility"""
        try:
            # Get recent price data to calculate volatility
            end_time = datetime.now()
            start_time = end_time - pd.Timedelta(hours=1)  # Last hour of data
            
            df = self._get_mt5_data(start_time, end_time)
            
            if df.empty or len(df) < 10:
                return MT5_CONFIG['BASE_DEVIATION']
            
            # Calculate price volatility (standard deviation of price changes)
            price_changes = df['close'].pct_change().dropna()
            volatility = price_changes.std()
            
            # Convert volatility to deviation multiplier
            # Higher volatility = higher deviation needed
            volatility_multiplier = 1 + (volatility * 100)  # Scale volatility
            
            # Calculate dynamic deviation
            dynamic_deviation = int(MT5_CONFIG['BASE_DEVIATION'] * volatility_multiplier)
            
            # Ensure deviation stays within bounds
            dynamic_deviation = max(MT5_CONFIG['MIN_DEVIATION'], 
                                  min(MT5_CONFIG['MAX_DEVIATION'], dynamic_deviation))
            
            logger.debug(f"Volatility: {volatility:.4f}, Deviation: {dynamic_deviation}")
            return dynamic_deviation
            
        except Exception as e:
            logger.warning(f"Failed to calculate dynamic deviation: {e}, using base deviation")
            return MT5_CONFIG['BASE_DEVIATION']
    
    def get_observation(self) -> np.ndarray:
        """Get current market state observation"""
        # Get recent market data
        end_time = datetime.now()
        start_time = end_time - pd.Timedelta(minutes=30)  # Get more data to ensure we have enough
        
        df = self._get_mt5_data(start_time, end_time)
        
        if df.empty:
            logger.error("No data received from MT5")
            raise RuntimeError("Failed to get market data")
        
        logger.info(f"Received {len(df)} data points from MT5")
        
        # Ensure we have enough data points
        if len(df) < 20:
            logger.warning(f"Not enough data points ({len(df)}), padding with last known values")
            # Pad with the last known values
            last_values = df.iloc[-1]
            padding = pd.DataFrame([last_values] * (20 - len(df)), index=pd.date_range(end=df.index[-1], periods=20-len(df), freq='1min'))
            df = pd.concat([df, padding])
        
        # Take the last 20 points
        df = df.tail(20)
        
        # Create observation array with shape (10, 20)
        observation = np.zeros((10, 20))
        
        # Use basic OHLCV features (no technical indicators for live trading)
        features = ['open', 'high', 'low', 'close', 'tick_volume']
        
        # Fill observation array with normalized features
        for i, feature in enumerate(features):
            if feature in df.columns:
                values = df[feature].values
                # Handle NaN values
                values = np.nan_to_num(values, nan=0.0)
                # Normalize values
                mean = np.mean(values)
                std = np.std(values)
                if std == 0:
                    std = 1
                normalized = (values - mean) / std
                # Clip values to prevent extreme values
                normalized = np.clip(normalized, -10, 10)
                observation[i] = normalized
        
        # Ensure no NaN values in final observation
        observation = np.nan_to_num(observation, nan=0.0)
        
        return observation
    
    def _calculate_position_size(self):
        """Calculate position size based on win rate"""
        if not self.trade_history:
            return MIN_POSITION_SIZE
        
        # Calculate win rate
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for trade in self.trade_history if trade['profit'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate position size based on win rate
        # Linear scaling from MIN_POSITION_SIZE to MAX_POSITION_SIZE based on win rate
        position_size = MIN_POSITION_SIZE + (MAX_POSITION_SIZE - MIN_POSITION_SIZE) * win_rate
        
        # Ensure position size stays within bounds
        return np.clip(position_size, MIN_POSITION_SIZE, MAX_POSITION_SIZE)
    
    def execute_trade(self, action: int):
        """Execute trading action based on model decision"""
        current_price = mt5.symbol_info_tick(self.symbol).ask
        
        # Hold
        if action == 0:
            return
        
        # Buy
        elif action == 1 and self.position <= 0:
            if self.position < 0:
                self._close_position()
            
            # Update position size based on win rate
            self.position_size = self._calculate_position_size()
            
            # Open long position with calculated size
            self._open_position(mt5.ORDER_TYPE_BUY, self.position_size)
        
        # Sell
        elif action == 2 and self.position >= 0:
            if self.position > 0:
                self._close_position()
            
            # Update position size based on win rate
            self.position_size = self._calculate_position_size()
            
            # Open short position with calculated size
            self._open_position(mt5.ORDER_TYPE_SELL, self.position_size)
        
        # Close
        elif action == 3 and self.position != 0:
            self._close_position()
    
    def run(self):
        """Main trading loop"""
        logger.info(f"Starting live trading for {self.symbol}")
        
        while True:
            try:
                # Get current market state
                observation = self.get_observation()
                
                # Get model's action
                action, _ = self.model.predict(observation, deterministic=True)
                
                # Execute trade
                self.execute_trade(action)
                
                # Wait for next tick
                self.wait_for_next_tick()
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(1)  # Wait before retrying
    
    def wait_for_next_tick(self):
        """Wait for next market tick"""
        while True:
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                time.sleep(0.1)
                continue
            
            if self.last_tick_time is None or tick.time > self.last_tick_time:
                self.last_tick_time = tick.time
                break
            
            time.sleep(0.1)
    
    def _open_position(self, order_type: int, position_size: float):
        """Open a new position"""
        price = mt5.symbol_info_tick(self.symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": position_size,
            "type": order_type,
            "price": price,
            "deviation": self.calculate_dynamic_deviation(),
            "magic": self.magic,
            "comment": MT5_CONFIG['COMMENT'] + " open",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to open position: {result.comment}")
            return False
        
        self.position = 1 if order_type == mt5.ORDER_TYPE_BUY else -1
        self.entry_price = price
        logger.info(f"Opened {'long' if order_type == mt5.ORDER_TYPE_BUY else 'short'} position at {price} with size {position_size}")
        return True
    
    def _close_position(self):
        """Close current position"""
        ticket = self._get_position_ticket()
        if ticket is None:
            return False
        
        position = mt5.positions_get(ticket=ticket)[0]
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": mt5.symbol_info_tick(self.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).ask,
            "deviation": self.calculate_dynamic_deviation(),
            "magic": self.magic,
            "comment": MT5_CONFIG['COMMENT'] + " close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close position: {result.comment}")
            return False
        
        # Calculate profit
        profit = position.profit
        
        # Update trade history
        self.trade_history.append({
            'type': 'long' if position.type == mt5.ORDER_TYPE_BUY else 'short',
            'entry_price': self.entry_price,
            'exit_price': result.price,
            'profit': profit,
            'timestamp': datetime.now()
        })
        
        self.position = 0
        self.entry_price = 0
        logger.info(f"Closed position at {result.price} with profit {profit}")
        return True
    
    def _get_position_ticket(self) -> int:
        """Get current position ticket"""
        positions = mt5.positions_get(symbol=self.symbol)
        return positions[0].ticket if positions else None
    
    def _get_account_balance(self) -> float:
        """Get current account balance"""
        account_info = mt5.account_info()
        return account_info.balance if account_info else 0.0

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize MT5 connection
    if not mt5.initialize(
        login=MT5_CONFIG['MT5LOGIN'],
        password=MT5_CONFIG['MT5PASSWORD'],
        server=MT5_CONFIG['MT5SERVER']
    ):
        logger.error("Failed to initialize MT5")
        exit(1)
    
    try:
        # Create trading environment
        trader = LiveTradingEnvironment(
            symbol=MT5_CONFIG['MT5SYMBOL'],  # Use symbol from MT5 config
            timeframe=MT5_CONFIG['MT5TIMEFRAME'],  # Use timeframe from MT5 config
            model_path=MT5_CONFIG['MODEL_PATH']  # Use model path from config
        )
        
        # Start trading
        trader.run()
        
    except KeyboardInterrupt:
        logger.info("Trading stopped by user")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        # Shutdown MT5
        mt5.shutdown() 