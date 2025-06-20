# Forex Trading Bot with Reinforcement Learning

A sophisticated Forex trading bot that uses Proximal Policy Optimization (PPO) reinforcement learning to make trading decisions. The bot features dynamic position sizing, AMD GPU acceleration via DirectML and OpenCL (specifically for AMD GPUs like the RX 6700/gfx1031, which do not support AMD ROCm), and live trading integration with MetaTrader 5.

GPU Acceleration (AMD) WSL doesn’t support DirectML or AMD ROCm. So GPU acceleration won’t work unless you’re using WSL2 with NVIDIA GPU + CUDA. Since you’re on AMD, you’re limited to CPU-only PyTorch inside WSL.

## ⚡️ Data Provider: Dukascopy (NEW)

**This project now uses [Dukascopy](https://www.dukascopy.com/) as the sole source for historical forex data.**

- Data is downloaded directly from Dukascopy in multi-year intervals (chunks) to avoid server and memory issues.
- All chunks are concatenated to create a single, continuous historical dataset covering the full requested date range (e.g., 10 years).
- The resulting CSV is saved in the `data/` directory (e.g., `data/EUR_USD_20150622_20250619.csv`).
- No API key or .env setup is required for data.

### How the Download Works

- The bot automatically downloads historical data for the configured symbol and timeframe.
- Data is fetched in 3-year intervals (by default) to avoid timeouts and large memory usage.
- Each chunk is a DataFrame with a datetime index and OHLCV columns.
- All chunks are concatenated, the datetime index is converted to a `time` column, and the final DataFrame is saved as a CSV.
- The CSV always includes: `time, open, high, low, close, volume`.

**Example CSV header:**
```csv
time,open,high,low,close,volume
2015-06-23 02:00:00+00:00,1.13214,1.13235,1.12549,1.12602,10090.10
...
```

**You do not need to manually download or prepare data.** The bot will handle all data fetching and formatting automatically.

## 🚀 Features

- **Reinforcement Learning**: Uses PPO algorithm for autonomous trading decisions
- **Dynamic Position Sizing**: Configurable position sizes based on market conditions and risk management
- **AMD GPU Acceleration**: Optimized for AMD hardware using DirectML and OpenCL (AMD ONLY)
- **Live Trading**: Direct integration with MetaTrader 5 for real-time trading
- **Comprehensive Analytics**: Detailed performance tracking and visualization
- **Risk Management**: Built-in stop-loss, take-profit, and trade frequency limits
- **Hyperparameter Optimization**: Automated tuning using Optuna

## ⚠️ Note on GPU Acceleration and test_directml.py

- **test_directml.py is currently not working** due to a bug in the latest torch-directml package ("TypeError: 'staticmethod' object is not callable").
- This bug affects static methods like `has_float64_support` and `gpu_memory` in torch-directml, causing the test script to fail.
- As a result, the backtesting and training will **default to using the CPU** until this issue is fixed in a future torch-directml release.
- This is a known issue and has been reported to the DirectML GitHub repository. If you need GPU acceleration, monitor the [DirectML GitHub issues](https://github.com/microsoft/DirectML/issues) for updates or fixes.

## Technologies and Libraries for GPU Acceleration

- **DirectML**: Provides hardware-accelerated deep learning on AMD GPUs (Windows only)
- **OpenCL**: Used for some low-level GPU operations and compatibility checks ([Khronos Group GitHub](https://github.com/KhronosGroup/OpenCL-ICD-Loader))
- **PyTorch**: Main deep learning framework; can use DirectML as a backend for AMD GPU support
- **torch-directml**: PyTorch extension enabling DirectML backend ([GitHub](https://github.com/microsoft/DirectML))
- **NumPy, pandas**: For data processing (CPU, but compatible with GPU workflows)

> **Note:**
> - This project does **not** use or require torchvision.
> - This project is built and tested for AMD GPUs like the RX 6700 (gfx1031) that do **not** support AMD ROCm.
> - GPU acceleration is achieved via OpenCL and DirectML, not ROCm.
> - To use Nvidia GPUs, you must modify the code to use the standard CUDA backend in PyTorch.
> - **GPU acceleration is tested and working with torch==2.0.1 and torch-directml==0.2.0.dev230426 as of June 2024.**

## Project Structure

```
Forex Trading Bot/
├── main.py                # Main entry point for the application
├── modules/               # Modular components
│   ├── __init__.py        # Module initialization
│   ├── config.py          # Configuration settings
│   ├── data_processing.py # Data fetching and preprocessing
│   ├── economic_calendar.py # Economic event handling
│   ├── execution.py       # Trade execution and management
│   ├── logger.py          # Logging utilities
│   ├── model.py           # Machine learning models
│   ├── signals.py         # Trading signal generation
│   └── visualization.py   # Performance visualization
├── data/                  # Stored market data
├── logs/                  # Application logs
├── models/                # Saved ML models
└── reports/               # Performance reports and charts
```

The bot consists of several key modules:
- **`modules/config.py`**: **All configurable variables** - trading parameters, model settings, risk management, etc.
- **`modules/environment.py`**: Trading environment for reinforcement learning
- **`modules/model.py`**: PPO model implementation with PyTorch Lightning
- **`modules/data_fetcher.py`**: Historical and real-time data fetching
- **`modules/live_trading.py`**: MetaTrader 5 integration for live trading
- **`modules/debug.py`**: Comprehensive analytics and visualization
- **`modules/logger.py`**: Application logging

## Setup

### Prerequisites

- Python 3.9+
- MetaTrader 5 account
- AMD GPU (for acceleration) 

### Weights & Biases (wandb) Setup

This project uses [Weights & Biases (wandb)](https://wandb.ai/) for experiment tracking, logging, and visualization of training runs and results.

Before running any training or backtesting, you will need to log in to your wandb account in the terminal:

```bash
wandb login
```

This command will prompt you for your wandb API key and authenticate your machine. It enables automatic logging of metrics, model checkpoints, and visualizations to your wandb dashboard for easy monitoring and analysis.

If you do not have a wandb account, you can create one for free at https://wandb.ai/.

## Data Provider
This bot now loads historical forex data from a single CSV file located in the `data/` directory (default: `data/forex_data.csv`).

**CSV format requirements:**
- Columns: `open`, `high`, `low`, `close`, `volume`, and either `time` or `date` (datetime).
- The datetime column will be used as the DataFrame index.
- Example header:
  ```csv
  time,open,high,low,close,volume
  2024-01-01 00:00:00,1.1000,1.1010,1.0990,1.1005,1000
  ...
  ```

You can use your own data or download from any source and save it in this format.

## Technologies and Libraries for GPU Acceleration

- **DirectML**: Provides hardware-accelerated deep learning on AMD GPUs (Windows only)
- **OpenCL**: Used for some low-level GPU operations and compatibility checks ([Khronos Group GitHub](https://github.com/KhronosGroup/OpenCL-ICD-Loader))
- **PyTorch**: Main deep learning framework; can use DirectML as a backend for AMD GPU support
- **torch-directml**: PyTorch extension enabling DirectML backend ([GitHub](https://github.com/microsoft/DirectML))
- **NumPy, pandas**: For data processing (CPU, but compatible with GPU workflows)

> **Note:**
> - This project does **not** use or require torchvision.
> - This project is built and tested for AMD GPUs like the RX 6700 (gfx1031) that do **not** support AMD ROCm.
> - GPU acceleration is achieved via OpenCL and DirectML, not ROCm.
> - To use Nvidia GPUs, you must modify the code to use the standard CUDA backend in PyTorch.
> - **GPU acceleration is tested and working with torch==2.0.1 and torch-directml==0.2.0.dev230426 as of June 2024.**

## Project Structure

```
Forex Trading Bot/
├── main.py                # Main entry point for the application
├── modules/               # Modular components
│   ├── __init__.py        # Module initialization
│   ├── config.py          # Configuration settings
│   ├── data_processing.py # Data fetching and preprocessing
│   ├── economic_calendar.py # Economic event handling
│   ├── execution.py       # Trade execution and management
│   ├── logger.py          # Logging utilities
│   ├── model.py           # Machine learning models
│   ├── signals.py         # Trading signal generation
│   └── visualization.py   # Performance visualization
├── data/                  # Stored market data
├── logs/                  # Application logs
├── models/                # Saved ML models
└── reports/               # Performance reports and charts
```

The bot consists of several key modules:
- **`modules/config.py`**: **All configurable variables** - trading parameters, model settings, risk management, etc.
- **`modules/environment.py`**: Trading environment for reinforcement learning
- **`modules/model.py`**: PPO model implementation with PyTorch Lightning
- **`modules/data_fetcher.py`**: Historical and real-time data fetching
- **`modules/live_trading.py`**: MetaTrader 5 integration for live trading
- **`modules/debug.py`**: Comprehensive analytics and visualization
- **`modules/logger.py`**: Application logging

## Setup

### Prerequisites

- Python 3.9+
- MetaTrader 5 account
- AMD GPU (for acceleration) 

### Weights & Biases (wandb) Setup

This project uses [Weights & Biases (wandb)](https://wandb.ai/) for experiment tracking, logging, and visualization of training runs and results.

Before running any training or backtesting, you will need to log in to your wandb account in the terminal:

```bash
wandb login
```

This command will prompt you for your wandb API key and authenticate your machine. It enables automatic logging of metrics, model checkpoints, and visualizations to your wandb dashboard for easy monitoring and analysis.

If you do not have a wandb account, you can create one for free at https://wandb.ai/.

## Data Source

This project now uses [Twelve Data](https://twelvedata.com/) for all historical and real-time forex data fetching. You must obtain a free API key from Twelve Data to use the bot.

### Setup
- Install the required packages:
  ```bash
  pip install twelvedata requests python-dotenv
  ```
- Add your Twelve Data API key to your configuration (see `modules/config.py`).
- Or, create a `.env` file at the project root with:
  ```env
  TWELVE_DATA_API_KEY=your_twelvedata_api_key
  ```

### Installation

1. Clone the repository
```
git clone https://github.com/Stefodan21/forex-trading-bot.git
cd forex-trading-bot
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Test GPU acceleration:
```bash
python test_directml.py
```

### Configuration

**All configurable variables are located in `modules/config.py`**, including:

- **Trading Parameters**: Symbols, timeframes, position sizes, trade limits
- **Risk Management**: Stop-loss, take-profit, maximum trades
- **Model Settings**: Learning rates, batch sizes, training parameters
- **MetaTrader 5**: Account credentials, server settings
- **Reward System**: Reward weights and penalties
- **Optimization**: Hyperparameter search spaces

> **Tip:** If model training uses too much memory or is too slow for your hardware, you can adjust a variety of parameters in `modules/config.py` to better fit your system:
> - `BATCH_SIZE`, `N_STEPS`, `PPO_BATCH_SIZE`, `PPO_N_STEPS`: Lowering these reduces memory usage and can speed up training.
> - `PPO_LEARNING_RATE`, `PPO_N_EPOCHS`, `PPO_GAMMA`, `PPO_CLIP_RANGE`, `PPO_ENT_COEF`, `PPO_VF_COEF`, `PPO_MAX_GRAD_NORM`, `PPO_TARGET_KL`, `PPO_USE_SDE`, `PPO_SDE_SAMPLE_FREQ`, `PPO_VERBOSE`: Tuning these can help optimize training performance and stability.
> - You can also adjust other training, model, and risk management parameters to suit your hardware and trading goals.
> - Lower values for batch size and steps are recommended for systems with less RAM or VRAM.

### Environment Variables

Create a `.env` file in the project root for secure credential storage:

I recommend you use a demo account credentials first to test the model after it has finished the backtesting in a demo environment before changing it to live account credentials
```env
MT5_LOGIN=your_mt5_account_number
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_broker_server
```

**Important**: Add `.env` to your `.gitignore` to keep credentials secure.

## Usage

### Backtesting/Training

To train and backtest the model on the number of years of historical data downloaded from Dukascopy, run:

```bash
cd modules
python main.py
```

### Live Trading

To run the bot in live trading mode with MetaTrader 5, use:

```bash
cd modules
python live_trading.py
```

> **Note:**
> - `main.py` is dedicated to backtesting and training only.
> - `live_trading.py` is dedicated to live trading only.
> - There is no need to use any `--mode` argument; each script is single-purpose.

## Configuration Parameters

**All parameters are configurable in `modules/config.py`**:

### Trading Configuration
- `TRADING_SYMBOL`: Default trading symbol (EURUSD)
- `TIMEFRAME`: Data timeframe (M15)
- `MIN_POSITION_SIZE` / `MAX_POSITION_SIZE`: Dynamic position sizing range
- `WEEKLY_TRADE_LIMIT`: Maximum trades per week
- `INITIAL_BALANCE`: Starting account balance
- `YEARS`: Number of years of historical data to use for backtesting/training

### Risk Management
- `STOP_LOSS`: Stop loss percentage per trade
- `PROFIT_TARGET`: Take profit percentage per trade
- `MAX_DAILY_TRADES`: Daily trade limit
- `MAX_WEEKLY_TRADES`: Weekly trade limit

### Model Parameters
- `PPO_LEARNING_RATE`: Learning rate for PPO algorithm
- `PPO_BATCH_SIZE`: Training batch size
- `PPO_N_EPOCHS`: Number of training epochs
- `PPO_GAMMA`: Discount factor for future rewards

### MetaTrader 5 Settings
- `MT5_CONFIG`: Account credentials and server settings
- `BASE_DEVIATION`: Order execution deviation
- `MAGIC_BASE`: Unique order identifier

## Performance Tracking

The bot provides comprehensive analytics through the `debug.py` module:

- Real-time performance metrics
- Trade analysis and statistics
- Portfolio visualization
- Risk-adjusted returns
- Drawdown analysis
- Position sizing tracking

## Troubleshooting

### GPU Issues
- Ensure AMD drivers are up to date
- Verify DirectML installation: `python test_directml.py`
- Check OpenCL compatibility

### MetaTrader 5 Connection
- Verify account credentials in `.env` file
- Ensure MT5 is running and logged in
- Check server settings match your broker

### Training Issues
- Verify data files exist in `data/` directory
- Check available memory for large datasets
- Monitor GPU temperature during training

## Security

- **Never commit credentials** to version control
- Use `.env` file for sensitive information
- Regularly update dependencies
- Monitor trading activity and account balance

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Disclaimer

This trading bot is for educational and research purposes. Past performance does not guarantee future results. Trading involves substantial risk of loss. Use at your own risk.

## Model Saving and Live Trading

- After training, the trained model is automatically saved in the `models/` folder.
- When you want to start using the live trading script, update the `MODEL_PATH` in `modules/config.py` to point to the specific trained model file you wish to use (e.g., `models/final_model.zip` or another checkpoint).
- This ensures the live trading script loads the correct model for real-time trading with MetaTrader 5.

