"""
Data fetcher module for retrieving market data
"""

import os
import requests
import pandas as pd
import pandas_ta as ta
import config
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any
import numpy as np

import logging
from logger import logger
from config import SYMBOL, TIMEFRAME, YEARS, DATA_CSV_PATH
from dukascopy_downloader import download_dukascopy_csv, download_dukascopy_range_csv
#from technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

def cov_to_corr(cov_matrix):
    """
    Convert a covariance matrix to a correlation matrix.
    
    Parameters
    ----------
    cov_matrix : array_like, 2d
        Covariance matrix, assumed to be symmetric
        
    Returns
    -------
    corr_matrix : ndarray, 2d
        Correlation matrix
    """
    cov_array = np.asarray(cov_matrix)
    std = np.sqrt(np.diag(cov_array))
    corr = cov_array / np.outer(std, std)
    return corr

class DataFetcher:
    def __init__(self):
        self.symbol = SYMBOL
        self.timeframe = TIMEFRAME
        self.cache = {}
        self.cache_timeout = timedelta(minutes=5)
        self.data_csv_path = DATA_CSV_PATH

    # แก้ไขไฟล์: modules/data_fetcher.py

# ในไฟล์ modules/data_fetcher.py

    def fetch_historical_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Loads historical data from a user-provided CSV and enriches it with technical indicators.
        """
        # --- ส่วนที่ 1: โหลดและจัดรูปแบบข้อมูล (เหมือนเดิม) ---
        timeframe_str = timeframe.replace("_", "")
        filename = f"{symbol}_{timeframe_str}.csv"
        csv_path = os.path.join(config.DATA_DIRECTORY, filename)

        logger.info(f"Attempting to load data from user-provided CSV: {csv_path}")

        if not os.path.exists(csv_path):
            logger.error(f"FATAL: CSV file not found at {csv_path}.")
            raise FileNotFoundError(f"Required data file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower()

        try:
            df['time'] = pd.to_datetime(df['time'], format='%Y%m%d %H%M%S')
            logger.info("Successfully parsed time column with YYYYMMDD HHMMSS format.")
        except Exception as e:
            logger.error(f"Could not parse time column. Error: {e}")
            raise

        # ---!!! ส่วนที่ 2: เพิ่ม Technical Indicators (ส่วนใหม่) !!!---
        logger.info("Enriching data with technical indicators using pandas-ta...")

        # คำนวณ Indicator ยอดนิยมและเพิ่มเข้าไปใน DataFrame
        # คุณสามารถเลือกเปิด/ปิด หรือปรับค่าพารามิเตอร์ของแต่ละ Indicator ได้ตามต้องการ
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)

        # การคำนวณ Indicator จะทำให้เกิดค่าว่าง (NaN) ในแถวแรกๆ ของข้อมูล
        # เราต้องลบแถวเหล่านี้ทิ้งเพื่อให้ AI ได้เรียนรู้จากข้อมูลที่สมบูรณ์เท่านั้น
        initial_rows = len(df)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True) # จัดเรียง index ใหม่หลังลบแถว
        final_rows = len(df)
        logger.info(f"Dropped {initial_rows - final_rows} rows with NaN values after indicator calculation.")

        logger.info(f"Data enriched successfully. Final columns: {df.columns.tolist()}")

        return df

    def get_latest_price(self, symbol: str) -> float:
        """Get the latest close price for a symbol from the most recent row in the Dukascopy-downloaded CSV file."""
        try:
            # Load the latest data from the CSV file for the given symbol
            df = self.fetch_historical_data(download_if_missing=False, symbol=symbol)
            if not df.empty:
                return float(df['close'].iloc[-1])
            else:
                logger.error(f"No data found for {symbol}")
                return 0.0
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {str(e)}")
            return 0.0

    def clear_cache(self):
        """Clear the data cache"""
        self.cache.clear()
    
    def _analyze_data_gaps(self, df: pd.DataFrame, start_date: datetime, end_date: datetime) -> dict:
        """Analyze gaps in the data and return detailed statistics (for debugging)"""
        # Create a complete date range
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Find missing dates
        missing_dates = all_dates.difference(df.index)

        # Calculate gap statistics
        gap_stats = {
            'total_days': len(all_dates),
            'available_days': len(df),
            'missing_days': len(missing_dates),
            'coverage': (len(df) / len(all_dates)) * 100,
            'gaps': []
        }

        # Find continuous gaps
        if len(missing_dates) > 0:
            missing_dates = sorted(missing_dates)
            current_gap_start = missing_dates[0]
            current_gap_end = missing_dates[0]

            for i in range(1, len(missing_dates)):
                if (missing_dates[i] - missing_dates[i - 1]).days == 1:
                    current_gap_end = missing_dates[i]
                else:
                    gap_stats['gaps'].append({
                        'start': current_gap_start,
                        'end': current_gap_end,
                        'duration': (current_gap_end - current_gap_start).days + 1
                    })
                    current_gap_start = missing_dates[i]
                    current_gap_end = missing_dates[i]

            # Add the last gap
            gap_stats['gaps'].append({
                'start': current_gap_start,
                'end': current_gap_end,
                'duration': (current_gap_end - current_gap_start).days + 1
            })

        return gap_stats
    
    def _log_gap_analysis(self, gap_stats: dict):
        """Log the results of gap analysis (for debugging)"""
        logger.info("\n=== Data Gap Analysis ===")
        logger.info(f"Total days: {gap_stats['total_days']:,}")
        logger.info(f"Available days: {gap_stats['available_days']:,}")
        logger.info(f"Missing days: {gap_stats['missing_days']:,}")
        logger.info(f"Data coverage: {gap_stats['coverage']:.2f}%")

        if gap_stats['gaps']:
            logger.info("\n=== Data Gaps Detected ===")
            # Sort gaps by duration
            sorted_gaps = sorted(
                gap_stats['gaps'],
                key=lambda x: x['duration'],
                reverse=True)

            # Log the top 5 largest gaps
            logger.info("Top 5 largest gaps:")
            for i, gap in enumerate(sorted_gaps[:5], 1):
                logger.info(f"Gap {i}:")
                logger.info(f"  Start: {gap['start']}")
                logger.info(f"  End: {gap['end']}")
                logger.info(f"  Duration: {gap['duration']} days") 