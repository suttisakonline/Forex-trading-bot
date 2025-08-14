#!/usr/bin/env python3
"""
Test script to verify technical indicators are properly integrated.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_fetcher import DataFetcher
import pandas as pd

def test_indicators():
    """Test that technical indicators are properly added to forex data."""
    print("Testing technical indicators integration...")
    
    # Initialize DataFetcher
    fetcher = DataFetcher(symbol='EURUSD', timeframe='H1')
    
    # Fetch data with indicators (using existing data file)
    try:
        # Use existing data file to avoid downloading
        csv_path = "data/EUR_USD_20150622_20250619.csv"
        if os.path.exists(csv_path):
            print(f"Using existing data file: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Prepare the data
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Add technical indicators manually
            from modules.technical_indicators import TechnicalIndicators
            ti = TechnicalIndicators()
            df_with_indicators = ti.add_all_indicators(df, price_col='close')
            
            print(f"\nOriginal data shape: {df.shape}")
            print(f"Data with indicators shape: {df_with_indicators.shape}")
            print(f"Added {df_with_indicators.shape[1] - df.shape[1]} technical indicators")
            
            # Show some indicator columns
            indicator_cols = [col for col in df_with_indicators.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            print(f"\nFirst 10 technical indicators:")
            for i, col in enumerate(indicator_cols[:10]):
                print(f"  {i+1}. {col}")
            
            print(f"\nSample values from last row:")
            last_row = df_with_indicators.iloc[-1]
            print(f"Close: {last_row['close']:.5f}")
            if 'sma_14' in last_row:
                print(f"SMA(14): {last_row['sma_14']:.5f}")
            if 'rsi_14' in last_row:
                print(f"RSI(14): {last_row['rsi_14']:.2f}")
            if 'macd' in last_row:
                print(f"MACD: {last_row['macd']:.5f}")
                
        else:
            print(f"Data file not found: {csv_path}")
            print("You can run the data fetcher to download fresh data with indicators")
            
    except Exception as e:
        print(f"Error testing indicators: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_indicators()
