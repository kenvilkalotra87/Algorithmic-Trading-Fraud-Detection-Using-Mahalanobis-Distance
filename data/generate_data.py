"""
Realistic High-Frequency Trading (HFT) Simulated Dataset Generator.
Designed for the academic project:
'Algorithmic Trading Fraud Detection Using Mahalanobis Distance'
Course: Pattern Recognition and Anomaly Detection (Semester 5)

Problem Statement:
Build a Mahalanobis distance pipeline to flag irregular high-frequency trades
and volume spikes in simulated financial markets.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_trading_dataset(
    num_records: int = 7500,
    anomaly_rate: float = 0.035,
    seed: int = 42,
    output_path: str = "data/trading_data.csv"
) -> pd.DataFrame:
    """
    Generates a realistic synthetic HFT dataset with normal market dynamics
    and controlled synthetic anomalies for academic evaluation.
    """
    np.random.seed(seed)
    
    # Asset configurations: base price, price volatility, base typical quantity
    symbols_config = {
        "TECH":   {"base_price": 182.50, "volatility": 0.35, "base_qty": 180},
        "BANK":   {"base_price": 46.20,  "volatility": 0.12, "base_qty": 350},
        "AUTO":   {"base_price": 68.75,  "volatility": 0.20, "base_qty": 220},
        "ENERGY": {"base_price": 93.10,  "volatility": 0.25, "base_qty": 200},
        "PHARMA": {"base_price": 138.40, "volatility": 0.28, "base_qty": 160}
    }
    
    symbols_list = list(symbols_config.keys())
    
    # Generate timestamp sequence across a simulated trading session (09:30:00 to 16:00:00)
    start_time = datetime(2026, 9, 30, 9, 30, 0)
    time_deltas = np.sort(np.random.uniform(0, 6.5 * 3600, size=num_records))
    
    records = []
    
    # Track current price walk for each symbol
    current_prices = {sym: cfg["base_price"] for sym, cfg in symbols_config.items()}
    
    # Determine anomaly indices (~3.5%)
    num_anomalies = int(num_records * anomaly_rate)
    anomaly_indices = set(np.random.choice(range(50, num_records - 50), size=num_anomalies, replace=False))
    
    for i in range(num_records):
        trade_id = f"TRD-{100000 + i}"
        trade_time = start_time + timedelta(seconds=float(time_deltas[i]))
        timestamp_str = trade_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Select symbol
        sym = np.random.choice(symbols_list, p=[0.32, 0.22, 0.18, 0.15, 0.13])
        cfg = symbols_config[sym]
        
        is_anomaly = i in anomaly_indices
        
        if not is_anomaly:
            # --- NORMAL TRADING DYNAMICS ---
            # Random walk price drift (micro-increments)
            price_tick = np.random.normal(0, cfg["volatility"])
            current_prices[sym] = max(5.0, current_prices[sym] + price_tick * 0.15)
            price = round(float(current_prices[sym]), 2)
            
            # Realistic log-normal trade quantity
            log_mean = np.log(cfg["base_qty"])
            quantity = int(np.clip(np.random.lognormal(mean=log_mean, sigma=0.55), 10, 1500))
            
            # Dollar Volume / Notional Trade Volume
            volume = round(price * quantity, 2)
            
            # Immediate price change relative to recent trade
            price_change = round(float(price_tick), 3)
            
            anomaly_label = 0
            anomaly_type = "normal"
            
        else:
            # --- SYNTHETIC ANOMALIES (Academic Demonstration) ---
            # 4 Distinct archetypes of market irregularity:
            anomaly_subtype = np.random.choice([
                "volume_spike",
                "price_shock",
                "multivariate_break",
                "liquidity_vacuum"
            ])
            
            if anomaly_subtype == "volume_spike":
                # Severe volume spike (fat-finger order / extreme institutional block sweep)
                price = round(float(current_prices[sym] + np.random.normal(0, 0.2)), 2)
                quantity = int(np.random.uniform(4500, 14000))
                volume = round(price * quantity, 2)
                price_change = round(float(np.random.normal(0, 0.15)), 3)
                
            elif anomaly_subtype == "price_shock":
                # Flash crash / abnormal abrupt jump without commensurate volume
                jump_direction = np.random.choice([-1, 1])
                price_jump = jump_direction * np.random.uniform(3.5, 9.5)
                price = round(float(max(5.0, current_prices[sym] + price_jump)), 2)
                current_prices[sym] = price  # shock registers
                quantity = int(np.random.uniform(80, 400))
                volume = round(price * quantity, 2)
                price_change = round(float(price_jump), 3)
                
            elif anomaly_subtype == "multivariate_break":
                # Multivariate anomaly: both volume and price change seem moderately high,
                # but their correlation/joint probability violates normal market depth covariance
                price = round(float(current_prices[sym]), 2)
                quantity = int(np.random.uniform(1800, 3200))
                volume = round(price * quantity, 2)
                price_change = round(float(np.random.choice([-1, 1]) * np.random.uniform(1.8, 3.2)), 3)
                
            else: # liquidity_vacuum
                # Micro-lot order moving the market disproportionately (illiquid gap)
                quantity = int(np.random.choice([1, 2, 5]))
                price_jump = np.random.choice([-1, 1]) * np.random.uniform(2.5, 5.0)
                price = round(float(max(5.0, current_prices[sym] + price_jump)), 2)
                volume = round(price * quantity, 2)
                price_change = round(float(price_jump), 3)
            
            anomaly_label = 1
            anomaly_type = anomaly_subtype
            
        records.append({
            "trade_id": trade_id,
            "timestamp": timestamp_str,
            "symbol": sym,
            "price": price,
            "quantity": quantity,
            "volume": volume,
            "price_change": price_change,
            "synthetic_anomaly": anomaly_label,
            "anomaly_type": anomaly_type
        })
        
    df = pd.DataFrame(records)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully at: {output_path}")
    print(f"Total records: {len(df)}")
    print(f"Synthetic anomalies: {df['synthetic_anomaly'].sum()} ({df['synthetic_anomaly'].mean()*100:.2f}%)")
    print(f"Symbols: {df['symbol'].value_counts().to_dict()}")
    return df

if __name__ == "__main__":
    generate_trading_dataset()
