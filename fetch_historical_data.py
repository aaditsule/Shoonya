import json
from api_helper import ShoonyaApiPy
import pandas as pd
import time
from datetime import datetime, timedelta
import pyotp
from loguru import logger

# ---- Logging Setup ----
logger.add("fetch_data.log", rotation="1 MB", retention="7 days", level="DEBUG")
logger.info("=== Script started ===")

# ---- Load Shoonya Config ----
def load_shoonya_config(config_path='config.json'):
    with open(config_path, "r") as file:
        return json.load(file)
    
# --- Split into 30-day chunks (Shoonya allows only up to 30 days per call) ---
def generate_date_chunks(start, end, days=30):
    chunks = []
    while start < end:
        chunk_end = min(start + timedelta(days=days), end)
        chunks.append((start, chunk_end))
        start = chunk_end
    return chunks

# ---- Fetch Historical Prices ----
def fetch_historical_price(exchange, token, interval, starttime, endtime):
    """
    Fetch historical intraday data using Shoonya API.

    Parameters:
    - exchange (str): NSE / NFO / BSE / CDS
    - token (str): Instrument token (e.g., '26000' for NIFTY50)
    - interval (str): '1' for 1-min, '5' for 5-min, etc.
    - starttime (datetime): Start datetime (datetime object)
    - endtime (datetime): End datetime (datetime object)

    Returns:
    - pd.DataFrame with columns:
    ['datetime', 'date', 'time', 'ssboe', 'open', 'high', 'low', 'close',
    'volume', 'vwap', 'interval_volume', 'oi_change', 'open_interest']
    """

    # --- Login ---
    config = load_shoonya_config()
    api = ShoonyaApiPy()
    response = api.login(
        userid=config["userid"],
        password=config["password"],
        twoFA=pyotp.TOTP(config["twoFA"]).now(),
        vendor_code=config["vendor_code"],
        api_secret=config["api_secret"],
        imei=config["imei"]
    )

    # print("Response:", response)

    if response is None:
        print("Login failed:", response)
        return pd.DataFrame()

    print("Login successful!")

    # --- Loop Through Chunks ---
    chunks = generate_date_chunks(starttime, endtime)
    symbol = map_token_to_symbol(token)
    all_data = []

    print(f"Fetching data for {symbol} ({token}) from {starttime.date()} to {endtime.date()} in {interval}-minute intervals")

    for start_dt, end_dt in chunks:
        print(f"Fetching {start_dt.date()} to {end_dt.date()}")

        start_unix = int(time.mktime(start_dt.timetuple()))
        end_unix = int(time.mktime(end_dt.timetuple()))

        try:
            response = api.get_time_price_series(
                exchange=exchange,
                token=token,
                interval=interval,
                starttime=str(start_unix),
                endtime=str(end_unix)
            )

            # print(f"Response: {response}")

            # Handle list of dicts format
            if isinstance(response, list) and len(response) > 0 and 'time' in response[0]:
                df = pd.DataFrame(response)
                df['datetime'] = pd.to_datetime(df['time'], format='%d-%m-%Y %H:%M:%S')
                df = df.rename(columns={
                    'into': 'open',
                    'inth': 'high',
                    'intl': 'low',
                    'intc': 'close',
                    'v': 'volume',
                    'intvwap': 'vwap',
                    'intv': 'interval_volume',
                    'intoi': 'oi_change',
                    'oi': 'open_interest',
                })

                # Drop unnecessary columns
                drop_cols = ['stat']
                df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

                # Add date & time columns
                df['date'] = df['datetime'].dt.date
                df['time'] = df['datetime'].dt.time

                # Add token and symbol columns
                df['token'] = token
                df['symbol'] = symbol

                # Reorder columns
                df = df[['datetime', 'token', 'symbol', 'date', 'time', 'ssboe', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'interval_volume', 'oi_change', 'open_interest']]
                # df = df[['datetime', 'date', 'time', 'ssboe', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'interval_volume', 'oi_change', 'open_interest']]

                all_data.append(df)
                print(f"Fetched {len(df)} rows")

            else:
                print(f"No valid data for {start_dt.date()} to {end_dt.date()}: {response}")

            time.sleep(2)

        except Exception as e:
            print(f"Error during fetch {start_dt.date()} to {end_dt.date()}: {e}")

    # --- Return Final Data ---
    if all_data:
        final_df = pd.concat(all_data)
        final_df.sort_values("datetime", inplace=True)
        return final_df
    else:
        print("No data collected.")
        return pd.DataFrame()

def clean_data(df: pd.DataFrame, keep: list = None, drop: list = None) -> pd.DataFrame:
    """
    Cleans a DataFrame by either keeping only specific columns or dropping specified ones.

    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - keep (list, optional): List of columns to keep (in the specified order)
    - drop (list, optional): List of columns to drop (ignored if `keep` is provided)

    Returns:
    - pd.DataFrame: Cleaned DataFrame
    """
    if keep is not None:
        missing = [col for col in keep if col not in df.columns]
        if missing:
            print(f"Warning: These 'keep' columns were not found in DataFrame: {missing}")
        df = df[[col for col in keep if col in df.columns]]
    elif drop is not None:
        df = df.drop(columns=[col for col in drop if col in df.columns], errors='ignore')
    return df

def map_token_to_symbol(token):
    try:
        # Load the CSV file
        df = pd.read_csv("NSE_Symbols.csv")

        # Ensure token column is string for consistent lookup
        df['Token'] = df['Token'].astype(str)

        # Create a mapping dictionary
        token_symbol_map = dict(zip(df['Token'], df['Symbol']))

        # Convert input token to string and lookup
        return token_symbol_map.get(str(token), None)
    
    except FileNotFoundError:
        print("Error: NSE_Symbols.csv not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
import pandas as pd

def calculate_rsi_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates RSI, MACD line, and MACD signal line for a DataFrame with close prices.

    Parameters:
        df (pd.DataFrame): DataFrame containing at least a 'close' column.

    Returns:
        df (pd.DataFrame): DataFrame with added columns: 'rsi', 'macd_line', 'macd_signal'
    """
    df = df.copy()  # Avoid modifying the original DataFrame

    # --- RSI Calculation (14-period) ---
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    delta = df['close'].diff()  # Change in closing price
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()  # Average gain over 14 periods
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()  # Average loss over 14 periods

    rs = gain / loss  # Relative strength
    df['rsi'] = 100 - (100 / (1 + rs))  # RSI formula

    # --- MACD Calculation ---
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()  # 12-period EMA
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()  # 26-period EMA

    df['macd_line'] = ema_12 - ema_26  # MACD line = 12 EMA - 26 EMA
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()  # 9-period EMA of MACD line
    df['macd_histogram'] = df['macd_line'] - df['macd_signal']  # MACD histogram

    return df

import pandas as pd

def add_nifty_value(df: pd.DataFrame, nifty_csv_path: str) -> pd.DataFrame:
    """
    Adds a 'nifty_value' column to df by mapping the corresponding NIFTY50 close price
    based on matching datetime values.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'datetime' column.
        nifty_csv_path (str): Path to the CSV file with NIFTY 50 minute data.

    Returns:
        pd.DataFrame: Original DataFrame with an additional 'nifty_value' column.
    """
    df = df.copy()

    # Load NIFTY50 data
    nifty_df = pd.read_csv(nifty_csv_path)

    # Ensure datetime column is in datetime format in both DataFrames
    df['datetime'] = pd.to_datetime(df['datetime'])
    nifty_df['datetime'] = pd.to_datetime(nifty_df['datetime'])

    # Select only datetime and close columns from NIFTY data
    nifty_df = nifty_df[['datetime', 'close']]
    nifty_df.rename(columns={'close': 'nifty_value'}, inplace=True)

    # Merge on datetime
    df = pd.merge(df, nifty_df, on='datetime', how='left')

    return df


# --- Parameters ---
# symbol = 'NIFTY'  # NIFTY 50
# exchange = 'NSE'
# token = '26000'  # NIFTY 50 token for NSE
# interval = '1'  # 1 minute
# start_date = datetime(2024, 1, 1)
# end_date = datetime.now()

# df = fetch_historical_price(
#     exchange = "NSE",
#     token = "3456",              
#     interval = "1",               # 1-minute
#     starttime = datetime(2025, 6, 1),
#     endtime = datetime(2025, 7, 5)
# )

# # Columns = [datetime,date,time,ssboe,open,high,low,close,volume,vwap,interval_volume,oi_change,open_interest]
# cols_to_drop = ['ssboe', 'volume', 'oi_change', 'open_interest']
# df_cleaned = clean_data(df, drop=cols_to_drop)

# # Save if needed
# if not df.empty:
#     df.to_csv("tatamotors_intraday_1min.csv", index=False)
#     print("Saved to CSV")
# Save if needed
# if not df.empty:
#     df_cleaned.to_csv("tatamotors_intraday_1min_cleaned.csv", index=False)
#     print("Saved to CSV")

# df = pd.read_csv("tatamotors_intraday_1min_cleaned.csv")
# df_nifty = add_nifty_value(df, "nifty50_minute_data.csv")
# df_final = calculate_rsi_macd(df_nifty)
# print(df_final.head())
# df_final.to_csv("tatamotors_intraday_1min_final.csv", index=False)    

def get_data(exchange, token, interval, starttime, endtime, save_to_csv=False):

    df = fetch_historical_price(
        exchange=exchange,
        token=token,
        interval=interval,
        starttime=starttime,
        endtime=endtime
    )

    if not df.empty:
        cols_to_drop = ['ssboe', 'volume', 'oi_change', 'open_interest']
        df_cleaned = clean_data(df, drop=cols_to_drop)
        df_nifty = add_nifty_value(df_cleaned, "nifty50_minute_data.csv")
        df_final = calculate_rsi_macd(df_nifty)

        if save_to_csv:
            symbol = map_token_to_symbol(token)
            filename = f"Data/{symbol}_data.csv"
            df_final.to_csv(filename, index=False)
            print(f"Data saved to {filename}")

    return df_final

# --- Parameters ---
symbol = 'NIFTY'  # NIFTY 50
exchange = 'NSE'
token = '2029'  
interval = '1'  # 1 minute
start_date = datetime(2025, 1, 1)
end_date = datetime.now()

data = get_data(exchange, token, interval, start_date, end_date, save_to_csv=True)




