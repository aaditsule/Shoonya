import json
import os
import time
import pandas as pd
from loguru import logger
from nsepythonserver import *
import threading
from filelock import FileLock
from broker import Shoonya

class Websocket:
    def __init__(self, segment, index="NIFTY", config_path='Shoonya/config.json'):
        self.broker = Shoonya(config_path)
        self.api = self.broker.api
        self.ltp_data = {}
        self.subscribe_list = ["NSE|26000", "NSE|26009"]
        self.websocket_opened = False
        self.index = index
        self.write_lock = threading.Lock() 
        self.write_timer = None
        self.segment = "NFO" if segment == "NFO" else "NSE" # NFO for Options and NSE for Equities
    
    def prepare_subscribe_list(self):
        """Prepare the subscription list from the appropriate symbols CSV file"""
        try:
            if self.segment == "NFO":
                csv_file = f"Shoonya/{self.index.capitalize()}_symbols.csv"
            else:
                csv_file = "Shoonya/NSE_Symbols.csv"
            
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                tokens = df["Token"]
                exchange_prefix = "BFO" if self.index == "SENSEX" else self.segment
                self.subscribe_list.extend([f"{exchange_prefix}|{token}" for token in tokens])
            else:
                logger.error(f"{csv_file} not found.")     
           
            # logger.debug(f"Formatted subscription list: {self.subscribe_list}")

        except Exception as e:
            logger.error(f"Error preparing subscription list: {e}")

    def start_websocket(self):
        """Start WebSocket for live data streaming."""
        if not self.websocket_opened:

            def quote_update_handler(data):
                """Handle incoming WebSocket updates and store LTP in ltp_data."""
                try:
                    token = data.get("tk")  # Token identifier
                    ltp = data.get("lp")   # Last traded price
                    if token and ltp is not None:
                        self.ltp_data[token] = ltp
                        if not self.write_timer:
                            self.schedule_write()

                except Exception as e:
                    logger.error(f"Error handling WebSocket data: {e}")

            def open_handler():
                """Handle WebSocket connection open event."""

                logger.debug("WebSocket connected...")
                self.websocket_opened = True
                self.api.subscribe(self.subscribe_list)

                # logger.debug(f"Subscribed to: {self.subscribe_list}")
                # logger.debug(f"Subscribed to subscribtion list...")

            # Start WebSocket connection
            self.api.start_websocket(
                subscribe_callback=quote_update_handler,
                socket_open_callback=open_handler,
            )

            # Wait for WebSocket connection to establish
            while not self.websocket_opened:
                logger.debug("Waiting for WebSocket connection...")
                time.sleep(1)

    def schedule_write(self):
        """Schedule periodic writes to reduce frequent writes."""
        self.write_timer = threading.Timer(1.0, self.periodic_write)  # Write every second
        self.write_timer.start()

    def periodic_write(self):
        """Write LTP data to file periodically."""
        with self.write_lock:
            self.dump_ltp_data()
        self.write_timer = None

    def login_and_ws(self):
        """Login and start WebSocket for real-time data updates."""
        try:
            if self.broker.login():
                self.prepare_subscribe_list()
                if not self.websocket_opened:
                    self.start_websocket()
                return True
            return False

        except Exception as e:
            logger.error(f"Exception in login_and_ws : {e}")
            return False
        
    def dump_ltp_data(self):
        """Dump the LTP data to a JSON file."""
        temp_file = "ltp_data.tmp"
        lock_file = "ltp_data.json.lock"  # Lock file to ensure exclusive access
        try:
            with FileLock(lock_file):  # Acquire a lock
                with open(temp_file, "w") as json_file:
                    json.dump(self.ltp_data, json_file)
                os.replace(temp_file, "Shoonya/ltp_data.json")  # Replace atomically
                # logger.debug("LTP data safely written to ltp_data.json")
        except Exception as e:
            logger.error(f"Error writing LTP data: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
    