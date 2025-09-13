import json
import os
import time
from datetime import datetime
import pandas as pd
import pyotp
import requests
from api_helper import ShoonyaApiPy
from loguru import logger
from nsepythonserver import *
import threading
from filelock import FileLock

class Websocket:
    def __init__(self, segment):
        self.api = None
        self.ltp_data = {}
        self.subscribe_list = ["NSE|26000", "NSE|26009"]
        self.account = {
            "client_code": "",
            "pwd": "",
            "totp_key": "",
            "vendor_code": "",
            "API_KEY": "",
            "IMEI": "",
        }
        # self.strategy_config = self.load_strategy(strategy_file) #Stores the configurations 
        self.websocket_opened = False
        # self.index = self.strategy_config.get("index", "NIFTY")
        self.index = "NIFTY"
        self.write_lock = threading.Lock() 
        self.write_timer = None
        self.segment = "NFO" if segment == "NFO" else "NSE" # NFO for Options and NSE for Equities
        
    def login(self):
        """Log in to the Shoonya API using customer credentials."""
        self.api = ShoonyaApiPy()
        totp = pyotp.TOTP(self.account["totp_key"]).now()
        # logger.debug(f"client_code {self.account['client_code']} : totp generated as {totp}")

        # Perform login
        ret = self.api.login(
            userid=self.account["client_code"],
            password=self.account["pwd"],
            twoFA=totp,
            vendor_code=self.account["vendor_code"],
            api_secret=self.account["API_KEY"],
            imei=self.account["IMEI"],
        )
        # logger.debug(f"Login returned: {ret}")
        logger.debug(f"Login Successful")
        return self.api
    
    def prepare_subscribe_list(self):
        """Prepare the subscription list from NSE_symbols.csv."""
        try:
            index = self.index
            if self.segment == "NFO":
                if index == "NIFTY":
                    csv_file = "Nifty_symbols.csv"
                # elif index == "BANKNIFTY":
                #     csv_file = "Banknifty_symbols.csv"
                # elif index == "SENSEX":
                #     csv_file = "Sensex_symbols.csv"
                else:
                    logger.error(f"Unknown index: {index}")
                    return
            else:
                csv_file = "NSE_Symbols.csv"
            
            df = pd.read_csv(csv_file)
            tokens = df["Token"]

            for t in tokens:
                if self.segment == "NFO":
                    item = f"BFO|{t}" if index == "SENSEX" else f"NFO|{t}"
                else:
                    item = f"NSE|{t}"
                self.subscribe_list.append(item)      
           
            # logger.debug(f"Formatted subscription list: {self.subscribe_list}")
        except Exception as e:
            logger.error(f"Error preparing subscription list: {e}")


    def load_strategy(self, file_path):
        """Load strategy configuration from a JSON file and initialize legs."""
        with open(file_path, 'r') as f:
            strategy = json.load(f)

        return strategy

    def start_websocket(self):
        """Start WebSocket for live data streaming."""
        if not self.websocket_opened:

            def quote_update_handler(data):
                """Handle incoming WebSocket updates and store LTP in ltp_data."""
                # # logger.debug(f"Raw data received: {data}")
                # token = data.get("tk")  # Token identifier
                # ltp = data.get("lp")   # Last traded price
                # if token and ltp is not None:
                #     self.ltp_data[token] = ltp
                #     self.dump_ltp_data()  # Dump LTP data to JSON file
                #     # logger.debug("LTP data successfully written to ltp_data.json")
                # # else:
                #     # logger.error(f"Invalid or missing data: {data}")
                try:
                    token = data.get("tk")  # Token identifier
                    ltp = data.get("lp")   # Last traded price
                    if token and ltp is not None:
                        self.ltp_data[token] = ltp
                        if not self.write_timer:
                            self.schedule_write()
                    # else:
                    #     logger.warning(f"Invalid WebSocket data received: {data}")
                except Exception as e:
                    logger.error(f"Error handling WebSocket data: {e}")

            def open_handler():
                """Handle WebSocket connection open event."""
                logger.debug("WebSocket connected.")
                self.websocket_opened = True
                self.api.subscribe(self.subscribe_list)
                # logger.debug(f"Subscribed to: {self.subscribe_list}")
                logger.debug(f"Subscribed to subscribtion list")

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
            self.prepare_subscribe_list()

            if self.websocket_opened:
                logger.debug("WebSocket already opened.")
                return True

            logger.debug("Starting login and WebSocket connection...")

            # Perform login
            self.api = self.login()
            if not self.api:
                logger.error("Login failed. WebSocket cannot be started.")
                return False

            # Start WebSocket
            self.start_websocket()

            # Allow some time for WebSocket to receive updates
            logger.debug("Waiting for WebSocket to receive updates...")
            # Log the LTP data after WebSocket starts
            # logger.debug(f"LTP Data after WebSocket starts: {self.ltp_data}")
            return True

        except Exception as e:
            logger.error(f"Exception {e} in login_and_ws.")
            return False
        
    def dump_ltp_data(self):
        """Dump the LTP data to a JSON file."""
        # with open("ltp_data.json", "w") as json_file:
        #     json.dump(self.ltp_data, json_file)
        temp_file = "ltp_data.tmp"
        lock_file = "ltp_data.json.lock"  # Lock file to ensure exclusive access
        try:
            with FileLock(lock_file):  # Acquire a lock
                with open(temp_file, "w") as json_file:
                    json.dump(self.ltp_data, json_file)
                os.replace(temp_file, "ltp_data.json")  # Replace atomically
                # logger.debug("LTP data safely written to ltp_data.json")
        except Exception as e:
            logger.error(f"Error writing LTP data: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def load_ltp_data(self):
        """Load LTP data from the shared dictionary."""
        self.login_and_ws()
        # logger.debug(f"LTP Data loaded: {self.ltp_data}")
        return self.ltp_data

# Websocket("NSE").login_and_ws()
# time.sleep(10)