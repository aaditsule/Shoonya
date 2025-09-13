from loguru import logger
from datetime import datetime
import pandas as pd
import pyotp
from api_helper import ShoonyaApiPy
from Websocket import Websocket
from nsepythonserver import *
import requests
import json
import time
import pandas as pd

class Shoonya:
    def __init__(self):
        self.api = None
        self.account = {
            "client_code": "",
            "pwd": "",
            "totp_key": "",
            "vendor_code": "",
            "API_KEY": "",
            "IMEI": "",
        }
        self.running_orders = {}
        self.market_orders = {}
        self.ordno = "norenordno"

    def login(self):
        """Log in to the Shoonya API using credentials."""
        self.api = ShoonyaApiPy()
        totp = pyotp.TOTP(self.account["totp_key"]).now()
        # logger.debug(f"Client_code {self.account['client_code']} : totp generated as {totp}")

        # Perform login
        ret = self.api.login(
            userid=self.account["client_code"],
            password=self.account["pwd"],
            twoFA=totp,
            vendor_code=self.account["vendor_code"],
            api_secret=self.account["API_KEY"],
            imei=self.account["IMEI"],
        )
        logger.debug(f"Login returned: {ret}")
        logger.debug(f"Login Successful")
        return self.api
    
    def place_market_order(self, symbol, order_type, quantity, unique_id=None):
        """Place a market order."""
        remarks = f"order_{unique_id}" if unique_id else f"order_{int(time.time())}"
        try:
            ret = self.api.place_order(
                buy_or_sell=order_type, # 'B' for Buy, 'S' for Sell
                product_type="I",
                exchange="NFO",
                tradingsymbol=symbol,
                quantity=quantity,
                discloseqty=0,
                price_type="MKT",
                price=0.00,
                trigger_price=None,
                retention="DAY",
                remarks=remarks,
            )
            # logger.debug(f"Raw response from API: {ret}")
            # logger.debug(f"Market order placed: {ret}")
            self.running_orders[unique_id] = ret
            self.market_orders[unique_id] = ret
            # logger.debug(f"running orders: {self.running_orders}")
            OrderID = ret.get(self.ordno)
            return ret, OrderID
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            logger.debug(self.api)
            return None

    def square_off_order(self, leg_data):
        """Square off a position by passing norenordno and quantity."""
        try:
            OrderId = leg_data["OrderID"]
            order_details = self.get_order_book(OrderId)
            if not order_details:
                logger.error(f"No order details found for {leg_data["symbol"]}")
                return None
            if order_details[0]["status"] != "Open":
                logger.debug(f"Order already squared off for {leg_data["symbol"]}")
                return None
            net_qty = int(leg_data["quantity"])
            order_type = "B" if net_qty < 0 else "S"
            symbol = leg_data["symbol"]
            if leg_data["OrderID"] == None:
                logger.debug(f"No net quantity to square off for: {symbol}")
                return None
            if net_qty == 0:
                logger.debug(f"No net quantity to square off for: {symbol}")
                return None
            # return self.place_market_order(symbol, order_type, abs(net_qty), unique_id)
            try:
                ret = self.api.place_order(
                    buy_or_sell=order_type, # 'B' for Buy, 'S' for Sell
                    product_type="I",
                    exchange="NFO",
                    tradingsymbol=symbol,
                    quantity=abs(net_qty),
                    discloseqty=0,
                    price_type="MKT",
                    price=0.00,
                    trigger_price=None,
                    retention="DAY",
                )
                logger.debug(f"Postion squared-off for {symbol}")
                return ret
            except Exception as e:
                logger.error(f"Error in squareoff: {e}")
                return None

        except Exception as e:
            logger.error(f"Error squaring off: {e}")
            return None
    
    def get_positions(self, unique_id=None):
        """Fetch positions based on unique identifier."""
        try:
            positions = self.api.get_positions()
            logger.debug(f"Fetched positions: {positions}")
            # logger.debug(f"Running Orders: {self.running_orders}")

            if not positions:
                logger.error("No positions fetched from API.")
                return None

            if unique_id:
                # Extract the symbol part from unique_id
                symbol = unique_id.split("_")[0]
                # logger.debug(f"Looking for symbol derived from unique_id: {symbol}")

                # Match symbol with tsym
                filtered_position = next(
                    (pos for pos in positions if pos.get("tsym") == symbol), 
                    None
                )

                if not filtered_position:
                    logger.error(f"No position found for unique_id: {unique_id} (symbol: {symbol})")
                return filtered_position

            return positions

        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return None
        
    def get_order_book(self, norenordno=None):
        """Fetch order book based on unique identifier."""
        try:
            order_book = self.api.get_order_book()
            if norenordno:
                return [order for order in order_book if order.get("norenordno") == norenordno]
            return order_book
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return None
        

bot = Shoonya()
bot.login()
# bot.place_market_order("INFY","B",1)
