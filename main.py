import time
import json
from datetime import datetime
import pandas as pd
from Websocket import Websocket
from broker import Shoonya
from loguru import logger
from nsepythonserver import *

logger.add("mtm_logs.log", level="DEBUG", format="{time} - {level} - {message}")

class Strategy:
    def __init__(self, strategy_file):
        """Initialize the trading bot with customer details and setup variables."""
        self.api = None
        self.websocket = Websocket(strategy_file)
        self.ltp_data = self.load_ltp()
        self.running_orders = {}
        self.market_orders = {}
        self.legs = {}
        self.strategy_config = self.load_strategy(strategy_file) #Stores the configurations 
        self.index = self.strategy_config.get("index", "NIFTY")
        self.strategy_type = "I" if self.strategy_config.get("strategyType", "OptionStraddle") == "Intraday" else "M"
        self.locked_profit = None  # Tracks the currently locked profit
        self.overall_reentry_attempts_sl = 0
        self.rangebreakout_legs = []
        self.simplemomentum_legs = []
        self.broker = Shoonya()

    def square_off_all_legs(self):
        """Square off all active positions."""
        for leg_id, leg_data in self.legs.items():
            if leg_data["status"] == "Active":
                unique_id = "Target_order_for_{leg_id}"
                self.broker.square_off_order(leg_data["symbol"],leg_data["orderID"], unique_id)
                self.legs[leg_id]["status"] = "Closed"  

    def calculate_mtm(self, order, ltp_data):
        """Calculate the MTM for a specific order."""
        try:
            if not order:
                logger.error("No order details provided.")
                return None

            avg_price = float(order.get('avgprc', 0))
            token = order.get('token')
            if not token or token not in ltp_data:
                logger.error(f"Token {token} not found in LTP data.")
                return None

            last_traded_price = float(ltp_data[token])
            symbol = order.get('tsym', "0")

            if last_traded_price is None:
                logger.error("Failed to fetch LTP for MTM calculation.")
                return None

            quantity = int(order.get('qty', 0))
            trantype = order.get('trantype')

            if trantype == 'B':  # Buy transaction
                mtm = (last_traded_price - avg_price) * quantity
            elif trantype == 'S':  # Sell transaction
                mtm = (avg_price - last_traded_price) * quantity
            else:
                logger.error(f"Unknown transaction type: {trantype}")
                return None

            # logger.debug(f"MTM for {symbol}: {mtm}")
            return mtm
        except Exception as e:
            logger.error(f"Error calculating MTM: {e}")
            return None
        
    def calculate_total_mtm(self):
        """Calculate the total MTM for all active legs."""
        try:
            total_mtm = 0
            legs = self.legs
            ltp_data = self.ltp_data
            order_book = self.broker.get_order_book()
            # logger.debug(f"Order Book : {order_book}")
            for leg_id, leg_data in legs.items():
                if leg_data["status"] != "Active":
                    continue

                orderID = leg_data.get("orderID")
                if not orderID:
                    logger.error(f"No orderID found for leg {leg_id}.")
                    continue

                order_details = [order for order in order_book if order.get('orderID') == orderID]
                # logger.debug(f"Order Details : {order_details}")
                if order_details[0].get('status') == "REJECTED":
                    continue

                mtm = self.calculate_mtm(order_details, ltp_data)
                if mtm is not None:
                    total_mtm += mtm

            # logger.info(f"Total MTM: {total_mtm}")
            return total_mtm
        except Exception as e:
            logger.error(f"Error calculating total MTM: {e}")
            return None

    def load_ltp(self):
        """Load strategy configuration from a JSON file and initialize legs."""
        with open("ltp_data.json", 'r') as f:
            strategy = json.load(f)

    def load_strategy(self, file_path):
        """Load strategy configuration from a JSON file and initialize legs."""
        with open(file_path, 'r') as f:
            strategy = json.load(f)
        
        # Populate self.legs
        for leg in strategy.get("legs", []):
            if leg["isEnabled"]:
                self.legs[leg["id"]] = {
                    "details": leg,  # Store original leg details
                    "symbol": None,
                    "average_price": None,
                    "orderID": None,
                    "quantity": None,
                    "range_data": {"high": None, "low": None},
                    "status": "Pending",
                }
        return strategy
          
    def evaluate_entry_conditions(self):    
        """Evaluate entry conditions based on strategy configuration."""
        # Get current time and strategy entry time
        entry_time = self.strategy_config['entryTime']
        while True:
            current_time = time.strftime("%H:%M", time.localtime())
            if current_time < entry_time:
                logger.debug("Waiting for entry time...")
                time.sleep(10)
            else:
                logger.debug("Entry time met")
                break
        self.place_entry_orders()
        return True

    def build_option_strikes(self, index_price, expiry_type):
        """Build option symbols for 20 strikes above and below the ATM price."""
        base_strike = round(index_price / 50) * 50 if self.index == "NIFTY" else round(index_price / 100) * 100
        expiry_date = self.get_expiry_date(expiry_type)

        strikes_ce = [f"{self.index}{expiry_date}C{base_strike + i * (50 if self.index == "NIFTY" else 100)}" for i in range(-20, 21)]
        strikes_pe = [f"{self.index}{expiry_date}P{base_strike + i * (50 if self.index == "NIFTY" else 100)}" for i in range(-20, 21)]
        # logger.debug(f"CE : {strikes_ce} , PE : {strikes_pe}")
        return strikes_ce, strikes_pe
    
    def get_expiry_date(self, expiry_type):
        """Fetch the expiry date based on the type."""
        from datetime import datetime, timedelta
        from calendar import monthrange
        today = datetime.now()
        weekday = today.weekday()

        if self.index == "NIFTY":
            if expiry_type == "Weekly":
                # Nifty weekly expiry is on Thursday
                days_until_thursday = (3 - weekday) % 7
                expiry_date = today + timedelta(days=days_until_thursday)
            elif expiry_type == "Next Weekly":
                # Next week's Nifty weekly expiry (Thursday of next week)
                days_until_this_thursday = (3 - weekday) % 7
                this_thursday = today + timedelta(days=days_until_this_thursday)
                expiry_date = this_thursday + timedelta(weeks=1)
            elif expiry_type == "Monthly":
                # Nifty monthly expiry is the last Thursday of the month
                year = today.year
                month = today.month

                # Get the last day of the current month
                last_day = monthrange(year, month)[1]
                last_date = datetime(year, month, last_day)

                # Find the last Thursday of the month
                while last_date.weekday() != 3:
                    last_date -= timedelta(days=1)

                if today > last_date:
                    month += 1
                    if month > 12:
                        month = 1
                        year += 1
                    last_day = monthrange(year, month)[1]
                    last_date = datetime(year, month, last_day)

                    while last_date.weekday() != 3:
                        last_date -= timedelta(days=1)

                expiry_date = last_date
            else:
                logger.error(f"Unknown expiry type: {expiry_type}")
                return None
        elif self.index == "BANKNIFTY":
            if expiry_type == "Weekly":
                # Bank Nifty weekly expiry is on Wednesday
                days_until_wednesday = (2 - weekday) % 7
                expiry_date = today + timedelta(days=days_until_wednesday)
            elif expiry_type == "Next Weekly":
                # Next week's Bank Nifty weekly expiry (Wednesday of next week)
                days_until_this_wednesday = (2 - weekday) % 7
                this_wednesday = today + timedelta(days=days_until_this_wednesday)
                expiry_date = this_wednesday + timedelta(weeks=1)
            elif expiry_type == "Monthly":
                # Bank Nifty monthly expiry is also the last Wednesday of the month
                year = today.year
                month = today.month

                # Get the last day of the current month
                last_day = monthrange(year, month)[1]
                last_date = datetime(year, month, last_day)

                # Find the last Wednesday of the month
                while last_date.weekday() != 2:
                    last_date -= timedelta(days=1)

                if today > last_date:
                    month += 1
                    if month > 12:
                        month = 1
                        year += 1
                    last_day = monthrange(year, month)[1]
                    last_date = datetime(year, month, last_day)

                    while last_date.weekday() != 2:
                        last_date -= timedelta(days=1)

                expiry_date = last_date
            else:
                logger.error(f"Unknown expiry type: {expiry_type}")
                return None
        else:
            logger.error(f"Unknown index: {self.index}")
            return None

        return expiry_date.strftime("%d%b%y").upper()
    
    def get_ltp(self, token):
        """Fetch LTP from the shared dictionary for the given token."""
        try:
            logger.debug(f"Fetching LTP for token: {token}")
            logger.debug(self.ltp_data)
            if token in self.ltp_data:
                return self.ltp_data[token]
            else:
                logger.error(f"Token {token} not found in LTP data.")
                # logger.error(f"Token {token} not found in LTP data: {self.ltp_data}")
                return None
        except Exception as e:
            logger.error(f"Error fetching LTP: {e}")
            return None
    
    def get_ltp_from_symbol(self, symbol):
        df = pd.read_csv("Nifty_symbols.csv") if self.index == "NIFTY" else pd.read_csv("Banknifty_symbols.csv")
        token = df.loc[df['TradingSymbol'] == symbol, 'Token'].iloc[0]
        return self.get_ltp(str(token))

    def closest_premium(self, strikes_ce, strikes_pe, leg_data):
        """Find CE and PE options closest to the desired premium."""
        closest_premium = leg_data["details"].get('StrikeType', 5)  # Default to 5 if not specified
        option_type = leg_data["details"]['optionType']
        strikes = strikes_ce if option_type == "CE" else strikes_pe

        symbol = min(
            strikes,
            key=lambda s: abs(float(self.get_ltp_from_symbol(s)) - closest_premium)
        )

        logger.debug(f"Closest {option_type}: {symbol} for target premium {closest_premium}")
        leg_data["closest_premium"] = symbol

        return symbol
    
    def atm_strike_offset(self, index_price, strikes_ce, strikes_pe, leg_data):
        """Find CE and PE options based on ATM strike offset."""
        offset = leg_data["details"].get('StrikeType', 0)  # Default to 0 (ATM) if not specified
        spot_price = index_price
        option_type = leg_data["details"]['optionType']
        strikes = strikes_ce if option_type == "CE" else strikes_pe
        atm_strike = round(index_price / 50) * 50 if self.index == "NIFTY" else round(index_price / 100) * 100

        if leg_data["details"]['optionType'] == "CE":
            target_strike = atm_strike - (offset * 50)  # Calculate target strike for Call options
        else:
            target_strike = atm_strike + (offset * 50)  # Calculate target strike for Put options

        symbol = next((s for s in strikes if float(s[-5:]) == target_strike), None)

        if symbol:
            logger.debug(f"ATM Strike Offset {leg_data['details']['optionType']}: {symbol} for offset {offset}")
            leg_data["atm_strike_offset"] = symbol
        else:
            logger.error(f"No matching strike found for target {target_strike} in {leg_data['details']['optionType']} options.")

        return symbol
        
    def premium_less_than(self, strikes_ce, strikes_pe, leg_data):
        """Find the CE or PE option with premium closest to but less than the specified value."""
        max_premium = leg_data["details"].get('StrikeType', 100)
        option_type = leg_data["details"]['optionType']
        strikes = strikes_ce if option_type == "CE" else strikes_pe

        symbol = min(
            [s for s in strikes if float(self.get_ltp_from_symbol(s)) < max_premium],
            key=lambda s: abs(float(self.get_ltp_from_symbol(s)) - max_premium),
            default=None
        )

        if symbol:
            logger.debug(f"Closest Premium Less Than {max_premium} for {option_type}: {symbol}")
        else:
            logger.error(f"No options found with premium less than {max_premium} for {option_type}.")

        leg_data["premium_less_than"] = symbol
        return symbol
    
    def premium_greater_than(self, strikes_ce, strikes_pe, leg_data):
        """Find the CE or PE option with premium closest to but greater than the specified value."""
        min_premium = leg_data["details"].get('StrikeType', 10)
        option_type = leg_data["details"]['optionType']
        strikes = strikes_ce if option_type == "CE" else strikes_pe

        symbol = min(
            [s for s in strikes if float(self.get_ltp_from_symbol(s)) > min_premium],
            key=lambda s: abs(float(self.get_ltp_from_symbol(s)) - min_premium),
            default=None
        )

        if symbol:
            logger.debug(f"Closest Premium Greater Than {min_premium} for {option_type}: {symbol}")
        else:
            logger.error(f"No options found with premium greater than {min_premium} for {option_type}.")

        leg_data["premium_greater_than"] = symbol
        return symbol
    
    def atm_straddle_premium(self, index_price, strikes_ce, strikes_pe, leg_data):
        """Find CE and PE options based on ATM straddle premium percentage."""
        percentage = leg_data["details"].get('StrikeType', 50)  # Percentage of ATM straddle premium

        # Calculate ATM strike by rounding the index price to the nearest 50
        atm_strike = round(index_price / 50) * 50 if self.index == "NIFTY" else round(index_price / 100) * 100

        # Find the symbols for the ATM strike in CE and PE
        atm_strike_ce = next((s for s in strikes_ce if float(s[-5:]) == atm_strike), None)
        atm_strike_pe = next((s for s in strikes_pe if float(s[-5:]) == atm_strike), None)

        if not atm_strike_ce or not atm_strike_pe:
            logger.error(f"ATM strike symbols not found for strike {atm_strike}")
            return None

        # Get premiums for the ATM CE and PE
        atm_premium_ce = float(self.get_ltp_from_symbol(atm_strike_ce))
        atm_premium_pe = float(self.get_ltp_from_symbol(atm_strike_pe))

        # Calculate ATM straddle premium and target premium
        atm_straddle_premium = atm_premium_ce + atm_premium_pe
        target_premium = atm_straddle_premium * (percentage / 100)

        # Determine option type and find the closest strike to the target premium
        option_type = leg_data["details"]['optionType']
        strikes = strikes_ce if option_type == "CE" else strikes_pe

        symbol = min(
            strikes,
            key=lambda s: abs(float(self.get_ltp_from_symbol(s)) - target_premium),
            default=None
        )

        if symbol:
            logger.debug(f"ATM Straddle Premium {option_type}: {symbol} for target premium {target_premium}")
            leg_data["atm_straddle_premium"] = symbol
        else:
            logger.error(f"No matching symbol found for target premium {target_premium}")

        return symbol

    def place_entry_orders(self):
        """Place entry and SL orders for CE and PE options."""

        for leg_id, leg_data in self.legs.items():

            if leg_data["status"] == "Active":
                continue
            
            if self.index == "NIFTY":
                index_price = float(self.get_ltp("26000"))
                if index_price:
                    logger.debug(f"Current NIFTY price : {index_price}")
                if not index_price:
                    logger.error("Failed to fetch NIFTY price.")
                    return False
            elif self.index == "BANKNIFTY":
                index_price = float(self.get_ltp("26009"))
                if index_price:
                    logger.debug(f"Current BANKNIFTY price : {index_price}")
                if not index_price:
                    logger.error("Failed to fetch BANKNIFTY price.")
                    return False
            elif self.index == "SENSEX":
                index_price = float(self.get_ltp("26000"))
                if index_price:
                    logger.debug(f"Current NIFTY price : {index_price}")
                if not index_price:
                    logger.error("Failed to fetch NIFTY price.")
                    return False
            else:
                logger.error(f"Unknown index: {self.index}")
                return False
            
            # Build option strikes for CE and PE
            expiry = leg_data["details"].get("Expiry", "Weekly")
            strikes_ce, strikes_pe = self.build_option_strikes(index_price, expiry)

            strike_criteria = leg_data["details"].get("StrikeCriteria", "closest_premium")
            if strike_criteria == "CLOSEST Premium":
                symbol = self.closest_premium(strikes_ce, strikes_pe, leg_data)
            elif strike_criteria == "ATM Strike offset":
                symbol = self.atm_strike_offset(index_price, strikes_ce, strikes_pe, leg_data)
            elif strike_criteria == "Premium less than":
                symbol = self.premium_less_than(strikes_ce, strikes_pe, leg_data)
            elif strike_criteria == "Premium greater than":
                symbol = self.premium_greater_than(strikes_ce, strikes_pe, leg_data)
            elif strike_criteria == "ATM Straddle Premium":
                symbol = self.atm_straddle_premium(index_price, strikes_ce, strikes_pe, leg_data)
            else:
                logger.error(f"Unknown StrikeCriteria: {strike_criteria}")
                return False
            if not symbol:
                logger.error("Failed to find options for entry.")
                return False
        
            if leg_data["details"].get("rangeBreakOut", {}).get("enabled", False):
                self.rangebreakout_legs.append(leg_id)
                self.legs[leg_id].update({
                    "symbol": symbol
                })
                continue  # Skip placing orders for range breakout legs

            if leg_data["details"].get("simpleMomentum", {}).get("enabled", False):
                self.simplemomentum_legs.append(leg_id)
                self.legs[leg_id].update({
                    "symbol": symbol,
                    "average_price": float(self.get_ltp_from_symbol(symbol)),
                    "underlying_price" : self.get_ltp("26000" if self.index == "NIFTY" else "26009")
                })
                continue  # Skip placing orders for momentum legs

            lots = leg_data["details"]['lots']
            position = "B" if leg_data["details"]['Position'] == "Buy" else "S"
            quantity = lots*75 if self.index == "NIFTY" else lots*15
            order_id = f"{symbol}_{time.strftime("%H%M%S", time.localtime())}"
            entry_price = float(self.get_ltp_from_symbol(symbol))

            response = self.broker.place_market_order(symbol, position, quantity, unique_id=order_id)
            if response:
                self.legs[leg_id].update({
                    "symbol": symbol,
                    "average_price": entry_price,
                    "orderID": response.get("orderID"),
                    "quantity": quantity,
                    "status": "Active",
                    "underlying_price" : self.get_ltp("26000" if self.index == "NIFTY" else "26009")
                })        
        # logger.debug(f"Legs : {self.legs}")

    def check_simple_momentum(self):
        """Check for simple momentum conditions and place entries accordingly."""
        for leg_id in self.simplemomentum_legs:
            leg_data = self.legs.get(leg_id, {})
            details = leg_data.get("details", {})
            momentum_config = details.get("simpleMomentum", {})

            if leg_data["status"] == "Active":
                return

            value = float(momentum_config.get("value", 0))
            unit = momentum_config.get("unit", "Pts ↑")

            if not value or not unit:
                logger.error(f"Invalid simple momentum configuration for leg {leg_id}.")
                continue

            ltp = float(self.get_ltp_from_symbol(leg_data["symbol"]))
            initial_price = float(leg_data.get("average_price"))
            underlying_price = float(leg_data.get("underlying_price"))

            # logger.debug(f"Initial Price : {initial_price} and ltp : {ltp}")

            if not initial_price:
                logger.warning(f"Initial price not set for leg {leg_id}. Skipping.")
                continue


            def place_order(symbol):
                """Place entry order for the given leg."""
                if leg_data["status"] == "Active":
                    return
                lots = leg_data["details"]['lots']
                position = "B" if leg_data["details"]['Position'] == "Buy" else "S"
                quantity = lots*75 if self.index == "NIFTY" else lots*15
                order_id = f"{symbol}_{time.strftime("%H%M%S", time.localtime())}"
                entry_price = float(self.get_ltp_from_symbol(symbol))
                response = self.broker.place_market_order(symbol, position, quantity, unique_id=order_id)
                if response:
                    self.legs[leg_id].update({
                        "symbol": symbol,
                        "average_price": entry_price,
                        "orderID": response.get("orderID"),
                        "quantity": quantity,
                        "status": "Active",
                    })
                else:
                    logger.error(f"Failed to place order for leg {leg_id}. Response: {response}")

            # Check momentum conditions
            if unit == "Pts â†‘" and ((ltp - initial_price) >= value):
                logger.info(f"Momentum condition met for leg {leg_id} with Pts ↑. Placing entry order.")
                place_order(leg_data["symbol"])

            if unit == "Pts â†“" and (ltp - initial_price) <= -value:
                logger.info(f"Momentum condition met for leg {leg_data["symbol"]} with Pts ↓. Placing entry order.")
                place_order(leg_data["symbol"])

            if unit == "Percentage (%) â†‘" and ((ltp - initial_price) / initial_price) * 100 >= value:
                logger.info(f"Momentum condition met for leg {leg_id} with Percentage (%) ↑. Placing entry order.")
                place_order(leg_data["symbol"])

            if unit == "Percentage (%) â†“" and ((ltp - initial_price) / initial_price) * 100 <= -value:
                logger.info(f"Momentum condition met for leg {leg_id} with Percentage (%) ↓. Placing entry order.")
                place_order(leg_data["symbol"])

            if unit == "Underlying Pts â†‘":
                underlying_ltp = self.get_ltp("26000" if self.index == "NIFTY" else "26009")
                if (underlying_ltp - underlying_price) >= value:
                    logger.info(f"Momentum condition met for leg {leg_id} with Underlying Pts ↑. Placing entry order.")
                    place_order(leg_data["symbol"])

            if unit == "Underlying Pts â†“":
                underlying_ltp = self.get_ltp("26000" if self.index == "NIFTY" else "26009")
                if (underlying_ltp - underlying_price) <= -value:
                    logger.info(f"Momentum condition met for leg {leg_id} with Underlying Pts ↓. Placing entry order.")
                    place_order(leg_data["symbol"])

            if unit == "Underlying % â†‘":
                underlying_ltp = self.get_ltp("26000" if self.index == "NIFTY" else "26009")
                if ((underlying_ltp - underlying_price  ) / underlying_price) * 100 >= value:
                    logger.info(f"Momentum condition met for leg {leg_id} with Underlying % ↑. Placing entry order.")
                    place_order(leg_data["symbol"])

            if unit == "Underlying % â†“":
                underlying_ltp = self.get_ltp("26000" if self.index == "NIFTY" else "26009")
                if ((underlying_ltp - underlying_price) / underlying_price) * 100 <= -value:
                    logger.info(f"Momentum condition met for leg {leg_id} with Underlying % ↓. Placing entry order.")
                    place_order(leg_data["symbol"])


    def check_range_breakout(self):
        """Check for range breakout conditions and place orders if met."""
        current_time = time.strftime("%H:%M", time.localtime())
        

        for leg_id in self.rangebreakout_legs:
            leg_data = self.legs[leg_id]
            details = leg_data["details"]
            range_config = details.get("rangeBreakOut", {})

            if leg_data["status"] != "Pending":
                continue

            start_time = range_config.get("entryTime")
            end_time = range_config.get("exitTime")
            entry_on = range_config.get("entryOn")
            method = range_config.get("method")

            lots = leg_data["details"]['lots']
            position = "B" if leg_data["details"]['Position'] == "Buy" else "S"
            quantity = lots*75 if self.index == "NIFTY" else lots*15
            symbol = leg_data["symbol"]
            order_id = f"{symbol}_{time.strftime("%H%M%S", time.localtime())}"

            symbol_to_monitor = (leg_data["symbol"] if method == "Instrument" else self.index)

            if not start_time or not end_time or not entry_on:
                logger.error(f"Invalid range breakout configuration for leg {leg_id}.")
                continue

            if current_time >= start_time and current_time < end_time:
                self.update_range_high_low(leg_id, symbol_to_monitor)

            elif current_time >= end_time:
                high = leg_data["range_data"]["high"]
                low = leg_data["range_data"]["low"]

                if high is None or low is None:
                    logger.warning(f"Incomplete range data for leg {leg_id}. Skipping.")
                    continue

                ltp = self.get_ltp_from_symbol(leg_data["symbol"]) if method == "Instrument" else (self.get_ltp("26000") if self.index == "NIFTY" else self.get_ltp("26009"))
                logger.debug(f"Current LTP for {symbol_to_monitor}: {ltp}")       
                if entry_on == "High" and ltp > high:
                    logger.info(f"High breakout detected for leg {leg_id}. Placing entry order.")
                    entry_price = float(self.get_ltp_from_symbol(symbol))
                    response = self.broker.place_market_order(symbol, position, quantity, unique_id=order_id)
                    if response:
                        self.legs[leg_id].update({
                            "symbol": symbol,
                            "average_price": entry_price,
                            "orderID": response.get("orderID"),
                            "quantity": quantity,
                            "status": "Active",
                        })   

                elif entry_on == "Low" and ltp < low:
                    logger.info(f"Low breakout detected for leg {leg_id}. Placing entry order.")
                    entry_price = float(self.get_ltp_from_symbol(symbol))
                    response = self.broker.place_market_order(symbol, position, quantity, unique_id=order_id)
                    if response:
                        self.legs[leg_id].update({
                            "symbol": symbol,
                            "average_price": entry_price,
                            "orderID": response.get("orderID"),
                            "quantity": quantity,
                            "status": "Active",
                        })

    def update_range_high_low(self, leg_id, symbol):
        """Update the high and low values for the range of a specific leg."""
        leg_data = self.legs[leg_id]

        current_price = self.get_ltp_from_symbol(symbol) if leg_data["details"]["rangeBreakOut"]["method"] == "Instrument" else self.get_ltp("26000" if self.index == "NIFTY" else "26009")
        range_data = leg_data["range_data"]

        if range_data["high"] is None or current_price > range_data["high"]:
            range_data["high"] = current_price

        if range_data["low"] is None or current_price < range_data["low"]:
            range_data["low"] = current_price

        logger.debug(f"Updated range for leg {leg_id}: High = {range_data['high']}, Low = {range_data['low']}.")

    def check_individual_leg_targets(self):
        """Check if individual leg targets are met and square off orders if targets are reached."""
        for leg_id, leg_data in self.legs.items():
            if leg_data["status"] != "Active":
                continue

            target_details = leg_data["details"].get("targetProfit", {})
            if not target_details.get("enabled", False):
                continue

            def square_off(leg_id):
                """Square off the position for the given leg."""
                logger.debug(f"Target reached for {leg_data["symbol"]}. Squaring off.")
                self.broker.square_off_order(leg_data["symbol"], leg_data["orderID"], unique_id=f"Target_order_for_{leg_id}")
                self.legs[leg_id]["status"] = "Closed"
                if self.check_square_off_all_legs():
                    self.square_off_all_legs()
                self.check_reentry_target(leg_id)
                # logger.debug(f"legs : {self.legs}")

            entry_price = float(leg_data["average_price"])
            current_price = float(self.get_ltp_from_symbol(leg_data["symbol"]))
            target_value = float(target_details["value"])
            target_unit = target_details["unit"]
            quantity = leg_data["quantity"]

            if target_unit == "Percent(%)" and ((current_price - entry_price) / entry_price) * 100 >= target_value:
                logger.info(f"Target met for leg {leg_id} (Percent). Closing position.")
                square_off(leg_id)

            elif target_unit == "Pts" and (current_price - entry_price) >= target_value:
                logger.info(f"Target met for leg {leg_id} (Points). Closing position.")
                square_off(leg_id)

            elif target_unit == "Underlying Pts":
                underlying_ltp = float(self.get_ltp("26000" if self.index == "NIFTY" else "26009"))
                initial_underlying_price = float(leg_data["underlying_price"])
                if (underlying_ltp - initial_underlying_price) >= target_value:
                    logger.info(f"Target met for leg {leg_id} (Underlying Points). Closing position.")
                    square_off(leg_id)

            elif target_unit == "Underlying Percent(%)":
                underlying_ltp = float(self.get_ltp("26000" if self.index == "NIFTY" else "26009"))
                initial_underlying_price = float(leg_data["underlying_price"])
                if ((underlying_ltp - initial_underlying_price) / initial_underlying_price) * 100 >= target_value:
                    logger.info(f"Target met for leg {leg_id} (Underlying Percent). Closing position.")
                    square_off(leg_id)

       
    def check_individual_leg_stoploss(self):
        """Check if individual leg stop-losses are met and square off orders if SL is reached."""
        for leg_id, leg_data in self.legs.items():
            if leg_data["status"] != "Active":
                continue

            sl_details = leg_data["details"].get("stopLoss", {})
            trail_sl_details = leg_data["details"].get("trailStopLoss", {})

            if not sl_details.get("enabled", False):
                continue

            entry_price = float(leg_data["average_price"])
            current_price = float(self.get_ltp_from_symbol(leg_data["symbol"]))
            sl_value = float(sl_details["value"])
            sl_unit = sl_details["unit"]
            quantity = leg_data["quantity"]

            def square_off(leg_id):
                """Square off the position for the given leg."""
                logger.debug(f"Stoploss reached for {leg_data["symbol"]}. Squaring off.")
                self.broker.square_off_order(leg_data["symbol"], leg_data["orderID"], unique_id=f"SL_order_for_{leg_id}")
                self.legs[leg_id]["status"] = "Closed"
                if self.check_square_off_all_legs():
                    self.square_off_all_legs()
                self.trail_sl_to_breakeven()
                self.check_reentry_sl(leg_id)
                # logger.debug(f"legs : {self.legs}")

            # Calculate initial SL Price
            if sl_unit == "Percent(%)" and ((entry_price - current_price) / entry_price) * 100 >= sl_value:
                logger.info(f"Stoploss met for leg {leg_id} (Percent). Closing position.")
                square_off(leg_id)

            elif sl_unit == "Pts" and (entry_price - current_price) >= sl_value:
                logger.info(f"Stoploss met for leg {leg_id} (Points). Closing position.")
                square_off(leg_id)

            elif sl_unit == "Underlying Pts":
                underlying_ltp = float(self.get_ltp("26000" if self.index == "NIFTY" else "26009"))
                initial_underlying_price = float(leg_data["underlying_price"])
                if (initial_underlying_price - underlying_ltp) >= sl_value:
                    logger.info(f"Stoploss met for leg {leg_id} (Underlying Points). Closing position.")
                    square_off(leg_id)

            elif sl_unit == "Underlying Percent(%)":
                underlying_ltp = float(self.get_ltp("26000" if self.index == "NIFTY" else "26009"))
                initial_underlying_price = float(leg_data["underlying_price"])
                if ((initial_underlying_price - underlying_ltp) / initial_underlying_price) * 100 >= sl_value:
                    logger.info(f"Stoploss met for leg {leg_id} (Underlying Percent). Closing position.")
                    square_off(leg_id)

            # Update SL Price for Trailing Stop-Loss if enabled
            if trail_sl_details.get("enabled", False):
                trail_movement_by = float(trail_sl_details["movementBy"])
                trail_sl_by = float(trail_sl_details["trailSlBy"])
                trail_unit = trail_sl_details["unit"]

                # Track the highest/lowest price reached for Buy/Sell positions
                if "trail_reference_price" not in leg_data:
                    leg_data["trail_reference_price"] = entry_price

                if trail_unit == "Percent(%)":
                    if leg_data["details"]["Position"] == "Buy":
                        if current_price > leg_data["trail_reference_price"] * (1 + trail_movement_by / 100):
                            leg_data["trail_reference_price"] = current_price
                            sl_value = max(sl_value, sl_value * (1 + trail_sl_by / 100))
                            logger.debug(f"Updated trailing SL for leg {leg_id}: {sl_value}")
                    else:  # Sell
                        if current_price < leg_data["trail_reference_price"] * (1 - trail_movement_by / 100):
                            leg_data["trail_reference_price"] = current_price
                            sl_value = min(sl_value, sl_value * (1 - trail_sl_by / 100))
                            logger.debug(f"Updated trailing SL for leg {leg_id}: {sl_value}")
                else:  # Points
                    if leg_data["details"]["Position"] == "Buy":
                        if current_price >= leg_data["trail_reference_price"] + trail_movement_by:
                             leg_data["trail_reference_price"] = current_price
                             sl_value = max(sl_value, sl_value + trail_sl_by)
                             logger.debug(f"Updated trailing SL for leg {leg_id}: {sl_value}")
                    else:  # Sell
                        if current_price <= leg_data["trail_reference_price"] - trail_movement_by:
                            leg_data["trail_reference_price"] = current_price
                            sl_value = min(sl_value, sl_value - trail_sl_by)
                            logger.debug(f"Updated trailing SL for leg {leg_id}: {sl_value}")
                leg_data["calculated_stop_loss"] = sl_value

                

            # logger.debug(f"Leg {leg_id} - Entry Price: {entry_price}, Current Price: {current_price}, Stop-Loss: {sl_value}, Trail Reference Price: {leg_data.get('trail_reference_price')}")

    def check_reentry_target(self, leg_id):
        """Check re-entry for a specific leg after a target hit."""

        if self.check_restrict_reentry():
            logger.debug("Reentry Restricted")
            return

        leg_data = self.legs.get(leg_id)
        if not leg_data or leg_data["status"] != "Closed":
            return

        reentry_config = leg_data["details"].get("reEntryOntarget", {})
        if not reentry_config.get("enabled", False):
            return

        reentry_count = reentry_config.get("value", 0)
        if "reentry_attempts_target" not in leg_data:
            leg_data["reentry_attempts_target"] = 0

        if leg_data["reentry_attempts_target"] >= reentry_count:
            logger.debug(f"Max re-entry attempts reached for leg {leg_id} on Target.")
            return

        # Increment re-entry attempts
        leg_data["reentry_attempts_target"] += 1

        # Re-enter in a new position
        index_price = float(self.get_ltp("26000")) if self.index == "NIFTY" else float(self.get_ltp("26009"))    
        strikes_ce, strikes_pe = self.build_option_strikes(index_price)
        strike_criteria = leg_data["details"].get("StrikeCriteria", "closest_premium")
        if strike_criteria == "CLOSEST Premium":
            symbol = self.closest_premium(strikes_ce, strikes_pe, leg_data)
        elif strike_criteria == "ATM Strike offset":
            symbol = self.atm_strike_offset(index_price, strikes_ce, strikes_pe, leg_data)
        elif strike_criteria == "Premium less than":
            symbol = self.premium_less_than(strikes_ce, strikes_pe, leg_data)
        elif strike_criteria == "Premium greater than":
            symbol = self.premium_greater_than(strikes_ce, strikes_pe, leg_data)
        elif strike_criteria == "ATM Straddle Premium":
            symbol = self.atm_straddle_premium(index_price, strikes_ce, strikes_pe, leg_data)
        else:
            logger.error(f"Unknown StrikeCriteria: {strike_criteria}")
            return False
        
        if leg_data["details"].get("rangeBreakOut", {}).get("enabled", False):
            self.rangebreakout_legs.append(leg_id)
            self.legs[leg_id].update({
                "symbol": symbol
            })
            return  # Skip placing orders for range breakout legs
        
        if leg_data["details"].get("simpleMomentum", {}).get("enabled", False):
            self.simplemomentum_legs.append(leg_id)
            self.legs[leg_id].update({
                "symbol": symbol,
                "average_price": float(self.get_ltp_from_symbol(symbol)),
                "underlying_price" : self.get_ltp("26000" if self.index == "NIFTY" else "26009")
            })
            return  # Skip placing orders for momentum legs
        
        
        position = "B" if leg_data["details"]["Position"] == "Buy" else "S"
        quantity = leg_data["details"]["lots"] * (75 if self.index == "NIFTY" else 15)
        response = self.broker.place_market_order(symbol, position, quantity, unique_id=f"Reentry_{leg_id}_{symbol}_Target")

        if response:
            logger.debug(f"Re-entered {symbol} for leg {leg_id} after Target.")
            self.legs[leg_id]["status"] = "Active"
            self.legs[leg_id]["orderID"] = response.get("orderID")

    def check_reentry_sl(self, leg_id):
        """Check re-entry for a specific leg after a stoploss hit."""

        if self.check_restrict_reentry():
            logger.debug("Reentry Restricted")
            return

        leg_data = self.legs.get(leg_id)
        # logger.debug(leg_data)
        if not leg_data or leg_data["status"] != "Closed":
            return

        reentry_config = leg_data["details"].get("reEntryOnSL", {})
        if not reentry_config.get("enabled", False):
            return

        reentry_count = reentry_config.get("value", 0)
        if "reentry_attempts_sl" not in leg_data:
            leg_data["reentry_attempts_sl"] = 0

        if leg_data["reentry_attempts_sl"] >= reentry_count:
            logger.debug(f"Max re-entry attempts reached for leg {leg_id} on Stoploss.")
            return

        # Increment re-entry attempts
        leg_data["reentry_attempts_sl"] += 1

        # Re-enter in a new position
        index_price = float(self.get_ltp("26000")) if self.index == "NIFTY" else float(self.get_ltp("26009"))
        strikes_ce, strikes_pe = self.build_option_strikes(index_price)
        strike_criteria = leg_data["details"].get("StrikeCriteria", "closest_premium")
        if strike_criteria == "CLOSEST Premium":
            symbol = self.closest_premium(strikes_ce, strikes_pe, leg_data)
        elif strike_criteria == "ATM Strike offset":
            symbol = self.atm_strike_offset(index_price, strikes_ce, strikes_pe, leg_data)
        elif strike_criteria == "Premium less than":
            symbol = self.premium_less_than(strikes_ce, strikes_pe, leg_data)
        elif strike_criteria == "Premium greater than":
            symbol = self.premium_greater_than(strikes_ce, strikes_pe, leg_data)
        elif strike_criteria == "ATM Straddle Premium":
            symbol = self.atm_straddle_premium(index_price, strikes_ce, strikes_pe, leg_data)
        else:
            logger.error(f"Unknown StrikeCriteria: {strike_criteria}")
            return False
        
        if leg_data["details"].get("rangeBreakOut", {}).get("enabled", False):
            self.rangebreakout_legs.append(leg_id)
            self.legs[leg_id].update({
                "symbol": symbol
            })
            return  # Skip placing orders for range breakout legs
        
        if leg_data["details"].get("simpleMomentum", {}).get("enabled", False):
            self.simplemomentum_legs.append(leg_id)
            self.legs[leg_id].update({
                "symbol": symbol,
                "average_price": float(self.get_ltp_from_symbol(symbol)),
                "underlying_price" : self.get_ltp("26000" if self.index == "NIFTY" else "26009")
            })
            return  # Skip placing orders for momentum legs
        position = "B" if leg_data["details"]["Position"] == "Buy" else "S"
        quantity = leg_data["details"]["lots"] * (75 if self.index == "NIFTY" else 15)
        response = self.broker.place_market_order(symbol, position, quantity, unique_id=f"Reentry_{leg_id}_{symbol}_Target")

        if response:
            logger.debug(f"Re-entered {symbol} for leg {leg_id} after Stoploss.")
            self.legs[leg_id]["status"] = "Active"
            self.legs[leg_id]["orderID"] = response.get("orderID")

    def check_square_off_all_legs(self):
        """Square off all legs in the strategy if any leg's Target or SL is hit"""
        leg_sl_square_off_all = self.strategy_config.get("legWiseSetting", {}).get("legSlSquareOffAllLegs", False)
        if leg_sl_square_off_all:
            logger.debug("Square off all legs is enabled") 
            return True
        else:
            return False
        
    def check_restrict_reentry(self):
        """Stop any re-entry if the SL/Target is hit after a specified time"""
        rentry_time = self.strategy_config.get("restrictReEntry", {}).get("time", False)
        current_time = time.strftime("%H:%M", time.localtime())
        if self.strategy_config.get("restrictReEntry", {}).get("enabled", False):
            if current_time >= rentry_time:
                return True
            else:
                return False
        else:
            return False

    def trail_sl_to_breakeven(self):
        if not self.strategy_config["legWiseSetting"]["trailSlToBreakeven"]:
            return

        for leg_id, leg_data in self.legs.items():
            if leg_data["status"] != "Active":
                continue

            # Check if the leg has a stop loss defined
            if "stopLoss" in leg_data["details"] and leg_data["details"]["stopLoss"].get("enabled", False):
                # Update stop loss to the breakeven price
                leg_data["calculated_stop_loss"] = leg_data["average_price"]
                logger.debug(f"Stop loss for leg {leg_id} moved to breakeven price: {leg_data['average_price']}")

    def check_mtm_setting(self):
        """Evaluate the MTM settings and execute actions based on the configuration"""
        strategy_limits = self.strategy_config['lockProfile']
        max_profit = float(strategy_limits.get('strategyMaxProfit', float('inf')))
        max_loss = float(strategy_limits.get('strategyMaxLoss', float('inf')))
        total_mtm = self.calculate_total_mtm()
        mtm_settings = self.strategy_config.get("mtMSettings", {})
    
        # if not mtm_settings.get("enabled", False):
        #     return

        # Lock Profit 
        lock_profit_config = mtm_settings.get("lockProfit", {})
        if lock_profit_config.get("enabled", False):
            self.lock_profit(lock_profit_config)

        # Trail MTM Loss Logic
        trail_mtm_loss_config = mtm_settings.get("trailMTMLoss", {})
        if trail_mtm_loss_config.get("enabled", False):
            # self.trail_mtm_loss(trail_mtm_loss_config)

            profit_increased_by = trail_mtm_loss_config["profitIncreasedBy"]
            trail_mtm_sl_by = trail_mtm_loss_config["trailMTMSLBy"]
            current_profit = self.calculate_total_mtm()  # Fetch total MTM profit

            initial_stop_loss = max_loss

            if not hasattr(self, "trail_stop_loss"):
                self.trail_stop_loss = initial_stop_loss

            if not hasattr(self, "profit_increase"):
                self.profit_increase = profit_increased_by

            # logger.info(f"Current Profit: {current_profit}, Trail Stop Loss: {self.trail_stop_loss}")

            if current_profit >= self.profit_increase:
                # Adjust the trailing stop loss
                self.trail_stop_loss -= trail_mtm_loss_config["trailMTMSLBy"]
                self.profit_increase += trail_mtm_loss_config["profitIncreasedBy"]
                logger.info(f"Trailing stop loss updated to {self.trail_stop_loss}. Next profit increment at {profit_increased_by}.")

            if current_profit < -self.trail_stop_loss:
                logger.info(f"Profit dropped below trailing stop loss {self.trail_stop_loss}. Exiting all positions.")
                self.square_off_all_legs()

        # Lock Profit and Trail : 
        lock_profit_and_trail_config = mtm_settings.get("lockProfitAndTrail", {})
        if lock_profit_and_trail_config.get("enabled", False):

            profit_achieved = lock_profit_and_trail_config["profitAchieved"]
            lock_profit = lock_profit_and_trail_config["lockProfit"]
            profit_increased_by = lock_profit_and_trail_config["profitIncreasedBy"]
            trail_locked_profit_by = lock_profit_and_trail_config["trailLockedProfitBy"]
                        
            # Logic: Lock profit initially, then trail as profit increases
            current_profit = self.calculate_total_mtm() 
            if current_profit >= profit_achieved and self.locked_profit is None:
                self.locked_profit = lock_profit
                logger.debug(f"Profit threshold {profit_achieved} reached. Locking profit at {self.locked_profit}.")

            if not hasattr(self, "profit_increase"):
                self.profit_increase = profit_increased_by

            if current_profit >= profit_achieved + self.profit_increase:
                self.locked_profit += trail_locked_profit_by
                self.profit_increase += profit_increased_by
                logger.debug(f"Trailing locked profit updated to {self.locked_profit}.")

            if self.locked_profit is not None and current_profit <= self.locked_profit:
                logger.debug(f"Profit dropped below locked level {self.locked_profit}. Exiting all positions.")
                self.square_off_all_legs()

        if total_mtm >= max_profit and max_profit != 0:
            logger.debug("Maximum profit reached. Exiting all positions.")
            self.square_off_all_legs()
            self.check_overall_reentry_target()

        if total_mtm <= -max_loss and max_loss != 0 and (trail_mtm_loss_config.get("enabled", False))== False:
            logger.debug("Maximum loss limit reached. Exiting all positions.")
            self.square_off_all_legs()
            self.check_overall_reentry_sl()
    
    def lock_profit(self, lock_profit_config):
        """Apply the Lock Profit logic"""

        profit_achieved = lock_profit_config["profitAchieved"]
        lock_profit = lock_profit_config["lockProfit"]

        current_profit = self.calculate_total_mtm()  
        # logger.debug(f"Current Profit: {current_profit}, Lock Profit Threshold: {profit_achieved}, Locked Profit: {self.locked_profit}")

        if current_profit >= profit_achieved and self.locked_profit is None:
            self.locked_profit = lock_profit
            logger.debug(f"Profit threshold {profit_achieved} reached. Locking profit at {self.locked_profit}.")

        if self.locked_profit is not None and current_profit <= self.locked_profit:
            # Profit falls below locked profit level
            logger.debug(f"Profit dropped below locked level {self.locked_profit}. Exiting all positions.")
            self.square_off_all_legs()

    def check_exit_time(self):
        """Exit all positions when exit time is reached"""
        current_time = time.strftime("%H:%M", time.localtime())
        exit_time = self.strategy_config['exitTime']
        if current_time >= exit_time:
            self.square_off_all_legs()
            logger.debug("Exit time reached. Squaring Off all Positons...")
            return
        return

    def check_overall_reentry_target(self):
            """Check overall re-entry after an overall Target is hit"""

            if self.check_restrict_reentry():
                logger.debug("Reentry Restricted")
                return
            
            overall_reentry_config = self.strategy_config.get("lockProfile", {}).get("overallReEntry", {}).get(f"overallReEntryOnTarget", {})
            if not overall_reentry_config.get("enabled", False):
                return
            
            max_reentries = overall_reentry_config.get("value", 0)
            if not hasattr(self, "overall_reentry_attempts_target"):
                    self.overall_reentry_attempts_target = 0

            if self.overall_reentry_attempts_target >= max_reentries:
                logger.info(f"Max overall re-entry attempts reached for Target")
                return

            # Increment re-entry attempts
            self.overall_reentry_attempts_target += 1

            # Re-enter in a new position
            self.evaluate_entry_conditions()
            logger.debug(f"Re-entered all positions again after Target")

    def check_overall_reentry_sl(self):
        """Check overall re-entry after an overall Stoploss is hit"""

        if self.check_restrict_reentry():
            logger.debug("Reentry Restricted")
            return
        
        overall_reentry_config = self.strategy_config.get("lockProfile", {}).get("overallReEntry", {}).get(f"overallReEntryOnSL", {})
        if not overall_reentry_config.get("enabled", False):
            return
        
        max_reentries = overall_reentry_config.get("value", 0)
        if not hasattr(self, "overall_reentry_attempts_sl"):
                self.overall_reentry_attempts_sl = 0

        if self.overall_reentry_attempts_sl >= max_reentries:
            logger.info(f"Max overall re-entry attempts reached for Stoploss")
            return

        # Increment re-entry attempts
        self.overall_reentry_attempts_sl += 1

        # Re-enter in a new position
        self.evaluate_entry_conditions()
        logger.debug(f"Re-entered all positions again after Stoploss")

    def monitor_mtm(self):
        """Monitor MTM continuously and exit on target/SL or individual leg target."""
        while True:

            total_mtm = self.calculate_total_mtm()
            logger.info(f"Current MTM: {round(total_mtm, 2)}")

            self.check_simple_momentum()
            self.check_range_breakout()

            self.check_individual_leg_targets()
            self.check_individual_leg_stoploss()
            self.check_mtm_setting()
            self.check_exit_time()

            time.sleep(2)

    def execute_strategy(self):
        """Execute the strategy by evaluating entry conditions and monitoring MTM."""
        try:
            logger.info("Starting strategy execution.")
            
            # Evaluate entry conditions and place initial orders
            if self.evaluate_entry_conditions():
                logger.debug("Orders placed successfully.")
                
                # Monitor MTM continuously
                self.monitor_mtm()
            else:
                logger.warning("Entry conditions not met. No orders placed.")

        except Exception as e:
            logger.error(f"Error during strategy execution: {e}")



strategy_file = "Strategy.json"
Websocket(strategy_file).login_and_ws()
bot1 = Strategy(strategy_file)

# bot1.broker.login()
bot1.execute_strategy()
