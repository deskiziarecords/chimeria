# ============================================================
# ORDER BOOK
# ============================================================

from collections import OrderedDict

class OrderBook:
    def __init__(self):
        self.instrument_id = 0
        self.bids = OrderedDict()  # Price * 10000 -> Volume
        self.asks = OrderedDict()
        self.last_price = 0.0
        self.timestamp_ns = 0

    def update(self, tick):
        self.timestamp_ns = tick.timestamp_ns
        price_key = int(tick.price * 10000.0)

        if tick.tick_type == 'Bid':
            if tick.volume > 0.0:
                self.bids[price_key] = tick.volume
            else:
                self.bids.pop(price_key, None)
        elif tick.tick_type == 'Ask':
            if tick.volume > 0.0:
                self.asks[price_key] = tick.volume
            else:
                self.asks.pop(price_key, None)
        elif tick.tick_type == 'Trade':
            self.last_price = tick.price

    def top_levels(self, depth):
        bid_vec = [(p / 10000.0, v) for p, v in list(self.bids.items())[-depth:][::-1]]
        ask_vec = [(p / 10000.0, v) for p, v in list(self.asks.items())[:depth]]
        return bid_vec, ask_vec

    def best_bid(self):
        return next(iter(reversed(self.bids)), 0) / 10000.0

    def best_ask(self):
        return next(iter(self.asks), 0) / 10000.0
