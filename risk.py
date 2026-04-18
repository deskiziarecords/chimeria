# ============================================================
# RISK ENGINE
# ============================================================
# Central risk management engine
# Real-time risk calculation and monitoring
# Integration with trading system
# ============================================================

from collections import deque
from threading import Lock

class RiskEngine:
    def __init__(self, config, event_sender):
        self.config = config.copy()
        self.metrics = Lock()  # Using Lock for thread safety
        self.var_calculator = HistoricalVaR(config['var_confidence'], config['var_horizon_seconds'])
        self.stress_tester = StressTester()
        self.position_limits = PositionLimits(config['max_position_size'], config['max_daily_loss'])
        self.pnl_calculator = PnLCalculator()
        self.history = deque(maxlen=1000)
        self.event_sender = event_sender

    # Update risk metrics with new market data
    def update(self, book, positions):
        start_time = time.time()
        
        # Update positions
        for instrument_id, position in positions.items():
            self.position_limits.update_position(instrument_id, position['quantity'])
        
        # Calculate current PnL
        current_pnl = self.pnl_calculator.calculate_total_pnl(positions, book)
        
        # Calculate VaR
        var_95 = self.var_calculator.calculate(positions, book, 0.95)
        var_99 = self.var_calculator.calculate(positions, book, 0.99)
        es = self.var_calculator.expected_shortfall(positions, book, 0.99)
        
        # Calculate drawdown
        drawdown = self.calculate_drawdown(current_pnl)
        
        # Update metrics
        with self.metrics:
            metrics = RiskMetrics()  # Assuming RiskMetrics is a defined class
            metrics.current_pnl = current_pnl
            metrics.daily_pnl += current_pnl
            metrics.daily_loss = max(0.0, -metrics.daily_pnl)
            metrics.drawdown = drawdown
            metrics.var_95 = var_95
            metrics.var_99 = var_99
            metrics.expected_shortfall = es
            metrics.timestamp_ns = get_hardware_timestamp()  # Assuming this function is defined
            
            # Store history
            self.history.append(metrics)
        
        # Check limits
        self.check_limits(metrics)
        
        elapsed_ns = (time.time() - start_time) * 1e9
        print(f"Risk update took {elapsed_ns:.0f} ns")
        
    # Check all risk limits
    def check_limits(self, metrics):
        # Check position limit
        breach = self.position_limits.check_position_limits()
        if breach:
            self.event_sender.send(RiskEvent('LimitBreached', breach))
            raise Exception(f"Position limit breached: {breach}")
        
        # Check daily loss limit
        if metrics.daily_loss > self.config['max_daily_loss']:
            self.event_sender.send(RiskEvent('LimitBreached', {'loss': metrics.daily_loss, 'limit': self.config['max_daily_loss']}))
            raise Exception(f"Daily loss limit exceeded: {metrics.daily_loss:.2f} > {self.config['max_daily_loss']:.2f}")
        
        # Check drawdown limit
        if metrics.drawdown > self.config['max_drawdown']:
            self.event_sender.send(RiskEvent('DrawdownLimitHit', {'drawdown': metrics.drawdown, 'limit': self.config['max_drawdown']}))
            raise Exception(f"Drawdown limit exceeded: {metrics.drawdown * 100:.2f}% > {self.config['max_drawdown'] * 100:.2f}%")
        
        # Check VaR exceedance
        if abs(metrics.current_pnl) > metrics.var_99:
            self.event_sender.send(RiskEvent('VaRExceeded', {'var_value': metrics.var_99, 'actual_loss': abs(metrics.current_pnl)}))
        
    # Calculate current drawdown
    def calculate_drawdown(self, current_pnl):
        peak_pnl = max(m.current_pnl for m in self.history) if self.history else current_pnl
        
        if peak_pnl > 0.0:
            return (peak_pnl - current_pnl) / peak_pnl
        return 0.0
    
    # Run stress tests
    def run_stress_test(self, book, positions):
        return self.stress_tester.run_all_scenarios(book, positions)
    
    # Get current risk metrics
    def metrics(self):
        with self.metrics:
            return self.metrics.copy()  # Assuming a copy method exists
    
    # Check if trading is allowed
    def can_trade(self, order_size):
        with self.metrics:
            if self.metrics.daily_loss + abs(order_size) > self.config['max_daily_loss'] * 0.9:
                return False
            
            if self.metrics.drawdown + 0.01 > self.config['max_drawdown']:
                return False
            
        return True
    
    # Record a trade
    def record_trade(self, trade):
        self.pnl_calculator.record_trade(trade)
        self.event_sender.send(RiskEvent('LimitBreached', {'trade_id': trade['trade_id'], 'size': trade['size']}))
    
    # Reset daily metrics
    def reset_daily(self):
        with self.metrics:
            self.metrics.daily_pnl = 0.0
            self.metrics.daily_loss = 0.0
            self.pnl_calculator.reset_daily()

# Test cases
def test_risk_engine():
    event_sender = EventSender()  # Assuming EventSender is defined
    engine = RiskEngine(RiskConfig.default(), event_sender)
  
"""    
# EV-ATR Confluence Model for Position Sizing
# Q_t = f_kelly * g_vol * h_conf * C_max
"""

class EVATRParams:
    def __init__(self):
        self.lambda_frac = 3.0
        self.avg_win = 0.015
        self.avg_loss = 0.005
        self.atr_ref = 0.005
        self.beta_vol = 0.5
        self.alpha_phi = 1.5
        self.phi_min = 0.60
        self.frisk = 0.01
        self.lmax = 50.0
        self.equity = 100_000.0

class EVATRModel:
    def __init__(self, params):
        self.params = params

    def compute_q_t(self, ev_t, atr_t, phi_t):
        f_kelly = 0.0 if ev_t <= 0.0 else ev_t / (self.params.lambda_frac * self.params.avg_win * self.params.avg_loss)

        g_vol = 1.0 if atr_t <= self.params.atr_ref else (self.params.atr_ref / atr_t) ** self.params.beta_vol

        h_conf = 0.0 if phi_t <= self.params.phi_min else phi_t ** self.params.alpha_phi

        c_max = min(self.params.frisk * self.params.equity, self.params.lmax * self.params.equity)

        return f_kelly * g_vol * h_conf * c_max

# ============================================================
# RISK GATE
# ============================================================

from enum import Enum
from typing import List

class TriggerType(Enum):
    Lambda1 = 1
    Lambda2 = 2

class GateStatus(Enum):
    Open = 1
    Closed = 2
    Emergency = 3

class GateDecision:
    def __init__(self, status: GateStatus, triggered_gates: List[TriggerType], signal_adjustment: float):
        self.status = status
        self.triggered_gates = triggered_gates
        self.signal_adjustment = signal_adjustment

class EVATRModel:
    @staticmethod
    def new(default):
        # Placeholder for actual implementation
        return EVATRModel()

    def compute_q_t(self, ev_t: float, atr_20: float, phi_t: float) -> float:
        # Placeholder for actual computation logic
        return ev_t + atr_20 + phi_t  # Example computation

class GateContext:
    def __init__(self, volatility_regime: int = 0, price_variation: float = 0.0, atr_20: float = 0.005,
                 delta_threshold: float = 0.3, kurtosis: float = 3.0, drift_bias: float = 0.0,
                 gamma: float = 0.2, ev_t: float = 0.0, phi_t: float = 0.0):
        self.volatility_regime = volatility_regime
        self.price_variation = price_variation
        self.atr_20 = atr_20
        self.delta_threshold = delta_threshold
        self.kurtosis = kurtosis
        self.drift_bias = drift_bias
        self.gamma = gamma
        self.ev_t = ev_t
        self.phi_t = phi_t

class RiskGate:
    def __init__(self):
        self.status = GateStatus.Open
        self.ev_atr = EVATRModel.new(None)

    def evaluate(self, ctx: GateContext) -> GateDecision:
        triggered = []

        # λ₁: Volatility regime gate
        if ctx.volatility_regime == 2 and (ctx.price_variation / (ctx.atr_20 + 1e-8)) < ctx.delta_threshold:
            triggered.append(TriggerType.Lambda1)

        # λ₂: Kurtosis/drift gate
        if abs(ctx.kurtosis - 1.0) < 0.1 and ctx.drift_bias < ctx.gamma:
            triggered.append(TriggerType.Lambda2)

        status = GateStatus.Open if not triggered else GateStatus.Closed

        # Real-time risk adjustment via EV-ATR confluence
        signal_adjustment = self.ev_atr.compute_q_t(ctx.ev_t, ctx.atr_20, ctx.phi_t)

        return GateDecision(status, triggered, signal_adjustment)

# ============================================================
# PNL CALCULATOR
# ============================================================
# Real-time profit and loss tracking
# ============================================================

from collections import defaultdict

class Position:
    def __init__(self, instrument_id: int, quantity: float, entry_price: float, current_price: float):
        self.instrument_id = instrument_id
        self.quantity = quantity
        self.entry_price = entry_price
        self.current_price = current_price
        self.unrealized_pnl = (current_price - entry_price) * quantity
        self.realized_pnl = 0.0

class TradeRecord:
    def __init__(self, trade_id: int, instrument_id: int, price: float, size: float, side: int, timestamp_ns: int):
        self.trade_id = trade_id
        self.instrument_id = instrument_id
        self.price = price
        self.size = size
        self.side = side
        self.timestamp_ns = timestamp_ns

class PnLCalculator:
    def __init__(self):
        self.total_realized_pnl = 0.0
        self.total_unrealized_pnl = 0.0

    def calculate_total_pnl(self, positions: defaultdict[int, Position]) -> float:
        unrealized = sum(position.unrealized_pnl for position in positions.values())
        return self.total_realized_pnl + unrealized

    def record_trade(self, trade: TradeRecord):
        # Update realized PnL logic
        pass

    def reset_daily(self):
        self.total_realized_pnl = 0.0
        self.total_unrealized_pnl = 0.0


