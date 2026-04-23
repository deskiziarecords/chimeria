# smk_risk_integration.py - Full integration with λ₇ (Macro) and λ₈ (Light-Cone)

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Import your existing modules
from risk_manager import RiskManager, RiskLevel, RiskMetrics, Position
from macro_causality_gate import Lambda7MacroGate, Lambda7Telemetry
from light_cone_violation import LightConeViolationDetector, Lambda8Telemetry


class SignalStrength(Enum):
    """Signal strength classification"""
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    INVALID = "INVALID"


@dataclass
class SMKSignal:
    """Unified trading signal from SMK system"""
    direction: str  # 'BUY', 'SELL', 'HOLD'
    strength: SignalStrength
    confidence: float  # 0-1
    position_size: float
    stop_loss_price: float
    take_profit_price: float
    lambda_signals: Dict[str, float]  # λ₁-λ₈ scores
    veto_reason: Optional[str] = None
    timestamp: datetime = datetime.now()


class SMKRiskIntegrator:
    """
    Full integration between SMK λ sensors and Risk Management System.
    
    Features:
    - Real-time λ₇ macro confirmation
    - λ₈ light-cone violation detection
    - Dynamic position sizing based on risk metrics
    - Stop-loss adjustment based on volatility (λ₁)
    - Macro regime-based trade filtering
    """
    
    def __init__(self,
                 initial_capital: float = 100_000,
                 max_portfolio_risk: float = 0.02,
                 min_confidence: float = 0.65):
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            initial_capital=initial_capital,
            max_position_size=0.10,
            max_portfolio_risk=max_portfolio_risk,
            stop_loss_percent=0.05
        )
        
        # Initialize λ sensors
        self.lambda7 = Lambda7MacroGate()
        self.lambda8 = LightConeViolationDetector()
        
        self.min_confidence = min_confidence
        self.trade_history: List[SMKSignal] = []
        
        # Dynamic adjustment factors
        self.volatility_adjustment = 1.0
        self.macro_adjustment = 1.0
        
    def evaluate_signal(self,
                        symbol: str,
                        current_price: float,
                        smk_fusion_p: float,  # From your lambda_fusion_engine
                        smk_entrapment: bool,
                        smk_manipulation: bool,
                        smk_displacement: float,
                        dxy_price: float,
                        spx_price: float,
                        volume: float) -> SMKSignal:
        """
        Evaluate trading signal with full λ₇ and λ₈ integration.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            smk_fusion_p: Fused signal from lambda_fusion_engine (-1 to 1)
            smk_entrapment: λ₁ entrapment status
            smk_manipulation: λ₄ manipulation status
            smk_displacement: λ₆ displacement score
            dxy_price: Current DXY price for λ₇
            spx_price: Current SPX price for λ₇
            volume: Current volume for λ₈
        """
        
        # 1. Run λ₇ Macro Causality Gate
        lambda7_result = self.lambda7.step(
            symbol=symbol,
            direction=1 if smk_fusion_p > 0 else -1,
            current_price=current_price,
            dxy_price=dxy_price,
            spx_price=spx_price
        )
        
        # 2. Run λ₈ Light-Cone Violation Detector
        lambda8_result = self.lambda8.step(
            target_price=current_price,
            dxy_price=dxy_price,
            spx_price=spx_price
        )
        
        # 3. Calculate base direction
        raw_direction = 'BUY' if smk_fusion_p > 0 else 'SELL' if smk_fusion_p < 0 else 'HOLD'
        base_confidence = abs(smk_fusion_p)
        
        # 4. Apply λ₇ macro veto
        macro_veto = False
        macro_reason = None
        
        if not lambda7_result.signal_valid:
            macro_veto = True
            macro_reason = lambda7_result.veto_reason
            base_confidence *= 0.3
            
        elif lambda7_result.risk_regime == 'RISK_OFF' and raw_direction == 'BUY':
            macro_veto = True
            macro_reason = 'Risk-Off regime: Long trades vetoed'
            base_confidence *= 0.2
            
        elif lambda7_result.risk_regime == 'RISK_ON' and raw_direction == 'SELL':
            macro_veto = True
            macro_reason = 'Risk-On regime: Short trades vetoed'
            base_confidence *= 0.2
        
        # 5. Apply λ₈ light-cone veto (highest authority - kills signals)
        light_cone_veto = False
        if lambda8_result.kill_switch_triggered:
            light_cone_veto = True
            macro_reason = f'λ₈ KILL SWITCH: {lambda8_result.kill_switch_reason}'
            base_confidence = 0
            
        elif lambda8_result.violation_detected:
            # Reduce confidence but don't kill completely
            base_confidence *= (1 - lambda8_result.violation_severity)
        
        # 6. Adjust for manipulation (λ₄)
        if smk_manipulation and raw_direction != 'HOLD':
            base_confidence *= 0.7
            
        # 7. Adjust for entrapment (λ₁)
        if smk_entrapment:
            base_confidence *= 1.2  # Boost confidence when entrapment confirmed
        
        # 8. Determine signal strength
        final_confidence = min(1.0, max(0, base_confidence))
        
        if final_confidence >= 0.7:
            strength = SignalStrength.STRONG
        elif final_confidence >= 0.5:
            strength = SignalStrength.MODERATE
        elif final_confidence >= 0.3:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.INVALID
        
        # 9. Dynamic position sizing based on risk metrics
        position_size = 0
        stop_loss = current_price
        take_profit = current_price
        
        if strength != SignalStrength.INVALID and not macro_veto and not light_cone_veto:
            # Get risk metrics
            port_metrics = self.risk_manager.calculate_portfolio_metrics()
            
            # Adjust position size based on volatility (λ₁)
            if smk_entrapment:
                self.volatility_adjustment = 1.2  # Can take larger position
            elif lambda8_result.violation_detected:
                self.volatility_adjustment = 0.5  # Reduce position
            else:
                self.volatility_adjustment = 1.0
            
            # Adjust for macro regime
            if lambda7_result.risk_regime in ['RISK_ON', 'RISK_OFF']:
                self.macro_adjustment = 0.8  # More conservative in clear regimes
            else:
                self.macro_adjustment = 1.0
            
            # Calculate risk per trade
            risk_per_trade = self.risk_manager.max_portfolio_risk * final_confidence
            risk_per_trade *= self.volatility_adjustment
            risk_per_trade *= self.macro_adjustment
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                price=current_price,
                risk_per_trade=risk_per_trade
            )
            
            # Set stop-loss based on λ₆ displacement
            sl_percent = 0.05  # Default 5%
            if smk_displacement > 0.5:
                sl_percent = 0.03  # Tighter stop for strong displacement
            elif smk_entrapment:
                sl_percent = 0.04  # Moderate stop for entrapment
            
            stop_loss = current_price * (1 - sl_percent) if raw_direction == 'BUY' else current_price * (1 + sl_percent)
            take_profit = current_price * (1 + sl_percent * 2) if raw_direction == 'BUY' else current_price * (1 - sl_percent * 2)
        
        # 10. Create final signal
        signal = SMKSignal(
            direction=raw_direction if strength != SignalStrength.INVALID else 'HOLD',
            strength=strength,
            confidence=final_confidence,
            position_size=position_size,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            lambda_signals={
                'λ₁_entrapment': float(smk_entrapment),
                'λ₄_manipulation': float(smk_manipulation),
                'λ₆_displacement': smk_displacement,
                'λ₇_macro_score': lambda7_result.score,
                'λ₈_light_cone': lambda8_result.score,
                'λ₇_smt': 1.0 if lambda7_result.smt_state != 'NO_DIVERGENCE' else 0.0
            },
            veto_reason=macro_reason if (macro_veto or light_cone_veto) else None,
            timestamp=datetime.now()
        )
        
        # Store history
        self.trade_history.append(signal)
        
        return signal
    
    def execute_trade(self, signal: SMKSignal) -> Dict:
        """Execute trade if signal is valid"""
        
        if signal.strength == SignalStrength.INVALID:
            return {
                'executed': False,
                'reason': f'Invalid signal: {signal.veto_reason}',
                'signal': signal
            }
        
        if signal.position_size <= 0:
            return {
                'executed': False,
                'reason': 'Position size zero based on risk limits',
                'signal': signal
            }
        
        # Add position to risk manager
        try:
            # Determine quantity (simplified - in reality, get from broker)
            quantity = signal.position_size
            
            self.risk_manager.add_position(
                symbol='EURUSD',
                quantity=quantity,
                price=signal.stop_loss_price if signal.direction == 'SELL' else signal.take_profit_price
            )
            
            return {
                'executed': True,
                'position_size': signal.position_size,
                'stop_loss': signal.stop_loss_price,
                'take_profit': signal.take_profit_price,
                'confidence': signal.confidence,
                'signal': signal
            }
            
        except Exception as e:
            return {
                'executed': False,
                'reason': str(e),
                'signal': signal
            }
    
    def update_market_prices(self, current_price: float):
        """Update all positions with current market price"""
        for symbol in self.risk_manager.positions:
            self.risk_manager.update_position_price(symbol, current_price)
    
    def get_risk_overlay(self) -> Dict:
        """Get current risk overlay for display"""
        metrics = self.risk_manager.calculate_portfolio_metrics()
        
        return {
            'portfolio_value': metrics.portfolio_value,
            'var_95': metrics.var_95,
            'max_drawdown': metrics.max_drawdown,
            'sharpe_ratio': metrics.sharpe_ratio,
            'risk_level': metrics.risk_level.value,
            'positions': len(self.risk_manager.positions),
            'volatility': metrics.volatility,
            'risk_adjusted_capital': self.risk_manager.current_capital
        }
    
    def get_lambda_dashboard(self) -> Dict:
        """Get current λ sensor readings for dashboard"""
        return {
            'λ₁_entrapment': self.lambda7.config.dxy_divergence_threshold,
            'λ₇_macro_score': self.lambda7.correlation_engine.update(0, 0),  # Placeholder
            'λ₈_status': 'ACTIVE'  # Placeholder
        }


# ---------------------------------------------------------------------------
# Integration with Your WebSocket Pipeline
# ---------------------------------------------------------------------------

class SMKLiveTradingSystem:
    """Complete live trading system with SMK + Risk Management"""
    
    def __init__(self, initial_capital: float = 100_000):
        self.integrator = SMKRiskIntegrator(initial_capital=initial_capital)
        self.active_trades = []
        
    def on_new_smk_bar(self, smk_result: Dict, current_price: float):
        """
        Call this on every new bar from your SMK pipeline.
        
        Args:
            smk_result: Output from SMKPipeline.step()
            current_price: Current market price
        """
        # Extract SMK signals
        fusion = smk_result.get('fusion', {})
        vol_decay = smk_result.get('vol_decay', {})
        manipulation = smk_result.get('manipulation', {})
        displacement = smk_result.get('displacement', {})
        
        # Get macro data (you need to feed this from your data source)
        dxy_price = smk_result.get('bar', {}).get('dxy', 105.0)
        spx_price = smk_result.get('bar', {}).get('spx', 4500.0)
        
        # Evaluate signal
        signal = self.integrator.evaluate_signal(
            symbol='EURUSD',
            current_price=current_price,
            smk_fusion_p=fusion.get('p_fused', 0),
            smk_entrapment=vol_decay.get('entrapped', False),
            smk_manipulation=manipulation.get('active', False),
            smk_displacement=displacement.get('body_ratio', 0),
            dxy_price=dxy_price,
            spx_price=spx_price,
            volume=smk_result.get('bar', {}).get('volume', 100)
        )
        
        # Execute if signal is strong
        if signal.strength == SignalStrength.STRONG and signal.confidence > 0.7:
            trade_result = self.integrator.execute_trade(signal)
            if trade_result['executed']:
                self.active_trades.append(trade_result)
                print(f"🔴 TRADE EXECUTED: {signal.direction} "
                      f"Size: {signal.position_size} "
                      f"Confidence: {signal.confidence:.2%}")
        
        # Check stop losses
        self.integrator.update_market_prices(current_price)
        
        # Log risk metrics
        risk_overlay = self.integrator.get_risk_overlay()
        print(f"📊 Risk: {risk_overlay['risk_level']} "
              f"VaR: ${risk_overlay['var_95']:.2f} "
              f"Drawdown: {risk_overlay['max_drawdown']:.2%}")
        
        return signal


# ---------------------------------------------------------------------------
# Usage Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("SMK + Risk Manager Integration Demo")
    print("="*60)
    
    # Initialize trading system
    trading_system = SMKLiveTradingSystem(initial_capital=100_000)
    
    # Simulate a few SMK bar results
    demo_signals = [
        {'p_fused': 0.75, 'entrapped': True, 'manipulation': False, 'body_ratio': 0.85},
        {'p_fused': -0.68, 'entrapped': True, 'manipulation': True, 'body_ratio': 0.72},
        {'p_fused': 0.32, 'entrapped': False, 'manipulation': False, 'body_ratio': 0.45},
        {'p_fused': 0.82, 'entrapped': True, 'manipulation': False, 'body_ratio': 0.91},
    ]
    
    prices = [1.0850, 1.0820, 1.0835, 1.0870]
    
    for i, (signal_data, price) in enumerate(zip(demo_signals, prices)):
        print(f"\n--- Bar {i+1} @ {price} ---")
        
        smk_result = {
            'fusion': {'p_fused': signal_data['p_fused']},
            'vol_decay': {'entrapped': signal_data['entrapped']},
            'manipulation': {'active': signal_data['manipulation']},
            'displacement': {'body_ratio': signal_data['body_ratio']},
            'bar': {'volume': 1500, 'dxy': 105.2, 'spx': 4500}
        }
        
        signal = trading_system.on_new_smk_bar(smk_result, price)
        
        print(f"Signal: {signal.direction} | "
              f"Strength: {signal.strength.value} | "
              f"Confidence: {signal.confidence:.2%}")
        
        if signal.veto_reason:
            print(f"⚠️ VETO: {signal.veto_reason}")
    
    print("\n" + "="*60)
    print("Final Risk Summary")
    print("="*60)
    
    summary = trading_system.integrator.get_risk_overlay()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
