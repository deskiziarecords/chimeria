"""
SovereignMarketKernel.py

The central brain of the Sovereign Market Kernel (SMK).
Orchestrates all layers: IPDA Compiler, Lambda Sensors, Lambda Fusion Engine,
Order Flow, and Risk Gate.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime

# Core imports
from core.bias_detector import BiasDetector
from core.dealing_range_detector import DealingRangeDetector
from core.ipda_phase_detector import IPDACompiler
from core.equilibrium_cross_detector import EquilibriumCrossDetector
from core.swing_detector import SwingDetector
from core.session_detector import SessionKillZoneDetector

# Lambda Sensors
from lambda_sensors.expansion_predictor import IPDAExpansionPredictor
from lambda_sensors.displacement_detector import DisplacementDetector
from lambda_sensors.volatility_decay_detector import VolatilityDecayDetector
from lambda_sensors.harmonic_trap_detector import HarmonicTrapDetector
from lambda_sensors.manipulation_detector import ManipulationPhaseDetector

# Advanced Engines
from lambda_fusion_engine import LambdaFusionEngine
from order_flow_visibility_engine import OrderFlowVisibilityEngine

# Risk
from risk.mandra_kernels import MandraGate


class SovereignMarketKernel:
    """
    Sovereign Market Kernel - Main Orchestrator
    Integrates IPDA Layer 1, all Lambda Sensors, Fusion Engine, Order Flow,
    and Risk Gate into one coherent decision engine.
    """

    def __init__(self):
        print("Initializing Sovereign Market Kernel v1.0...\n")

        # === Layer 1: IPDA Structural Compiler ===
        self.ipda = IPDACompiler()
        self.bias = BiasDetector()
        self.dealing_range = DealingRangeDetector()
        self.equilibrium_cross = EquilibriumCrossDetector()
        self.swing = SwingDetector(lookback=5)
        self.session = SessionKillZoneDetector()

        # === Lambda Sensors (λ1–λ7) ===
        self.expansion = IPDAExpansionPredictor()
        self.displacement = DisplacementDetector()
        self.vol_decay = VolatilityDecayDetector()
        self.harmonic = HarmonicTrapDetector()
        self.manipulation = ManipulationPhaseDetector()

        # === Ring 0: Lambda Fusion Engine ===
        self.fusion_engine = LambdaFusionEngine()

        # === Order Flow Visibility ===
        self.order_flow = OrderFlowVisibilityEngine()

        # === Risk Gate ===
        self.mandra = MandraGate()

        print("Sovereign Market Kernel fully loaded and ready.\n")

    def on_new_bar(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process one new bar through the entire SMK pipeline.
        Returns a comprehensive analysis result.
        """
        if len(df) < 60:
            return {"error": "Insufficient data (need ≥60 bars)", "status": "waiting"}

        try:
            # 1. IPDA Layer 1 - Structural Context
            phase_state = self.ipda.process_market_state(df)
            bias_state = self.bias.detect_bias(df)
            range_state = self.dealing_range.update_ranges(df)
            cross_state = self.equilibrium_cross.process_tick(df)
            session_state = self.session.detect_session(df.index[-1])

            # 2. Lambda Sensors
            magnets = {
                'H60': float(df['high'].tail(60).max()),
                'L60': float(df['low'].tail(60).min()),
                'EQ': float((df['high'].tail(60).max() + df['low'].tail(60).min()) / 2)
            }

            expansion_state = self.expansion.predict_expansion(df, magnets)
            displacement_state = self.displacement.analyze_candle(
                df.iloc[-1].to_dict(),
                atr20=float(df.get('atr', pd.Series([0.001])).iloc[-1])
            )
            entrapment_state = self.vol_decay.detect_entrapment(df)

            # Harmonic trap (simplified - uses last 64 bars)
            harmonic_state = self.harmonic.detect_trap(
                predicted_prices=df['close'].tail(64).values * 1.005,  # dummy prediction
                actual_prices=df['close'].tail(64).values
            )

            manipulation_state = self.manipulation.scan_for_manipulation(df, df['volume'].mean())

            # 3. Order Flow Analysis (if tick data available, otherwise placeholder)
            # For now we use a simple placeholder. You can extend this later with real ticks.
            order_flow_state = {
                "delta": 0.0,
                "is_absorption": False,
                "institutional_pulse": False,
                "status": "NO_TICK_DATA"
            }

            # 4. Lambda Signals for Fusion
            lambda_signals = {
                "λ1_vol_decay": {
                    "score": 0.9 if getattr(entrapment_state, 'is_entrapped', False) else 0.2,
                    "confidence": min(1.0, getattr(entrapment_state, 'latent_energy_score', 0) / 50),
                    "veto": False
                },
                "λ3_harmonic": {
                    "score": -1.0 if getattr(harmonic_state, 'is_inverted', False) else 0.4,
                    "confidence": 0.75,
                    "veto": getattr(harmonic_state, 'is_inverted', False)
                },
                "λ4_manipulation": {
                    "score": 0.8 if getattr(manipulation_state, 'is_active', False) else -0.3,
                    "confidence": getattr(manipulation_state, 'confidence_score', 50) / 100,
                    "veto": getattr(manipulation_state, 'is_active', False)
                },
                "λ5_displacement": {
                    "score": getattr(displacement_state, 'direction', 0),
                    "confidence": 0.85 if getattr(displacement_state, 'is_displacement', False) else 0.5,
                    "veto": getattr(displacement_state, 'is_vetoed', False)
                },
                "λ6_bias": {
                    "score": 1.0 if bias_state.bias == "BULLISH" else -1.0 if bias_state.bias == "BEARISH" else 0.0,
                    "confidence": getattr(bias_state, 'coherence', 0.6),
                    "veto": False
                },
            }

            # 5. Ring 0 Fusion + Veto
            fused = self.fusion_engine.fuse(
                lambda_signals=lambda_signals,
                ipda_phase_confidence=getattr(phase_state, 'confidence', 0.7)
            )

            # 6. Mandra Risk Gate
            gate = self.mandra.evaluate_gate(
                current_phi=np.array([fused.p_fused]),
                stability=getattr(phase_state, 'confidence', 0.65),
                raw_size=0.02
            )

            # Final Decision
            trade_allowed = (not fused.veto_active) and gate.is_open and abs(fused.p_fused) > 0.4

            # Build comprehensive result
            result = {
                "timestamp": df.index[-1].isoformat() if hasattr(df.index[-1], 'isoformat') else str(df.index[-1]),
                "phase": getattr(phase_state, 'phase', 'UNKNOWN'),
                "bias": bias_state.bias,
                "fused_signal": fused.p_fused,
                "confidence": fused.confidence,
                "regime": fused.regime,
                "veto_active": fused.veto_active,
                "trade_allowed": trade_allowed,
                "session": getattr(session_state, 'session_name', 'UNKNOWN'),
                "expansion_prob": getattr(expansion_state, 'expansion_prob', 0.0),
                "order_flow": order_flow_state,
                "status": fused.status,
                "gate_status": "OPEN" if gate.is_open else "CLOSED"
            }

            return result

        except Exception as e:
            print(f"❌ Error in on_new_bar: {e}")
            return {"error": str(e), "status": "failed"}

    def get_system_info(self) -> Dict:
        """Return basic system metadata"""
        return {
            "name": "Sovereign Market Kernel",
            "version": "1.0",
            "components": [
                "IPDA Layer 1", "Lambda Sensors", "Lambda Fusion Engine",
                "Order Flow Visibility", "Mandra Risk Gate"
            ],
            "status": "ready"
        }


# Quick test when running directly
if __name__ == "__main__":
    kernel = SovereignMarketKernel()
    print("\n" + "="*60)
    print("SovereignMarketKernel is ready for use in smk_pipeline.py")
    print("Use: kernel.on_new_bar(your_dataframe)")
    print("="*60)
