#!/usr/bin/env python3
"""
Sovereign Market Kernel - Main Orchestrator (Updated with Lambda Fusion Engine)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Core IPDA Layer 1
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

# Fusion Engine (Ring 0 Veto Authority)
from lambda_fusion_engine import LambdaFusionEngine

# Risk
from risk.mandra_kernels import MandraGate


class SovereignMarketKernel:
    """Main orchestrator integrating all SMK components + Lambda Fusion Engine."""

    def __init__(self):
        print(" Initializing Sovereign Market Kernel (with Lambda Fusion)...\n")

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

        # === Risk Gate ===
        self.mandra = MandraGate()

        print("✅ Sovereign Market Kernel fully initialized with Lambda Fusion Engine\n")

    def on_new_bar(self, df: pd.DataFrame):
        """Main processing pipeline - call on every new candle"""
        if len(df) < 60:
            print("⚠️ Insufficient data (need at least 60 bars)")
            return None

        # 1. IPDA Layer 1 - Structural Analysis
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

        # Placeholder for harmonic (needs predicted vs actual - simplified here)
        harmonic_state = self.harmonic.detect_trap(
            predicted_prices=df['close'].tail(64).values * 1.01,  # dummy prediction
            actual_prices=df['close'].tail(64).values
        )

        manipulation_state = self.manipulation.scan_for_manipulation(df, df['volume'].mean())

        # 3. Prepare Lambda Signals for Fusion Engine
        lambda_signals = {
            "λ1_vol_decay": {
                "score": 0.9 if entrapment_state.is_entrapped else 0.2,
                "confidence": min(1.0, entrapment_state.latent_energy_score / 50),
                "veto": False
            },
            "λ3_harmonic": {
                "score": -1.0 if harmonic_state.is_inverted else 0.4,
                "confidence": 0.75,
                "veto": harmonic_state.is_inverted
            },
            "λ4_manipulation": {
                "score": 0.8 if manipulation_state.is_active else -0.3,
                "confidence": manipulation_state.confidence_score / 100,
                "veto": manipulation_state.is_active
            },
            "λ5_displacement": {
                "score": displacement_state.direction,
                "confidence": 0.85 if displacement_state.is_displacement else 0.5,
                "veto": displacement_state.is_vetoed
            },
            "λ6_bias": {
                "score": 1.0 if bias_state.bias == "BULLISH" else -1.0 if bias_state.bias == "BEARISH" else 0.0,
                "confidence": bias_state.coherence,
                "veto": False
            },
        }

        # 4. Ring 0: Lambda Fusion + Veto
        fused = self.fusion_engine.fuse(
            lambda_signals=lambda_signals,
            ipda_phase_confidence=phase_state.confidence if hasattr(phase_state, 'confidence') else 0.7
        )

        # 5. Mandra Risk Gate (Information-Theoretic veto)
        gate = self.mandra.evaluate_gate(
            current_phi=np.array([fused.p_fused]),
            stability=phase_state.confidence if hasattr(phase_state, 'confidence') else 0.65,
            raw_size=0.02
        )

        # Final Decision
        trade_allowed = fused.veto_active is False and gate.is_open and abs(fused.p_fused) > 0.4

        # Output Summary
        print(f"[{df.index[-1].strftime('%Y-%m-%d %H:%M')}] "
              f"Phase: {phase_state.phase} | "
              f"Bias: {bias_state.bias} | "
              f"Fused: {fused.p_fused:+.3f} | "
              f"Confidence: {fused.confidence:.2f} | "
              f"Gate: {'OPEN' if gate.is_open else 'CLOSED'} | "
              f"Signal: {'✅ TRADE ALLOWED' if trade_allowed else '⛔ HALTED'}")

        return {
            "timestamp": df.index[-1],
            "phase": phase_state.phase,
            "bias": bias_state.bias,
            "fused_signal": fused.p_fused,
            "confidence": fused.confidence,
            "veto_active": fused.veto_active,
            "regime": fused.regime,
            "trade_allowed": trade_allowed,
            "status": fused.status
        }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    kernel = SovereignMarketKernel()
    
    print("\n" + "="*70)
    print("Sovereign Market Kernel is ready!")
    print("To test with real data:")
    print("   df = pd.read_csv('your_ohlcv_data.csv', parse_dates=True, index_col=0)")
    print("   result = kernel.on_new_bar(df)")
    print("="*70)