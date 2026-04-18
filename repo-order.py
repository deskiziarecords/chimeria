#!/usr/bin/env python3
"""
SMK/IPDA Trading System Organizer + Main Orchestrator
"""

import os
import shutil
from pathlib import Path

ROOT = Path("smk_ipda_trading_system")

# Original filename → target path
FILE_MAPPING = {
    "bias_detector.py": "core/bias_detector.py",
    "dealing_range_detector.py": "core/dealing_range_detector.py",
    "premium_discount_detector.py": "core/premium_discount_detector.py",
    "equilibrium_cross_detector.py": "core/equilibrium_cross_detector.py",
    "ipda_phase_detector.py": "core/ipda_phase_detector.py",
    "swing_detector.py": "core/swing_detector.py",
    "session_detector.py": "core/session_detector.py",

    "volatility_decay_detector.py": "lambda_sensors/volatility_decay_detector.py",
    "displacement_detector.py": "lambda_sensors/displacement_detector.py",
    "harmonic_trap_detector.py": "lambda_sensors/harmonic_trap_detector.py",
    "manipulation_detector.py": "lambda_sensors/manipulation_detector.py",
    "expansion_predictor.py": "lambda_sensors/expansion_predictor.py",

    "ccm_engine.py": "causality/ccm_engine.py",
    "granger_causality.py": "causality/granger_causality.py",
    "transfer_entropy.py": "causality/transfer_entropy.py",
    "spearman_lag_engine.py": "causality/spearman_lag_engine.py",
    "signal_fusion_kernel.py": "causality/signal_fusion_kernel.py",

    "fvg_detector_engine.py": "liquidity/fvg_detector_engine.py",
    "liquidity_void_scanner.py": "liquidity/liquidity_void_scanner.py",
    "order_block_detector.py": "liquidity/order_block_detector.py",
    "volume_profile_memory_engine.py": "liquidity/volume_profile_memory_engine.py",

    "risk.py": "risk/risk.py",
    "mandra_kernels.py": "risk/mandra_kernels.py",

    "depth.py": "market/depth.py",
    "order_book.py": "market/order_book.py",

    "kl_divergence_detector.py": "detectors/kl_divergence_detector.py",
    "topological_fracture_detector.py": "detectors/topological_fracture_detector.py",

    "README.md": "README.md",
    "modules.md": "docs/modules.md",
}

PACKAGES = ["core", "lambda_sensors", "causality", "liquidity", "risk", "market", "detectors", "utils"]

def create_structure():
    print(f" Creating Sovereign Market Kernel structure in: {ROOT.resolve()}\n")
    ROOT.mkdir(exist_ok=True)

    # Create directories
    for d in PACKAGES + ["config", "tests", "docs", "utils"]:
        (ROOT / d).mkdir(exist_ok=True)

    # Create __init__.py
    for pkg in PACKAGES:
        (ROOT / pkg / "__init__.py").touch()

    # Move files
    moved = 0
    for src_name, dst_rel in FILE_MAPPING.items():
        src = Path(src_name)
        dst = ROOT / dst_rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f"   ✅ {src_name} → {dst_rel}")
            moved += 1
        else:
            print(f"   ⚠️  Missing: {src_name}")

    # requirements.txt
    (ROOT / "requirements.txt").write_text("""numpy>=1.24
pandas>=2.0
scipy>=1.10
statsmodels>=0.14
jax>=0.4.0
ripser
scikit-learn
pytz
""")

    print(f"\n🎉 Structure created! {moved} files moved.\n")

def create_main_orchestrator():
    main_content = '''#!/usr/bin/env python3
"""
Sovereign Market Kernel - Main Orchestrator
"""

import pandas as pd
from pathlib import Path

# Import core components
from core.bias_detector import BiasDetector
from core.dealing_range_detector import DealingRangeDetector
from core.ipda_phase_detector import IPDACompiler
from core.equilibrium_cross_detector import EquilibriumCrossDetector
from core.swing_detector import SwingDetector
from core.session_detector import SessionKillZoneDetector

from lambda_sensors.expansion_predictor import IPDAExpansionPredictor
from lambda_sensors.displacement_detector import DisplacementDetector
from lambda_sensors.volatility_decay_detector import VolatilityDecayDetector

from causality.signal_fusion_kernel import SignalFusionEngine
from risk.mandra_kernels import MandraGate
from risk.risk import RiskEngine

class SovereignMarketKernel:
    """Main orchestrator for the entire SMK system."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        
        # Layer 1: Structural Compiler
        self.ipda = IPDACompiler()
        self.bias = BiasDetector()
        self.dealing_range = DealingRangeDetector()
        self.equilibrium_cross = EquilibriumCrossDetector()
        self.swing = SwingDetector()
        self.session = SessionKillZoneDetector()

        # Lambda Sensors
        self.expansion = IPDAExpansionPredictor()
        self.displacement = DisplacementDetector()
        self.vol_decay = VolatilityDecayDetector()

        # Fusion & Risk
        self.fusion = SignalFusionEngine()
        self.mandra = MandraGate()
        self.risk = RiskEngine(self.config.get('risk', {}), None)  # event_sender placeholder

        print(" Sovereign Market Kernel initialized")

    def on_new_bar(self, df: pd.DataFrame):
        """Main processing loop - call on every new candle/tick"""
        
        # 1. Structural Analysis (IPDA Layer 1)
        phase_state = self.ipda.process_market_state(df)
        bias_state = self.bias.detect_bias(df)
        range_state = self.dealing_range.update_ranges(df)
        cross_state = self.equilibrium_cross.process_tick(df)
        session_state = self.session.detect_session(df.index[-1])

        # 2. Lambda Sensors
        expansion_state = self.expansion.predict_expansion(df, {'H60': 0, 'L60': 0, 'EQ': 0})
        displacement_state = self.displacement.analyze_candle(
            df.iloc[-1].to_dict(), 
            df['atr'].iloc[-1] if 'atr' in df else 0
        )
        entrapment_state = self.vol_decay.detect_entrapment(df)

        # 3. Signal Fusion
        fused_signal = self.fusion.fuse(
            p_ipda=bias_state.bias == "BULLISH",
            causal_signals=[],           # add your causal signals here
            regime_stability=phase_state.confidence
        )

        # 4. Risk & Execution Gate
        gate = self.mandra.evaluate_gate(
            current_phi=np.array([fused_signal.p_fused]),
            stability=phase_state.confidence,
            raw_size=0.02
        )

        # 5. Risk Check
        self.risk.update(None, {})  # placeholder

        print(f"[{df.index[-1]}] Phase: {phase_state.phase} | Bias: {bias_state.bias} | "
              f"Fused: {fused_signal.p_fused:.3f} | Gate: {gate.status}")

        return {
            "phase": phase_state,
            "bias": bias_state,
            "fused": fused_signal,
            "gate": gate,
            "risk_ok": gate.is_open
        }


if __name__ == "__main__":
    # Example usage
    kernel = SovereignMarketKernel()
    
    # Simulate with dummy data
    print(" SMK ready. Add your OHLCV DataFrame and call kernel.on_new_bar(df)")
'''

    (ROOT / "main.py").write_text(main_content)
    print("    Created main.py (orchestrator)")

    # Create a simple run example
    (ROOT / "run.py").write_text('''from main import SovereignMarketKernel
import pandas as pd

kernel = SovereignMarketKernel()
print("Sovereign Market Kernel is ready!")
# df = pd.read_csv("your_data.csv", parse_dates=True, index_col=0)
# kernel.on_new_bar(df)
''')

if __name__ == "__main__":
    create_structure()
    create_main_orchestrator()
    print("\n All done! Your repo is now organized and ready.")
    print(f"   cd {ROOT}")
    print("   pip install -r requirements.txt")
    print("   python run.py")