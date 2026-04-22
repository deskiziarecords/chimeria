import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import jax
import jax.numpy as jnp

@dataclass
class FusionTelemetry:
    p_fused: float          # Final master signal probability [-1.0, 1.0]
    confidence: float       # Overall system confidence (0.0 - 1.0)
    veto_active: bool       # True if any λ sensor triggered reverse
    active_lambdas: List[str]  # Which lambdas fired
    regime: str             # "SINCERE", "LIAR_STATE", "REVERSE_PERIOD", "FRACTURE"
    status: str


class LambdaFusionEngine:
    """
    Ring 0 Veto Authority: Online Bayesian Network Fusion Engine (OBNFE)
    
    Monitors λ1 to λ7 in parallel and computes a unified truth signal.
    If any critical sensor detects inversion (Liar State), it forces veto (u_t = 0).
    """

    def __init__(self, 
                 lambda_weights: Optional[Dict[str, float]] = None,
                 veto_threshold: float = 0.65,
                 bayesian_prior: float = 0.5):
        
        # Default weights for the 7 lambda sensors (can be adapted online)
        self.lambda_weights = lambda_weights or {
              "λ1_vol_decay": 0.14,
              "λ2_session": 0.08,
              "λ3_harmonic": 0.12,
              "λ4_manipulation": 0.10,
              "λ5_displacement": 0.12,
              "λ6_bias": 0.10,
              "λ7_macro": 0.18,      # ← NEW: Macro causality gate
              "λ8_light_cone": 0.16, 
        }
        
        self.veto_threshold = veto_threshold      # Confidence required to override veto
        self.prior = bayesian_prior
        
        # Online Bayesian state (simple Dirichlet-style adaptive weights)
        self.posterior_weights = jnp.array(list(self.lambda_weights.values()))
        
        print("✅ Lambda Fusion Engine (Ring 0) initialized")

    def fuse(self,
             lambda_signals: Dict[str, Dict[str, float]],
             ipda_phase_confidence: float = 0.7,
             current_price: Optional[float] = None) -> FusionTelemetry:
        """
        Master Fusion Logic:
        
        1. Collect raw signals from all λ sensors
        2. Compute Bayesian-weighted fused probability
        3. Apply Unified Reverse Trigger (veto logic)
        4. Return final executable signal + diagnostics
        """
        
        active_lambdas = []
        weighted_sum = 0.0
        total_weight = 0.0
        veto_reasons = []

        # Process each lambda sensor
        for lambda_name, signal in lambda_signals.items():
            if lambda_name not in self.lambda_weights:
                continue
                
            raw_score = signal.get("score", 0.0)        # -1.0 to 1.0 (bearish to bullish)
            confidence = signal.get("confidence", 0.5)
            is_veto = signal.get("veto", False)
            
            w = self.lambda_weights[lambda_name]
            
            # Adaptive weighting: dampen low-confidence sensors
            effective_w = w * confidence
            weighted_sum += raw_score * effective_w
            total_weight += effective_w
            
            if is_veto:
                veto_reasons.append(lambda_name)
                active_lambdas.append(f"{lambda_name}(VETO)")
            elif abs(raw_score) > 0.6:
                active_lambdas.append(lambda_name)

        # Bayesian fusion with prior
        if total_weight > 0:
            p_structural = weighted_sum / total_weight
        else:
            p_structural = 0.0

        # Blend with IPDA Layer 1 confidence
        p_fused = (0.65 * p_structural) + (0.35 * (ipda_phase_confidence * 2 - 1))

        # Unified Reverse Trigger Logic (Ring 0 Veto)
        veto_active = len(veto_reasons) > 0
        
        if veto_active:
            regime = "REVERSE_PERIOD"
            final_p = 0.0  # Force halt
            status = f"VETO ACTIVATED by: {', '.join(veto_reasons)}"
        elif abs(p_fused) < 0.25:
            regime = "LIAR_STATE"
            final_p = 0.0
            status = "LOW CONVERGENCE - HALT"
        elif p_fused > 0.55 and ipda_phase_confidence > 0.75:
            regime = "SINCERE"
            final_p = p_fused
            status = "STRONG LONG BIAS - EXECUTION ENABLED"
        elif p_fused < -0.55 and ipda_phase_confidence > 0.75:
            regime = "SINCERE"
            final_p = p_fused
            status = "STRONG SHORT BIAS - EXECUTION ENABLED"
        else:
            regime = "NEUTRAL"
            final_p = 0.0
            status = "INSUFFICIENT CONVERGENCE"

        confidence_score = min(1.0, (ipda_phase_confidence + (total_weight / sum(self.lambda_weights.values()))) / 2)

        return FusionTelemetry(
            p_fused=round(final_p, 4),
            confidence=round(confidence_score, 4),
            veto_active=veto_active,
            active_lambdas=active_lambdas,
            regime=regime,
            status=status
        )

    def update_weights_online(self, performance_feedback: float):
        """Online adaptation of lambda weights based on recent P&L or accuracy feedback."""
        # Simple Bayesian update (Dirichlet-style)
        self.posterior_weights = self.posterior_weights * (0.95 + 0.1 * performance_feedback)
        self.posterior_weights /= self.posterior_weights.sum()
        
        # Update dict weights
        for i, key in enumerate(self.lambda_weights.keys()):
            self.lambda_weights[key] = float(self.posterior_weights[i])

    def get_sensor_status(self) -> str:
        """Diagnostic summary"""
        return f"Active weights: { {k: round(v,3) for k,v in self.lambda_weights.items()} }"


# ─────────────────────────────────────────────────────────────────────────────
# Example Usage (integrate with your main.py or SMK orchestrator)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fusion = LambdaFusionEngine()
    
    # Example signals from your lambda sensors
    example_signals = {
        "λ1_vol_decay": {"score": 0.85, "confidence": 0.82, "veto": False},   # Strong entrapment → expansion coming
        "λ3_harmonic": {"score": -0.3, "confidence": 0.65, "veto": False},
        "λ5_displacement": {"score": 0.92, "confidence": 0.88, "veto": False},
        "λ6_bias": {"score": 0.75, "confidence": 0.91, "veto": False},
        "λ7_regime": {"score": 0.4, "confidence": 0.7, "veto": False},
    }
    
    result = fusion.fuse(
        lambda_signals=example_signals,
        ipda_phase_confidence=0.83
    )
    
    print(result)
    print(f"\nFinal Decision → {result.status}")
