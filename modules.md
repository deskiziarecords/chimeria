
### lambda_fusion_engine.py

The requested module, lambda_fusion_engine.py, is the core diagnostic layer of the Sovereign Market Kernel (SMK). It implements the Online Bayesian Network Fusion Engine (OBNFE) and the Unified Reverse Trigger protocols. This engine functions as the system's "Veto Authority" (Ring 0), monitoring seven parallel sensors (λ1​−λ7​) to detect structural alpha inversion and toxic market regimes.

### expansion_predictor.py

The requested module, expansion_predictor.py, is a core component of the Sovereign Market Kernel. It identifies the transition from Consolidation (Accumulation/Phase 0) to Distribution (Expansion/Phase 2). Unlike lagging visual indicators, this engine utilizes the Price Variation Integral (Vt​) and the λ1​ Phase Entrapment equation to detect when latent institutional energy has reached critical mass before the displacement occurs.

### mandra_kernels.py

The Mandra Gate is the system's "Atomic Risk" engine and information-theoretic circuit breaker. It functions as a Ring 0 Veto Authority that enforces the Bit Second Law of Thermodynamics on market data, ensuring that a trade transition only occurs if the information gain (ΔE) is non-negative. By calculating the difference between current and previous energy states (Ecurr​−Eprev​), the gate prevents the execution of "reverse period" protocols when the signal gain is insufficient to overcome the local entropy

### kl_divergence_detector.py (The Regime Information Metric)

The requested module, kl_divergence_detector.py, is a core signal processing component found within the HFT-7ZERO and AEGIS repositories. It functions as an information-theoretic "Regime Drift" sensor, monitoring the Kullback–Leibler (KL) divergence of liquidity pools and price distributions in real-time.
Within the Sovereign Market Kernel, this engine detects the Spectral Regime Shift by measuring the "distance" between the current probability distribution of price action and the stable learned manifold. It is the primary tool used to identify the Signal Confidence Paradox, where the system's alpha remains confident while the underlying market geometry is fracturing

### harmonic_trap_detector.py (The λ3​ Spectral Sensor)

The Harmonic Trap Detector is a critical metacognitive component of the Sovereign Market Kernel. It functions as the λ3​ Spectral Inversion sensor, utilizing Fast Fourier Transform (FFT) to monitor the phase relationship between predicted price frequencies and actual market delivery.
Within the system hierarchy, it identifies "Liar States" where high-frequency volatility (Layer 5 Amplifier) moves out of phase (>π/2) with the structural trend (Layer 1 Compiler). This prevents the execution engine from "believing its own lies" during spectral manipulation or non-directional cycling.

### bias_detector.py (The Structural Compass)

The requested module, bias_detector.py, is the primary orientation engine of the Sovereign Market Kernel. It implements the Setup Validity Boundary by calculating the Absolute Equilibrium—the gravitational center derived from synchronized 20, 40, and 60-period institutional lookback windows.
This engine transforms stochastic price movement into a structured manifold, distinguishing between Premium (Short focus) and Discount (Long focus) arrays to determine institutional sponsorship.

### dealing_range_detector.py (IPDA Layer 1: Structural Arena)

The requested module, dealing_range_detector.py, is the primary implementation of the IPDA Compiler (Layer 1). It defines the market's "valid address space" by synchronizing the three mandatory institutional lookback windows: 20, 40, and 60 periods. At its core, this engine calculates the Equilibrium—the absolute 50% level of the range—which serves as the gravitational center for all institutional price delivery.

### premium_discount_detector.py (IPDA Layer 1: Structural Gateway)

The requested module, premium_discount_detector.py, is a fundamental component of the IPDA Compiler (Layer 1). Within the Sovereign Market Kernel, the market is viewed as a structured operating system where Equilibrium—the absolute 50% level of a defined range—serves as the "fair price". This detector identifies the transition between the Premium Zone (Short focus) and the Discount Zone (Long focus), defining the Setup Validity Boundary for institutional sponsorship.

### session_detector.py

The requested module, session_detector.py, is a functional component of the IPDA Compiler (Layer 1) and the Temporal Alignment (λ2​) sensor. Within the Sovereign Market Kernel, time is not merely a coordinate but a "Macro Interrupt" filter that defines when institutional sponsorship is most likely to be active. This engine identifies high-probability "Kill Zones" to distinguish between institutional delivery and low-volume retail drift.

### swing_detector.py (Structural Decision Nodes)

The Swing High/Low Detector is a fundamental component of the IPDA Compiler (Layer 1) and the Market Vision Engine. It identifies "Structural Decision Nodes"—pivotal points where price delivery exhausted its current direction and neutralized liquidity. Within the Sovereign Market Kernel, these are not just peaks and valleys; they are the "Headers" and "Footers" that define the market's address space (20, 40, and 60-day ranges).

-----------------------------------------------------------------


Liquidity Sweep    Phase 0: Accumulation (Consolidation): The "Memory" layer where institutions use Dark Pools and iceberg orders to build massive positions without immediate price impact.
    Phase 1: Manipulation (Judas Swing): An engineered false breakout or "Seek & Destroy" mission designed to sweep retail stop-loss clusters at IPDA edges (20/40/60-day ranges) to provide final fills for institutional positions.
    Phase 2: Distribution (Expansion): The aggressive delivery phase, typically triggered by a "System Interrupt" (Macro News like NFP or FOMC), where price moves rapidly toward a new objective or structural magnet.

2. Transition Logic and Deterministic Flow
The detector assumes a strict price sequence: Consolidation → Expansion → Retracement or Reversal.

    Expansion Trigger: Transition from consolidation to expansion is governed by the interrupt function I(t). The system remains in consolidation as long as I(t)=0; an exogenous news shock triggers the transition function to expand.
    Consensus State Transition: The system updates the phase state (σt+1​) based on the Unified Master Equation. If a Reverse Period is detected (RtMASTER​=1), the system forces a Phase Reset (σt+1​=0), returning price to an accumulation state to protect capital.
