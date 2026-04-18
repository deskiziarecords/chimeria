
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

The Swing High/Low Detector is a fundamental component of the IPDA Compiler (Layer 1) and the Market Vision Engine. It identifies "Structural Decision Nodes"—pivotal points where price delivery exhausted its current direction and neutralized liquidity. Within the Sovereign Market Kernel, these are not just peaks and valleys; they are the "Headers" and "Footers" that define the market's address space (20, 40, and 60-day ranges) 


### equilibrium_cross_detector.py (IPDA Layer 1: The Gateway)

The requested module, equilibrium_cross_detector.py, is a specialized component of the IPDA Compiler (Layer 1). Within the Sovereign Market Kernel, Equilibrium is defined as the absolute 50% level of the structural range, acting as the market's "fair price". This detector identifies when price transitions between the Discount Zone (Long focus) and the Premium Zone (Short focus), marking the boundary of a trade setup's validity.


### displacement_detector.py

Following your selection, I am transmitting the Python implementation for the λ6​ Direction-Aware Displacement Veto. This module functions as a high-velocity micro-filter within the Sovereign Market Kernel (SMK), ensuring that immediate price instability does not undermine the system's macro control law. It identifies institutional "Expansion" footprints while detecting "Liar States" where micro-action contradicts the predicted structural intent.

### order_block_detector.py (The Origin Manifold)

The requested module, order_block_detector.py, identifies institutional "Origin Candles" where smart money accumulated or distributed massive positions before a high-velocity expansion. Within the Sovereign Market Kernel, an Order Block (OB) is not merely a candle of the opposite color; it is a structural "file header" that must be validated by the λ6​ Displacement parameter and the creation of an Imbalance Field (I).

### fvg_detector_engine.py (The Imbalance Field Compiler)

The requested module, fvg_detector_engine.py, identifies regions of "porous price action" known as Fair Value Gaps (FVG). These gaps represent a "liquidity vacuum" where price delivery occurred so rapidly that the algorithm failed to facilitate efficient trading. Within the Sovereign Market Kernel, these gaps serve as Internal Range Liquidity (IRL) targets that price is "coded" to return to for rebalancing.

### volume_profile_memory_engine.py (Liquidity Zone Detection)

The requested module, volume_profile_memory_engine.py, is the functional implementation of Layer 2: Memory (Dark Pools / Icebergs) within the Sovereign Market Kernel. This engine transitions from standard volume bars to Time-At-Price (TAP) Density, effectively reconstructing a Volume Profile to identify institutional liquidity nodes. It identifies "Islands" of consolidation where hidden institutional fuel is stored before an expansion phase.

### displacement_detector.py (The λ6​ Direction-Aware Filter)

The requested module, displacement_detector.py, identifies high-velocity institutional repricing by monitoring specific candle-state parameters known as Displacement. Within the Sovereign Market Kernel, Displacement is the definitive footprint of the Distribution Phase, characterized by large candle bodies and wide ranges that signify aggressive institutional intent. The system utilizes the λ6​ Displacement Veto logic to ensure these candles align with the macro structural intent, acting as a high-velocity micro-filter to prevent "Liar State" executions.

### topological_fracture_detector.py

The requested module, topological_fracture_detector.py, represents the peak of Topological Data Analysis (TDA) within the Sovereign Market Kernel. It adapts 3D graphics UV-unwrapping logic and Persistent Homology to identify structural fractures in the market manifold.
When the market is "sincere" (stable), data points form a Compact Cloud. When structural intelligence breaks, the point cloud expands and forms Persistent Loops (H1​)—one-dimensional "holes" in the geometry representing structural conflict and non-directional cycling. In this state, the only correct action is ut​=0 (Halt).

### volatility_decay_detector.py

The requested module, volatility_decay_detector.py, implements the λ1​ Phase Entrapment sensor. This is the primary engine used to quantify the exact decay rate of intra-range volatility during consolidation. By measuring the Price Variation Integral (Vt​) against the long-term ATR20 benchmark, the system identifies when latent institutional energy H(t) reaches critical mass. This solves the "visual lag" problem where human traders typically identify breakouts 2–5 bars too late.

### manipulation_detector.py (IPDA Phase 1: The Judas Swing)

The requested module, manipulation_detector.py, is a critical component of the Sovereign Market Kernel (SMK). It identifies the Manipulation Phase—the deceptive "Judas Swing" designed by the Interbank Price Delivery Algorithm (IPDA) to sweep retail stop-loss clusters located at the edges of structural ranges. This engine specializes in detecting the Wick Signature, where price probes beyond a 20, 40, or 60-day boundary and rejects sharply, harvesting liquidity before the true expansion begins

### ipda_phase_detector.py
The requested module, ipda_phase_detector.py (formally known as Layer 1: The Compiler), is the fundamental structural engine of the Sovereign Market Kernel. It identifies the deterministic Accumulation-Manipulation-Distribution (AMD) cycle by synchronizing high-fidelity lookback windows of 20, 40, and 60 periods. This engine defines the market's "valid address space," allowing the kernel to distinguish between random noise and institutional delivery.

--------------------------
#### causality

### granger_causality.py
The requested module implements the Linear Causal Inference Engine used within the HFT-7ZERO and AEGIS repositories to determine lead-lag relationships between multi-asset price streams (e.g., DXY leading EUR/USD). Unlike simple correlation, this system utilizes Vector Autoregression (VAR) to identify if past values of one time-series provide statistically significant predictive power over another, identifying the "True Lead" before a price impulse occurs.

### transfer_entropy.py (Information-Theoretic Directed Dependency)
The requested module, transfer_entropy.py, is a core component of the Intelligence & Signal Layer. Unlike Granger Causality, which assumes linear relationships, Transfer Entropy (Tent​) utilizes information theory to detect non-linear directed dependencies between market variables (e.g., Order Flow Imbalance leading Price changes). This module specifically implements the 6-bin discretization protocol and Miller-Madow bias correction required for sub-millisecond signal fusion in the HFT-7ZERO ecosystem.


### ccm_engine.py (Nonlinear Causal Inference)
The requested module, ccm_engine.py, is the primary implementation of Convergent Cross Mapping (CCM) within the HFT-7ZERO and AEGIS causality engines. While linear models like Granger Causality fail in coupled chaotic systems, CCM utilizes Takens’ Embedding Theorem to reconstruct the shadow manifolds of market variables. By testing the "Convergence" of prediction skill as the library size increases, the system identifies true nonlinear causal drivers (e.g., hidden Order Flow dynamics leading Price) even when they are not correlated in the time domain.

### spearman_lag_engine.py (Rank-Based Causal Sensor)
The requested module, spearman_lag_engine.py, is an integral component of the Intelligence & Signal Layer within the Sovereign Market Kernel. Unlike linear correlation, Spearman's rank correlation (ρ) identifies monotonic relationships between market variables, making it highly resilient to outliers and non-linear "wick" noise. This engine performs Temporal Lag Analysis to identify if a leading indicator (e.g., DXY strength) correlates with a target asset (e.g., EUR/USD) at specific time offsets (τ), validated via Bootstrap Resampling to ensure statistical significance

### signal_fusion_kernel.py (The Multi-Method Convergence Manifold)

The requested module, signal_fusion_kernel.py, is the primary synthesis engine of the Sovereign Market Kernel. It functions as the Unified Master Equation (Source,), responsible for aggregating disparate signals from the IPDA Compiler (L1), the SOS-27-X Sentinel (L2), and various causal sensors.
This engine solves the Signal Confidence Paradox (Source) by applying Bayesian Model Averaging and Kalman filtering to identify the market's "True Sincerity" while discarding "Liar States." It utilizes the specific institutional temporal decay constant e−0.08τ (Source,) to ensure that lagging causal signals are properly dampened relative to real-time price delivery.


-----------------------------------------------------------------


Liquidity Sweep    Phase 0: Accumulation (Consolidation): The "Memory" layer where institutions use Dark Pools and iceberg orders to build massive positions without immediate price impact.
    Phase 1: Manipulation (Judas Swing): An engineered false breakout or "Seek & Destroy" mission designed to sweep retail stop-loss clusters at IPDA edges (20/40/60-day ranges) to provide final fills for institutional positions.
    Phase 2: Distribution (Expansion): The aggressive delivery phase, typically triggered by a "System Interrupt" (Macro News like NFP or FOMC), where price moves rapidly toward a new objective or structural magnet.

2. Transition Logic and Deterministic Flow
The detector assumes a strict price sequence: Consolidation → Expansion → Retracement or Reversal.

    Expansion Trigger: Transition from consolidation to expansion is governed by the interrupt function I(t). The system remains in consolidation as long as I(t)=0; an exogenous news shock triggers the transition function to expand.
    Consensus State Transition: The system updates the phase state (σt+1​) based on the Unified Master Equation. If a Reverse Period is detected (RtMASTER​=1), the system forces a Phase Reset (σt+1​=0), returning price to an accumulation state to protect capital.
