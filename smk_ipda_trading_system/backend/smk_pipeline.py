    def _harmonic(self, df):
        """FIXED: Use proper phase extraction instead of fake prediction signal."""
        m = self.modules.get('harmonic')
        if m and len(df) >= 64:
            try:
                actual = df['close'].values
                # FIXED: Use a simple prediction model (random walk with drift)
                # instead of the nonsensical sin wave
                returns = np.diff(actual) / actual[:-1]
                drift = np.mean(returns[-20:]) if len(returns) >= 20 else 0
                pred = actual.copy()
                pred[-len(returns):] = actual[:-1] * (1 + drift)
                
                t = m.detect_trap(pred, actual)
                return {'phase_diff': float(t.phase_difference), 'inverted': bool(t.is_inverted),
                        'freq': float(t.dominant_frequency), 'trap': str(t.trap_type), 'status': str(t.status)}
            except Exception:
                pass
        
        # FIXED FALLBACK: Proper FFT phase extraction with detrending
        closes = df['close'].values[-min(64, len(df)):]
        # Detrend to remove non-stationary component
        detrended = closes - np.linspace(closes[0], closes[-1], len(closes))
        fft = np.fft.rfft(detrended)
        
        # Find dominant frequency (excluding DC)
        magnitudes = np.abs(fft[1:])
        if len(magnitudes) > 0:
            dom_idx = np.argmax(magnitudes) + 1
            phi = float(np.angle(fft[dom_idx]))
            freq = float(dom_idx / len(closes))
        else:
            phi = 0.0
            freq = 0.0
            
        inv = abs(phi) > np.pi / 2
        return {'phase_diff': round(abs(phi), 3), 'inverted': inv, 'freq': round(freq, 4),
                'trap': 'PHASE_INVERSION' if inv else 'NONE',
                'status': 'DISSONANT' if inv else 'IN_HARMONY'}
