def step(self):
        if self.cursor >= len(self.raw_bars):
            return None
        idx = self.cursor
        self.cursor += 1
        W = min(60, idx + 1)
        window = self.raw_bars[max(0, idx - W + 1): idx + 1]
        if len(window) < 3:
            return self._blank(self.raw_bars[idx], idx)
        df = _to_df(window)
        cur = self.raw_bars[idx]
        r = {'bar': cur, 'bar_index': idx, 'total_bars': len(self.raw_bars)}
        r['dealing_range'] = self._dealing_range(df)
        r['bias']          = self._bias(df)
        r['ipda_phase']    = self._ipda(df)
        r['eq_cross']      = self._eq_cross(df)
        r['session']       = self._session(df)
        r['swings']        = self._swings(df)
        r['fvg']           = self._fvg(df)
        r['ob']            = self._ob(df)
        r['vol_profile']   = self._vol_profile(df, cur)
        r['vol_decay']     = self._vol_decay(df)
        r['displacement']  = self._displacement(cur, df)
        r['harmonic']      = self._harmonic(df)
        r['expansion']     = self._expansion(df, r['dealing_range'])
        r['manipulation']  = self._manipulation(df)
        r['kl']            = self._kl(df)
        r['topology']      = self._topology(df)
        r['amd']           = self._amd(r)
        r['fusion']        = self._fusion(r)
        r['mandra']        = self._mandra(r)
        r['veto']          = self._veto(r)
        r['sensors']       = self._sensors(r)

        # ── NEW: Run λ₇ Macro Gate and λ₈ Light-Cone Detector ──────────────
        # Get current direction from bias
        direction = 1 if r['bias']['bias'] == 'BULLISH' else (-1 if r['bias']['bias'] == 'BEARISH' else 0)
        self._lambda7_macro(r, cur, direction)
        self._lambda8_light_cone(r, cur)

        # ── Plugin layer ──────────────────────────────────────────────────────
        try:
            mgr = _get_plugins()
            if mgr:
                # Build a minimal DataFrame for plugins from the current window
                window = self.raw_bars[max(0, idx-59):idx+1]
                import pandas as _pd
                df_plugin = _pd.DataFrame(window)
                if 'time' in df_plugin.columns:
                    df_plugin['datetime'] = _pd.to_datetime(df_plugin['time'], unit='s', utc=True)
                    df_plugin = df_plugin.set_index('datetime')
                for col in ['open','high','low','close','volume']:
                    if col in df_plugin.columns:
                        df_plugin[col] = _pd.to_numeric(df_plugin[col], errors='coerce').fillna(0)
                df_plugin['atr'] = (df_plugin['high']-df_plugin['low']).rolling(14).mean().fillna(
                    (df_plugin['high']-df_plugin['low']).mean())
                df_plugin['atr20'] = (df_plugin['high']-df_plugin['low']).rolling(20).mean().fillna(
                    (df_plugin['high']-df_plugin['low']).mean())

                plugin_results = mgr.run(cur, df_plugin, r)
                r['plugins'] = plugin_results
                # Append plugin sensors to sensor list
                r['sensors'] += mgr.to_sensor_rows(plugin_results)
        except Exception as _pe:
            r['plugins'] = {}

        # ── AEGIS Execution Bridge ────────────────────────────────────────────
        try:
            bridge = _get_bridge()
            if bridge:
                # Feed ATR on every bar regardless of veto
                bridge.update_atr(cur)
                # Only run full evaluation on PROCEED bars
                if r['veto']['decision'] == 'Proceed':
                    exe = bridge.evaluate(r, self.raw_bars[:idx+1])
                    r['execution'] = exe
                else:
                    r['execution'] = {
                        "action": "HALT",
                        "reason": r['veto']['decision'],
                        "is_armed": False,
                        "lot_size": 0.0,
                        "stop_loss_price": 0.0,
                        "take_profit_price": 0.0,
                        "kelly_size": 0.0,
                        "pattern": "",
                        "dominant": "X",
                        "direction": 0,
                        "venue_allocation": [],
                        "risk_profile": "",
                        "risk_pips": 0.0,
                        "rr_ratio": 0.0,
                        "delta_e": 0.0,
                        "rev_score": 0.0,
                    }
        except Exception as _be:
            r['execution'] = {"action": "HALT", "reason": str(_be), "is_armed": False,
                              "lot_size": 0.0, "stop_loss_price": 0.0,
                              "take_profit_price": 0.0, "kelly_size": 0.0,
                              "pattern": "", "dominant": "X", "direction": 0,
                              "venue_allocation": [], "risk_profile": "", "risk_pips": 0.0,
                              "rr_ratio": 0.0, "delta_e": 0.0, "rev_score": 0.0}

        return _sanitize(r)

    def _lambda7_macro(self, r, bar, direction):
        """Run λ₇ Macro Causality Gate"""
        try:
            gate = self.modules.get("lambda7")
            if gate:
                # Need DXY price from somewhere - you'll need to add this to your bars
                dxy_price = bar.get('dxy', 105.0)  # Fetch from data source
                spx_price = bar.get('spx', 4500.0)
                
                telemetry = gate.step(
                    symbol="EURUSD",
                    direction=direction,
                    current_price=bar['close'],
                    dxy_price=dxy_price,
                    spx_price=spx_price
                )
                
                r['lambda_7'] = {
                    'score': telemetry.score,
                    'active': telemetry.active,
                    'status': telemetry.status,
                    'dxy_correlation': telemetry.dxy_correlation,
                    'dxy_veto': telemetry.dxy_veto_triggered,
                    'signal_valid': telemetry.signal_valid,
                    'risk_regime': telemetry.risk_regime
                }
                
                # Add to sensors
                r['sensors'].append({
                    'id': 'λ₇',
                    'name': 'Macro Gate',
                    'score': telemetry.score,
                    'active': telemetry.active,
                    'layer': 'L5-MACRO',
                    'status': telemetry.status
                })
                
                # Apply veto
                if telemetry.dxy_veto_triggered:
                    r['veto']['decision'] = 'Halt'
                    r['veto']['reasons'].append(f"λ₇: {telemetry.veto_reason}")
        except Exception as e:
            print(f"[SMK] λ₇ error: {e}")

    def _lambda8_light_cone(self, r, bar):
        """Run λ₈ Light-Cone Violation Detector"""
        try:
            detector = self.modules.get("lambda8")
            if detector:
                dxy_price = bar.get('dxy', 105.0)
                spx_price = bar.get('spx', 4500.0)
                
                telemetry = detector.step(
                    target_price=bar['close'],
                    dxy_price=dxy_price,
                    spx_price=spx_price
                )
                
                r['lambda_8'] = {
                    'score': telemetry.score,
                    'active': telemetry.active,
                    'status': telemetry.status,
                    'violation_detected': telemetry.violation_detected,
                    'violation_type': telemetry.violation_type,
                    'dxy_z_score': telemetry.dxy_z_score,
                    'target_z_score': telemetry.target_z_score,
                    'kill_switch': telemetry.kill_switch_triggered
                }
                
                r['sensors'].append({
                    'id': 'λ₈',
                    'name': 'Light-Cone',
                    'score': telemetry.score,
                    'active': telemetry.active,
                    'layer': 'L0-ALPHA',
                    'status': telemetry.status
                })
                
                # Kill switch overrides everything
                if telemetry.kill_switch_triggered:
                    r['veto']['decision'] = 'Halt'
                    r['veto']['reasons'].append(f"λ₈: {telemetry.kill_switch_reason}")
                    r['execution']['action'] = 'HALT'
        except Exception as e:
            print(f"[SMK] λ₈ error: {e}")

    def _sensors(self, r):
        vd = r['vol_decay']; ex = r['expansion']; ha = r['harmonic']
        dr = r['dealing_range']; di = r['displacement']; fv = r['fvg']
        ob = r['ob']; kl = r['kl']; tp = r['topology']
        ma = r['mandra']; se = r['session']; mn = r['manipulation']; sw = r['swings']
        return [
            {'id': 's01', 'name': 'PHASE ENTRAP',  'score': vd['ratio'],                     'active': vd['entrapped']},
            {'id': 's02', 'name': 'EXPANSION',      'score': ex['prob'],                      'active': ex['prob'] > 0.5},
            {'id': 's03', 'name': 'HARMONIC L3',    'score': min(1, ha['phase_diff'] / 3.14), 'active': ha['inverted']},
            {'id': 's04', 'name': 'DEAL RANGE',     'score': dr['coherence'],                 'active': True},
            {'id': 's05', 'name': 'PREM/DISC',      'score': 0.9,                             'active': dr['zone'] != 'NEUTRAL'},
            {'id': 's06', 'name': 'DISPLACEMENT',   'score': di['body_ratio'],                'active': di['is_disp']},
            {'id': 's07', 'name': 'FVG DETECT',     'score': min(1, fv['count'] / 5),         'active': fv['active']},
            {'id': 's08', 'name': 'ORDER BLOCK',    'score': min(1, ob['count'] / 5),         'active': ob['active']},
            {'id': 's09', 'name': 'KL DIVERGE',     'score': min(1, kl['score']),             'active': not kl['stable']},
            {'id': 's10', 'name': 'TOPO FRACT',     'score': min(1, tp['h1_score'] / 5),      'active': tp['fractured']},
            {'id': 's11', 'name': 'MANDRA GATE',    'score': 0.9 if ma['open'] else 0.1,      'active': ma['open']},
            {'id': 's12', 'name': 'SESSION L2',     'score': se['score'],                     'active': se['killzone']},
            {'id': 's13', 'name': 'MANIPULATION',   'score': mn['score'] / 100,               'active': mn['active']},
            {'id': 's14', 'name': 'SWING NODES',    'score': min(1, sw['count'] / 10),        'active': sw['count'] > 0},
        ]
