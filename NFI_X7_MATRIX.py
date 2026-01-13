from __future__ import annotations

import json
import logging
import pathlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pandas_ta as pta
import talib.abstract as ta
from pandas import DataFrame, DatetimeIndex, Timestamp

from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IStrategy, IntParameter
import freqtrade.strategy.strategy_helper as strategy_helper
from freqtrade.strategy.strategy_helper import merge_informative_pair as _orig_merge_inf

# Parche NFI Compat: pandas-ta 0.4 -> 0.3 (Bollinger naming)
_orig_bbands = pta.bbands


def _bbands_compat(*args, **kwargs):
    res = _orig_bbands(*args, **kwargs)
    if isinstance(res, pd.DataFrame):
        res.columns = [c.replace("_2.0_2.0", "_2.0") for c in res.columns]
    return res


pta.bbands = _bbands_compat

logger = logging.getLogger(__name__)

# Add agent directory to path for gate_engine
try:
    agent_path = pathlib.Path(__file__).parent.parent.parent / "matrix" / "agent"
    if str(agent_path) not in sys.path:
        sys.path.insert(0, str(agent_path))
    from gate_engine import get_effective_gate_cached

    GATE_ENGINE_AVAILABLE = True
except Exception as e:
    logger.warning(f"Failed to import gate_engine: {e}. Using legacy gate check.")
    GATE_ENGINE_AVAILABLE = False

# ============================================================
#  PARCHE GLOBAL merge_informative_pair (FAIL-OPEN + DATE SAFE)
# ============================================================

def _merge_informative_pair_safe(
    base_df: DataFrame,
    informative_df: DataFrame,
    *args,
    **kwargs,
) -> DataFrame:
    """
    FAIL-OPEN REAL:
    - Si el informative está vacío -> NO rompe el bot
    - Garantiza columna 'date' si es posible
    - Si algo falla -> se salta el merge
    """
    try:
        if informative_df is None or informative_df.empty:
            logger.warning("NFI_X7_AI_Hopt: Informative vacío -> merge omitido (fail-open).")
            return base_df

        # Garantizar 'date' en base_df
        if "date" not in base_df.columns and isinstance(base_df.index, DatetimeIndex):
            base_df = base_df.copy()
            base_df["date"] = base_df.index

        # Garantizar 'date' en informative_df
        if "date" not in informative_df.columns:
            inf = informative_df.copy()
            if isinstance(inf.index, DatetimeIndex):
                inf["date"] = inf.index
            elif "timestamp" in inf.columns:
                inf["date"] = Timestamp.utcfromtimestamp(0) + (
                    inf["timestamp"].astype("int64") / 1000
                ).astype("timedelta64[s]")
            else:
                logger.warning("NFI_X7_AI_Hopt: Informative sin 'date' -> merge omitido.")
                return base_df
            informative_df = inf

        return _orig_merge_inf(base_df, informative_df, *args, **kwargs)

    except Exception as exc:
        logger.error(
            f"NFI_X7_AI_Hopt: merge_informative_pair falló -> omitido. Error: {exc}"
        )
        return base_df


strategy_helper.merge_informative_pair = _merge_informative_pair_safe


def _state_dir() -> Path:
    # Adjust based on file location: user_data/strategies/NFI_X7_MATRIX.py -> ... -> matrix/state
    # relative to THIS file: ../../matrix/state (if in user_data/strategies)
    # BUT we need project root check.
    # Current file: c:\freqtrade_ia\NFI_MATRIX\user_data\strategies\NFI_X7_MATRIX.py
    # Matrix root: c:\freqtrade_ia\NFI_MATRIX\matrix
    return Path(__file__).resolve().parents[2] / "matrix" / "state"


def _load_json_safe(p: Path) -> dict:
    try:
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _is_fresh(ts_iso: str, ttl_min: int) -> bool:
    try:
        ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - ts).total_seconds() / 60
        return age <= ttl_min
    except Exception:
        return False


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _safe_float(x, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


# ============================================================
# ============================================================
#  IMPORT BASE STRATEGY (NostalgiaForInfinityX7) + PARCHE NFI
# ============================================================
try:
    import NostalgiaForInfinityX7 as _nfi_mod  # module

    if hasattr(_nfi_mod, "merge_informative_pair"):
        _nfi_mod.merge_informative_pair = _merge_informative_pair_safe
        logger.info("NFI_MATRIX: merge_informative_pair parcheado en NostalgiaForInfinityX7.")
    from NostalgiaForInfinityX7 import NostalgiaForInfinityX7  # class

except Exception as e:
    logger.error(f"NFI_MATRIX: No se pudo importar NostalgiaForInfinityX7: {e}")
    raise


# ============================================================
#  STRATEGIA
# ============================================================
class NFI_X7_MATRIX(NostalgiaForInfinityX7):
    INTERFACE_VERSION = 3

    startup_candle_count: int = 2000

    # ----------------------------
    # FreqAI base
    # ----------------------------
    use_freqai: bool = True
    freqai_model_path: str = "NFI_MATRIX_v1"
    ai_fail_closed: bool = False

    freqai_target_column: str = "&-s_close"
    freqai_pred_col: str = "&-s_close"
    freqai_prediction_columns = ["&-s_close"]

    # ----------------------------
    # Runmode helpers (single source of truth)
    # ----------------------------
    def _runmode(self) -> str:
        return str(self.config.get("runmode", "")).lower()

    def _is_optimizing(self) -> bool:
        """True in backtest/backtesting/hyperopt."""
        rm = self._runmode()
        return ("backtest" in rm) or ("backtesting" in rm) or ("hyperopt" in rm)

    def _is_trade_live(self) -> bool:
        """True in live + dry-run trade execution."""
        return not self._is_optimizing()

    def bot_start(self, **kwargs) -> None:
        super().bot_start(**kwargs)
        logger.info("MATRIX: bot_start executed successfully. Strategy loaded.")

        # -------------------------------------------------------
        # Sentinel risk_level cache (aplicado SOLO al inicio de vela)
        # -------------------------------------------------------
        # Regla LIVE: risk_level NO cambia a mitad de vela.
        # Se actualiza cuando cambia la última vela (date) del dataframe 5m.
        self._risk_level_cached: int = 1
        self._max_trades_cached: int = 6
        self._risk_level_last_candle_ts: Optional[pd.Timestamp] = None

    # FORCE EXIT AUDIT
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # ----------------------------
    # Core AI thresholds (optimizable)
    # ----------------------------
    ai_entry_threshold = DecimalParameter(0.0, 0.0058, default=0.005, space="buy")

    ai_filter_adx_threshold = IntParameter(20, 30, default=30, space="buy")
    ai_threshold_sideways = DecimalParameter(
        0.01, 0.10, default=0.010, space="buy"
    )

    # ----------------------------
    # Dynamic EMA Threshold Parameters (Hyperopt)
    # ----------------------------
    buy_ema_dist_deep_min = DecimalParameter(
        0.010, 0.050, default=0.030, space="buy", optimize=True, load=True
    )
    buy_ema_dist_shallow_min = DecimalParameter(
        0.001, 0.020, default=0.015, space="buy", optimize=True, load=True
    )

    # ----------------------------
    # ----------------------------
    # Dynamic EMA Thresholds (EMA Agent)
    # ----------------------------
    # Estos umbrales se modulan por ema_thresholds.json (Matrix) via get_pair_ema_thresholds().

    # ----------------------------
    # Explorador (oportunista) - habilita entradas nuevas
    # ----------------------------
    explorer_enabled = False  # Base switch (can be overridden by Explorer Agent)
    explorer_threshold = DecimalParameter(
        0.001, 0.04, default=0.0075, space="buy"
    )
    explorer_stake_mult = DecimalParameter(0.10, 3.0, default=2.0, space="buy")

    # Momentum gatillo (rápido)
    explorer_roc_min = DecimalParameter(0.001, 0.005, default=0.0020, space="buy")
    explorer_rsi_min = IntParameter(10, 60, default=38, space="buy")

    # Anti-FOMO (Techos)
    explorer_roc_max = DecimalParameter(0.03, 0.20, default=0.120, space="buy")
    explorer_rsi_max = IntParameter(70, 99, default=98, space="buy")

    # Explorer Agent state file path
    explorer_state_path = "/freqtrade/matrix/state/explorer_state.json"
    sentinel_config_path = "/freqtrade/matrix/state/sentinel_config.json"

    # ------------------------------------------------------------------
    # VERSION + EMA THRESHOLDS (Fusion desde NFI_X7_MATRIX)
    # ------------------------------------------------------------------
    def version(self) -> str:
        try:
            return f"{super().version()}_MATRIX_v5_FUSED"
        except Exception:
            return "NFI_MATRIX_v5_FUSED"

    def get_pair_ema_thresholds(self, pair: str) -> tuple[float, float]:
        """
        Calculates dynamic EMA thresholds for a pair based on EMA Agent state.
        Returns: (deep_thr, shallow_thr)

        Fuente: matrix/state/ema_thresholds.json
        - Si el fichero no existe / está stale -> usa los valores base (params hyperopt).
        """
        base_deep = float(self.buy_ema_dist_deep_min.value)
        base_shallow = float(self.buy_ema_dist_shallow_min.value)

        cfg = _load_json_safe(_state_dir() / "ema_thresholds.json")
        ts = str(cfg.get("ts", ""))
        ttl = int(cfg.get("ttl_min", 0) or 0)

        if not ts or ttl <= 0 or not _is_fresh(ts, ttl):
            return base_deep, base_shallow

        data = cfg.get("data", {}) or {}
        per_pair = data.get("pairs", {}) or {}
        g = data.get("global", {}) or {}

        p = per_pair.get(pair, {}) or {}
        deep_mult = _safe_float(p.get("deep_mult"), _safe_float(g.get("deep_mult"), 1.0))
        shallow_mult = _safe_float(
            p.get("shallow_mult"), _safe_float(g.get("shallow_mult"), 1.0)
        )

        # Optional range guards
        rmin = _safe_float(g.get("mult_min"), 0.5)
        rmax = _safe_float(g.get("mult_max"), 2.0)

        deep_mult = _clamp(deep_mult, rmin, rmax)
        shallow_mult = _clamp(shallow_mult, rmin, rmax)

        deep_thr = base_deep * deep_mult
        shallow_thr = base_shallow * shallow_mult

        # Airbags (Absolute limits)
        deep_thr = _clamp(deep_thr, 0.004, 0.080)
        shallow_thr = _clamp(shallow_thr, 0.002, 0.050)

        # Coherence check
        if deep_thr < shallow_thr:
            deep_thr, shallow_thr = shallow_thr, deep_thr

        return deep_thr, shallow_thr

    def _load_explorer_state(self) -> dict:
        """
        Load the Explorer Agent's dynamic decisions from explorer_state.json.
        Returns the state dict or defaults if file doesn't exist.
        """
        try:
            p = pathlib.Path(self.explorer_state_path)
            if p.exists():
                with open(p, "r") as f:
                    state = json.load(f)
                    # Check freshness (max 30 min old)
                    ts_str = state.get("ts", "")
                    if ts_str:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        age_min = (datetime.now(timezone.utc) - ts).total_seconds() / 60
                        if age_min > 30:
                            logger.warning("ExplorerAgent state is stale. Using defaults.")
                            return {}
                    return state
        except Exception as e:
            logger.warning(f"Failed to load explorer_state.json: {e}")
        return {}

    # ==========================================================
    # SENTINEL: risk_level 0-3 (FUENTE DE VERDAD: sentinel_config.json)
    # ==========================================================
    def _load_sentinel_config(self) -> dict:
        """
        Lee `sentinel_config.json` (contrato v1.0) desde `matrix/state/`.

        Esperado (según diseño acordado):
        {
          "schema_version": "1.0",
          "ttl_seconds": 86400,
          "expires_at_utc": "2025-12-27T22:00:00Z",
          "data": { "risk_level": 0-3, ... }
        }

        Política LIVE:
        - Si falta / está corrupto / expirado -> devolver {} (y se usará fallback seguro).
        """
        try:
            p = pathlib.Path(self.sentinel_config_path)
            if not p.exists():
                return {}
            raw = json.loads(p.read_text())
            data = raw.get("data", raw)

            # Expiración por expires_at_utc (si existe)
            exp = raw.get("expires_at_utc")
            if exp:
                try:
                    exp_dt = datetime.fromisoformat(exp.replace("Z", "+00:00"))
                    if datetime.now(timezone.utc) > exp_dt:
                        logger.warning("Sentinel config expirado -> ignorado (fallback seguro).")
                        return {}
                except Exception:
                    # Si no se puede parsear, preferimos ignorar el contrato (fail-closed a nivel de config)
                    logger.warning(
                        "Sentinel expires_at_utc inválido -> ignorado (fallback seguro)."
                    )
                    return {}

            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning(f"Fallo leyendo sentinel_config.json: {e}")
            return {}

    def _get_risk_level_for_candle(self, candle_ts: pd.Timestamp) -> int:
        """
        Devuelve el risk_level aplicable a ESTA vela.

        Requisito acordado:
        - Se evalúa SOLO al inicio de cada vela (5m).
        - Si el usuario cambia risk_level en Sentinel, se aplica en la siguiente vela.
        """
        try:
            # Si no hay timestamp, usar cache actual
            if candle_ts is None:
                return int(getattr(self, "_risk_level_cached", 1))

            last_ts = getattr(self, "_risk_level_last_candle_ts", None)
            if last_ts is not None and candle_ts == last_ts:
                return int(getattr(self, "_risk_level_cached", 1))

            # Nueva vela -> leer Sentinel
            cfg = self._load_sentinel_config()
            rl = int(cfg.get("risk_level", 1))
            if rl not in (0, 1, 2, 3):
                rl = 1

            self._risk_level_cached = rl
            self._max_trades_cached = int(cfg.get("max_concurrent_trades", 6))
            self._risk_level_last_candle_ts = candle_ts
            return rl
        except Exception:
            return int(getattr(self, "_risk_level_cached", 1))

    # AI Smart Exit
    ai_secure_profit = DecimalParameter(0.005, 0.02, default=0.006, space="sell")
    ai_secure_score = DecimalParameter(
        0.0, 0.5, default=0.15, space="sell"
    )  # v5: 0.2 -> 0.15

    # ----------------------------
    # Shock BTC+ETH (15m, 2 velas, adaptativo)
    # ----------------------------
    shock_tf = "15m"
    shock_confirm_candles = 2

    # Ventana adaptativa (15m): 192 ~ 2 días. Ajustable si quieres.
    shock_adapt_window = 192
    shock_q_med = 0.90
    shock_q_high = 0.97

    # Gate operativo: ambos >= MED
    # HIGH existe desde día 1 SOLO telemetría (action_taken:false)
    shock_action_on_high = False

    # Carpeta eventos (dentro contenedor)
    events_dir = "/freqtrade/user_data/events/btc_eth_shock"

    # ----------------------------------------------------------
    # Informative pairs
    # ----------------------------------------------------------
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        inf = []
        for p in pairs:
            inf.append((p, "15m"))
            inf.append((p, "1h"))
            inf.append((p, "4h"))
            inf.append((p, "1d"))

        # BTC completo
        inf.append(("BTC/USDC", "15m"))
        inf.append(("BTC/USDC", "1h"))
        inf.append(("BTC/USDC", "4h"))
        inf.append(("BTC/USDC", "1d"))

        # ETH para shock gate
        inf.append(("ETH/USDC", "15m"))

        return inf

    # ----------------------------------------------------------
    # Target IA
    # ----------------------------------------------------------
    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        lp = int(self.config["freqai"]["feature_parameters"].get("label_period_candles", 12))
        label = (
            dataframe["high"].shift(-lp).rolling(lp).max() / dataframe["close"] - 1.0
        )
        dataframe["&-s_close"] = label
        return dataframe

    # ----------------------------------------------------------
    # Feature engineering
    # ----------------------------------------------------------
    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int = None, metadata: Dict = None, **kwargs
    ) -> DataFrame:
        if "date" not in dataframe.columns and isinstance(dataframe.index, DatetimeIndex):
            dataframe["date"] = dataframe.index

        # Base
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["ema_25"] = ta.EMA(dataframe, timeperiod=25)

        # Momentum rápido (para explorador y para que el modelo tenga dinámica)
        dataframe["roc_1"] = ta.ROC(dataframe, timeperiod=1)
        dataframe["roc_3"] = ta.ROC(dataframe, timeperiod=3)

        # Bollinger
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upperband"] = bb["upperband"]
        dataframe["bb_middleband"] = bb["middleband"]
        dataframe["bb_lowerband"] = bb["lowerband"]

        # Sanitización sin fillna(0) masivo
        dataframe.replace([np.inf, -np.inf], 0, inplace=True)

        # Prefijos FreqAI
        dataframe["%-rsi"] = dataframe["rsi"]
        dataframe["%-mfi"] = dataframe["mfi"]
        dataframe["%-adx"] = dataframe["adx"]
        dataframe["%-ema_25"] = dataframe["ema_25"]
        dataframe["%-roc_1"] = dataframe["roc_1"]
        dataframe["%-roc_3"] = dataframe["roc_3"]
        dataframe["%-bb_upperband"] = dataframe["bb_upperband"]
        dataframe["%-bb_middleband"] = dataframe["bb_middleband"]
        dataframe["%-bb_lowerband"] = dataframe["bb_lowerband"]

        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, period: int = None, metadata: Dict = None, **kwargs
    ) -> DataFrame:
        return self.feature_engineering_expand_all(dataframe, period, metadata, **kwargs)

    # ============================================================
    #  SHOCK ENGINE (BTC+ETH 15m)
    # ============================================================
    def _utc_now_iso(self) -> str:
        return (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )

    def _ensure_dir(self, p: str) -> None:
        pathlib.Path(p).mkdir(parents=True, exist_ok=True)

    def _write_event_json(self, event: Dict[str, Any]) -> None:
        try:
            self._ensure_dir(self.events_dir)
            ts = event.get("timestamp", self._utc_now_iso()).replace(":", "-")
            name = f"{ts}_{event.get('event', 'EVENT')}.json"
            path = pathlib.Path(self.events_dir) / name
            path.write_text(json.dumps(event, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Event JSON write failed (fail-open): %s", exc)

    def _calc_shock_level_15m(self, df: DataFrame, prefix: str) -> DataFrame:
        out = df.copy()

        # Garantizar date
        if "date" not in out.columns and isinstance(out.index, DatetimeIndex):
            out["date"] = out.index

        # Velocidad (abs retorno)
        out[f"{prefix}_vel"] = (out["close"] / out["close"].shift(1) - 1.0).abs()

        # ATR ratio adaptativo
        atr = ta.ATR(out, timeperiod=14)
        base = atr.rolling(self.shock_adapt_window, min_periods=50).mean()
        out[f"{prefix}_atr_ratio"] = (atr / base).replace([np.inf, -np.inf], np.nan)

        # Umbrales adaptativos por cuantiles
        vel_med = (
            out[f"{prefix}_vel"]
            .rolling(self.shock_adapt_window, min_periods=50)
            .quantile(self.shock_q_med)
        )
        vel_high = (
            out[f"{prefix}_vel"]
            .rolling(self.shock_adapt_window, min_periods=50)
            .quantile(self.shock_q_high)
        )

        ar_med = (
            out[f"{prefix}_atr_ratio"]
            .rolling(self.shock_adapt_window, min_periods=50)
            .quantile(self.shock_q_med)
        )
        ar_high = (
            out[f"{prefix}_atr_ratio"]
            .rolling(self.shock_adapt_window, min_periods=50)
            .quantile(self.shock_q_high)
        )

        out[f"{prefix}_shock"] = 0  # 0 low, 1 med, 2 high
        out.loc[
            (out[f"{prefix}_vel"] > vel_med)
            & (out[f"{prefix}_atr_ratio"] > ar_med),
            f"{prefix}_shock",
        ] = 1
        out.loc[
            (out[f"{prefix}_vel"] > vel_high)
            & (out[f"{prefix}_atr_ratio"] > ar_high),
            f"{prefix}_shock",
        ] = 2

        # Confirmación 2 velas seguidas MED+
        out[f"{prefix}_shock_2c"] = (
            (out[f"{prefix}_shock"] >= 1) & (out[f"{prefix}_shock"].shift(1) >= 1)
        ).astype("int8")
        out[f"{prefix}_shock_high_2c"] = (
            (out[f"{prefix}_shock"] >= 2) & (out[f"{prefix}_shock"].shift(1) >= 2)
        ).astype("int8")

        return out

    def _update_market_state_from_btc_eth(self) -> Dict[str, Any]:
        """
        Calcula estado mercado global (NORMAL / SHOCK_MED / SHOCK_HIGH) usando BTC+ETH 15m.
        Gate: ambos en shock MED (confirmado 2 velas).
        """
        # Estado persistente
        if not hasattr(self, "custom_info") or self.custom_info is None:
            self.custom_info = {}
        st = self.custom_info.setdefault(
            "market_state",
            {
                "shock_on": False,
                "level": "NORMAL",
                "since": None,
            },
        )

        try:
            btc15 = self.dp.get_pair_dataframe(pair="BTC/USDC", timeframe=self.shock_tf)
            eth15 = self.dp.get_pair_dataframe(pair="ETH/USDC", timeframe=self.shock_tf)

            if btc15 is None or btc15.empty or eth15 is None or eth15.empty:
                return st

            btc = self._calc_shock_level_15m(btc15, "btc").iloc[-1]
            eth = self._calc_shock_level_15m(eth15, "eth").iloc[-1]

            btc_med2 = int(btc.get("btc_shock_2c", 0)) == 1
            eth_med2 = int(eth.get("eth_shock_2c", 0)) == 1
            btc_high2 = int(btc.get("btc_shock_high_2c", 0)) == 1
            eth_high2 = int(eth.get("eth_shock_high_2c", 0)) == 1

            # Activación: ambos MED2
            shock_on = btc_med2 and eth_med2
            level = "NORMAL"
            if shock_on:
                level = "HIGH" if (btc_high2 and eth_high2) else "MED"

            # Transiciones ON/OFF con logging por evento
            if shock_on and not st["shock_on"]:
                st["shock_on"] = True
                st["level"] = level
                st["since"] = self._utc_now_iso()

                evt = {
                    "event": "BTC_ETH_SHOCK_ON",
                    "timestamp": st["since"],
                    "timeframe": self.shock_tf,
                    "confirmation_candles": self.shock_confirm_candles,
                    "level": level,
                    "action_taken": (level == "MED")
                    or (level == "HIGH" and self.shock_action_on_high),
                    "reason": "telemetry_only"
                    if (level == "HIGH" and not self.shock_action_on_high)
                    else "operational_gate",
                    "btc": {
                        "shock": int(btc.get("btc_shock", 0)),
                        "vel": float(btc.get("btc_vel", np.nan))
                        if not np.isnan(btc.get("btc_vel", np.nan))
                        else None,
                        "atr_ratio": float(btc.get("btc_atr_ratio", np.nan))
                        if not np.isnan(btc.get("btc_atr_ratio", np.nan))
                        else None,
                    },
                    "eth": {
                        "shock": int(eth.get("eth_shock", 0)),
                        "vel": float(eth.get("eth_vel", np.nan))
                        if not np.isnan(eth.get("eth_vel", np.nan))
                        else None,
                        "atr_ratio": float(eth.get("eth_atr_ratio", np.nan))
                        if not np.isnan(eth.get("eth_atr_ratio", np.nan))
                        else None,
                    },
                }
                self._write_event_json(evt)

            elif (not shock_on) and st["shock_on"]:
                off_ts = self._utc_now_iso()
                # duración aproximada
                dur_min = None
                try:
                    t0 = datetime.fromisoformat(st["since"].replace("Z", "+00:00"))
                    t1 = datetime.fromisoformat(off_ts.replace("Z", "+00:00"))
                    dur_min = int((t1 - t0).total_seconds() / 60)
                except Exception:
                    pass

                evt = {
                    "event": "BTC_ETH_SHOCK_OFF",
                    "timestamp": off_ts,
                    "duration_minutes": dur_min,
                    "prev_level": st.get("level", "UNKNOWN"),
                }
                self._write_event_json(evt)

                st["shock_on"] = False
                st["level"] = "NORMAL"
                st["since"] = None

            else:
                # Si sigue ON, actualizamos level si cambia (sin evento para no spamear)
                if st["shock_on"]:
                    st["level"] = level

            return st

        except Exception as exc:
            logger.warning("Shock state update failed (fail-open): %s", exc)
            return st

    # ----------------------------------------------------------
    # INDICATORS
    # ----------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if "date" not in dataframe.columns and isinstance(dataframe.index, DatetimeIndex):
            dataframe = dataframe.copy()
            dataframe["date"] = dataframe.index

        # NFI base
        df = super().populate_indicators(dataframe, metadata)
        # MATRIX SAFETY: Ensure 'date' exists after super()
        if "date" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df["date"] = df.index
            elif "timestamp" in df.columns:
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # TA Column Mapper (pandas-ta 0.4 -> 0.3 compatibility)
        for col in df.columns:
            if "_2.0_2.0" in col:
                legacy_name = col.replace("_2.0_2.0", "_2.0")
                if legacy_name not in df.columns:
                    df[legacy_name] = df[col]

        # Asegurar indicadores rápidos para explorador a nivel trading df
        try:
            df["roc_3"] = ta.ROC(df, timeperiod=3)
            df["rsi_7"] = ta.RSI(df, timeperiod=7)
        except Exception:
            pass

        # ----------------------------------------------------------
        # CORE TREND FILTER 15m (Solo para Core, no Explorer)
        # ----------------------------------------------------------
        df["core_trend15_ok"] = False  # Default: block
        try:
            pair = metadata.get("pair")
            inf_15m = self.dp.get_pair_dataframe(pair=pair, timeframe="15m")

            if inf_15m is not None and len(inf_15m) > 210:
                # EMAs en 15m
                inf_15m["ema50_15m"] = ta.EMA(inf_15m, timeperiod=50)
                inf_15m["ema200_15m"] = ta.EMA(inf_15m, timeperiod=200)
                inf_15m["ema200_15m_prev"] = inf_15m["ema200_15m"].shift(1)

                # Condición BULL 15m (solo velas cerradas -> shift(1))
                inf_15m["trend_bull_15m"] = (
                    (inf_15m["close"] > inf_15m["ema200_15m"])
                    & (inf_15m["ema50_15m"] > inf_15m["ema200_15m"])
                    & (inf_15m["ema200_15m"] > inf_15m["ema200_15m_prev"])
                )

                # Histeresis: 2 cierres 15m consecutivos
                inf_15m["trend_15m_confirmed"] = (
                    inf_15m["trend_bull_15m"].shift(1)
                    & inf_15m["trend_bull_15m"].shift(2)
                ).fillna(False)

                # Guardar valores para log
                inf_15m["close_15m"] = inf_15m["close"].shift(1)  # Solo vela cerrada

                # Merge al 5m (forward fill desde última señal 15m)
                inf_15m = inf_15m[
                    [
                        "date",
                        "trend_15m_confirmed",
                        "close_15m",
                        "ema50_15m",
                        "ema200_15m",
                    ]
                ].copy()
                inf_15m = inf_15m.rename(columns={"date": "date_15m"})

                # Merge por tiempo más cercano (5m recibe el estado del 15m)
                # MATRIX SAFETY: Final date check before merge
                if "date" not in df.columns:
                    if isinstance(df.index, pd.DatetimeIndex):
                        df["date"] = df.index

                df = pd.merge_asof(
                    df.sort_values("date"),
                    inf_15m.sort_values("date_15m"),
                    left_on="date",
                    right_on="date_15m",
                    direction="backward",
                )

                df["core_trend15_ok"] = df["trend_15m_confirmed"].fillna(False)
            else:
                logger.warning(
                    f"15m data insufficient for {pair}, defaulting core_trend15_ok=False"
                )

        except Exception as e:
            logger.warning(
                f"CoreTrendFilter15m failed for {metadata.get('pair')}: {e}. Defaulting to False."
            )
            df["core_trend15_ok"] = False

        # FreqAI start
        if self.config.get("freqai", {}).get("enabled", False) and getattr(
            self, "freqai", None
        ):
            try:
                # MATRIX SAFETY: Ensure 'date' exists before FreqAI start
                if "date" not in df.columns:
                    if isinstance(df.index, pd.DatetimeIndex):
                        df["date"] = df.index
                    elif "timestamp" in df.columns:
                        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

                df = self.freqai.start(df, metadata, self)

            except Exception as exc:
                logger.warning(
                    "FreqAI error en %s -> skip (fail-open): %s", metadata.get("pair"), exc
                )

        # Actualizar estado global mercado (BTC+ETH shock)
        st = self._update_market_state_from_btc_eth()
        shock_on = bool(st.get("shock_on", False))
        level = st.get("level", "NORMAL")

        df["market_shock"] = 1 if shock_on else 0
        df["market_shock_level"] = 0
        if level == "MED":
            df["market_shock_level"] = 1
        elif level == "HIGH":
            df["market_shock_level"] = 2

        # ---------------------------
        # Filtro IA Core (ai_long_ok)
        # ---------------------------
        # Fail-closed: initialized as False, must be proven True by AI logic
        cond_ok = pd.Series(False, index=df.index)

        # -------------------------------------------------------
        # risk_level (0-3) desde Sentinel - SOLO inicio de vela
        # -------------------------------------------------------
        # Se cachea por vela (5m) para evitar cambios intra-vela en LIVE.
        try:
            candle_ts = df["date"].iloc[-1] if "date" in df.columns else None
        except Exception:
            candle_ts = None

        risk_level = self._get_risk_level_for_candle(candle_ts)
        df["risk_level"] = risk_level
        if metadata["pair"] in ["BTC/USDC", "ETH/USDC"]:
            logger.info(f" SENTINEL RISK: Pair={metadata['pair']} | level={risk_level}")

        # --- GATE RELAXATION DETECTION (v5.0) ---
        gate_relaxed = False
        regime_found = "UNKNOWN"
        if GATE_ENGINE_AVAILABLE:
            try:
                state_dir = pathlib.Path(__file__).parent.parent.parent / "matrix" / "state"
                egate = get_effective_gate_cached(state_dir)
                edata = egate.get("data", egate)
                regime_found = edata.get("regime", "N/A")
                allow_entries = edata.get("allow_entries", False)
                if regime_found == "MANUAL_OVERRIDE" and allow_entries:
                    gate_relaxed = True
            except Exception as e:
                logger.warning(f"Gate Relax Logic Error: {e}")

        # --- DYNAMIC EMA THRESHOLDS (Per-Pair) ---
        # "Aplicar umbrales de entrada basados en dist_below_ema por par a toda la estrategia MATRIX"
        try:
            deep_thr, shallow_thr = self.get_pair_ema_thresholds(metadata["pair"])
        except Exception:
            # Fallback if method missing (should not happen with NFI_X7_MATRIX updated)
            deep_thr = float(self.buy_ema_dist_deep_min.value)
            shallow_thr = float(self.buy_ema_dist_shallow_min.value)

        # Calculate dist_below_ema (using EMA_200 from base strategy if available)
        # NFI base puts EMA_200 in dataframe.
        dist_col = "dist_below_ema_gen"
        if "EMA_200" in df.columns:
            df[dist_col] = (df["EMA_200"] - df["close"]) / df["close"]
        else:
            # Fallback (should typically have EMA_200)
            df[dist_col] = 0.0

        # Apply Threshold to Core (Buy-The-Dip) modulado por risk_level
        # ------------------------------------------------------------
        # - RL 0-1: CORE exige DIP profundo (deep_thr).
        # - RL 2-3: CORE permite MOMENTUM (precio por encima de EMA).
        # - Seguridad: Limitamos el momentum para no entrar en burbujas (1-2% max).
        rl = int(risk_level)
        if rl == 0:
            # Strict: require a deep dip below EMA200 (classic buy-the-dip)
            required_dip = deep_thr
        elif rl == 1:
            # RL1: High Activity (-0.5% momentum).
            # Allow entries up to 0.5% above EMA to ensure signal flow.
            required_dip = -0.0050
        elif rl == 2:
            # RL2: MÁXIMA PERMISIVIDAD - Requiere DIP del 0.15% BAJO EMA200
            # Objetivo: Capturar virtualmente cualquier caída o apoyo cerca de EMA200
            required_dip = 0.0015  # Mínimo 0.15% por debajo de EMA200
        else:
            # RL3+: aggressive: allow up to ~2% above EMA200
            required_dip = -0.0200

        if gate_relaxed and required_dip > shallow_thr:
            required_dip = shallow_thr

        # Monitor critical pairs
        if metadata["pair"] in ["BTC/USDC", "ETH/USDC", "SOL/USDC"] or gate_relaxed:
            logger.info(
                f" GATE MONITOR [{metadata['pair']}]: regime={regime_found}, relaxed={gate_relaxed}, engine={GATE_ENGINE_AVAILABLE}"
            )

        # Initialize cond_ok as False (default)
        base_thr = float(self.ai_entry_threshold.value)
        sideways_thr = float(self.ai_threshold_sideways.value)
        cond_ok = df["close"] == 0  # All False

        if self.freqai_pred_col in df.columns:
            df["ai_long_score"] = df[self.freqai_pred_col].astype("float64")

            # ------------------------------------------------------------
            # required_score (umbral IA) modulado por risk_level (0-3)
            # ------------------------------------------------------------
            base_thr = float(self.ai_entry_threshold.value)
            sideways_thr = float(self.ai_threshold_sideways.value)

            if int(risk_level) <= 1:
                # RL0-1: standard high threshold
                df["required_score"] = max(base_thr, sideways_thr)
            elif int(risk_level) == 2:
                # RL2: SELECTIVE RESTORE - Stricter than RL1
                df["required_score"] = max(base_thr, sideways_thr) * 1.10
            else:
                # RL3+: Most aggressive
                df["required_score"] = base_thr * 0.25

            mask_ai = df[self.freqai_pred_col].fillna(-1.0) > df["required_score"]
            if "do_predict" in df.columns:
                # Visibilidad total: permitir señales incluso si do_predict es 0 para el gráfico
                mask_ai &= (df["do_predict"] == 1) | (df["do_predict"] == 0)

            # Combine technical (dip) with AI score
            cond_ok = (df[dist_col] >= required_dip) & mask_ai
        else:
            df["ai_long_score"] = np.nan
            df["required_score"] = base_thr if hasattr(self, "ai_entry_threshold") else 0.0

        df["ai_long_ok"] = cond_ok.astype("int8")
        df["ai_short_ok"] = 0
        df["ai_short_score"] = 0.0

        # DEBUG: Log logic scores (ALWAYS for diagnostics)
        if ("date" in df.columns or isinstance(df.index, DatetimeIndex)):
            score_val = (
                df[self.freqai_pred_col].iloc[-1]
                if self.freqai_pred_col in df.columns
                else np.nan
            )
            do_pred = df["do_predict"].iloc[-1] if "do_predict" in df.columns else "N/A"
            final_ok = df["ai_long_ok"].iloc[-1]
            dist_val = df[dist_col].iloc[-1]
            req_dip = required_dip

            # Detailed log for diagnostics
            dip_ok = "OK" if dist_val >= req_dip else "WAIT"
            ai_ok = "OK" if score_val > df["required_score"].iloc[-1] else "LOW"

            logger.info(
                f"[MATRIX DIAG] [{metadata['pair']}]: "
                f"Result={'APPROVED' if final_ok else 'REJECTED'} | "
                f"AI={score_val:.4f}/{df['required_score'].iloc[-1]:.4f} ({ai_ok}) | "
                f"Dip={dist_val:.4f}/{req_dip:.4f} ({dip_ok}) | "
                f"do_pred={do_pred}"
            )

        # ---------------------------
        # Explorador: habilitar entradas nuevas
        # Ahora controlado por Explorer Agent (autónomo)
        # ---------------------------
        df["ai_explorer_ok"] = 0

        # 1. Load Explorer Agent's decision
        explorer_state = self._load_explorer_state()
        agent_enabled = explorer_state.get("explorer_enabled", True)  # Default to enabled

        # Store agent state in custom_info for use in custom_exit and custom_stake_amount
        if not hasattr(self, "custom_info") or self.custom_info is None:
            self.custom_info = {}
        self.custom_info["explorer_agent"] = explorer_state

        if not agent_enabled:
            logger.info(
                "Explorer PAUSED by Explorer Agent. Reason: "
                + explorer_state.get("reason", "Unknown")
            )

        # 2. Base conditions
        if (
            self.explorer_enabled
            and agent_enabled
            and ("do_predict" in df.columns)
            and (self.freqai_pred_col in df.columns)
        ):
            # No BTC dependency - Explorer is INDEPENDENT
            # Maverick Mode: IGNORE BTC SHOCK. Hunt whenever prey is found.
            explorer_allowed = True

            # --- HUNTER LOGIC: Check Opportunity Queue ---
            # v4: EXPLORER LIBRE MODE - Can enter if:
            #   1. Pair is in Hunter's queue (priority), OR
            #   2. AI score is strong enough (free exploration)
            in_queue = False
            queue_data = explorer_state.get("opportunity_queue", {})
            candidates = queue_data.get("candidates", [])

            if candidates:
                current_pair = metadata.get("pair")
                for cand in candidates:
                    if cand["pair"] == current_pair:
                        in_queue = True
                        break

            # PROD: SELECTIVO - Umbrales IA conservadores
            score_val = df[self.freqai_pred_col].fillna(-1.0)

            # Use dynamic threshold from agent if available, else fallback to hyperopt param
            threshold_val = float(
                explorer_state.get("explorer_threshold", self.explorer_threshold.value)
            )

            # SCALPING FILTERS: Tighten for better quality
            # 1. AI Score High Confidence or consistent (v6.0)
            score_ok = score_val > threshold_val

            # 2. ROC check: MAXIMUM DROP ACTION (Entries "en las caídas")
            # Ultra-Relaxed to 0.10 to maximize visibility
            current_roc_max = 0.10
            roc_val = df.get("roc_3", 0).fillna(0)
            roc_ok = (roc_val < current_roc_max) & (roc_val > -0.15)

            # 3. RSI check: Maximum Flow (Ceiling 80)
            current_rsi_max = 80
            rsi_val = df.get("rsi_7", 0).fillna(0)
            rsi_ok = (rsi_val > 30) & (rsi_val < current_rsi_max)

            # 4. BREAKOUT CONFIRMATION (Anti-Rango)
            # Close must be > max high of last 30 min (6 candles)
            rolling_high = df["high"].rolling(6).max().shift(1).fillna(0)
            breakout_ok = df["close"] > rolling_high

            # 5. PRESSURE CANDLE (Body > 0.10%)
            body_pct = ((df["close"] - df["open"]) / df["close"]).abs()
            candle_ok = (df["close"] > df["open"]) & (body_pct >= 0.0010)

            # 6. VOLUME SPIKE (1.0x Avg - No requirement)
            vol_avg = df["volume"].rolling(24).mean().shift(1).fillna(0)
            vol_ok = df["volume"] >= (vol_avg * 1.0)

            # EXPLORER SCALPING: Combine filters
            df["ai_explorer_ok"] = (
                (df["do_predict"] == 1)
                & explorer_allowed
                & score_ok
                & roc_ok
                & rsi_ok
                & breakout_ok
                & candle_ok
                & vol_ok
            ).astype("int8")

            if df["ai_explorer_ok"].iloc[-1] == 1:
                logger.info(
                    f" EXPLORER SIGNAL [{metadata['pair']}]: score={score_val.iloc[-1]:.4f}, breakout=OK, vol_spike=OK"
                )

        return df

    # ----------------------------------------------------------
    # ENTRY TREND
    # ----------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        MATRIX Entry Logic v5.0 - Full Autonomy

        NO llama a super().populate_entry_trend().
        Las entradas son 100% controladas por MATRIX:
        - matrix_core: Señales FreqAI con ai_long_ok=1
        - ai_explorer: Señales Explorer con ai_explorer_ok=1

        El padre (NFI) solo aporta indicadores, NO decide entradas.
        """
        df = dataframe.copy()

        # Inicializar columnas de protección (fail-open para compatibilidad con salidas del padre)
        global_protection_cols = [
            "protections_long_global",
            "protections_short_global",
            "global_protections_long_pump",
            "global_protections_long_dump",
            "global_protections_short_pump",
            "global_protections_short_dump",
        ]
        for col in global_protection_cols:
            if col not in df.columns:
                df[col] = True

        # Inicializar columnas de entrada
        df["enter_long"] = 0
        df["enter_short"] = 0
        df["enter_tag"] = ""

        # ========================================
        # CORE ENTRIES (FreqAI)
        # ========================================
        if "ai_long_ok" in df.columns:
            # Seleccionamos solo velas donde ai_long_ok sea 1
            mask_core = df["ai_long_ok"] == 1

            df.loc[mask_core, "enter_long"] = 1
            df.loc[mask_core, "enter_tag"] = "matrix_core"

            if mask_core.any():
                logger.info(
                    f"  MATRIX CORE [{metadata.get('pair')}]: {mask_core.sum()} entry signals"
                )
                if df["enter_long"].iloc[-1] == 1:
                    logger.info(
                        f" SIGNAL ACTIVE on last candle for {metadata.get('pair')}"
                    )

        # ========================================
        # EXPLORER ENTRIES (Override si no hay Core)
        # ========================================
        if self.explorer_enabled and "ai_explorer_ok" in df.columns:
            shock_col = df.get("market_shock", 0)
            if isinstance(shock_col, int):
                shock_col = pd.Series([shock_col] * len(df), index=df.index)

            mask_explorer = (
                (df["ai_explorer_ok"] == 1)
                & (shock_col == 0)
                & (df["enter_long"] != 1)
            )

            df.loc[mask_explorer, "enter_long"] = 1
            df.loc[mask_explorer, "enter_tag"] = "ai_explorer"

            if mask_explorer.any() and df["enter_long"].iloc[-1] == 1:
                logger.info(
                    f" SIGNAL ACTIVE (Explorer) on last candle for {metadata.get('pair')}"
                )

        return df

    # ----------------------------------------------------------
    # SALIDAS - DELEGADAS 100% A NFI (v6.0)
    # ----------------------------------------------------------
    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        """Delegado 100% a NFI."""
        return super().confirm_trade_exit(
            pair,
            trade,
            order_type,
            amount,
            rate,
            time_in_force,
            exit_reason,
            current_time,
            **kwargs,
        )

    # ----------------------------------------------------------
    # QUANTITY LIMITS - MATRIX Entry Gate v5.0
    # ----------------------------------------------------------
    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str,
        side: str,
        **kwargs,
    ) -> bool:
        """
        MATRIX Entry Gate v5.0 - Bypass Quirúrgico

        Para tags MATRIX (matrix_core, ai_explorer):
        - Aplica checks propios (gate, límites, DI, PROBE)
        - NO llama a super() - El padre NO puede bloquear

        Para tags legacy:
        - Delega a super() (compatibilidad futura)
        """

        # ========================================
        # 1. HARD GATE CHECK (Siempre primero)
        # ========================================
        gate_open = False
        gate_relaxed = False

        if GATE_ENGINE_AVAILABLE:
            try:
                state_dir = pathlib.Path(__file__).parent.parent.parent / "matrix" / "state"
                effective_gate = get_effective_gate_cached(state_dir)
                edata = effective_gate.get("data", effective_gate)
                gate_open = edata.get("allow_entries", False)
                gate_relaxed = edata.get("regime") == "MANUAL_OVERRIDE" and gate_open
            except Exception as e:
                logger.warning(f"Gate read failed: {e}. Defaulting to CLOSED.")
                gate_open = False
        else:
            # Si no hay gate engine, BLOQUEAMOS (fail-closed para LIVE)
            gate_open = False

        gate_relaxed_confirm = gate_relaxed

        # ========================================
        # 2. MATRIX TAGS - Bypass Quirúrgico
        # ========================================
        is_matrix_entry = entry_tag in ["matrix_core", "ai_explorer", "ai_hunter"]

        if is_matrix_entry:
            # --- GATE ENFORCEMENT ---
            if not gate_open:
                # Bypass during backtesting/optimization to allow Ollama to gather data
                # Backtest/Hyperopt: accept to gather data
                if self._is_optimizing():
                    logger.info(
                        f"[BACKTEST BYPASS] Accepting {entry_tag} for {pair} (Gate was closed)"
                    )
                else:
                    logger.info(f"[GATE_CLOSED] Rejecting {entry_tag} for {pair}")
                    return False

            # --- EXPLORER SPECIFIC CHECKS ---
            if entry_tag == "ai_explorer":
                try:
                    # A. Límite de trades Explorer
                    state = self._load_explorer_state()
                    confidence = state.get("confidence", "NORMAL")

                    # PROBE mode: máximo 1 trade
                    if confidence == "PROBE":
                        explorer_limit = 1
                    else:
                        explorer_limit = state.get("max_concurrent_trades", 2)

                    open_explorers = Trade.get_trades(
                        [Trade.is_open.is_(True), Trade.enter_tag == "ai_explorer"]
                    ).all()

                    if len(open_explorers) >= explorer_limit:
                        logger.info(
                            f"Explorer limit reached ({explorer_limit}, confidence={confidence}). Rejecting {pair}."
                        )
                        return False

                    # B. DI Check (Territorio desconocido)
                    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                    last = dataframe.iloc[-1].squeeze()

                    # Risk-Dynamic DI Thresholds
                    rl = int(getattr(self, "_risk_level_cached", 1))
                    close_15m = float(last.get("close_15m", 0) or 0)
                    ema50_15m = float(last.get("ema50_15m", 0) or 0)
                    ema200_15m = float(last.get("ema200_15m", 0) or 0)
                    core_trend15_ok = bool(last.get("core_trend15_ok", False))

                    missing_15m = (close_15m <= 0) or (ema200_15m <= 0)

                    # FAIL-CLOSED in trade live: if 15m informatives are missing, block entries and log it.
                    if missing_15m and self._is_trade_live() and (not gate_relaxed_confirm):
                        logger.info(
                            f"[CORE_BLOCK_15M_MISSING] {pair} | "
                            f"close_15m={close_15m:.6f}, ema50_15m={ema50_15m:.6f}, ema200_15m={ema200_15m:.6f} | "
                            f"rl={rl} ts={current_time.isoformat()} | Action=Download 15m data / check datadir"
                        )
                        return False

                    # Risk-level trend gating (MONOTONIC permissiveness: higher RL => fewer restrictions)
                    if gate_relaxed_confirm:
                        trend15_pass = True
                    elif missing_15m and self._is_optimizing():
                        # Backtest/Hyperopt: warn but do not block to keep dataset generation moving.
                        logger.info(
                            f"[BACKTEST_WARN_15M_MISSING] {pair} | "
                            f"close_15m={close_15m:.6f}, ema50_15m={ema50_15m:.6f}, ema200_15m={ema200_15m:.6f} | "
                            f"rl={rl} ts={current_time.isoformat()} | Action=Download 15m data"
                        )
                        trend15_pass = True
                    else:
                        if rl == 0:
                            # Strict: require confirmed 15m trend
                            trend15_pass = bool(core_trend15_ok)
                        elif rl == 1:
                            # Relax: confirmed trend OR above EMA50 15m
                            trend15_pass = bool(core_trend15_ok) or (
                                close_15m > ema50_15m and ema50_15m > 0
                            )
                        elif rl == 2:
                            # More permissive: confirmed trend OR above EMA200 15m
                            trend15_pass = bool(core_trend15_ok) or (
                                close_15m > ema200_15m and ema200_15m > 0
                            )
                        else:
                            # RL3+: maximum permissiveness -> no 15m filter
                            trend15_pass = True

                    if (not trend15_pass) and (not gate_relaxed_confirm):
                        # Log detallado para auditoría
                        close_15m = last.get("close_15m", 0)
                        ema50_15m = last.get("ema50_15m", 0)
                        ema200_15m = last.get("ema200_15m", 0)
                        logger.info(
                            f"[CORE_BLOCK_15M_TREND] {pair} | "
                            f"close_15m={close_15m:.2f}, ema50_15m={ema50_15m:.2f}, ema200_15m={ema200_15m:.2f} | "
                            f"ts={current_time.isoformat()}"
                        )
                        return False

                except Exception as e:
                    logger.warning(f"Core checks failed: {e}")

            # --- CORE SPECIFIC CHECKS ---
            if entry_tag == "matrix_core":
                try:
                    core_limit = int(getattr(self, "_max_trades_cached", 6))
                    open_core = Trade.get_trades(
                        [Trade.is_open.is_(True), Trade.enter_tag == "matrix_core"]
                    ).all()

                    if len(open_core) >= core_limit:
                        logger.info(f"Core limit reached ({core_limit}). Rejecting {pair}.")
                        return False
                except Exception as e:
                    logger.warning(f"Core checks failed: {e}")

            # MATRIX ENTRY APPROVED - NO SUPER() CALL
            logger.info(f"[MATRIX APPROVED] [{entry_tag}]: {pair} @ {rate:.6f}")
            return True

        # ========================================
        # 3. LEGACY TAGS - Delegate to Parent
        # ========================================
        # Para cualquier tag que no sea MATRIX, dejamos que el padre decida
        # (Esto mantiene compatibilidad futura si se usan otras estrategias)
        return super().confirm_trade_entry(
            pair, order_type, amount, rate, time_in_force, current_time, entry_tag, side, **kwargs
        )
