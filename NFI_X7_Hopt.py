from __future__ import annotations

"""
Estrategia mejorada `NFI_X7_Hopt` con ajustes basados en la auditoría.

- Parámetros validados y con registro de valores efectivos.
- Señales long diferenciando retrocesos profundos/superficiales.
- Filtros de tendencia y control de rebuys con límites de drawdown.
- Integración FreqAI con opción de fail-open/fail-closed y logging explícito.
- Features ampliadas y normalizadas para FreqAI.
- Telemetría ligera de señales generadas y filtradas.
"""

import importlib.util
import json
import logging
import os
import pathlib
from typing import Any, Dict, Optional

import numpy as np
from pandas import DataFrame
import talib.abstract as ta
from freqtrade.strategy.parameters import DecimalParameter, IntParameter

logger = logging.getLogger(__name__)

try:
    from freqtrade.freqai.freqai_interface import FreqaiStrategy
except Exception:  # pragma: no cover - fallback si no está instalado
    class FreqaiStrategy:  # type: ignore
        pass

try:
    from .NostalgiaForInfinityX7 import NostalgiaForInfinityX7
except Exception:
    _base_path = pathlib.Path(__file__).resolve().parent / "NostalgiaForInfinityX7.py"
    spec = importlib.util.spec_from_file_location("NostalgiaForInfinityX7", str(_base_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    NostalgiaForInfinityX7 = getattr(mod, "NostalgiaForInfinityX7")


class NFI_X7_Hopt(NostalgiaForInfinityX7):
    """
    Wrapper optimizado para NostalgiaForInfinityX7 - SPOT ONLY.

    Cambios clave respecto a la versión anterior:
    - Parámetros separados para retrocesos profundos y superficiales.
    - Filtro de tendencia (EMA rápida vs lenta + ADX) opcional.
    - Control de rebuys con límite de drawdown y enfriamiento.
    - Telemetría básica de señales.
    """

    max_open_trades_global: int = 12
    max_open_trades_long: int = 12

    # Hyperopt parameters
    buy_ema_dist_deep_min = DecimalParameter(0.010, 0.050, default=0.020, space="buy", optimize=True)
    buy_ema_dist_shallow_min = DecimalParameter(0.004, 0.020, default=0.010, space="buy", optimize=True)
    buy_rsi_deep_max = IntParameter(28, 45, default=40, space="buy", optimize=True)
    buy_rsi_shallow_max = IntParameter(40, 55, default=50, space="buy", optimize=True)
    entry_ema_len = IntParameter(40, 90, default=50, space="buy", optimize=True)
    rsi_len = IntParameter(10, 28, default=14, space="buy", optimize=True)
    trend_ema_fast_len = IntParameter(30, 80, default=50, space="buy", optimize=True)
    trend_ema_slow_len = IntParameter(150, 260, default=200, space="buy", optimize=True)
    trend_adx_threshold = IntParameter(15, 30, default=20, space="buy", optimize=True)
    rebuy_drawdown_max = DecimalParameter(0.04, 0.12, default=0.08, space="buy", optimize=True)
    rebuy_cooldown_candles = IntParameter(5, 30, default=10, space="buy", optimize=True)

    freqai_pred_col: str = os.getenv("NFI_FREQAI_PRED_COL", "fai_pred_entry")
    ai_fail_closed: bool = os.getenv("NFI_AI_FAIL_CLOSED", "true").lower() == "true"
    telemetry_enabled: bool = True

    freqai_thresholds_default: Dict[str, float] = {
        "core_bluechips": 0.45,
        "core_alt": 0.48,
        "high_beta": 0.52,
        "shit_performers": 0.57,
        "default": 0.50,
    }
    _pair_map: Dict[str, str] = {}
    _freqai_thresholds: Dict[str, float] = freqai_thresholds_default.copy()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.signal_counters: Dict[str, int] = {
            "generated": 0,
            "filtered_ai": 0,
            "filtered_trend": 0,
            "rebuy": 0,
        }

    # ------------------------------------------------------------------
    #            BOT START
    # ------------------------------------------------------------------
    def bot_start(self, **kwargs) -> None:
        super().bot_start(**kwargs)
        cfg: Dict[str, Any] = getattr(self, "config", {}) or {}
        strat_params: Dict[str, Any] = cfg.get("strategy_parameters", {}) or {}

        def _bounds(param_obj, value):
            low = getattr(param_obj, "low", getattr(param_obj, "min", None))
            high = getattr(param_obj, "high", getattr(param_obj, "max", None))
            if low is not None and high is not None:
                return max(float(low), min(float(high), float(value)))
            return float(value)

        def _apply_override(param_obj, key: str, section: str) -> None:
            section_values: Dict[str, Any] = strat_params.get(section, {}) or {}
            if key not in section_values:
                return
            raw_val = section_values[key]
            if not isinstance(raw_val, (int, float)):
                logger.warning("Parámetro %s debe ser numérico; se ignora override.", key)
                return
            clamped = _bounds(param_obj, raw_val)
            if hasattr(param_obj, "value"):
                param_obj.value = type(param_obj.value)(clamped)
            logger.info("Parámetro %s ajustado a %.4f (entrada=%.4f)", key, clamped, raw_val)

        buy_params = "buy"
        _apply_override(self.buy_ema_dist_deep_min, "buy_ema_dist_deep_min", buy_params)
        _apply_override(self.buy_ema_dist_shallow_min, "buy_ema_dist_shallow_min", buy_params)
        _apply_override(self.buy_rsi_deep_max, "buy_rsi_deep_max", buy_params)
        _apply_override(self.buy_rsi_shallow_max, "buy_rsi_shallow_max", buy_params)
        _apply_override(self.entry_ema_len, "entry_ema_len", buy_params)
        _apply_override(self.rsi_len, "rsi_len", buy_params)
        _apply_override(self.trend_ema_fast_len, "trend_ema_fast_len", buy_params)
        _apply_override(self.trend_ema_slow_len, "trend_ema_slow_len", buy_params)
        _apply_override(self.trend_adx_threshold, "trend_adx_threshold", buy_params)
        _apply_override(self.rebuy_drawdown_max, "rebuy_drawdown_max", buy_params)
        _apply_override(self.rebuy_cooldown_candles, "rebuy_cooldown_candles", buy_params)

        merged_path: Optional[str] = getattr(self, "merged_params_path", None)
        if not merged_path:
            merged_path = os.getenv("NFI_MERGED_PARAMS", "")
        if merged_path and os.path.isfile(merged_path):
            try:
                with open(merged_path, "r", encoding="utf-8") as f:
                    merged = json.load(f)
                self._pair_map = merged.get("pair_map", {}) or {}
                fai = merged.get("freqai", {}) or {}
                entry_th = fai.get("entry_thresholds", {}) or {}
                if entry_th:
                    self._freqai_thresholds.update({k: float(v) for k, v in entry_th.items()})
                self.freqai_pred_col = fai.get("prediction_column", self.freqai_pred_col)
                logger.info(
                    "NFI_X7_Hopt: thresholds FreqAI cargados desde merged y pred_col=%s",
                    self.freqai_pred_col,
                )
            except Exception as exc:  # pragma: no cover - logging defensivo
                logger.warning("No se pudo leer merged params en '%s': %s", merged_path, exc)

    # ------------------------------------------------------------------
    #           HELPERS DE CATEGORÍAS / FREQAI (USADOS POR NFI_X7_AI)
    # ------------------------------------------------------------------
    def _freqai_enabled_in_config(self) -> bool:
        cfg: Dict[str, Any] = getattr(self, "config", {}) or {}
        fcfg: Dict[str, Any] = cfg.get("freqai", {}) or {}
        return bool(fcfg.get("enabled", False))

    def _get_category_for_pair(self, pair: str) -> str:
        return self._pair_map.get(pair, "default")

    def _get_entry_threshold(self, pair: str) -> float:
        return float(self._freqai_thresholds.get(self._get_category_for_pair(pair), self._freqai_thresholds["default"]))

    # ------------------------------------------------------------------
    #          NUEVA API FREQAI: FEATURES + TARGET
    # ------------------------------------------------------------------
    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        df = dataframe.copy()

        if "rsi" in df.columns:
            df["%-rsi"] = df["rsi"].astype("float64")
        if "dist_below_ema" in df.columns:
            series = df["dist_below_ema"].astype("float64")
            df["%-dist_below_ema"] = series
            df["%-dist_below_ema_z"] = (series - series.rolling(50).mean()) / (series.rolling(50).std() + 1e-9)
        if "volume" in df.columns:
            df["%-volume"] = df["volume"].astype("float64")
            df["%-vol_ema_ratio"] = df["volume"] / (df["volume"].rolling(30).mean() + 1e-9)
        if "close" in df.columns:
            df["%-close"] = df["close"].astype("float64")
            df["%-returns_1"] = df["close"].pct_change().fillna(0)
        if "atr_pct" in df.columns:
            df["%-atr_pct"] = df["atr_pct"].astype("float64")
        if "ema_slope" in df.columns:
            df["%-ema_slope"] = df["ema_slope"].astype("float64")

        return df

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        if "close" not in dataframe.columns:
            return dataframe

        period = 1
        try:
            period = int(self.freqai_info.get("feature_parameters", {}).get("label_period_candles", 1))
        except Exception:
            period = 1
        period = max(period, 1)

        dataframe["fai_target_entry"] = (
            dataframe["close"].shift(-period).rolling(period).mean() / dataframe["close"] - 1.0
        )
        return dataframe

    # ------------------------------------------------------------------
    #                 INDICADORES
    # ------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = super().populate_indicators(dataframe, metadata)

        if "ema" not in df.columns or df["ema"].isnull().all():
            df["ema"] = ta.EMA(df["close"], timeperiod=int(self.entry_ema_len.value))
        else:
            df["ema"] = df["ema"].fillna(method="ffill").fillna(method="bfill")

        if "rsi" not in df.columns or df["rsi"].isnull().all():
            df["rsi"] = ta.RSI(df["close"], timeperiod=int(self.rsi_len.value))
        else:
            df["rsi"] = df["rsi"].fillna(method="ffill").fillna(method="bfill")

        df["dist_below_ema"] = 1.0 - (df["close"] / df["ema"])

        df["ema_fast"] = ta.EMA(df["close"], timeperiod=int(self.trend_ema_fast_len.value))
        df["ema_slow"] = ta.EMA(df["close"], timeperiod=int(self.trend_ema_slow_len.value))
        df["adx"] = ta.ADX(df)

        atr = ta.ATR(df)
        df["atr_pct"] = atr / (df["close"] + 1e-9)
        df["ema_slope"] = df["ema"].diff().fillna(0)

        return df

    # ------------------------------------------------------------------
    #                           ENTRADAS (SPOT - LONG ONLY)
    # ------------------------------------------------------------------
    def _update_telemetry(self, key: str, amount: int = 1) -> None:
        if not self.telemetry_enabled:
            return
        self.signal_counters[key] = self.signal_counters.get(key, 0) + amount

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        ema = df.get("ema")
        rsi = df.get("rsi")
        dist_below = df.get("dist_below_ema")
        ema_fast = df.get("ema_fast")
        ema_slow = df.get("ema_slow")
        adx = df.get("adx")

        if ema is None or rsi is None or dist_below is None:
            df["enter_long"] = 0
            df["enter_tag"] = ""
            return df

        trend_mask = ema_fast.notnull() & ema_slow.notnull()
        trend_filter = trend_mask & (ema_fast > ema_slow)
        if adx is not None:
            trend_filter &= adx > int(self.trend_adx_threshold.value)

        deep_pullback = (dist_below >= float(self.buy_ema_dist_deep_min.value)) & (rsi < int(self.buy_rsi_deep_max.value))
        shallow_pullback = (dist_below >= float(self.buy_ema_dist_shallow_min.value)) & (
            rsi < int(self.buy_rsi_shallow_max.value)
        )

        long_cond = (deep_pullback | shallow_pullback) & trend_filter

        df["enter_long"] = 0
        df["enter_tag"] = ""

        long_edge = long_cond & (~long_cond.shift(1, fill_value=False))

        cooldown = ~deep_pullback.shift(int(self.rebuy_cooldown_candles.value), fill_value=False)
        rebuy_guard = dist_below <= float(self.rebuy_drawdown_max.value)
        rebuy_cond = long_edge & deep_pullback & rebuy_guard & cooldown

        normal_cond = long_edge & (~rebuy_cond)

        df.loc[rebuy_cond, "enter_tag"] += "long_rebuy "
        df.loc[normal_cond, "enter_tag"] += "long_normal "
        df.loc[rebuy_cond | normal_cond, "enter_long"] = 1

        self._update_telemetry("generated", int((rebuy_cond | normal_cond).sum()))
        self._update_telemetry("rebuy", int(rebuy_cond.sum()))

        ai_long_ok = df.get("ai_long_ok")
        if ai_long_ok is not None:
            normal_mask = (df["enter_long"] == 1) & df["enter_tag"].str.contains("long_normal", na=False)
            blocked = normal_mask & (ai_long_ok <= 0)
            df.loc[blocked, "enter_long"] = 0
            self._update_telemetry("filtered_ai", int(blocked.sum()))

        trend_blocked = (df["enter_long"] == 1) & (~trend_filter)
        df.loc[trend_blocked, "enter_long"] = 0
        self._update_telemetry("filtered_trend", int(trend_blocked.sum()))

        return df


class NFI_X7_AI(NFI_X7_Hopt, FreqaiStrategy):
    """
    Variante de NFI_X7_Hopt con integración FreqAI (nueva API).
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = super().populate_indicators(dataframe, metadata)

        if self._freqai_enabled_in_config() and getattr(self, "freqai", None) is not None:
            pair = (metadata or {}).get("pair", "UNKNOWN")
            try:
                df = self.freqai.start(df, metadata, self)
            except KeyError as exc:
                logger.warning(
                    "FreqAI KeyError para el par %s: %s. Se omite FreqAI para este par. Comprueba 'freqai.whitelist'.",
                    pair,
                    exc,
                )
            except Exception as exc:  # pragma: no cover - logging defensivo
                logger.warning("FreqAI error inesperado para el par %s: %s. Se omite FreqAI para este par.", pair, exc)

        pred_col = self.freqai_pred_col
        pair = (metadata or {}).get("pair", "")
        th = self._get_entry_threshold(pair) if pair else self._freqai_thresholds["default"]

        if "ai_long_score" not in df.columns:
            df["ai_long_score"] = np.nan
        if "ai_long_ok" not in df.columns:
            df["ai_long_ok"] = 0
        if "ai_short_score" not in df.columns:
            df["ai_short_score"] = 0.0
        if "ai_short_ok" not in df.columns:
            df["ai_short_ok"] = 0

        if "close" in df.columns:
            cond_ok = df["close"].notnull()
        else:
            cond_ok = df.index == df.index  # Serie booleana vacía si no hay datos

        if "&s_close_mean" in df.columns and "do_predict" in df.columns:
            bad_delta = (df["do_predict"] == 1) & (df["&s_close_mean"] < 0)
            cond_ok = cond_ok & (~bad_delta)

        if pred_col in df.columns:
            df["ai_long_score"] = df[pred_col].astype("float64")
            cond_ok = cond_ok & (df[pred_col].fillna(float("-inf")) > float(th))
        else:
            logger.info(
                "NFI_X7_AI: columna de predicción '%s' no encontrada para %s. fail_closed=%s",
                pred_col,
                pair or "UNKNOWN",
                self.ai_fail_closed,
            )
            if self.ai_fail_closed:
                cond_ok = cond_ok & False

        df["ai_long_ok"] = cond_ok.astype("int8")
        df["ai_short_score"] = 0.0
        df["ai_short_ok"] = 0

        if self.telemetry_enabled:
            blocked = (df["ai_long_ok"] == 0).sum()
            self._update_telemetry("filtered_ai", int(blocked))

        return df
