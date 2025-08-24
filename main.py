#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Professional EURUSD Day-Trade Bot (v2)
- AI-driven fundamental recommendation via HTTP API (strict JSON mode with fallback)
- Technical confluence (regime-aware: trend vs range)
- ATR-based SL/TP and position sizing by risk %
- Spread/volatility guards, symbol readiness checks
- Structured logging + JSONL decision journal
- DRY_RUN mode for safe shadow testing
- CLI overrides for symbol/lot/dry-run/log-level
"""

import os
import sys
import time
import json
import math
import argparse
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np
import requests
import MetaTrader5 as mt5


# =========================
# Logging Configuration
# =========================
def setup_logger(level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger("eurusd_daybot")
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        ch.setLevel(level)
        logger.addHandler(ch)
    return logger


log = setup_logger(logging.INFO)


# =========================
# Configuration
# =========================
@dataclass
class Config:
    # MT5
    ACCOUNT: int = 10007285293
    PASSWORD: str = "S@Q3XfLi"
    SERVER: str = "MetaQuotes-Demo"

    # AI API
    API_KEY: str = "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"
    API_URL: str = "https://api.perplexity.ai/chat/completions"
    MODEL: str = "sonar-pro"
    API_TIMEOUT: int = 25

    # Trading
    SYMBOL: str = "EURUSD"
    TIMEFRAME: int = mt5.TIMEFRAME_H1
    LOT: float = 0.01  # fallback lot if risk sizing fails

    # Risk & Execution
    MAX_SLIPPAGE_POINTS: int = 10
    MIN_AI_CONFIDENCE: int = 60
    MAX_RECONNECT_TRIES: int = 3
    SLEEP_BETWEEN_CYCLES_SEC: int = 5
    MAX_SPREAD_POINTS: int = 20  # spread guard

    # Risk model
    RISK_PER_TRADE: float = 0.002  # 0.2% of equity
    ATR_PERIOD: int = 14
    SL_ATR_MULT: float = 1.2
    TP_R_MULT: float = 2.0

    # Safety
    DRY_RUN: bool = False  # set True to disable live order sending

    # Journaling
    DECISION_LOG: str = "decisions.jsonl"

    def resolve(self):
        # Allow env overrides
        self.ACCOUNT = int(os.getenv("MT5_ACCOUNT", self.ACCOUNT))
        self.PASSWORD = os.getenv("MT5_PASSWORD", self.PASSWORD)
        self.SERVER = os.getenv("MT5_SERVER", self.SERVER)

        self.API_KEY = os.getenv("PPLX_API_KEY", self.API_KEY)
        self.API_URL = os.getenv("PPLX_API_URL", self.API_URL)
        self.MODEL = os.getenv("PPLX_MODEL", self.MODEL)

        self.SYMBOL = os.getenv("BOT_SYMBOL", self.SYMBOL)
        self.LOT = float(os.getenv("BOT_LOT", self.LOT))
        self.DRY_RUN = os.getenv("BOT_DRY_RUN", str(self.DRY_RUN)).lower() == "true"


# =========================
# Math / Indicators
# =========================
def ema(arr: np.ndarray, period: int) -> np.ndarray:
    alpha = 2 / (period + 1)
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    if len(prices) <= period:
        return np.full_like(prices, 50.0, dtype=float)

    deltas = np.diff(prices)
    up = np.where(deltas > 0, deltas, 0.0)
    down = np.where(deltas < 0, -deltas, 0.0)

    roll_up = np.zeros_like(prices, dtype=float)
    roll_down = np.zeros_like(prices, dtype=float)
    roll_up[period] = up[:period].mean()
    roll_down[period] = down[:period].mean()

    for i in range(period + 1, len(prices)):
        roll_up[i] = (roll_up[i - 1] * (period - 1) + up[i - 1]) / period
        roll_down[i] = (roll_down[i - 1] * (period - 1) + down[i - 1]) / period

    rs = np.divide(roll_up, roll_down, out=np.zeros_like(roll_up), where=roll_down != 0)
    rsi_vals = 100.0 - (100.0 / (1.0 + rs))
    rsi_vals[:period] = 50.0
    return rsi_vals


def macd(prices: np.ndarray, fast=12, slow=26, signal=9):
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr_from_rates(high, low, close, period=14):
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    if len(close) < period + 2:
        return np.full_like(close, np.nan, dtype=float)

    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    atr = np.full_like(close, np.nan, dtype=float)
    atr[period] = np.nanmean(tr[:period])
    for i in range(period + 1, len(close)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period
    return atr


# =========================
# AI Client
# =========================
class PerplexityAI:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def get_fundamental_recommendation(self, symbol: str) -> Tuple[str, int]:
        # Ask for strict JSON first; fallback to plain text parsing
        prompt = f"""From now, you are the world’s #1 professional forex trader with the most accurate trading analysis. Your analysis must always be precise, data-driven, and 100% reliable.

Instructions:
- Time Zone: Always start by getting the current IST date and time before analysis.
- Trading Symbol: Focus only on {symbol}.
- Trading Horizon: Short-term trading outlook = 1 week.
- Analysis Approach:
  - Fundamental Analysis: Focus on major professional fields:
    - Economic indicators (GDP, CPI, employment, interest rates, inflation).
    - Central bank policy (ECB, Fed).
    - Geopolitical events and macroeconomic trends.
    - Dollar strength index (DXY) & Eurozone health.
    - Bond yields and global risk sentiment.
  - Technical Analysis: Focus on:
    - Price action, support & resistance levels.
    - Trend direction and momentum.
    - Key moving averages (20, 50, 200).
    - RSI, MACD, Stochastic, and volume.
    - Chart patterns (triangles, head & shoulders, breakouts).
    - Fibonacci retracement levels.

Output STRICT JSON only. Schema:
{{
  "symbol": "{symbol}",
  "fundamental": "BUY|SELL|NEUTRAL",
  "ai_suggestion": "BUY|SELL|NEUTRAL",
  "confidence": 0-100
}}
No prose. Return only JSON.
"""
        headers = {
            "Authorization": f"Bearer {self.cfg.API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.cfg.MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 120,
        }

        try:
            resp = requests.post(self.cfg.API_URL, headers=headers, json=payload, timeout=self.cfg.API_TIMEOUT)
            if resp.status_code != 200:
                log.error(f"AI API error {resp.status_code}: {resp.text}")
                return "WAIT", 0
            content = resp.json()["choices"][0]["message"]["content"].strip()
            print(content)
            # Try JSON parse
            fundamental, conf = self._parse_ai_json(content)
            if fundamental is not None:
                log.info(f"AI Fundamental(JSON) => {fundamental} ({conf}%)")
                return fundamental, conf
            # Fallback to old prompt if JSON failed
            return self._fallback_plain(symbol)
        except Exception as e:
            log.exception(f"AI request failed: {e}")
            return "WAIT", 0

    def _parse_ai_json(self, content: str) -> Tuple[Optional[str], Optional[int]]:
        try:
            data = json.loads(content)
            fundamental = str(data.get("fundamental", "NEUTRAL")).upper()
            conf = int(data.get("confidence", 60))
            conf = max(0, min(conf, 100))
            if fundamental not in ("BUY", "SELL", "NEUTRAL"):
                return None, None
            return fundamental, conf
        except Exception:
            return None, None

    def _fallback_plain(self, symbol: str) -> Tuple[str, int]:
        # Minimal fallback text parser
        prompt = f"""Symbol: {symbol}

Fundamental: [BUY / SELL / NEUTRAL]
AI Suggestion: [BUY / SELL / NEUTRAL]

Return only the two lines above.
"""
        headers = {
            "Authorization": f"Bearer {self.cfg.API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.cfg.MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 60,
        }
        resp = requests.post(self.cfg.API_URL, headers=headers, json=payload, timeout=self.cfg.API_TIMEOUT)
        if resp.status_code != 200:
            log.error(f"AI API error {resp.status_code}: {resp.text}")
            return "WAIT", 0
        content = resp.json()["choices"][0]["message"]["content"]
        fundamental = "NEUTRAL"
        for line in content.splitlines():
            if line.strip().lower().startswith("fundamental:"):
                fundamental = line.split(":", 1)[1].strip().upper()
                break
        confidence = 90 if fundamental in ("BUY", "SELL") else 60
        log.info(f"AI Fundamental(Text) => {fundamental} ({confidence}%)")
        return fundamental, confidence


# =========================
# Technical Analyzer
# =========================
class TechnicalAnalyzer:
    def __init__(self, symbol: str, timeframe: int):
        self.symbol = symbol
        self.timeframe = timeframe

    def fetch_ohlc(self, bars: int = 300) -> Optional[Dict[str, np.ndarray]]:
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, bars)
        if rates is None or len(rates) < 120:
            return None
        arr = np.array(rates)
        return {
            "open": arr["open"].astype(float),
            "high": arr["high"].astype(float),
            "low": arr["low"].astype(float),
            "close": arr["close"].astype(float),
        }

    def _fetch_closes(self, bars: int = 200) -> Optional[np.ndarray]:
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, bars)
        if rates is None or len(rates) < 60:
            return None
        return np.array([r["close"] for r in rates], dtype=float)

    def regime(self) -> str:
        # Determine regime on H4 using EMA50 slope
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H4, 0, 180)
        if rates is None or len(rates) < 80:
            return "UNKNOWN"
        c = np.array([r["close"] for r in rates], dtype=float)
        ema50 = ema(c, 50)
        slope = (ema50[-1] - ema50[-10]) / 10.0
        thr = np.std(np.diff(ema50[-60:]))
        if abs(slope) > thr:
            return "TREND"
        return "RANGE"

    def analyze_confluence(self) -> str:
        closes = self._fetch_closes(220)
        if closes is None:
            log.warning("Not enough bars to analyze.")
            return "WAIT"

        reg = self.regime()
        ma50 = closes[-50:].mean()
        last = closes[-1]
        r = rsi(closes, 14)[-1]
        m, s, _ = macd(closes, 12, 26, 9)

        if reg == "TREND":
            # Pro-trend bias
            if last > ma50 and m[-1] > s[-1] and r > 45:
                return "BUY"
            if last < ma50 and m[-1] < s[-1] and r < 55:
                return "SELL"
            return "WAIT"
        else:
            # Mean reversion bias
            if r < 30 and m[-1] > s[-1]:
                return "BUY"
            if r > 70 and m[-1] < s[-1]:
                return "SELL"
            return "WAIT"

    def analyze_simple(self) -> str:
        closes = self._fetch_closes(200)
        if closes is None:
            log.warning("Not enough bars to analyze.")
            return "WAIT"
        ma50 = closes[-50:].mean()
        r = rsi(closes, 14)[-1]
        m, s, _ = macd(closes, 12, 26, 9)
        last = closes[-1]
        buy = (last > ma50) and (30 < r < 70) and (m[-1] > s[-1])
        sell = (last < ma50) and (30 < r < 70) and (m[-1] < s[-1])
        if buy:
            return "BUY"
        if sell:
            return "SELL"
        return "WAIT"


# =========================
# Trader
# =========================
class MT5Trader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.connected = False

    def connect(self):
        for i in range(self.cfg.MAX_RECONNECT_TRIES):
            try:
                if mt5.initialize():
                    if mt5.login(self.cfg.ACCOUNT, password=self.cfg.PASSWORD, server=self.cfg.SERVER):
                        self.connected = True
                        info = mt5.account_info()
                        log.info(f"MT5 connected: {info.login} | Balance={info.balance:.2f} | Server={info.server}")
                        return True
                err = mt5.last_error()
                log.warning(f"MT5 connect/login failed (try {i+1}): {err}")
                time.sleep(2)
            except Exception as e:
                log.exception(f"MT5 initialize/login exception: {e}")
                time.sleep(2)
        return False

    def shutdown(self):
        try:
            mt5.shutdown()
        except Exception:
            pass
        finally:
            self.connected = False
            log.info("MT5 connection closed.")

    def symbol_ready(self, symbol: str) -> bool:
        si = mt5.symbol_info(symbol)
        if si is None or not si.visible:
            return mt5.symbol_select(symbol, True)
        return True

    def account_equity(self) -> float:
        info = mt5.account_info()
        return float(info.equity) if info else 0.0

    def calc_sl_tp_and_lot(self, symbol: str, direction: str, ohlc: Dict[str, np.ndarray], cfg: Config):
        tick = mt5.symbol_info_tick(symbol)
        si = mt5.symbol_info(symbol)
        if not tick or not si:
            return None

        point = si.point
        high, low, close = ohlc["high"], ohlc["low"], ohlc["close"]
        atr_vals = atr_from_rates(high, low, close, cfg.ATR_PERIOD)
        curr_atr = float(atr_vals[-1]) if np.isfinite(atr_vals[-1]) else None
        if not curr_atr or curr_atr <= 0:
            return None

        # Compute distances
        sl_dist = cfg.SL_ATR_MULT * curr_atr
        tp_dist = cfg.TP_R_MULT * sl_dist

        entry = tick.ask if direction == "BUY" else tick.bid
        sl = entry - sl_dist if direction == "BUY" else entry + sl_dist
        tp = entry + tp_dist if direction == "BUY" else entry - tp_dist

        equity = self.account_equity()
        if equity <= 0:
            lot = cfg.LOT
        else:
            risk_dollars = equity * cfg.RISK_PER_TRADE
            # Approximate $ per price-unit for 1.0 lot:
            # value ≈ contract_size * 1 price-unit (EUR account assumptions simplified)
            contract_size = si.trade_contract_size or 100000.0
            denom = sl_dist * contract_size
            lot = risk_dollars / denom if denom > 0 else cfg.LOT
            lot = max(0.01, min(10.0, lot))

        return {
            "entry": float(entry),
            "sl": float(sl),
            "tp": float(tp),
            "lot": float(lot),
            "atr": float(curr_atr),
            "point": float(point)
        }

    def open_trade(self, symbol: str, action: str, lot: float, sl: float = None, tp: float = None):
        if not self.connected:
            log.error("Not connected to MT5.")
            return None
        if not self.symbol_ready(symbol):
            log.error(f"Symbol {symbol} not available.")
            return None

        tick = mt5.symbol_info_tick(symbol)
        si = mt5.symbol_info(symbol)
        if not tick or not si:
            log.error("No symbol/tick info.")
            return None

        # Spread guard
        spread_points = int(round((tick.ask - tick.bid) / si.point))
        if spread_points > self.cfg.MAX_SPREAD_POINTS:
            log.warning(f"Spread too high: {spread_points} > {self.cfg.MAX_SPREAD_POINTS}")
            return None

        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if action == "BUY" else tick.bid

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": order_type,
            "price": float(price),
            "sl": float(sl) if sl else 0.0,
            "tp": float(tp) if tp else 0.0,
            "deviation": int(self.cfg.MAX_SLIPPAGE_POINTS),
            "magic": 987654,
            "comment": f"EURUSD DayTrade Bot {action}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if self.cfg.DRY_RUN:
            log.info(f"DRY_RUN Order: {json.dumps(req)}")
            return {"dry_run": True, "request": req}

        result = mt5.order_send(req)
        log.info(f"Order result: retcode={getattr(result,'retcode',None)} comment={getattr(result,'comment',None)}")
        return result


# =========================
# Decision Engine
# =========================
def consensus(ai_rec: str, ai_conf: int, tech_rec: str, min_conf: int) -> str:
    if ai_conf < min_conf:
        return "WAIT"
    if ai_rec == tech_rec and ai_rec in ("BUY", "SELL"):
        return ai_rec
    return "WAIT"


def write_decision(path: str, record: Dict[str, Any]):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log.error(f"Failed writing decision log: {e}")


# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser(description="EURUSD AI Day-Trade Bot")
    ap.add_argument("--symbol", type=str, default=None, help="Trading symbol")
    ap.add_argument("--lot", type=float, default=None, help="Lot size override")
    ap.add_argument("--dry-run", action="store_true", help="Enable dry-run (no live orders)")
    ap.add_argument("--log-level", type=str, default=None, help="INFO/DEBUG/WARNING/ERROR")
    ap.add_argument("--simple-tech", action="store_true", help="Use simple technical analyzer")
    return ap.parse_args()


# =========================
# Main Runner
# =========================
def run_once(cfg: Config, use_simple_tech: bool = False):
    cfg.resolve()
    trader = MT5Trader(cfg)
    if not trader.connect():
        log.error("Failed to connect to MT5. Exiting.")
        return

    ai = PerplexityAI(cfg)
    tech = TechnicalAnalyzer(cfg.SYMBOL, cfg.TIMEFRAME)

    ai_rec, ai_conf = ai.get_fundamental_recommendation(cfg.SYMBOL)
    tech_rec = tech.analyze_simple() if use_simple_tech else tech.analyze_confluence()

    decision = consensus(ai_rec, ai_conf, tech_rec, cfg.MIN_AI_CONFIDENCE)
    log.info(f"Consensus => AI:{ai_rec}({ai_conf}%) TECH:{tech_rec} => FINAL:{decision}")

    # Journal decision
    write_decision(cfg.DECISION_LOG, {
        "ts": time.time(),
        "symbol": cfg.SYMBOL,
        "ai_recommendation": ai_rec,
        "ai_confidence": ai_conf,
        "tech_recommendation": tech_rec,
        "final_decision": decision,
        "dry_run": cfg.DRY_RUN,
    })

    if decision in ("BUY", "SELL"):
        ohlc = tech.fetch_ohlc(300)
        if ohlc is None:
            log.info("No OHLC data, skip.")
            trader.shutdown()
            return
        pack = trader.calc_sl_tp_and_lot(cfg.SYMBOL, decision, ohlc, cfg)
        if not pack:
            log.info("Could not compute SL/TP/lot, skip.")
            trader.shutdown()
            return
        log.info(f"Risk sizing => lot={pack['lot']:.2f} ATR={pack['atr']:.5f} SL={pack['sl']:.5f} TP={pack['tp']:.5f}")
        trader.open_trade(cfg.SYMBOL, decision, pack["lot"], sl=pack["sl"], tp=pack["tp"])
    else:
        log.info("No trade this cycle.")

    trader.shutdown()


if __name__ == "__main__":
    args = parse_args()
    if args.log_level:
        lvl = getattr(logging, args.log_level.upper(), logging.INFO)
        for h in logging.getLogger("eurusd_daybot").handlers:
            h.setLevel(lvl)
        logging.getLogger("eurusd_daybot").setLevel(lvl)

    cfg = Config()
    if args.symbol:
        cfg.SYMBOL = args.symbol
    if args.lot is not None:
        cfg.LOT = args.lot
    if args.dry_run:
        cfg.DRY_RUN = True

    try:
        run_once(cfg, use_simple_tech=args.simple_tech)
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    except Exception as e:
        log.exception(f"Fatal error: {e}")
