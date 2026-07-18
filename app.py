#!/usr/bin/env python3
"""
Stock Deep Diagnostic Report Generator
- Flask backend serving report generation and AI chat
- Data source: akshare (free, no API key required)
- AI chat: OpenAI-compatible API (user configurable)
"""

import json
import os
import time
import threading
from datetime import datetime, timedelta, timezone
from functools import lru_cache

# China timezone (UTC+8) — ensures consistent timestamps regardless of server location
CN_TZ = timezone(timedelta(hours=8))

def now_cn():
    """Return current datetime in China timezone (UTC+8)."""
    return datetime.now(CN_TZ)

# ========== Proxy workaround: must happen BEFORE any import of requests/akshare ==========
# Nuke ALL proxy-related env vars — they break Chinese financial data APIs
for var in list(os.environ.keys()):
    if "PROXY" in var.upper() or "proxy" in var.lower():
        os.environ.pop(var, None)
# Also set NO_PROXY as belt-and-suspenders
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

# Now safe to import
import requests
import re
import math
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory, Response

# Try to import pypinyin for comprehensive Chinese pinyin support
try:
    from pypinyin import pinyin as _pypinyin_func, Style as _PinyinStyle
    _HAS_PYPINYIN = True
except ImportError:
    _HAS_PYPINYIN = False
    print("[pinyin] pypinyin not installed, falling back to manual _PINYIN_FULL dictionary")

# API fallback module
import api_fallback

# Curated HK/US stock list for local fuzzy search (pinyin + fuzzy, like A-shares)
try:
    from hk_us_list import HK_US_STOCKS
except ImportError:
    print("[hk_us_list] hk_us_list.py not found, HK/US fuzzy search limited to smartbox")
    HK_US_STOCKS = {"HK": [], "US": []}

# ----- Trust no proxy: create a dedicated session that never touches system proxy settings -----
_http_session = requests.Session()
_http_session.trust_env = False

app = Flask(__name__)

# ---- Safe JSON provider: never emit NaN / Infinity (invalid JSON) ----
# yfinance / EastMoney occasionally yield NaN; Python's json.dumps serializes
# those as the literal `NaN`, which JSON.parse on the frontend rejects
# ("Unexpected token 'N'"). Recursively coerce non-finite floats to null.
from flask.json.provider import DefaultJSONProvider

def _sanitize_json(obj):
    # numpy scalars (e.g. np.float64 nan from EastMoney's financial parser)
    # are NOT Python float subclasses, so normalize them to native types first.
    if isinstance(obj, np.generic):
        try:
            obj = obj.item()
        except Exception:
            return obj
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_json(v) for v in obj]
    return obj

class _SafeJSONProvider(DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        kwargs.setdefault("allow_nan", False)
        return super().dumps(_sanitize_json(obj), **kwargs)
    def dump(self, obj, fp, **kwargs):
        kwargs.setdefault("allow_nan", False)
        return super().dump(_sanitize_json(obj), fp, **kwargs)

app.json = _SafeJSONProvider(app)


# ---- After-request hook: inject API fallback status header ----
@app.after_request
def add_fallback_header(response):
    """Add X-API-Fallback header so frontend JS can detect fallback usage."""
    recent = api_fallback.get_fallback_log()
    if recent:
        # Get the last few events from current request context
        # We use a simple heuristic: events from the last 5 seconds
        now = datetime.now()
        recent_events = []
        for e in reversed(recent):
            try:
                et = datetime.strptime(e["time"], "%H:%M:%S").replace(
                    year=now.year, month=now.month, day=now.day
                )
                if (now - et).total_seconds() < 30:
                    recent_events.append(e)
            except ValueError:
                pass
            if len(recent_events) >= 10:
                break

        if recent_events:
            # Minimal header: count and key events
            fb_sources = set(e["source"] for e in recent_events if not e["ok"])
            fb_count = sum(1 for e in recent_events if e["source"] != "primary")
            header_val = f"fb={fb_count}"
            if fb_sources:
                header_val += f";fail={','.join(sorted(fb_sources))}"
            response.headers["X-API-Fallback"] = header_val

    return response

# ---------- Config ----------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
CACHE = {}
CACHE_TTL = 1800  # 30 minutes


def load_config():
    """Load config from config.json, with env var overrides (for Vercel deployment)."""
    cfg = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    # Environment variable overrides (Vercel / cloud deployment)
    if os.environ.get("AI_CHAT_API_KEY"):
        cfg.setdefault("ai_chat", {})["api_key"] = os.environ["AI_CHAT_API_KEY"]
    if os.environ.get("AI_CHAT_API_BASE"):
        cfg.setdefault("ai_chat", {})["api_base"] = os.environ["AI_CHAT_API_BASE"]
    if os.environ.get("AI_CHAT_MODEL"):
        cfg.setdefault("ai_chat", {})["model"] = os.environ["AI_CHAT_MODEL"]
    if os.environ.get("AI_CHAT_PROVIDER"):
        cfg.setdefault("ai_chat", {})["provider"] = os.environ["AI_CHAT_PROVIDER"]

    return cfg


def save_config(cfg):
    """Save config. On Vercel (filesystem read-only), this is a no-op;
    use environment variables instead."""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except (OSError, IOError):
        # Read-only filesystem (e.g. Vercel) — silently skip
        pass



def get_ai_config():
    """Get AI config: prefer request body, fall back to config.json."""
    try:
        data = request.get_json(silent=True)
        if data and isinstance(data, dict) and data.get("ai_chat"):
            cc = data["ai_chat"]
            if cc.get("api_key") and not str(cc["api_key"]).startswith("sk-your"):
                return cc
    except Exception:
        pass
    cfg = load_config()
    return cfg.get("ai_chat", {})
def cached(key, ttl=CACHE_TTL):
    """Simple in-memory cache decorator."""
    now = time.time()
    if key in CACHE:
        val, ts = CACHE[key]
        if now - ts < ttl:
            return val
    return None


def cache_set(key, val):
    CACHE[key] = (val, time.time())


# ---------- Data Fetching (Tencent Stock API, no proxy issues) ----------

TENCENT_QT = "https://qt.gtimg.cn/q="
REQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://gu.qq.com/",
}

# ========== Cross-Market Support (Beta) ==========
MARKET_CONFIG = {
    "A": {
        "name": "A股",
        "suffixes": [".SZ", ".SH", ".BJ"],
        "tc_prefixes": {"SZ": "sz", "SH": "sh", "BJ": "bj"},
        "default_prefix": "sz",
        "indices": {
            "shanghai": ("sh000001", "上证指数"),
            "shenzhen": ("sz399001", "深证成指"),
            "chinext": ("sz399006", "创业板指"),
        },
        "currency": "¥",
        "currency_label": "元",
        "mv_unit": "亿",
        "search_t_code": "gp",
        "has_limit_price": True,
    },
    "HK": {
        "name": "港股",
        "suffixes": [".HK"],
        "tc_prefix": "hk",
        "default_prefix": "hk",
        "indices": {
            "hsi": ("hkHSI", "恒生指数"),
            "hscei": ("hkHSCEI", "恒生中国企业"),
            "hstech": ("hkHSTECH", "恒生科技"),
        },
        "currency": "HK$",
        "currency_label": "港元",
        "mv_unit": "亿",
        "search_t_code": "hk",
        "has_limit_price": False,
    },
    "US": {
        "name": "美股",
        "suffixes": [".US"],
        "tc_prefix": "us",
        "default_prefix": "us",
        "indices": {
            "sp500": ("usINX", "标普500"),
            "nasdaq": ("usIXIC", "纳斯达克"),
            "dow": ("usDJI", "道琼斯"),
        },
        "currency": "$",
        "currency_label": "美元",
        "mv_unit": "亿",
        "search_t_code": "us",
        "has_limit_price": False,
    },
}

def detect_market(code):
    """Return (market_key, symbol) from a full stock code."""
    code = code.strip().upper()
    if ".HK" in code:
        return "HK", code.replace(".HK", "")
    if ".US" in code:
        return "US", code.replace(".US", "")
    if any(s in code for s in [".SZ", ".SH", ".BJ"]):
        return "A", code
    # Bare numeric → A-share
    if code.isdigit() and 1 <= len(code) <= 6:
        return "A", code
    # Alphabetic ticker → US
    if any(c.isalpha() for c in code) and not code.isdigit():
        return "US", code
    return "A", code

def _http_get(url, params=None, timeout=15, headers_extra=None):
    """HTTP GET using trust_env=False session — never touches system proxy."""
    headers = dict(REQ_HEADERS)
    if headers_extra:
        headers.update(headers_extra)
    return _http_session.get(url, params=params, headers=headers, timeout=timeout)

def _tencent_code(code, market="A"):
    """Convert standard code to Tencent format: sz000001, sh600519, hk00700, usAAPL"""
    if market == "HK":
        symbol = code.replace(".HK", "")
        return f"hk{symbol}"
    if market == "US":
        symbol = code.replace(".US", "")
        return f"us{symbol}"
    # A-share
    symbol = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "")
    if ".SZ" in code:
        prefix = "sz"
    elif ".SH" in code:
        prefix = "sh"
    elif ".BJ" in code:
        prefix = "bj"
    elif symbol.startswith(("6", "9")):
        prefix = "sh"
    elif symbol.startswith(("0", "3")):
        prefix = "sz"
    elif symbol.startswith(("4", "8")):
        prefix = "bj"
    else:
        prefix = "sz"
    return f"{prefix}{symbol}"

def get_stock_info(code, market="A"):
    """Get stock basic info via Tencent real-time quote API. Market-aware field mapping."""
    cache_key = f"info_{code}"
    cached_val = cached(cache_key)
    if cached_val:
        return cached_val

    try:
        tc = _tencent_code(code, market)
        symbol = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "").replace(".HK", "").replace(".US", "")
        resp = _http_get(f"{TENCENT_QT}{tc}")
        text = resp.text

        if len(text) < 10 or "none" in text.lower() or "v_" not in text:
            return {}

        # Parse Tencent format: v_sz000001="field1~field2~field3~..."
        start = text.index('"') + 1
        end = text.rindex('"')
        fields = text[start:end].split("~")

        if len(fields) < 45:
            return {}

        # Market-specific field mapping
        if market in ("HK", "US"):
            # HK/US: similar to A-share but some fields shifted
            # 1:name, 2:code, 3:price, 4:prev_close, 5:open, 6:volume(lots),
            # 30:date, 31:change, 32:change%, 33:high, 34:low,
            # 36:volume, 37:amount, 38:turnover%, 39:PE,
            # 44:circ_mv, 45:total_mv, 46:eng_name, 47:(varies),
            # For HK: PB may be in extended fields; for US, different structure
            _safe_float = lambda i: float(fields[i]) if len(fields) > i and fields[i] else 0
            # Try to determine PB from available fields
            pb_val = 0
            # For HK, PB is typically at index 72; turnover is at index 59 (not 38)
            if market == "HK" and len(fields) > 72 and fields[72]:
                try: pb_val = float(fields[72])
                except: pass
            hk_turnover = _safe_float(59) if market == "HK" else _safe_float(38)
            info = {
                "股票简称": fields[1] if len(fields) > 1 else "",
                "最新价": _safe_float(3),
                "昨收": _safe_float(4),
                "今开": _safe_float(5),
                "最高价": _safe_float(33),
                "最低价": _safe_float(34),
                "涨跌幅": _safe_float(32),
                "涨跌额": _safe_float(31),
                "换手率": hk_turnover,
                "市盈率-动态": _safe_float(39),
                "市净率": pb_val,
                "总市值": _safe_float(45),
                "流通市值": _safe_float(44),
                "成交量": int(_safe_float(36)) if len(fields) > 36 and fields[36] else 0,
                "成交额": _safe_float(37),
            }
        else:
            # A-share original mapping (index-based):
            # 0: unknown, 1: name, 2: code, 3: current price, 4: prev close,
            # 5: open, 6: volume(lots), 7: outer, 8: inner,
            # 30: date, 31: change, 32: change%, 33: high, 34: low,
            # 38: turnover%, 39: PE, 40: unknown, 41: high2, 42: low2,
            # 43: amplitude%, 44: circulation market cap, 45: total market cap,
            # 46: PB, 47:涨停价, 48:跌停价
            info = {
                "股票简称": fields[1] if len(fields) > 1 else "",
                "最新价": float(fields[3]) if fields[3] else 0,
                "昨收": float(fields[4]) if fields[4] else 0,
                "今开": float(fields[5]) if fields[5] else 0,
                "最高价": float(fields[33]) if len(fields) > 33 and fields[33] else 0,
                "最低价": float(fields[34]) if len(fields) > 34 and fields[34] else 0,
                "涨跌幅": float(fields[32]) if len(fields) > 32 and fields[32] else 0,
                "换手率": float(fields[38]) if len(fields) > 38 and fields[38] else 0,
                "市盈率-动态": float(fields[39]) if len(fields) > 39 and fields[39] else 0,
                "市净率": float(fields[46]) if len(fields) > 46 and fields[46] else 0,
                "总市值": float(fields[45]) if len(fields) > 45 and fields[45] else 0,
                "流通市值": float(fields[44]) if len(fields) > 44 and fields[44] else 0,
                "成交量": int(float(fields[6])) if len(fields) > 6 and fields[6] else 0,
                "成交额": float(fields[37]) if len(fields) > 37 and fields[37] else 0,
            }

        # US PB fallback via yfinance (Tencent doesn't provide PB for US)
        if market == "US" and info.get("市净率", 0) == 0:
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                fast_pb = (ticker.fast_info or {}).get('price_to_book', 0)
                if fast_pb:
                    info["市净率"] = round(float(fast_pb), 2)
                else:
                    # fallback to info dict
                    yf_info = ticker.info or {}
                    yf_pb = yf_info.get('priceToBook', 0)
                    if yf_pb:
                        info["市净率"] = round(float(yf_pb), 2)
            except Exception:
                pass  # keep PB = 0 if yfinance fails

        cache_set(cache_key, info)
        return info
    except Exception as e:
        print(f"[PRIMARY] Error fetching stock info for {code}: {e}")

    # ---- Fallback: Sina Finance (A-share only) ----
    if market == "A":
        try:
            fb_info = api_fallback.sina_get_stock_info(code)
            if fb_info and fb_info.get("最新价"):
                fb_info["_fb_source"] = "sina"
                cache_set(cache_key, fb_info)
                return fb_info
        except Exception as fe:
            print(f"[FALLBACK] Sina also failed for {code}: {fe}")

    return {}


def get_price_history(code, days=250, market="A"):
    """Get historical K-line data.
    Primary: EastMoney API (supports IPO-level, up to ~7000 bars) — A-share only.
    Fallback: Tencent kline API (limited to ~640 bars, 2.5 years) — all markets.
    """
    cache_key = f"price_{code}"
    cached_val = cached(cache_key, ttl=600)
    if cached_val:
        return cached_val

    # ---- Layer 1: EastMoney K-line (A / HK / US) ----
    # EastMoney provides full IPO-level history for all three markets.
    # secid: A=0./1., HK=116.xxxxx, US=105.TICKER
    try:
        result = api_fallback.em_get_price_history(code, 8000, market=market)
        if result and len(result) >= 10:
            # HK/US: EastMoney uses fqt=2 (后复权) which inflates prices for
            # high-dividend/split-heavy stocks (e.g. Tencent recent close ~2600 vs
            # real ~460). Rescale the whole series so the latest close matches
            # the real-time spot price, so moving averages align with the
            # displayed price. A-share keeps fqt=1 (already correct).
            if market in ("HK", "US"):
                try:
                    spot = get_stock_info(code, market).get("最新价") or 0
                    last_c = result[-1].get("收盘") or 0
                    if spot and last_c and abs(last_c - spot) / spot > 0.01:
                        f = spot / last_c
                        for b in result:
                            b["开盘"] = round(b["开盘"] * f, 3)
                            b["收盘"] = round(b["收盘"] * f, 3)
                            b["最高"] = round(b["最高"] * f, 3)
                            b["最低"] = round(b["最低"] * f, 3)
                except Exception as e:
                    print(f"[get_price_history] HK/US rescale skip: {e}")
            print(f"[get_price_history] EastMoney({market}): {len(result)} klines for {code}")
            cache_set(cache_key, result)
            return result
    except Exception as e:
        print(f"[get_price_history] EastMoney failed: {e}")

    # ---- Layer 2: Tencent K-line (all markets) ----
    try:
        tc = _tencent_code(code, market)
        # Request 800 to get max available (~640 for most stocks, ~2.5 years)
        tc_days = max(days, 800)
        kline_url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        params = {
            "param": f"{tc},day,,,{tc_days + 30},qfq",
            "_": str(int(time.time() * 1000)),
        }
        resp = _http_get(kline_url, params=params, timeout=15)
        data = resp.json()

        stock_data = data.get("data", {}).get(tc, {})
        klines = stock_data.get("qfqday", stock_data.get("day", []))

        if not klines:
            cache_set(cache_key, [])
            return []

        result = []
        for k in klines:
            result.append({
                "日期": k[0],
                "开盘": float(k[1]),
                "收盘": float(k[2]),
                "最高": float(k[3]),
                "最低": float(k[4]),
                "成交量": int(float(k[5])),
            })

        print(f"[get_price_history] Tencent: {len(result)} klines for {code}")
        cache_set(cache_key, result)
        return result
    except Exception as e:
        print(f"[get_price_history] Tencent also failed: {e}")

    return []


# ---------- EastMoney Datacenter API ----------
EM_DATACENTER = "https://datacenter.eastmoney.com/securities/api/data/v1/get"

def _em_fetch(report_name, columns, filter_str, pagesize=8, sort_col="REPORTDATE"):
    """Generic EastMoney datacenter API fetch."""
    cache_key = f"em_{report_name}_{filter_str}"
    cached_val = cached(cache_key, ttl=600)
    if cached_val:
        return cached_val

    try:
        params = {
            "reportName": report_name,
            "columns": columns,
            "filter": filter_str,
            "pageNumber": 1,
            "pageSize": pagesize,
            "sortTypes": -1,
            "sortColumns": sort_col,
            "source": "HSF10",
            "client": "PC",
        }
        if report_name in ("RPT_DMSK_FN_BALANCE", "RPT_DMSK_FN_INCOME"):
            params["sortColumns"] = "REPORT_DATE"

        resp = _http_get(EM_DATACENTER, params=params, timeout=15)
        data = resp.json()
        if data.get("success"):
            records = data.get("result", {}).get("data", [])
            cache_set(cache_key, records)
            return records
        # If sort column fails, retry without sort
        if "排序列不存在" in data.get("message", ""):
            del params["sortColumns"]
            del params["sortTypes"]
            resp2 = _http_get(EM_DATACENTER, params=params, timeout=15)
            data2 = resp2.json()
            if data2.get("success"):
                records = data2.get("result", {}).get("data", [])
                cache_set(cache_key, records)
                return records
        return []
    except Exception as e:
        print(f"_em_fetch error ({report_name}): {e}")
        return []


# ========== Cross-Market Financial Data (yfinance) ==========
# NOTE: yfinance import at module level to avoid Flask-context threading issues

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

YF_CACHE = {}


def _fy_code(code, market):
    """Convert code to yfinance ticker format. 00700.HK→0700.HK (4-digit), AAPL.US→AAPL."""
    symbol = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "").replace(".HK", "").replace(".US", "")
    if market == "HK":
        # yfinance expects exactly 4-digit HK code: 0700.HK, 9988.HK, 0005.HK
        num = str(int(symbol))  # strip leading zeros
        padded = num.zfill(4)   # pad to 4 digits
        return f"{padded}.HK"
    return symbol  # US: plain ticker


def _fetch_financials_yfinance(code, market):
    """Fetch financial statements via yfinance (subprocess isolation to avoid Flask threading issues)."""
    cache_key = f"yf_{code}"
    if cache_key in YF_CACHE:
        return YF_CACHE[cache_key]

    if not _YF_AVAILABLE:
        return None

    try:
        ticker_sym = _fy_code(code, market)
        ticker = yf.Ticker(ticker_sym)
        inc = ticker.income_stmt
        bal = ticker.balance_sheet

        # yfinance data access may fail in Flask's parent process context;
        # fall back to subprocess isolation if DataFrames are empty
        if inc is not None and not inc.empty and bal is not None and not bal.empty:
            return _parse_yfinance_data(inc, bal, ticker_sym, ticker)

        # ---- Subprocess isolation fallback ----
        print(f"[yfinance] Flask-context empty, trying subprocess for {ticker_sym}", flush=True)
        import json, subprocess, os, sys as _sys
        py_code = f"""
import yfinance as yf, json, sys, os, traceback
try:
    # Clear yfinance cache
    yf.set_tz_cache_location(None)
    t = yf.Ticker('{ticker_sym}')
    inc = t.income_stmt
    bal = t.balance_sheet
    if hasattr(inc, 'empty'):
        empty_flag = inc.empty
    else:
        empty_flag = inc is None
    print(json.dumps({{'status': 'ok', 'inc_empty': empty_flag, 'bal_empty': bal.empty if hasattr(bal,'empty') else True, 'inc_type': str(type(inc).__name__) }}), flush=True)
    if empty_flag or (hasattr(bal,'empty') and bal.empty):
        sys.exit(0)
    inc_json = inc.to_dict()
    bal_json = bal.to_dict()
    print(json.dumps({{'income': inc_json, 'balance': bal_json}}), flush=True)
except Exception as e:
    print(json.dumps({{'status': 'error', 'msg': str(e), 'trace': traceback.format_exc()}}), flush=True)
"""
        result = subprocess.run(
            [_sys.executable, "-c", py_code],
            capture_output=True, text=True, timeout=60
        )

        if result.returncode != 0:
            print(f"[yfinance] Subprocess error: stderr={result.stderr[:300]}", flush=True)
            return None

        if not result.stdout.strip():
            print(f"[yfinance] Subprocess returned empty stdout", flush=True)
            return None

        data = json.loads(result.stdout.splitlines()[-1].strip())  # take last line (JSON data)
        if data.get("status") == "error":
            print(f"[yfinance] Subprocess error: {data.get('msg','?')[:200]}", flush=True)
            return None
        if data.get("inc_empty", True):
            print(f"[yfinance] Subprocess: yfinance returned empty DataFrames (rate-limited?)", flush=True)
            return None

        # Reconstruct DataFrames from JSON
        import pandas as pd
        inc_df = pd.DataFrame.from_dict(data["income"])
        bal_df = pd.DataFrame.from_dict(data["balance"])
        qinc_df = pd.DataFrame.from_dict(data.get("quarterly", {}))

        # Parse with the dedicated helper
        return _parse_yfinance_data(inc_df, bal_df, ticker_sym, ticker, qinc_df)

    except Exception as e:
        print(f"[yfinance] Financial fetch failed for {code} ({market}): {e}")
        return None


def _parse_yfinance_data(inc, bal, ticker_sym, ticker, quarterly_df=None):
    """Parse yfinance DataFrames into A-share-compatible financial dict."""
    # ---- Row name mappings ----
    def _find_row(df, candidates):
        for c in candidates:
            if c in df.index:
                return c
        return None

    rev_row = _find_row(inc, ["Total Revenue", "Operating Revenue"])
    ni_row = _find_row(inc, ["Net Income Common Stockholders", "Net Income"])
    eps_row = _find_row(inc, ["Diluted EPS", "Basic EPS"])
    cost_row = _find_row(inc, ["Cost Of Revenue", "Reconciled Cost Of Revenue"])
    asset_row = _find_row(bal, ["Total Assets"])
    liab_row = _find_row(bal, ["Total Liabilities Net Minority Interest", "Total Liabilities"])
    equity_row = _find_row(bal, ["Stockholders Equity", "Total Equity Gross Minority Interest"])

    if not rev_row or not ni_row or not asset_row:
        return None

    # ---- Build income list (annual reports, latest 5 years) ----
    income_cols = list(inc.columns)[:5]
    income = []
    for idx, col in enumerate(income_cols):
        rev = float(inc.loc[rev_row, col]) if rev_row in inc.index else 0
        ni = float(inc.loc[ni_row, col]) if ni_row in inc.index else 0
        eps = float(inc.loc[eps_row, col]) if eps_row and eps_row in inc.index else 0
        cost = float(inc.loc[cost_row, col]) if cost_row and cost_row in inc.index else 0
        equity_val = float(bal.loc[equity_row, col]) if equity_row and equity_row in bal.index and col in bal.columns else None

        roe = (ni / equity_val * 100) if equity_val and equity_val != 0 else 0
        rev_yoy = ni_yoy = 0
        if idx < len(income_cols) - 1:
            prev_col = income_cols[idx + 1]
            prev_rev = float(inc.loc[rev_row, prev_col]) if rev_row in inc.index else 0
            prev_ni = float(inc.loc[ni_row, prev_col]) if ni_row in inc.index else 0
            rev_yoy = ((rev - prev_rev) / prev_rev * 100) if prev_rev != 0 else 0
            ni_yoy = ((ni - prev_ni) / prev_ni * 100) if prev_ni != 0 else 0

        income.append({
            "报告期": str(col)[:10], "报告类型": "年报", "年份": str(col)[:4],
            "营业总收入": rev, "归母净利润": ni, "基本每股收益": eps,
            "加权ROE": round(roe, 2),
            "每股净资产": round(equity_val / (float(inc.loc["Diluted Average Shares", col]) if "Diluted Average Shares" in inc.index else 1), 2) if equity_val else 0,
            "每股经营现金流": 0,
            "营收同比": round(rev_yoy, 2), "净利同比": round(ni_yoy, 2),
        })

    # ---- Build balance list ----
    bal_cols = list(bal.columns)[:5]
    balance = []
    for col in bal_cols:
        assets = float(bal.loc[asset_row, col]) if asset_row in bal.index else 0
        liab = float(bal.loc[liab_row, col]) if liab_row and liab_row in bal.index else 0
        eq = float(bal.loc[equity_row, col]) if equity_row and equity_row in bal.index else 0
        balance.append({
            "报告期": str(col)[:10], "报告类型": "年报", "年份": str(col)[:4],
            "总资产": assets, "总负债": liab, "净资产": eq,
            "资产负债率": round((liab / assets * 100) if assets else 0, 2),
        })

    # ---- Quarterly income ----
    quarterly_inc = []
    if quarterly_df is not None and not quarterly_df.empty:
        for col in list(quarterly_df.columns)[:8]:
            if rev_row in quarterly_df.index and ni_row in quarterly_df.index:
                quarterly_inc.append({
                    "报告期": str(col)[:10], "报告类型": "季报",
                    "营业总收入": float(quarterly_df.loc[rev_row, col]),
                    "归母净利润": float(quarterly_df.loc[ni_row, col]),
                    "基本每股收益": float(quarterly_df.loc[eps_row, col]) if eps_row and eps_row in quarterly_df.index else 0,
                    "加权ROE": 0,
                })

    # ---- Gross/Net Margin ----
    latest_col = income_cols[0]
    rev_latest = float(inc.loc[rev_row, latest_col]) if rev_row in inc.index else 0
    cost_latest = float(inc.loc[cost_row, latest_col]) if cost_row and cost_row in inc.index else 0
    ni_latest = float(inc.loc[ni_row, latest_col]) if ni_row in inc.index else 0
    gross_margin = round((rev_latest - cost_latest) / rev_latest * 100, 2) if rev_latest and cost_latest else None
    net_margin = round(ni_latest / rev_latest * 100, 2) if rev_latest and rev_latest != 0 else None

    return {
        "income": income, "balance": balance, "quarterly": quarterly_inc,
        "gross_margin": gross_margin, "net_margin": net_margin,
        "_source": "yfinance",
    }


def _fetch_fundamentals_http(code, market, yahoo_symbol):
    """Fetch fundamentals via direct Yahoo Finance HTTP API (query1/v10)."""
    try:
        url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{yahoo_symbol}"
        params = {"modules": "financialData,defaultKeyStatistics,summaryDetail,incomeStatementHistory,balanceSheetHistory"}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = _http_get(url, params=params, timeout=15, headers_extra=headers)
        if resp.status_code != 200:
            return None
        data = resp.json()
        result = data.get("quoteSummary", {}).get("result", [])
        if not result: return None

        summary = result[0]
        income_hist = summary.get("incomeStatementHistory", {}).get("incomeStatementHistory", [])
        balance_hist = summary.get("balanceSheetHistory", {}).get("balanceSheetHistory", [])
        if not income_hist or not balance_hist: return None

        # Income
        income = []
        for entry in income_hist[:5]:
            e = entry.get("endDate", {}).get("fmt", "") or ""
            rev = (entry.get("totalRevenue", {}) or {}).get("raw", 0)
            ni = (entry.get("netIncomeToCommon", {}) or {}).get("raw", 0)
            income.append({"报告期": e[:10], "报告类型": "年报", "年份": e[:4],
                "营业总收入": rev, "归母净利润": ni, "基本每股收益": 0, "加权ROE": 0,
                "每股净资产": 0, "每股经营现金流": 0, "营收同比": 0, "净利同比": 0})
        for i in range(len(income) - 1):
            if income[i+1]["营业总收入"] and income[i]["营业总收入"]:
                income[i]["营收同比"] = round((income[i]["营业总收入"] - income[i+1]["营业总收入"]) / income[i+1]["营业总收入"] * 100, 2)
            if income[i+1]["归母净利润"] and income[i]["归母净利润"]:
                income[i]["净利同比"] = round((income[i]["归母净利润"] - income[i+1]["归母净利润"]) / abs(income[i+1]["归母净利润"]) * 100, 2)

        # Balance
        balance = []
        for entry in balance_hist[:5]:
            e = entry.get("endDate", {}).get("fmt", "") or ""
            assets = (entry.get("totalAssets", {}) or {}).get("raw", 0)
            liab = (entry.get("totalLiab", {}) or {}).get("raw", 0)
            eq = (entry.get("totalStockholderEquity", {}) or {}).get("raw", 0)
            balance.append({"报告期": e[:10], "报告类型": "年报", "年份": e[:4],
                "总资产": assets, "总负债": liab, "净资产": eq,
                "资产负债率": round((liab / assets * 100) if assets else 0, 2)})

        if income and balance:
            eq_l = balance[0].get("净资产", 0); ni_l = income[0].get("归母净利润", 0)
            if eq_l and ni_l: income[0]["加权ROE"] = round(ni_l / eq_l * 100, 2)

        print(f"[yahoo_http] OK for {yahoo_symbol}: {len(income)} income items", flush=True)
        return {"income": income, "balance": balance, "quarterly": [],
                "gross_margin": None, "net_margin": None, "_source": "yahoo_http"}
    except Exception as e:
        print(f"[yahoo_http] Failed: {e}", flush=True)
        return None


def _fetch_from_yf_info(code, market, ticker_sym):
    """Fetch key financial metrics from yfinance ticker.info (lightweight, no DataFrame issues).
    Uses a fresh requests Session to avoid Flask-context threading issues."""
    try:
        import yfinance as yf, requests as _req
        # Create a fresh Ticker with a clean session
        sess = _req.Session()
        sess.trust_env = False
        t = yf.Ticker(ticker_sym, session=sess)
        info = t.info or {}
        if not info or 'totalRevenue' not in info:
            return None

        # _to_float kills NaN/Inf returned by yfinance (e.g. missing returnOnEquity
        # comes back as float('nan'), and `nan or 0` is still nan in Python).
        rev = _to_float(info.get('totalRevenue'))
        ni = _to_float(info.get('netIncomeToCommon'))
        roe = _to_float(info.get('returnOnEquity')) * 100  # yfinance gives decimal
        gm = _to_float(info.get('grossMargins')) * 100
        nm = _to_float(info.get('profitMargins')) * 100
        rev_growth = _to_float(info.get('revenueGrowth')) * 100
        dte = _to_float(info.get('debtToEquity'))
        dar = (dte / (1 + dte) * 100) if dte else 0  # debt ratio from D/E
        bps = _to_float(info.get('bookValue'))
        eps = _to_float(info.get('trailingEps'))

        income = [{
            "报告期": now_cn().strftime("%Y-%m-%d"), "报告类型": "年报",
            "年份": str(now_cn().year), "营业总收入": rev, "归母净利润": ni,
            "基本每股收益": eps, "加权ROE": round(roe, 2),
            "每股净资产": bps, "每股经营现金流": 0,
            "营收同比": round(rev_growth, 2), "净利同比": 0,
        }]
        balance = [{
            "报告期": now_cn().strftime("%Y-%m-%d"), "报告类型": "年报",
            "年份": str(now_cn().year),
            "总资产": 0, "总负债": 0, "净资产": 0,
            "资产负债率": round(dar, 2),
        }]
        print(f"[yf_info] OK for {ticker_sym}: ROE={roe:.1f}% rev_growth={rev_growth:.1f}% GM={gm:.1f}%", flush=True)
        return {
            "income": income, "balance": balance, "quarterly": [],
            "gross_margin": round(gm, 2) if gm else None,
            "net_margin": round(nm, 2) if nm else None,
            "_source": "yf_info",
        }
    except Exception as e:
        print(f"[yf_info] Failed for {ticker_sym}: {e}", flush=True)
        return None


# ========== Cross-market API key helpers ==========

def _get_api_key(key_name):
    """Get API key from config.json data_source section."""
    try:
        cfg = load_config()
        ds = cfg.get("data_source", {})
        return ds.get(key_name, "") or ""
    except Exception:
        return ""


def _fetch_alpha_vantage_financials(code, market):
    """Fallback: Alpha Vantage free API (needs key in config.json)."""
    api_key = _get_api_key("alpha_vantage_key")
    if not api_key:
        return None
    try:
        symbol = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "").replace(".HK", "").replace(".US", "")
        url = "https://www.alphavantage.co/query"
        params = {"function": "OVERVIEW", "symbol": symbol, "apikey": api_key}
        resp = _http_get(url, params=params, timeout=15)
        data = resp.json()
        if "Error" in data or "Note" in data or not data.get("Symbol"):
            return None

        roe_r = float(data.get("ReturnOnEquityTTM", 0) or 0)  # in percent
        gm = float(data.get("GrossProfitTTM", 0) or 0)  # raw value
        rev = float(data.get("RevenueTTM", 0) or 0)
        ni = float(data.get("NetIncomeTTM", 0) or 0)
        eps = float(data.get("EPS", 0) or 0)
        rev_growth = float(data.get("QuarterlyRevenueGrowthYOY", 0) or 0) * 100

        income = [{
            "报告期": now_cn().strftime("%Y-%m-%d"), "报告类型": "年报",
            "年份": str(now_cn().year), "营业总收入": rev, "归母净利润": ni,
            "基本每股收益": eps, "加权ROE": round(roe_r, 2),
            "每股净资产": 0, "每股经营现金流": 0,
            "营收同比": round(rev_growth, 2), "净利同比": 0,
        }]
        return {"income": income, "balance": [], "quarterly": [],
                "gross_margin": round(gm / rev * 100, 2) if rev and gm else None,
                "net_margin": round(ni / rev * 100, 2) if rev and ni else None,
                "_source": "alpha_vantage"}
    except Exception as e:
        print(f"[alpha_vantage] Failed: {e}", flush=True)
        return None


def _fetch_fmp_financials(code, market):
    """Fallback: Financial Modeling Prep free API (needs key in config.json)."""
    api_key = _get_api_key("fmp_key")
    if not api_key:
        return None
    try:
        symbol = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "").replace(".HK", "").replace(".US", "")
        url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}"
        params = {"period": "annual", "limit": 3, "apikey": api_key}
        resp = _http_get(url, params=params, timeout=15)
        data = resp.json()
        if not data or isinstance(data, dict) and "Error" in data:
            return None

        income = []
        for entry in data[:3]:
            rev = entry.get("revenue", 0) or 0
            ni = entry.get("netIncome", 0) or 0
            eps = entry.get("eps", 0) or 0
            gross_profit = entry.get("grossProfit", 0) or 0
            income.append({
                "报告期": entry.get("date", "")[:10], "报告类型": "年报",
                "年份": entry.get("date", "")[:4],
                "营业总收入": rev, "归母净利润": ni, "基本每股收益": eps,
                "加权ROE": round(entry.get("roe", 0) or 0, 2),
                "每股净资产": 0, "每股经营现金流": 0,
                "营收同比": round(entry.get("revenueGrowth", 0) or 0, 2),
                "净利同比": round(entry.get("netIncomeGrowth", 0) or 0, 2),
            })

        latest = income[0] if income else {}
        rev_l = latest.get("营业总收入", 0)
        ni_l = latest.get("归母净利润", 0)
        gross_l = data[0].get("grossProfit", 0) if data else 0
        return {"income": income, "balance": [], "quarterly": [],
                "gross_margin": round(gross_l / rev_l * 100, 2) if rev_l and gross_l else None,
                "net_margin": round(ni_l / rev_l * 100, 2) if rev_l and ni_l else None,
                "_source": "fmp"}
    except Exception as e:
        print(f"[fmp] Failed: {e}", flush=True)
        return None


def _estimate_fundamentals_from_tencent(code, market):
    """Guaranteed fallback: estimate ROE & metrics from Tencent PE/PB (always available).
    ROE ≈ PB / PE × 100 — derived from: ROE = EPS/BVPS = (Price/PE)/(Price/PB) = PB/PE."""
    try:
        info = get_stock_info(code, market)
        if not info or not info.get("最新价"):
            return None
        pe = info.get("市盈率-动态", 0) or 0
        pb = info.get("市净率", 0) or 0
        price = info.get("最新价", 0) or 0

        # US PB fix: Tencent doesn't provide PB for US, try yfinance via subprocess (10s timeout)
        if pb == 0 and market == "US":
            try:
                import subprocess as _sp, json as _j
                ticker_yf = _fy_code(code, market)
                pb_result = _sp.run(
                    [sys.executable, "-c", f"import yfinance as yf; t=yf.Ticker('{ticker_yf}'); print(t.info.get('priceToBook', 0) or 0)"],
                    capture_output=True, text=True, timeout=12
                )
                if pb_result.returncode == 0 and pb_result.stdout.strip():
                    pb = float(pb_result.stdout.strip())
            except Exception:
                pass  # keep pb=0, will use sector default ROE

        # Guard: need at minimum PE and price
        if pe == 0 or price == 0:
            return None
        # PB may be 0 for US if yfinance also fails — use sector-default ROE
        if pb == 0:
            # Conservative sector defaults (科技15%, 金融10%, 其他12%)
            roe_est = 15.0 if market == "US" else 12.0
            bps_est = 0
        else:
            roe_est = round(pb / pe * 100, 2)
            bps_est = round(price / pb, 2)

        eps_est = round(price / pe, 2)

        income = [{
            "报告期": now_cn().strftime("%Y-%m-%d"), "报告类型": "年报",
            "年份": str(now_cn().year),
            "营业总收入": 0,  # cannot estimate
            "归母净利润": 0,  # cannot estimate from PE/PB alone
            "基本每股收益": eps_est,
            "加权ROE": roe_est,
            "每股净资产": bps_est,
            "每股经营现金流": 0,
            "营收同比": 0, "净利同比": 0,
        }]
        # Debt ratio estimation: from D/E ratio in info if available
        dar = 0
        if pe and pb and pe > 0 and pb > 0:
            # Conservative: assume median D/E = 1.0 → DAR = 50%
            dar = 50.0
        balance = [{
            "报告期": now_cn().strftime("%Y-%m-%d"), "报告类型": "年报",
            "年份": str(now_cn().year),
            "总资产": 0, "总负债": 0, "净资产": 0,
            "资产负债率": round(dar, 2),
        }]
        print(f"[tencent_est] ROE={roe_est:.1f}% EPS={eps_est:.2f} BPS={bps_est:.2f} for {code}", flush=True)
        return {
            "income": income, "balance": balance, "quarterly": [],
            "gross_margin": None, "net_margin": None,
            "_source": "tencent_estimation",
        }
    except Exception as e:
        print(f"[tencent_est] Failed: {e}", flush=True)
        return None


def _em_datacenter(report_name, filter_str, source="F10", v="01975982096513973",
                    pagesize=30, sort_col="STD_REPORT_DATE"):
    """Direct EastMoney datacenter API (no akshare, Vercel-safe). Returns list of row-dicts or []."""
    cache_key = f"emdc_{report_name}_{filter_str}"
    cv = cached(cache_key, ttl=600)
    if cv is not None:
        return cv
    try:
        params = {
            "reportName": report_name,
            "columns": "ALL",
            "quoteColumns": "",
            "filter": filter_str,
            "pageNumber": "1",
            "pageSize": str(pagesize),
            "sortTypes": "-1",
            "sortColumns": sort_col,
            "source": source,
            "client": "PC",
            "v": v,
        }
        resp = _http_get(EM_DATACENTER, params=params, timeout=15)
        data = resp.json()
        rows = (data.get("result") or {}).get("data") or []
        cache_set(cache_key, rows)
        return rows
    except Exception as e:
        print(f"[_em_datacenter] {report_name} failed: {e}")
        return []


def _fetch_em_hk_us_financials(code, market):
    """Primary: EastMoney HK/US financial indicators via datacenter HTTP (no akshare, Vercel-safe).
    HK: RPT_HKF10_FN_MAININDICATOR ; US: RPT_USF10_FN_GMAININDICATOR
    (SECUCODE resolved via RPT_USF10_INFO_ORGPROFILE)."""
    symbol = code.replace(".HK", "").replace(".US", "")
    try:
        if market == "HK":
            rows = _em_datacenter(
                "RPT_HKF10_FN_MAININDICATOR",
                f'(SECUCODE="{symbol}.HK")(DATE_TYPE_CODE="001")',
                source="F10", v="01975982096513973", pagesize=20,
                sort_col="STD_REPORT_DATE",
            )
        else:
            org = _em_datacenter(
                "RPT_USF10_INFO_ORGPROFILE",
                f'(SECURITY_CODE="{symbol}")',
                source="SECURITIES", v="04406064331266868", pagesize=5,
                sort_col="REPORT_DATE",
            )
            secucode = (org[0].get("SECUCODE", "") if org else "") or f"{symbol}.O"
            rows = _em_datacenter(
                "RPT_USF10_FN_GMAININDICATOR",
                f'(SECUCODE="{secucode}")(DATE_TYPE_CODE="001")',
                source="SECURITIES", v="04406064331266868", pagesize=30,
                sort_col="REPORT_DATE",
            )

        if not rows:
            return None

        # Defensive annual-only filter (query already restricts 001)
        rows = [r for r in rows if r.get("DATE_TYPE_CODE") in (None, "", "001")]

        if market == "HK":
            rev_col = "OPERATE_INCOME"
            ni_col = "HOLDER_PROFIT"
            eps_col = "BASIC_EPS"
            bps_col = "BPS"
            yoy_ni = "HOLDER_PROFIT_YOY"
        else:
            rev_col = "OPERATE_INCOME"
            ni_col = "PARENT_HOLDER_NETPROFIT"
            eps_col = "BASIC_EPS"
            bps_col = None  # US doesn't have BPS in this dataset
            yoy_ni = "PARENT_HOLDER_NETPROFIT_YOY"

        income = []
        for row in rows:
            report_date = str(row.get("REPORT_DATE", "") or "")
            rev = float(row.get(rev_col, 0) or 0)
            ni = float(row.get(ni_col, 0) or 0)
            eps = float(row.get(eps_col, 0) or 0)
            bps = float(row.get(bps_col, 0) or 0) if bps_col else 0
            roe = float(row.get("ROE_AVG", 0) or 0)
            rev_yoy = float(row.get("OPERATE_INCOME_YOY", 0) or 0)
            ni_yoy = float(row.get(yoy_ni, 0) or 0)

            income.append({
                "报告期": report_date[:10] if report_date else "",
                "报告类型": "年报",
                "年份": report_date[:4] if report_date else "",
                "营业总收入": rev,
                "归母净利润": ni,
                "基本每股收益": eps,
                "加权ROE": round(roe, 2),
                "每股净资产": round(bps, 2),
                "每股经营现金流": 0,
                "营收同比": round(rev_yoy, 2),
                "净利同比": round(ni_yoy, 2),
            })

        latest = income[0] if income else {}
        gm = float(rows[0].get("GROSS_PROFIT_RATIO", 0) or 0)
        nm = float(rows[0].get("NET_PROFIT_RATIO", 0) or 0)
        dar = float(rows[0].get("DEBT_ASSET_RATIO", 0) or 0)

        balance = [{
            "报告期": income[0]["报告期"] if income else "",
            "报告类型": "年报",
            "年份": income[0]["年份"] if income else "",
            "总资产": 0, "总负债": 0, "净资产": 0,
            "资产负债率": round(dar, 2),
        }]

        print(f"[em_http] OK for {code}: {len(income)} years, ROE={latest.get('加权ROE','?')}%, GM={gm}%", flush=True)
        return {
            "income": income, "balance": balance, "quarterly": [],
            "gross_margin": round(gm, 2) if gm else None,
            "net_margin": round(nm, 2) if nm else None,
            "_source": "eastmoney_http",
        }
    except Exception as e:
        print(f"[em_http] Failed for {code}: {e}", flush=True)
        return None




def get_financial_data(code, market="A"):
    """Fetch financial statements. A-share: EastMoney direct. HK/US: AKShare(EastMoney) → Tencent estimation."""
    if market in ("HK", "US"):
        # L1: AKShare EastMoney global — real financial data, no API key needed
        result = _fetch_em_hk_us_financials(code, market)
        if result: return result

        # L2: Tencent PE/PB estimation — 100% guaranteed, always instant
        result = _estimate_fundamentals_from_tencent(code, market)
        if result: return result
        return {"income": [], "balance": [], "quarterly": [], "gross_margin": None, "net_margin": None, "_source": "empty"}

    symbol = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "")
    security_filter = f'(SECURITY_TYPE_CODE="058001001")(SECURITY_CODE="{symbol}")'

    income_cols = (
        "SECURITY_CODE,SECURITY_NAME_ABBR,NOTICE_DATE,REPORTDATE,DATATYPE,"
        "DATEMMDD,DATAYEAR,BASIC_EPS,DEDUCT_BASIC_EPS,"
        "TOTAL_OPERATE_INCOME,PARENT_NETPROFIT,WEIGHTAVG_ROE,BPS,MGJYXJJE,"
        "YSHZ,YSTZ,SJLHZ,SJLTZ"
    )
    income_raw = _em_fetch("RPT_LICO_FN_CPD", income_cols, security_filter, pagesize=12)

    balance_cols = (
        "SECURITY_CODE,SECURITY_NAME_ABBR,REPORT_DATE,DATE_TYPE_CODE,NOTICE_DATE,"
        "TOTAL_ASSETS,TOTAL_LIABILITIES,TOTAL_EQUITY,DEBT_ASSET_RATIO"
    )
    balance_raw = _em_fetch("RPT_DMSK_FN_BALANCE", balance_cols, security_filter, pagesize=8)

    income = []
    for r in income_raw:
        dmm = r.get("DATEMMDD", "")
        if dmm == "年报":
            income.append({
                "报告期": r.get("REPORTDATE", "")[:10] if r.get("REPORTDATE") else "",
                "报告类型": dmm,
                "年份": r.get("DATAYEAR", ""),
                "营业总收入": r.get("TOTAL_OPERATE_INCOME"),
                "归母净利润": r.get("PARENT_NETPROFIT"),
                "基本每股收益": r.get("BASIC_EPS"),
                "扣非每股收益": r.get("DEDUCT_BASIC_EPS"),
                "加权ROE": r.get("WEIGHTAVG_ROE"),
                "每股净资产": r.get("BPS"),
                "每股经营现金流": r.get("MGJYXJJE"),
                "营收同比": r.get("YSTZ"),
                "净利同比": r.get("SJLTZ"),
            })

    balance = []
    for r in balance_raw:
        if r.get("DATE_TYPE_CODE") == "001":
            report_date = r.get("REPORT_DATE", "")[:10] if r.get("REPORT_DATE") else ""
            year = report_date[:4] if report_date else ""
            balance.append({
                "报告期": report_date,
                "报告类型": "年报",
                "年份": year,
                "总资产": r.get("TOTAL_ASSETS"),
                "总负债": r.get("TOTAL_LIABILITIES"),
                "净资产": r.get("TOTAL_EQUITY"),
                "资产负债率": r.get("DEBT_ASSET_RATIO"),
            })

    quarterly_income = []
    for r in income_raw:
        dmm = r.get("DATEMMDD", "")
        if dmm != "年报":
            quarterly_income.append({
                "报告期": r.get("REPORTDATE", "")[:10] if r.get("REPORTDATE") else "",
                "报告类型": r.get("DATEMMDD", ""),
                "营业总收入": r.get("TOTAL_OPERATE_INCOME"),
                "归母净利润": r.get("PARENT_NETPROFIT"),
                "基本每股收益": r.get("BASIC_EPS"),
                "每股净资产": r.get("BPS"),
                "加权ROE": r.get("WEIGHTAVG_ROE"),
            })

    # ---- Calculate gross/net margins from annual income statement ----
    # Use RPT_DMSK_FN_INCOME with annual filter (DATE_TYPE_CODE="001") to get
    # the latest annual report's raw figures, then calculate margins ourselves.
    # This matches Tongdaxin F10's methodology (annual, not latest quarterly).
    gross_margin = None
    net_margin = None
    annual_filter = f'{security_filter}(DATE_TYPE_CODE="001")'
    income_detail_raw = _em_fetch(
        "RPT_DMSK_FN_INCOME",
        "TOTAL_OPERATE_INCOME,OPERATE_COST,PARENT_NETPROFIT",
        annual_filter,
        pagesize=1,
        sort_col="REPORT_DATE",
    )
    if income_detail_raw:
        ar = income_detail_raw[0]
        ti = ar.get("TOTAL_OPERATE_INCOME")
        oc = ar.get("OPERATE_COST")
        pn = ar.get("PARENT_NETPROFIT")
        if ti and float(ti) != 0 and oc is not None:
            gm_val = (float(ti) - float(oc)) / float(ti) * 100
            gross_margin = round(gm_val, 2)
        if ti and float(ti) != 0 and pn is not None:
            nm_val = float(pn) / float(ti) * 100
            net_margin = round(nm_val, 2)

    # Fallback: if annual income statement fetch failed, use pre-calculated XSMLL/XSJLL
    # (Note: these return latest quarterly data, which may differ from annual)
    if gross_margin is None or net_margin is None:
        margins_raw = _em_fetch(
            "RPT_F10_FINANCE_MAINFINADATA",
            "XSMLL,XSJLL",
            security_filter,
            pagesize=1,
        )
        if margins_raw:
            mr = margins_raw[0]
            if gross_margin is None:
                gm = mr.get("XSMLL")
                if gm is not None and float(gm) != 0:
                    gross_margin = round(float(gm), 2)
            if net_margin is None:
                nm = mr.get("XSJLL")
                if nm is not None and float(nm) != 0:
                    net_margin = round(float(nm), 2)

    return {
        "income": income,
        "balance": balance,
        "quarterly": quarterly_income,
        "gross_margin": gross_margin,
        "net_margin": net_margin,
    }


def _sina_money_flow(code):
    """A-share money flow history via Sina (fallback for when EastMoney push2his is down).

    Sina endpoint `MoneyFlow.ssl_qsfx_zjlrqs` returns multi-year daily history with:
      - netamount = 主力净额 (元)
      - r0_net    = 超大单净额 (元)
    We derive 大单 ≈ 主力-超大单, 散户 ≈ -主力 (净额守恒近似). 新浪无中单拆分 -> 0.
    Returns the same dict shape as get_money_flow, or None on failure.
    """
    try:
        symbol = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "")
        if code.endswith(".SH") or symbol.startswith(("6", "9")):
            daima = f"sh{symbol}"
        elif code.endswith(".BJ") or symbol.startswith(("4", "8")):
            daima = f"bj{symbol}"
        else:
            daima = f"sz{symbol}"
        url = ("http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
               f"MoneyFlow.ssl_qsfx_zjlrqs?daima={daima}")
        resp = _http_get(url, timeout=12)
        rows = json.loads(resp.text)
        if not rows or not isinstance(rows, list):
            return None
        # Sina returns newest-first; reverse to oldest-first (align with EastMoney klines).
        rows = list(reversed(rows))
        records = []
        for r in rows:
            try:
                main = float(r.get("netamount") or 0)
                r0 = float(r.get("r0_net") or 0)
            except (TypeError, ValueError):
                continue
            records.append({
                "date": r.get("opendate", ""),
                "main_net": main,               # 主力净流入 (元)
                "retail_net": -main,            # 散户 ≈ -主力 (净额守恒近似)
                "medium_net": 0,                # 新浪无中单拆分
                "large_net": main - r0,         # 大单 ≈ 主力 - 超大单
                "super_large_net": r0,          # 超大单
            })
        if not records:
            return None
        recent = records[-20:]
        last5 = recent[-5:] if len(recent) >= 5 else recent
        result = {
            "records": records[-60:],
            "last5": last5,
            "latest": recent[-1] if recent else None,
            "summary": {
                "main_5d_net": round(sum(x["main_net"] for x in last5) / 1e4, 2),
                "super_large_5d_net": round(sum(x["super_large_net"] for x in last5) / 1e4, 2),
                "large_5d_net": round(sum(x["large_net"] for x in last5) / 1e4, 2),
                "medium_5d_net": 0,
                "retail_5d_net": round(sum(x["retail_net"] for x in last5) / 1e4, 2),
                "main_inflow_days": sum(1 for x in last5 if x["main_net"] > 0),
                "main_outflow_days": sum(1 for x in last5 if x["main_net"] < 0),
                "trend": "流入为主" if sum(1 for x in recent[-10:] if x["main_net"] > 0) >= 6
                         else "流出为主" if sum(1 for x in recent[-10:] if x["main_net"] < 0) >= 6
                         else "震荡平衡",
            },
            "source": "sina",
            "note": "东方财富资金流暂不可用，已切换新浪历史资金流（无中单拆分）",
        }
        return result
    except Exception as e:
        print(f"[SINA] Money flow error for {code}: {e}")
        return None


def get_money_flow(code):
    """Daily capital flow via EastMoney fund flow API: main/super-large/large/medium/retail net flow."""
    cache_key = f"flow_{code}"
    cached_val = cached(cache_key, ttl=300)
    if cached_val:
        return cached_val

    try:
        symbol = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "")
        # secid: 0=SH, 1=SZ (actually EastMoney uses 1 for SH, 0 for SZ for real-time)
        # But for his (historical) API it's 0.SHxxxxx or 1.SZxxxxx
        market = "0" if code.endswith(".SZ") else "1"
        secid = f"{market}.{symbol}"

        url = "http://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
        params = {
            "lmt": "0",
            "klt": "1",
            "secid": secid,
            "fields1": "f1,f2,f3,f7",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
        }
        resp = _http_get(url, params=params, timeout=15)
        data = resp.json()

        if not data.get("data") or not data["data"].get("klines"):
            cache_set(cache_key, {"error": "无资金流向数据"})
            return {"error": "无资金流向数据"}

        klines = data["data"]["klines"]
        records = []
        for line in klines:
            parts = line.split(",")
            if len(parts) >= 7:
                records.append({
                    "date": parts[0],
                    "main_net": float(parts[1]) if parts[1] != "-" else 0,       # 主力净流入
                    "retail_net": float(parts[2]) if parts[2] != "-" else 0,      # 小单净流入(散户)
                    "medium_net": float(parts[3]) if parts[3] != "-" else 0,      # 中单净流入
                    "large_net": float(parts[4]) if parts[4] != "-" else 0,       # 大单净流入
                    "super_large_net": float(parts[5]) if parts[5] != "-" else 0, # 超大单净流入
                })

        # Summary: last 5 days
        recent = records[-20:]  # last 20 trading days
        last5 = recent[-5:] if len(recent) >= 5 else recent

        result = {
            "records": records[-60:],  # last 60 days for trend
            "last5": last5,
            "latest": recent[-1] if recent else None,
            "summary": {
                "main_5d_net": round(sum(r["main_net"] for r in last5) / 1e4, 2),
                "super_large_5d_net": round(sum(r["super_large_net"] for r in last5) / 1e4, 2),
                "large_5d_net": round(sum(r["large_net"] for r in last5) / 1e4, 2),
                "retail_5d_net": round(sum(r["retail_net"] for r in last5) / 1e4, 2),
                "main_inflow_days": sum(1 for r in last5 if r["main_net"] > 0),
                "main_outflow_days": sum(1 for r in last5 if r["main_net"] < 0),
                "trend": "流入为主" if sum(1 for r in recent[-10:] if r["main_net"] > 0) >= 6
                         else "流出为主" if sum(1 for r in recent[-10:] if r["main_net"] < 0) >= 6
                         else "震荡平衡",
            },
        }
        cache_set(cache_key, result)
        return result
    except Exception as e:
        print(f"[PRIMARY] Money flow error for {code}: {e}")

    # ---- Fallback 1: Sina money flow history (multi-year daily) ----
    sina = _sina_money_flow(code)
    if sina and sina.get("records"):
        print(f"[FALLBACK] Sina money flow -> OK: {len(sina['records'])} records for {code}")
        cache_set(cache_key, sina)
        return sina

    # ---- Fallback 2: Graceful degradation ----
    fb_result = api_fallback.graceful_money_flow(code)
    cache_set(cache_key, fb_result)
    return fb_result


def get_news_events(code):
    """Recent stock announcements via EastMoney notice API."""
    cache_key = f"news_{code}"
    cached_val = cached(cache_key, ttl=600)
    if cached_val:
        return cached_val

    try:
        symbol = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "")
        url = "https://np-anotice-stock.eastmoney.com/api/security/ann"
        params = {
            "sr": "-1",
            "page_size": 20,
            "page_index": 1,
            "ann_type": "A",
            "client_source": "web",
            "stock_list": symbol,
        }
        resp = _http_get(url, params=params, timeout=15)
        data = resp.json()

        if not data.get("data") or not data["data"].get("list"):
            cache_set(cache_key, [])
            return []

        items = data["data"]["list"]
        events = []
        for item in items[:15]:
            title = item.get("title", "")
            # Skip routine boring announcements
            skip_keywords = ["独立董事", "审计报告", "内部控制", "董事会议", "监事会", "募集资金"]
            if any(kw in title for kw in skip_keywords):
                continue
            art_code = item.get("art_code", "")
            # Construct EastMoney notice detail URL
            notice_url = f"https://data.eastmoney.com/notices/detail/{symbol}/{art_code}.html" if art_code else ""
            events.append({
                "date": (item.get("notice_date") or "")[:10],
                "title": title,
                "type": item.get("ann_type_name", "公告"),
                "url": notice_url,
            })

        # Categorize with expanded keyword library
        # Weighted keywords: (keyword, weight) where weight is importance
        positive_kw = [
            ("重大资产重组", 5), ("借壳上市", 5), ("摘帽", 5), ("业绩大涨", 4),
            ("增持", 3), ("回购完成", 4), ("回购实施结果", 4), ("回购股份实施", 4), ("回购注销", 4), ("股份注销", 3), ("回购", 3), ("股权激励", 3), ("中标", 3),
            ("战略合作", 3), ("签署重大合同", 3), ("获得订单", 3),
            ("高分红", 3), ("高送转", 2), ("派息", 2), ("权益分派", 2), ("分红", 2),
            ("业绩预增", 3), ("业绩预告.*增长", 3), ("扭亏为盈", 4),
            ("新产品获批", 3), ("获得专利", 2), ("技术突破", 3),
            ("产能扩张", 2), ("新建项目", 2), ("投产", 2),
            ("政府补贴", 2), ("减税", 2), ("政策利好", 3),
            ("机构调研", 2), ("券商推荐", 2), ("买入评级", 2),
            ("外资增持", 3), ("北向资金增持", 3),
            ("送股", 2), ("转增", 2),
            ("业绩说明会", 1), ("投资者关系", 1),
            ("超预期", 3), ("好于预期", 3), ("上调评级", 2),
            ("获得资质", 2), ("入选", 2),
            ("签署协议", 2), ("合作协议", 2), ("共同投资", 2),
            ("配股", 1), ("可转债", 1),
            ("业绩快报.*增长", 2), ("营收增长", 2), ("利润增长", 2),
            ("投资者调研", 1), ("接待调研", 1),
            ("设立", 1), ("投资设立", 1), ("设立子公司", 2),
        ]
        negative_kw = [
            ("退市风险", 5), ("ST警告", 5), ("*ST", 5), ("暂停上市", 5),
            ("减持", 3), ("大股东减持", 4), ("清仓减持", 5),
            ("亏损", 3), ("业绩预亏", 4), ("业绩预告.*下降", 3),
            ("商誉减值", 4), ("资产减值", 3), ("计提减值", 3),
            ("处罚", 4), ("罚款", 3), ("立案调查", 5), ("证监会监管", 4),
            ("诉讼", 3), ("仲裁", 3), ("债务违约", 5), ("债券违约", 5),
            ("资金占用", 4), ("违规担保", 4), ("财务造假", 5),
            ("限售解禁", 2), ("大额解禁", 3),
            ("高管辞职", 3), ("高管减持", 3), ("实际控制人变更", 3),
            ("重组失败", 4), ("终止重组", 4), ("撤回申请", 3),
            ("业绩下滑", 3), ("增速放缓", 2), ("不及预期", 3),
            ("下调评级", 2), ("目标价下调", 2),
            ("停工", 3), ("停产", 4), ("客户流失", 3),
            ("原材料涨价", 2), ("成本上升", 2),
            ("担保", 2), ("质押", 2), ("冻结", 3),
            ("违约", 4), ("到期未偿付", 4),
            ("退市", 5), ("破产", 5), ("债务危机", 5),
            ("澄清", -1), ("风险提示", 1), ("异常波动", 1),
            ("修正业绩", 2), ("业绩快报.*下降", 2),
            ("被实施", 4), ("退市风险警示", 5),
        ]
        
        for evt in events:
            title = evt["title"]
            pos_score = 0
            neg_score = 0
            for kw, w in positive_kw:
                try:
                    if re.search(kw, title):
                        pos_score += w
                except:
                    if kw in title:
                        pos_score += w
            for kw, w in negative_kw:
                try:
                    if re.search(kw, title):
                        neg_score += w
                except:
                    if kw in title:
                        neg_score += w
            
            if pos_score > neg_score:
                evt["sentiment"] = "positive"
                evt["sentiment_score"] = min(pos_score, 5)
            elif neg_score > pos_score:
                evt["sentiment"] = "negative"
                evt["sentiment_score"] = min(neg_score, 5)
            else:
                evt["sentiment"] = "neutral"
                evt["sentiment_score"] = 0

        cache_set(cache_key, events)
        return events
    except Exception as e:
        return []


def get_industry_data(code):
    """Fetch industry classification and key comparison data."""
    cache_key = f"ind_{code}"
    cached_val = cached(cache_key, ttl=1800)
    if cached_val:
        return cached_val

    symbol = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "")
    security_filter = f'(SECURITY_TYPE_CODE="058001001")(SECURITY_CODE="{symbol}")'

    # 1. Get industry classification
    ind_raw = _em_fetch(
        "RPT_F10_ORG_BASICINFO",
        "CSRC_INDUSTRY_NAME,BOARD_NAME_2LEVEL,BOARD_CODE_BK_2LEVEL,BOARD_CODE_BK_1LEVEL,TRADE_MARKET",
        security_filter, pagesize=1
    )

    ind_name = ""
    board_name = ""
    board_code = ""
    if ind_raw:
        ind_name = ind_raw[0].get("CSRC_INDUSTRY_NAME", "") or ""
        board_name = ind_raw[0].get("BOARD_NAME_2LEVEL", "") or ""
        board_code = ind_raw[0].get("BOARD_CODE_BK_2LEVEL", "") or ""

    # 2. Get dividend history
    div_raw = _em_fetch(
        "RPT_SHAREBONUS_DET",
        "EX_DIVIDEND_DATE,PRETAX_BONUS_RMB,DIVIDENT_RATIO,PLAN_NOTICE_DATE",
        f'(SECURITY_CODE="{symbol}")', pagesize=10, sort_col="NOTICE_DATE"
    )

    dividends = []
    for d in div_raw[:5]:  # top 5 most recent
        ex_date = (d.get("EX_DIVIDEND_DATE") or "")[:10]
        bonus = d.get("PRETAX_BONUS_RMB")
        div_ratio = d.get("DIVIDENT_RATIO")
        dividends.append({
            "ex_date": ex_date,
            "cash_per_share": float(bonus) / 10.0 if bonus else 0,
            "dividend_ratio": float(div_ratio) * 100 if div_ratio else 0,
        })

    result = {
        "industry_name": ind_name,
        "board_name": board_name,
        "board_code": board_code,
        "dividends": dividends,
        "latest_dividend": {
            "date": dividends[0]["ex_date"] if dividends else "",
            "cash_per_share": dividends[0]["cash_per_share"] if dividends else 0,
            "dividend_ratio": dividends[0]["dividend_ratio"] if dividends else 0,
        } if dividends else None,
    }

    cache_set(cache_key, result)
    return result


# ---------- Analysis Functions ----------

# Industry-specific PE/PB benchmark ranges (CSRC industry classification)
def _industry_pe_range(industry_name, board_name=None):
    """Return (low, high, pb_low, pb_high, roe_avg, roe_good) for given industry.

    Matches on BOTH industry_name (e.g. "制造业-酒、饮料和精制茶制造业")
    and board_name (e.g. "白酒Ⅱ"). Some A-share industry_name strings do NOT
    contain the short keyword (e.g. 白酒), but board_name does -- previously this
    fell through to the generic default (pb_high=4.0) and wrongly flagged high-PB
    names like 茅台 as "PB偏高".
    """
    mapping = {
        "金融":    {"low": 5,  "high": 15, "pb_low": 0.5, "pb_high": 2.0, "roe_avg": 8,  "roe_good": 12},
        "银行":    {"low": 4,  "high": 10, "pb_low": 0.3, "pb_high": 1.5, "roe_avg": 8,  "roe_good": 12},
        "保险":    {"low": 8,  "high": 20, "pb_low": 0.8, "pb_high": 2.5, "roe_avg": 10, "roe_good": 15},
        "证券":    {"low": 12, "high": 30, "pb_low": 0.8, "pb_high": 3.0, "roe_avg": 6,  "roe_good": 10},
        "白酒":    {"low": 15, "high": 35, "pb_low": 2.0, "pb_high": 10.0,"roe_avg": 15, "roe_good": 25},
        "食品":    {"low": 12, "high": 30, "pb_low": 1.5, "pb_high": 6.0, "roe_avg": 10, "roe_good": 18},
        "医药":    {"low": 15, "high": 40, "pb_low": 1.5, "pb_high": 6.0, "roe_avg": 8,  "roe_good": 15},
        "医疗器械":{"low": 18, "high": 45, "pb_low": 2.0, "pb_high": 7.0, "roe_avg": 10, "roe_good": 18},
        "电子":    {"low": 15, "high": 40, "pb_low": 1.5, "pb_high": 5.0, "roe_avg": 8,  "roe_good": 15},
        "半导体":  {"low": 20, "high": 60, "pb_low": 2.0, "pb_high": 8.0, "roe_avg": 8,  "roe_good": 15},
        "计算机":  {"low": 20, "high": 55, "pb_low": 1.5, "pb_high": 6.0, "roe_avg": 6,  "roe_good": 12},
        "软件":    {"low": 18, "high": 50, "pb_low": 2.0, "pb_high": 7.0, "roe_avg": 6,  "roe_good": 12},
        "通信":    {"low": 15, "high": 35, "pb_low": 1.2, "pb_high": 4.0, "roe_avg": 6,  "roe_good": 12},
        "新能源":  {"low": 12, "high": 35, "pb_low": 1.5, "pb_high": 5.0, "roe_avg": 8,  "roe_good": 15},
        "电池":    {"low": 15, "high": 40, "pb_low": 1.5, "pb_high": 5.0, "roe_avg": 10, "roe_good": 18},
        "电力":    {"low": 10, "high": 25, "pb_low": 0.8, "pb_high": 3.0, "roe_avg": 6,  "roe_good": 10},
        "石油":    {"low": 8,  "high": 20, "pb_low": 0.6, "pb_high": 2.5, "roe_avg": 8,  "roe_good": 12},
        "化工":    {"low": 10, "high": 28, "pb_low": 1.0, "pb_high": 4.0, "roe_avg": 8,  "roe_good": 15},
        "钢铁":    {"low": 5,  "high": 15, "pb_low": 0.4, "pb_high": 2.0, "roe_avg": 5,  "roe_good": 10},
        "煤炭":    {"low": 5,  "high": 15, "pb_low": 0.5, "pb_high": 2.5, "roe_avg": 8,  "roe_good": 15},
        "有色金属":{"low": 12, "high": 30, "pb_low": 1.0, "pb_high": 4.0, "roe_avg": 8,  "roe_good": 15},
        "房地产":  {"low": 5,  "high": 15, "pb_low": 0.3, "pb_high": 1.5, "roe_avg": 5,  "roe_good": 10},
        "建筑":    {"low": 5,  "high": 15, "pb_low": 0.5, "pb_high": 2.0, "roe_avg": 6,  "roe_good": 12},
        "交通运输":{"low": 8,  "high": 20, "pb_low": 0.8, "pb_high": 3.0, "roe_avg": 6,  "roe_good": 12},
        "汽车":    {"low": 10, "high": 30, "pb_low": 1.0, "pb_high": 4.0, "roe_avg": 8,  "roe_good": 15},
        "家电":    {"low": 10, "high": 25, "pb_low": 1.0, "pb_high": 4.0, "roe_avg": 10, "roe_good": 18},
        "传媒":    {"low": 10, "high": 30, "pb_low": 1.0, "pb_high": 4.0, "roe_avg": 5,  "roe_good": 10},
        "军工":    {"low": 20, "high": 55, "pb_low": 1.5, "pb_high": 5.0, "roe_avg": 5,  "roe_good": 10},
        "农业":    {"low": 10, "high": 30, "pb_low": 1.0, "pb_high": 4.0, "roe_avg": 5,  "roe_good": 10},
        "环保":    {"low": 12, "high": 30, "pb_low": 1.0, "pb_high": 3.5, "roe_avg": 6,  "roe_good": 12},
    }
    # Match by keyword (industry_name OR board_name)
    _search = f"{industry_name or ''} {board_name or ''}"
    for key, val in mapping.items():
        if key in _search:
            return val
    # Default: general industry
    return {"low": 10, "high": 30, "pb_low": 1.0, "pb_high": 4.0, "roe_avg": 8, "roe_good": 15}


def _a_share_pe_pb(financial, price):
    """A-share PE(动态) & PB from EastMoney QUARTERLY financials (own data, no extra deps).

    PB uses the LATEST reported-quarter BPS (RPT_LICO_FN_CPD raw `BPS`),
    which matches 同花顺 caliber (latest-quarter BPS). The quote's 市净率
    field[46] is stale/higher (annual-ish BPS) and understates PB.

    PE(动态) = price / (latest-quarter STANDALONE EPS * 4). Quarterly
    BASIC_EPS in RPT_LICO_FN_CPD is cumulative YTD, so standalone =
    diff vs prior quarter (Q1 is standalone by definition). This matches
    同花顺 市盈率(动) exactly (e.g. 茅台 14.4 vs 同花顺 14.37).

    Returns (pe, pb); either may be None if not computable -> caller keeps fallback.
    """
    pe = None
    pb = None
    try:
        q = [x for x in (financial.get("quarterly") or []) if x.get("报告期")]
        q.sort(key=lambda x: x["报告期"], reverse=True)
        if q and price > 0:
            # PB from latest-quarter BPS
            _bps = q[0].get("每股净资产") or 0
            if isinstance(_bps, (int, float)) and _bps > 0:
                pb = round(price / float(_bps), 2)
            # PE(动态) from latest-quarter standalone EPS annualized
            _lm = int(str(q[0]["报告期"])[5:7] or 0)  # month: 3/6/9
            _eps = float(q[0].get("基本每股收益") or 0)
            _dyn = 0.0
            if _eps > 0:
                if _lm == 3:
                    _dyn = _eps * 4
                elif _lm in (6, 9) and len(q) >= 2:
                    _prev = float(q[1].get("基本每股收益") or 0)
                    _dyn = (_eps - _prev) * 4
                # Q4 is in annual (income), not quarterly -> skip
            if _dyn > 0:
                pe = round(price / _dyn, 2)
    except Exception as e:
        print(f"[_a_share_pe_pb] skip: {e}", flush=True)
    return pe, pb


def calc_ma(prices, window):
    """Calculate moving average."""
    if len(prices) < window:
        return None
    return round(sum(prices[-window:]) / window, 2)


def calc_macd(prices):
    """Calculate MACD."""
    if len(prices) < 26:
        return None, None, None
    ema12 = pd.Series(prices).ewm(span=12, adjust=False).mean().iloc[-1]
    ema26 = pd.Series(prices).ewm(span=26, adjust=False).mean().iloc[-1]
    dif = round(ema12 - ema26, 4)
    # Need history for DEA
    ema12_series = pd.Series(prices).ewm(span=12, adjust=False).mean()
    ema26_series = pd.Series(prices).ewm(span=26, adjust=False).mean()
    dif_series = ema12_series - ema26_series
    dea = round(dif_series.ewm(span=9, adjust=False).mean().iloc[-1], 4)
    macd_bar = round(2 * (dif - dea), 4)
    return dif, dea, macd_bar


def calc_kdj(prices, highs, lows):
    """Calculate KDJ."""
    if len(prices) < 9:
        return None, None, None
    n = 9
    low_n = min(lows[-n:])
    high_n = max(highs[-n:])
    if high_n == low_n:
        return 50, 50, 50
    rsv = (prices[-1] - low_n) / (high_n - low_n) * 100

    # Simplified: use single-point calculation
    k = rsv * 1/3 + 50 * 2/3
    d = k * 1/3 + 50 * 2/3
    j = 3 * k - 2 * d
    return round(k, 2), round(d, 2), round(j, 2)


def calc_bollinger(prices, window=20):
    """Calculate Bollinger Bands."""
    if len(prices) < window:
        return None, None, None
    ma = np.mean(prices[-window:])
    std = np.std(prices[-window:], ddof=1)
    upper = round(ma + 2 * std, 2)
    lower = round(ma - 2 * std, 2)
    middle = round(ma, 2)
    return upper, middle, lower


def analyze_stock(code):
    """Main analysis function - generates comprehensive report. Supports A股/HK/US."""
    # Clean code & detect market
    code = code.strip().upper()

    # ---- Market Detection ----
    market, bare_code = detect_market(code)
    cfg = MARKET_CONFIG.get(market, MARKET_CONFIG["A"])

    # A-share auto-suffix
    if market == "A" and not any(s in code for s in cfg["suffixes"]):
        if bare_code.startswith(("6", "9")):
            code = f"{bare_code}.SH"
        elif bare_code.startswith(("0", "3")):
            code = f"{bare_code}.SZ"
        elif bare_code.startswith(("4", "8")):
            code = f"{bare_code}.BJ"
        else:
            return {"error": f"无法识别股票代码: {code}。A股请输入如 000607.SZ；港股如 00700.HK；美股如 AAPL.US"}
    elif market == "HK" and not code.endswith(".HK"):
        code = f"{bare_code}.HK"
    elif market == "US" and not code.endswith(".US"):
        code = f"{bare_code}.US"

    symbol = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "").replace(".HK", "").replace(".US", "")

    # ---- Init warnings list for fallback tracking ----
    warnings = []

    # Fetch all data (market-aware)
    info = get_stock_info(code, market)
    financial = get_financial_data(code, market)
    prices_data = get_price_history(code, days=250, market=market)
    money_flow = get_money_flow(code) if market == "A" else _get_market_money_flow(code, market, prices_data)
    news = get_news_events(code) if market == "A" else _get_market_news(code, market)
    _flag_buyback_anomaly(news)  # tag buyback records with abnormal (out-of-scale) amounts
    industry_data = get_industry_data(code) if market == "A" else _get_market_industry(code, market)

    # Adjust money_flow warning for non-A markets
    _mfw = money_flow.get("error", "") if isinstance(money_flow, dict) else ""

    # Check for data fetch warnings (adjusted per market)
    if market == "A":
        if not financial.get("income"):
            warnings.append({"dim": "基本面", "msg": "利润表数据获取失败，基本面分析可能不完整"})
        if money_flow.get("error"):
            warnings.append({"dim": "资金面", "msg": money_flow["error"]})
        if not news:
            warnings.append({"dim": "事件催化", "msg": "新闻公告数据获取失败"})
        if not industry_data.get("industry_name"):
            warnings.append({"dim": "同业对标", "msg": "行业分类数据获取失败，同业对比可能不完整"})
    else:
        if not financial.get("income"):
            warnings.append({"dim": "基本面", "msg": f"{cfg['name']}财报加载中，当前使用估值指标分析"})
        if isinstance(money_flow, dict) and money_flow.get("error"):
            warnings.append({"dim": "资金面", "msg": f"{cfg['name']}资金流：{money_flow['error']}"})
        if not news:
            pass  # HK/US news may legitimately return empty for now
        if not industry_data.get("industry_name"):
            pass  # Industry data not loaded for non-A markets yet

    if not prices_data:
        warnings.append({"dim": "技术面", "msg": "历史K线数据获取失败，技术分析不完整"})

    if not info or not info.get("最新价"):
        return {"error": f"无法获取股票 {code} 的数据，请检查代码是否正确或稍后重试"}

    # ---- Basic Info ----
    price = info.get("最新价", 0)
    pe = info.get("市盈率-动态", 0)
    pb = info.get("市净率", 0) or 0
    # HK/US: Tencent quote 市净率 field is mis-aligned (returns ~5.3 for
    # Tencent vs real ~3.3). Prefer BPS from EastMoney financials (Vercel-safe).
    if market in ("HK", "US"):
        try:
            _inc0 = (financial.get("income") or [])[0]
            _bps = _inc0.get("每股净资产") or 0
            if isinstance(_bps, (int, float)) and _bps > 0 and price > 0:
                pb = round(price / float(_bps), 2)
        except Exception:
            pass
    # A-share: recompute PE(动态) & PB from EastMoney QUARTERLY financials
    # (own reliable data, no extra deps). Quote 市盈率-动态 uses annualized
    # annual EPS (static-ish, ~18.9 for 茅台); quote 市净率 uses a
    # stale/higher BPS. Quarterly BPS -> PB 5.79 (matches 同花顺),
    # latest-quarter standalone EPS*4 -> PE(动态) 14.4 (matches 同花顺 动态).
    if market == "A":
        _ap = _a_share_pe_pb(financial, price)
        if _ap[0]:
            pe = _ap[0]
        if _ap[1]:
            pb = _ap[1]
    total_mv = info.get("总市值", 0)
    circ_mv = info.get("流通市值", 0)
    change_pct = info.get("涨跌幅", 0)

    report = {
        "code": code,
        "symbol": symbol,
        "market": market,
        "market_name": cfg["name"],
        "currency": cfg["currency"],
        "currency_label": cfg["currency_label"],
        "name": info.get("股票简称", code),
        "price": price,
        "pe": pe,
        "pb": pb,
        "total_mv": total_mv,
        "circ_mv": circ_mv,
        "change_pct": change_pct,
        "report_time": now_cn().strftime("%Y-%m-%d %H:%M"),
        "prices_data": prices_data,  # K-line data for chart rendering
        "scores": {},
    }

    # ---- 1. Fundamental Analysis (25 pts) ----
    fund_score = 15
    fund_detail = {}
    fund_breakdown = [{"item": "起始分", "change": 15, "score_after": 15, "detail": "公式: 起始分 = 15 (固定)"}]

    income_rows = financial.get("income", [])
    balance_rows = financial.get("balance", [])
    quarterly = financial.get("quarterly", [])

    if income_rows:
        # Latest annual report
        latest_income = income_rows[0]
        latest_rev = latest_income.get("营业总收入")
        latest_np = latest_income.get("归母净利润")
        latest_eps = latest_income.get("基本每股收益")
        latest_roe = latest_income.get("加权ROE") or 0
        fund_detail["latest_report"] = latest_income.get("报告期", "")
        fund_detail["latest_revenue"] = round(float(latest_rev) / 1e8, 2) if latest_rev else 0
        fund_detail["latest_net_profit"] = round(float(latest_np) / 1e8, 2) if latest_np else 0
        fund_detail["latest_eps"] = latest_eps
        fund_detail["latest_roe"] = float(latest_roe) if latest_roe else 0

        # Dividend yield (calculate here so it's available in fundamental detail)
        latest_div = industry_data.get("latest_dividend") or {}
        div_cash = latest_div.get("cash_per_share", 0)
        if div_cash > 0 and price > 0:
            fund_detail["dividend_yield"] = round((div_cash / price) * 100, 2)
            fund_detail["dividend_cash_per_share"] = div_cash
        else:
            fund_detail["dividend_yield"] = 0

        # Revenue trend over years
        revenue_trend = []
        for row in income_rows[:5]:
            rev = row.get("营业总收入")
            np_val = row.get("归母净利润")
            revenue_trend.append({
                "period": row.get("报告期", "")[:10] if row.get("报告期") else row.get("年份", ""),
                "revenue": round(float(rev) / 1e8, 2) if rev else 0,
                "net_profit": round(float(np_val) / 1e8, 2) if np_val else 0,
                "roe": float(row.get("加权ROE") or 0),
                "eps": float(row.get("基本每股收益") or 0),
                "yoy_growth": float(row.get("营收同比") or 0),
            })
        fund_detail["revenue_trend"] = revenue_trend

        # Revenue growth scoring (latest vs previous year)
        if len(revenue_trend) >= 2:
            latest_rev_val = revenue_trend[0]["revenue"]
            prev_rev_val = revenue_trend[1]["revenue"]
            if prev_rev_val > 0:
                rev_growth = (latest_rev_val - prev_rev_val) / prev_rev_val * 100
                fund_detail["revenue_growth"] = round(rev_growth, 1)
                if rev_growth < -10:
                    fund_score -= 5
                    fund_detail["growth_note"] = "营收大幅下滑"
                    fund_breakdown.append({"item": "营收增速", "change": -5, "score_after": fund_score, "detail": f"规则: if 营收同比 < -10%: -5分; elif < 0%: -2分; elif > 15%: +3分; else: 0分\n当前: 营收同比={rev_growth:.1f}% < -10% → 触发 -5分"})
                elif rev_growth < 0:
                    fund_score -= 2
                    fund_detail["growth_note"] = "营收小幅下滑"
                    fund_breakdown.append({"item": "营收增速", "change": -2, "score_after": fund_score, "detail": f"规则: if 营收同比 < -10%: -5分; elif < 0%: -2分; elif > 15%: +3分; else: 0分\n当前: 营收同比={rev_growth:.1f}%，-10%≤增速<0% → 触发 -2分"})
                elif rev_growth > 15:
                    fund_score += 3
                    fund_detail["growth_note"] = "营收快速增长"
                    fund_breakdown.append({"item": "营收增速", "change": +3, "score_after": fund_score, "detail": f"规则: if 营收同比 < -10%: -5分; elif < 0%: -2分; elif > 15%: +3分; else: 0分\n当前: 营收同比=+{rev_growth:.1f}% > 15% → 触发 +3分"})
                else:
                    fund_breakdown.append({"item": "营收增速", "change": 0, "score_after": fund_score, "detail": f"规则: if 营收同比 < -10%: -5分; elif < 0%: -2分; elif > 15%: +3分; else: 0分\n当前: 营收同比={rev_growth:.1f}%，处于0%~15%正常范围 → 不加减"})

        # Latest ROE scoring
        if latest_roe and float(latest_roe) > 15:
            fund_score += 5
            fund_breakdown.append({"item": "ROE评分", "change": +5, "score_after": fund_score, "detail": f"规则: if ROE > 15%: +5分; elif > 8%: +2分; elif < 0%: -8分; else: 0分\n当前: ROE={float(latest_roe):.1f}% > 15% → 触发 +5分"})
        elif latest_roe and float(latest_roe) > 8:
            fund_score += 2
            fund_breakdown.append({"item": "ROE评分", "change": +2, "score_after": fund_score, "detail": f"规则: if ROE > 15%: +5分; elif > 8%: +2分; elif < 0%: -8分; else: 0分\n当前: ROE={float(latest_roe):.1f}%，8%<ROE≤15% → 触发 +2分"})
        elif latest_roe and float(latest_roe) < 0:
            fund_score -= 8
            fund_breakdown.append({"item": "ROE评分", "change": -8, "score_after": fund_score, "detail": f"规则: if ROE > 15%: +5分; elif > 8%: +2分; elif < 0%: -8分; else: 0分\n当前: ROE={float(latest_roe):.1f}% < 0% → 触发 -8分"})
        else:
            fund_breakdown.append({"item": "ROE评分", "change": 0, "score_after": fund_score, "detail": f"规则: if ROE > 15%: +5分; elif > 8%: +2分; elif < 0%: -8分; else: 0分\n当前: ROE={float(latest_roe) if latest_roe else 'N/A'}%，0%≤ROE≤8% → 不加减"})

        # EPS trend check
        if len(income_rows) >= 3:
            eps_trend = [r.get("基本每股收益") for r in income_rows[:3] if r.get("基本每股收益")]
            if len(eps_trend) >= 2 and all(float(x) for x in eps_trend):
                eps_list = [float(x) for x in eps_trend]
                if eps_list[0] > eps_list[1] and eps_list[1] > eps_list[2]:
                    fund_score += 2
                    fund_detail["eps_trend"] = "连续增长"
                    fund_breakdown.append({"item": "EPS趋势", "change": +2, "score_after": fund_score, "detail": f"规则: 近3年EPS逐年递增 → +2分; 逐年递减 → 0分(记录)\n当前: {eps_list[0]:.2f} > {eps_list[1]:.2f} > {eps_list[2]:.2f} → 连续增长，触发 +2分"})
                elif eps_list[0] <= eps_list[1] and eps_list[1] <= eps_list[2]:
                    fund_detail["eps_trend"] = "连续下降"
                    fund_breakdown.append({"item": "EPS趋势", "change": 0, "score_after": fund_score, "detail": f"规则: 近3年EPS逐年递增 → +2分; 逐年递减 → 0分(记录)\n当前: {eps_list[0]:.2f} → {eps_list[1]:.2f} → {eps_list[2]:.2f} → 连续下降，不加分"})

        # Balance sheet check
        if balance_rows:
            latest_balance = balance_rows[0]
            dar = latest_balance.get("资产负债率")
            if dar is not None:
                dar = float(dar)
                fund_detail["debt_ratio"] = round(dar, 2)
                if dar > 85:
                    fund_score -= 2
                    fund_breakdown.append({"item": "资产负债率", "change": -2, "score_after": fund_score, "detail": f"规则: if 负债率 > 85%: -2分; elif < 30%: +1分; else: 0分\n当前: 负债率={dar:.1f}% > 85% → 触发 -2分"})
                elif dar < 30:
                    fund_score += 1
                    fund_breakdown.append({"item": "资产负债率", "change": +1, "score_after": fund_score, "detail": f"规则: if 负债率 > 85%: -2分; elif < 30%: +1分; else: 0分\n当前: 负债率={dar:.1f}% < 30% → 触发 +1分"})
                else:
                    fund_breakdown.append({"item": "资产负债率", "change": 0, "score_after": fund_score, "detail": f"规则: if 负债率 > 85%: -2分; elif < 30%: +1分; else: 0分\n当前: 负债率={dar:.1f}%，30%~85%正常 → 不加减"})
    else:
        # Fallback: use real-time ROE from quote
        roe = info.get("净资产收益率", 0)
        if isinstance(roe, str):
            try:
                roe = float(roe.replace("%", ""))
            except ValueError:
                roe = 0
        fund_detail["roe"] = roe
        fund_detail["note"] = "未获取到详细财务数据，使用实时行情估算"
        fund_breakdown.append({"item": "数据来源", "change": 0, "score_after": fund_score, "detail": "注意: 未获取到详细财务数据，以下使用实时行情估算"})
        if roe > 15:
            fund_score += 5
            fund_breakdown.append({"item": "ROE估算", "change": +5, "score_after": fund_score, "detail": f"规则: if ROE > 15%: +5分; elif > 8%: +2分; elif < 0%: -8分\n当前: ROE(实时)={roe:.1f}% > 15% → 触发 +5分"})
        elif roe > 8:
            fund_score += 2
            fund_breakdown.append({"item": "ROE估算", "change": +2, "score_after": fund_score, "detail": f"规则: if ROE > 15%: +5分; elif > 8%: +2分; elif < 0%: -8分\n当前: ROE(实时)={roe:.1f}%，8%<ROE≤15% → 触发 +2分"})
        elif roe < 0:
            fund_score -= 8
            fund_breakdown.append({"item": "ROE估算", "change": -8, "score_after": fund_score, "detail": f"规则: if ROE > 15%: +5分; elif > 8%: +2分; elif < 0%: -8分\n当前: ROE(实时)={roe:.1f}% < 0% → 触发 -8分"})

        # Dividend yield (fallback branch)
        latest_div = industry_data.get("latest_dividend") or {}
        div_cash = latest_div.get("cash_per_share", 0)
        if div_cash > 0 and price > 0:
            fund_detail["dividend_yield"] = round((div_cash / price) * 100, 2)
        else:
            fund_detail["dividend_yield"] = 0

    # PE check
    if pe > 0 and pe < 15:
        fund_score += 3
        fund_breakdown.append({"item": "PE估值", "change": +3, "score_after": fund_score, "detail": f"规则: if PE < 15: +3分; elif PE < 0: -5分; else: 0分\n当前: 动态PE={pe:.1f} < 15 → 触发 +3分"})
    elif pe < 0:
        fund_score -= 5
        fund_breakdown.append({"item": "PE估值", "change": -5, "score_after": fund_score, "detail": f"规则: if PE < 15: +3分; elif PE < 0: -5分; else: 0分\n当前: PE={pe:.1f}(亏损) < 0 → 触发 -5分"})
    else:
        fund_breakdown.append({"item": "PE估值", "change": 0, "score_after": fund_score, "detail": f"规则: if PE < 15: +3分; elif PE < 0: -5分; else: 0分\n当前: 动态PE={pe:.1f} ≥ 15 → 不加减"})

    fund_score = max(0, min(25, fund_score))
    fund_breakdown.append({"item": "最终得分", "change": 0, "score_after": fund_score, "detail": f"公式: clamp(原始分, 0, 25) = {fund_score}/25"})
    fund_detail["score_breakdown"] = fund_breakdown
    report["scores"]["fundamental"] = {
        "score": fund_score, "max": 25, "detail": fund_detail,
        "summary": "基本面优秀" if fund_score >= 20 else "基本面良好" if fund_score >= 15 else "基本面一般" if fund_score >= 8 else "基本面堪忧"
    }

    # ---- 2. Technical Analysis (20 pts) ----
    tech_score = 12
    tech_detail = {}
    tech_breakdown = [{"item": "起始分", "change": 12, "score_after": 12, "detail": "公式: 起始分 = 12 (固定)"}]

    if prices_data:
        closes = [d["收盘"] for d in prices_data if d.get("收盘")]
        opens = [d["开盘"] for d in prices_data if d.get("开盘")]
        highs = [d["最高"] for d in prices_data if d.get("最高")]
        lows = [d["最低"] for d in prices_data if d.get("最低")]

        ma5 = calc_ma(closes, 5)
        ma10 = calc_ma(closes, 10)
        ma20 = calc_ma(closes, 20)
        ma60 = calc_ma(closes, 60)
        ma120 = calc_ma(closes, 120)
        ma250 = calc_ma(closes, 250)

        tech_detail["mas"] = {
            "MA5": ma5, "MA10": ma10, "MA20": ma20,
            "MA60": ma60, "MA120": ma120, "MA250": ma250
        }

        # Position vs MAs
        current = closes[-1]
        above_count = 0
        if ma5 and current > ma5: above_count += 1
        if ma10 and current > ma10: above_count += 1
        if ma20 and current > ma20: above_count += 1
        if ma60 and current > ma60: above_count += 1
        if ma120 and current > ma120: above_count += 1
        tech_detail["above_ma_count"] = above_count

        if above_count >= 5:
            tech_score += 5
            tech_breakdown.append({"item": "均线多头排列", "change": +5, "score_after": tech_score, "detail": f"规则: 站上≥5条均线→+5分; ≥3条→+2分; ≤1条→-5分\n当前: 站上{above_count}/6条 → 触发 +5分"})
        elif above_count >= 3:
            tech_score += 2
            tech_breakdown.append({"item": "均线排列", "change": +2, "score_after": tech_score, "detail": f"规则: 站上≥5条均线→+5分; ≥3条→+2分; ≤1条→-5分\n当前: 站上{above_count}/6条 → 触发 +2分"})
        elif above_count <= 1:
            tech_score -= 5
            tech_breakdown.append({"item": "均线空头排列", "change": -5, "score_after": tech_score, "detail": f"规则: 站上≥5条均线→+5分; ≥3条→+2分; ≤1条→-5分\n当前: 仅站上{above_count}/6条 → 触发 -5分"})
        else:
            tech_breakdown.append({"item": "均线排列", "change": 0, "score_after": tech_score, "detail": f"规则: 站上≥5条→+5; ≥3条→+2; ≤1条→-5\n当前: 站上{above_count}/6条 → 2~4条无加减"})

        # MACD
        dif, dea, bar = calc_macd(closes)
        tech_detail["macd"] = {"DIF": dif, "DEA": dea, "BAR": bar}
        if dif and dea:
            if dif > dea and dif > 0:
                tech_score += 3
                tech_breakdown.append({"item": "MACD金叉", "change": +3, "score_after": tech_score, "detail": f"规则: if DIF>DEA且DIF>0: +3分; elif DIF<DEA且DIF<0: -3分; else: 0分\n当前: DIF={dif:.3f} > DEA={dea:.3f}，且DIF>0 → 触发 +3分"})
            elif dif < dea and dif < 0:
                tech_score -= 3
                tech_breakdown.append({"item": "MACD死叉", "change": -3, "score_after": tech_score, "detail": f"规则: if DIF>DEA且DIF>0: +3分; elif DIF<DEA且DIF<0: -3分; else: 0分\n当前: DIF={dif:.3f} < DEA={dea:.3f}，且DIF<0 → 触发 -3分"})
            else:
                tech_breakdown.append({"item": "MACD", "change": 0, "score_after": tech_score, "detail": f"规则: if DIF>DEA且DIF>0: +3分; elif DIF<DEA且DIF<0: -3分; else: 0分\n当前: DIF={dif:.3f}, DEA={dea:.3f} → 不满足加减条件"})

        # KDJ
        k, d, j = calc_kdj(closes, highs, lows)
        tech_detail["kdj"] = {"K": k, "D": d, "J": j}

        # Bollinger
        bu, bm, bl = calc_bollinger(closes)
        tech_detail["bollinger"] = {"upper": bu, "middle": bm, "lower": bl}

        # Volatility
        if len(closes) >= 120:
            start_price = closes[-120] if len(closes) >= 120 else closes[0]
            half_year_return = (closes[-1] - start_price) / start_price * 100
            tech_detail["half_year_return"] = round(half_year_return, 2)
            if half_year_return < -15:
                tech_score -= 3
                tech_breakdown.append({"item": "半年涨跌幅", "change": -3, "score_after": tech_score, "detail": f"规则: if 半年跌幅 < -15%: -3分; else: 0分\n当前: 半年涨跌={half_year_return:.1f}% < -15% → 触发 -3分"})
            else:
                tech_breakdown.append({"item": "半年涨跌幅", "change": 0, "score_after": tech_score, "detail": f"规则: if 半年跌幅 < -15%: -3分; else: 0分\n当前: 半年涨跌={half_year_return:.1f}% ≥ -15% → 不触发"})

        tech_detail["turnover"] = info.get("换手率", 0)
    else:
        tech_detail["error"] = "无历史价格数据"
        tech_breakdown.append({"item": "数据缺失", "change": 0, "score_after": tech_score, "detail": "无历史价格数据 → 无法进行技术分析"})

    tech_score = max(0, min(20, tech_score))
    tech_breakdown.append({"item": "最终得分", "change": 0, "score_after": tech_score, "detail": f"公式: clamp(原始分, 0, 20) = {tech_score}/20"})
    tech_detail["score_breakdown"] = tech_breakdown
    report["scores"]["technical"] = {
        "score": tech_score, "max": 20, "detail": tech_detail,
        "summary": "技术面偏多" if tech_score >= 14 else "技术面中性" if tech_score >= 8 else "技术面偏空"
    }

    # ---- 3. Capital Flow (15 pts) ----
    flow_score = 8
    flow_detail = {}
    flow_breakdown = [{"item": "基础分", "change": 8, "score_after": 8, "detail": "公式: 起始分 = 8 (固定)"}]
    flow_data_ok = money_flow and not money_flow.get("error") and money_flow.get("summary")
    flow_detail["data_ok"] = flow_data_ok

    if flow_data_ok:
        summary = money_flow["summary"]
        flow_detail["main_5d_net"] = summary["main_5d_net"]
        flow_detail["super_large_5d_net"] = summary["super_large_5d_net"]
        flow_detail["large_5d_net"] = summary["large_5d_net"]
        flow_detail["medium_5d_net"] = summary.get("medium_5d_net", 0)
        flow_detail["retail_5d_net"] = summary["retail_5d_net"]
        flow_detail["main_inflow_days"] = summary["main_inflow_days"]
        flow_detail["main_outflow_days"] = summary.get("main_outflow_days", 0)
        flow_detail["trend"] = summary["trend"]
        flow_detail["records"] = money_flow.get("records", [])[-10:]  # last 10 days for chart

        main_5d = summary["main_5d_net"]
        if main_5d > 5000:  # >5000万
            flow_score += 5
            flow_breakdown.append({"item": "主力5日净流入", "change": +5, "score_after": flow_score, "detail": f"规则: if 5日主力净>5000万: +5; >1000万: +2; <-5000万: -5; <-1000万: -3\n当前: 主力{main_5d:.0f}万 > 5000万 → +5分"})
        elif main_5d > 1000:
            flow_score += 2
            flow_breakdown.append({"item": "主力5日净流入", "change": +2, "score_after": flow_score, "detail": f"规则: if 5日主力净>5000万: +5; >1000万: +2; <-5000万: -5; <-1000万: -3\n当前: 主力{main_5d:.0f}万，1000~5000万 → +2分"})
        elif main_5d < -5000:
            flow_score -= 5
            flow_breakdown.append({"item": "主力5日净流出", "change": -5, "score_after": flow_score, "detail": f"规则: if 5日主力净>5000万: +5; >1000万: +2; <-5000万: -5; <-1000万: -3\n当前: 主力{main_5d:.0f}万 < -5000万 → -5分"})
        elif main_5d < -1000:
            flow_score -= 3
            flow_breakdown.append({"item": "主力5日净流出", "change": -3, "score_after": flow_score, "detail": f"规则: if 5日主力净>5000万: +5; >1000万: +2; <-5000万: -5; <-1000万: -3\n当前: 主力{main_5d:.0f}万，-5000~-1000万 → -3分"})
        else:
            flow_breakdown.append({"item": "主力5日净流量", "change": 0, "score_after": flow_score, "detail": f"规则: 金额在-1000~+1000万之间无加减\n当前: 主力{main_5d:.0f}万 → 不触发"})

        # Divergence check: main outflow but price up
        latest_flow = money_flow.get("latest")
        if latest_flow and latest_flow["main_net"] < 0 and change_pct > 1:
            flow_detail["divergence"] = True
            flow_detail["divergence_msg"] = "主力流出但股价上涨，量价背离需警惕"
            flow_score -= 2
            flow_breakdown.append({"item": "量价背离", "change": -2, "score_after": flow_score, "detail": "规则: if 主力流出且涨幅>1%: -2分(量价背离)\n当前: 触发量价背离 → -2分"})

        # Main/retail divergence check
        if main_5d > 0 and summary["retail_5d_net"] < 0:
            flow_detail["structure"] = "主力吸筹、散户出货，偏多信号"
            flow_score += 1
            flow_breakdown.append({"item": "筹码结构", "change": +1, "score_after": flow_score, "detail": f"规则: 主力流入+散户流出→+1分; 主力流出+散户流入→-1分\n当前: 主力{main_5d:.0f}万 vs 散户{summary['retail_5d_net']:.0f}万 → 吸筹+1"})
        elif main_5d < 0 and summary["retail_5d_net"] > 0:
            flow_detail["structure"] = "主力出货、散户接盘，偏空信号"
            flow_score -= 1
            flow_breakdown.append({"item": "筹码结构", "change": -1, "score_after": flow_score, "detail": f"规则: 主力流入+散户流出→+1分; 主力流出+散户流入→-1分\n当前: 主力{main_5d:.0f}万 vs 散户{summary['retail_5d_net']:.0f}万 → 出货-1"})
    else:
        flow_detail["error"] = money_flow.get("error", "无资金流向数据")
        warnings.append({"dim": "资金面", "msg": flow_detail["error"]})
        flow_breakdown.append({"item": "数据缺失", "change": 0, "score_after": flow_score, "detail": f"无法获取资金流向: {flow_detail['error']}"})

    flow_score = max(0, min(15, flow_score))
    flow_breakdown.append({"item": "最终得分", "change": 0, "score_after": flow_score, "detail": f"公式: clamp(原始分, 0, 15) = {flow_score}/15"})
    flow_detail["score_breakdown"] = flow_breakdown
    report["scores"]["capital"] = {
        "score": flow_score, "max": 15, "detail": flow_detail,
        "summary": "资金面偏多" if flow_score >= 10 else "资金面中性" if flow_score >= 6 else "资金面偏空"
    }

    # ---- 4. Event Catalyst (10 pts) ----
    event_score = 4  # baseline lowered from 6
    event_detail = {"events": [], "data_ok": bool(news), "method": "keyword_matching"}
    event_breakdown = [{"item": "基础分", "change": 4, "score_after": 4, "detail": "公式: 起始分 = 4 (配置LLM后深度分析可突破上限)"}]

    if news:
        # Weighted sentiment analysis with expanded keyword library
        total_pos_weight = sum(n.get("sentiment_score", 0) for n in news if n.get("sentiment") == "positive")
        total_neg_weight = sum(n.get("sentiment_score", 0) for n in news if n.get("sentiment") == "negative")
        positive_count = sum(1 for n in news if n.get("sentiment") == "positive")
        negative_count = sum(1 for n in news if n.get("sentiment") == "negative")
        
        event_detail["events"] = news[:10]
        event_detail["positive_count"] = positive_count
        event_detail["negative_count"] = negative_count
        event_detail["total_count"] = len(news)
        event_detail["positive_weight"] = total_pos_weight
        event_detail["negative_weight"] = total_neg_weight

        # Score based on weighted sentiment
        # Net sentiment = positive weight - negative weight
        net_sentiment = total_pos_weight - total_neg_weight
        
        if net_sentiment >= 8:
            event_score = 9  # strongly positive
        elif net_sentiment >= 5:
            event_score = 7  # moderately positive
        elif net_sentiment >= 2:
            event_score = 6  # slightly positive
        elif net_sentiment >= -2:
            event_score = 4  # neutral / mixed
        elif net_sentiment >= -5:
            event_score = 3  # slightly negative
        elif net_sentiment >= -8:
            event_score = 2  # moderately negative
        else:
            event_score = 1  # strongly negative

        event_breakdown.append({
            "item": "加权情感分析",
            "change": event_score - 4,
            "score_after": event_score,
            "detail": f"规则: 净情感=正权重-负权重; if>=8→9分;>=5→7;>=2→6;>=-2→4;>=-5→3;>=-8→2;else→1\n当前: +{total_pos_weight}-{total_neg_weight}=净{net_sentiment} → {event_score}/10"
        })

        # Highlight key events
        key_events = [n for n in news if n.get("sentiment_score", 0) >= 3]
        if key_events:
            event_detail["key_events"] = key_events[:5]
    else:
        event_detail["events"].append({"date": "", "title": "近30日无重大公告或获取失败", "sentiment": "neutral"})
        warnings.append({"dim": "事件催化", "msg": "新闻公告获取失败"})
        event_breakdown.append({"item": "数据缺失", "change": 0, "score_after": event_score, "detail": "无公告数据，维持基准分"})

    event_score = max(0, min(10, event_score))
    event_breakdown.append({"item": "最终得分", "change": 0, "score_after": event_score, "detail": f"公式: clamp(原始分, 0, 10) = {event_score}/10"})
    event_detail["method_note"] = "当前使用关键词匹配引擎。配置 LLM API 后可启用网络搜索+深度分析，获得更精准的事件催化评分。"
    event_detail["score_breakdown"] = event_breakdown
    report["scores"]["events"] = {
        "score": event_score, "max": 10, "detail": event_detail,
        "summary": "近期偏多事件较多" if event_score >= 7 else "事件面相对平静" if event_score >= 4 else "近期偏空事件较多"
    }

    # ---- 5. Industry Comparison (15 pts) ----
    ind_score = 8
    ind_detail = {}
    ind_breakdown = [{"item": "基础分", "change": 8, "score_after": 8, "detail": "公式: 起始分 = 8 (固定)"}]
    ind_data_ok = bool(industry_data.get("industry_name"))
    ind_detail["data_ok"] = ind_data_ok
    ind_detail["score_breakdown"] = ind_breakdown

    if ind_data_ok:
        ind_detail["industry_name"] = industry_data["industry_name"]
        ind_detail["board_name"] = industry_data["board_name"]

        # PE-based industry assessment with industry-specific benchmarks
        # Different industries have different normal PE ranges
        ind_name = industry_data["industry_name"]
        # Use international benchmarks for HK/US markets
        pe_range = _industry_pe_range_intl(ind_name) if market in ("HK", "US") else _industry_pe_range(ind_name, industry_data.get("board_name"))
        ind_detail["pe_benchmark"] = pe_range

        if pe > 0:
            if pe < pe_range["low"]:
                ind_detail["pe_assessment"] = "PE低于行业均值，可能被低估"
                ind_score += 3
                ind_breakdown.append({"item": "PE行业评估", "change": +3, "score_after": ind_score, "detail": f"规则: if PE<行业下限: +3分; elif PE<行业上限: +1分; else: -2分\n当前: PE={pe:.1f} < {pe_range['low']} → +3分"})
            elif pe < pe_range["high"]:
                ind_detail["pe_assessment"] = "PE处于行业合理区间"
                ind_score += 1
                ind_breakdown.append({"item": "PE行业评估", "change": +1, "score_after": ind_score, "detail": f"规则: if PE<行业下限: +3分; elif PE<行业上限: +1分; else: -2分\n当前: PE={pe:.1f}在[{pe_range['low']}~{pe_range['high']}]内 → +1分"})
            else:
                ind_detail["pe_assessment"] = f"PE({pe:.1f})高于行业合理区间({pe_range['low']}~{pe_range['high']})"
                ind_score -= 2
                ind_breakdown.append({"item": "PE行业评估", "change": -2, "score_after": ind_score, "detail": f"规则: if PE<行业下限: +3分; elif PE<行业上限: +1分; else: -2分\n当前: PE={pe:.1f} > {pe_range['high']} → -2分"})
        else:
            ind_detail["pe_assessment"] = "亏损状态无法用PE估值"
            ind_breakdown.append({"item": "PE行业评估", "change": 0, "score_after": ind_score, "detail": "亏损状态，无法用PE进行行业对标"})

        # PB assessment
        if pb > 0:
            ind_detail["pb"] = pb
            if pb < pe_range["pb_low"]:
                ind_detail["pb_assessment"] = "破净或PB极低，价值洼地信号"
                ind_score += 2
                ind_breakdown.append({"item": "PB行业评估", "change": +2, "score_after": ind_score, "detail": f"规则: if PB<PB下限: +2分; elif PB<PB上限: 0分; else: -1分\n当前: PB={pb:.2f} < {pe_range['pb_low']} → +2分"})
            elif pb < pe_range["pb_high"]:
                ind_detail["pb_assessment"] = "PB处于合理区间"
                ind_breakdown.append({"item": "PB行业评估", "change": 0, "score_after": ind_score, "detail": f"规则: if PB<PB下限: +2分; elif PB<PB上限: 0分; else: -1分\n当前: PB={pb:.2f}在[{pe_range['pb_low']}~{pe_range['pb_high']}]内 → 不加减"})
            else:
                ind_detail["pb_assessment"] = f"PB({pb:.1f})偏高"
                ind_score -= 1
                ind_breakdown.append({"item": "PB行业评估", "change": -1, "score_after": ind_score, "detail": f"规则: if PB<PB下限: +2分; elif PB<PB上限: 0分; else: -1分\n当前: PB={pb:.2f} > {pe_range['pb_high']} → -1分"})

        # ROE vs industry
        fin_roe_val = fund_detail.get("latest_roe", 0) or 0
        ind_detail["company_roe"] = fin_roe_val
        if fin_roe_val > pe_range["roe_good"]:
            ind_detail["roe_assessment"] = f"ROE({fin_roe_val:.1f}%)高于行业优秀线({pe_range['roe_good']}%)"
            ind_score += 2
            ind_breakdown.append({"item": "ROE行业对标", "change": +2, "score_after": ind_score, "detail": f"规则: if ROE>优秀线: +2分; elif ROE>平均线: +1分; elif ROE>0: -1分\n当前: ROE={fin_roe_val:.1f}% > {pe_range['roe_good']}% → +2分"})
        elif fin_roe_val > pe_range["roe_avg"]:
            ind_detail["roe_assessment"] = "ROE处于行业中上水平"
            ind_score += 1
            ind_breakdown.append({"item": "ROE行业对标", "change": +1, "score_after": ind_score, "detail": f"规则: if ROE>优秀线: +2分; elif ROE>平均线: +1分; elif ROE>0: -1分\n当前: ROE={fin_roe_val:.1f}%，平均~优秀 → +1分"})
        elif fin_roe_val > 0:
            ind_detail["roe_assessment"] = "ROE低于行业平均水平"
            ind_score -= 1
            ind_breakdown.append({"item": "ROE行业对标", "change": -1, "score_after": ind_score, "detail": f"规则: if ROE>优秀线: +2分; elif ROE>平均线: +1分; elif ROE>0: -1分\n当前: ROE={fin_roe_val:.1f}%，0~平均 → -1分"})
    else:
        # Fallback: basic PE/PB assessment
        ind_detail["industry_name"] = "行业分类获取中"
        ind_breakdown.append({"item": "数据缺失", "change": 0, "score_after": ind_score, "detail": "行业分类数据获取失败，使用默认PE/PB基准"})
        if pe > 0 and pe < 25:
            ind_score += 2
            ind_detail["pe_assessment"] = "PE处于合理偏低区间"
            ind_breakdown.append({"item": "PE简易评估", "change": +2, "score_after": ind_score, "detail": f"规则(fallback): if PE<25: +2分; elif PE<0: -2分\n当前: PE={pe:.1f} < 25 → +2分"})
        elif pe < 0:
            ind_score -= 2
            ind_detail["pe_assessment"] = "亏损状态无法评估PE"
            ind_breakdown.append({"item": "PE简易评估", "change": -2, "score_after": ind_score, "detail": "亏损状态"})

        if pb > 0 and pb < 3:
            ind_score += 1
            ind_breakdown.append({"item": "PB简易评估", "change": +1, "score_after": ind_score, "detail": f"规则(fallback): if PB<3: +1分\n当前: PB={pb:.2f} < 3 → +1分"})

    ind_score = min(15, max(0, ind_score))
    ind_breakdown.append({"item": "最终得分", "change": 0, "score_after": ind_score, "detail": f"公式: clamp(原始分, 0, 15) = {ind_score}/15"})
    report["scores"]["industry"] = {
        "score": ind_score, "max": 15, "detail": ind_detail,
        "summary": "行业排名靠前" if ind_score >= 10 else "行业排名中等" if ind_score >= 6 else "行业排名靠后"
    }

    # ---- 6. Investment Value (15 pts) ----
    value_score = 6
    value_detail = {}
    value_detail["data_ok"] = True
    value_breakdown = [{"item": "基础分", "change": 6, "score_after": 6, "detail": "公式: 起始分 = 6 (固定)"}]

    # Dividend yield from industry_data
    latest_div = industry_data.get("latest_dividend") or {}
    dividends = industry_data.get("dividends", [])
    value_detail["dividend_history"] = dividends[:3]

    div_cash = latest_div.get("cash_per_share", 0)
    if div_cash > 0 and price > 0:
        div_yield = (div_cash / price) * 100
        value_detail["dividend_yield"] = round(div_yield, 2)
        value_detail["dividend_cash_per_share"] = div_cash
        if div_yield > 3:
            value_score += 3
            value_breakdown.append({"item": "股息率", "change": +3, "score_after": value_score, "detail": f"规则: if 股息率>3%: +3分; elif >1.5%: +1分; else: 0分\n当前: 股息率={div_yield:.1f}% > 3% → +3分"})
        elif div_yield > 1.5:
            value_score += 1
            value_breakdown.append({"item": "股息率", "change": +1, "score_after": value_score, "detail": f"规则: if 股息率>3%: +3分; elif >1.5%: +1分; else: 0分\n当前: 股息率={div_yield:.1f}%，1.5~3% → +1分"})
    else:
        value_detail["dividend_yield"] = 0
        if not dividends:
            if market == "US":
                value_detail["dividend_note"] = "美股股息数据暂未覆盖（免费数据源限制，非代表无分红）"
            elif market == "HK":
                value_detail["dividend_note"] = "暂无分红记录"
            else:
                value_detail["dividend_note"] = "无分红记录"
        value_breakdown.append({"item": "股息率", "change": 0, "score_after": value_score, "detail": "无分红数据或每股分红为0"})

    # ---- 毛利率 (Gross Margin) & 净利率 (Net Margin) ----
    gross_margin = financial.get("gross_margin")
    net_margin = financial.get("net_margin")

    if gross_margin is not None or net_margin is not None:
        if gross_margin is not None:
            value_detail["gross_margin"] = gross_margin
        else:
            value_detail["gross_margin"] = None
        if net_margin is not None:
            value_detail["net_margin"] = net_margin
        else:
            value_detail["net_margin"] = None

        # Score bonus for high margins
        if gross_margin is not None and gross_margin > 50:
            value_score += 1
            value_breakdown.append({"item": "毛利率", "change": +1, "score_after": value_score,
                "detail": f"规则: if 毛利率>50%: +1分\n当前: 毛利率={gross_margin:.1f}% > 50% → +1分"})
        if net_margin is not None and net_margin > 15:
            value_score += 1
            value_breakdown.append({"item": "净利率", "change": +1, "score_after": value_score,
                "detail": f"规则: if 净利率>15%: +1分\n当前: 净利率={net_margin:.1f}% > 15% → +1分"})
    else:
        value_detail["gross_margin"] = None
        value_detail["net_margin"] = None

    # PE-based valuation
    if pe > 0 and pe < 15:
        value_score += 4
        value_detail["pe_assessment"] = "PE处于低估区域"
        value_breakdown.append({"item": "PE估值分位", "change": +4, "score_after": value_score, "detail": f"规则: if PE<15: +4分; elif PE<25: +1分; elif PE<0: -3分; else: 0分\n当前: PE={pe:.1f} < 15 → +4分"})
    elif pe > 0 and pe < 25:
        value_score += 1
        value_detail["pe_assessment"] = "PE处于合理区域"
        value_breakdown.append({"item": "PE估值分位", "change": +1, "score_after": value_score, "detail": f"规则: if PE<15: +4分; elif PE<25: +1分; elif PE<0: -3分; else: 0分\n当前: PE={pe:.1f}，15~25 → +1分"})
    elif pe < 0:
        value_detail["pe_assessment"] = "当前亏损，无法用PE估值"
        value_score -= 3
        value_breakdown.append({"item": "PE估值分位", "change": -3, "score_after": value_score, "detail": f"规则: if PE<15: +4分; elif PE<25: +1分; elif PE<0: -3分; else: 0分\n当前: PE为负(亏损) → -3分"})
    else:
        value_detail["pe_assessment"] = "PE偏高"
        value_breakdown.append({"item": "PE估值分位", "change": 0, "score_after": value_score, "detail": f"规则: if PE<15: +4分; elif PE<25: +1分; elif PE<0: -3分; else: 0分\n当前: PE={pe:.1f} >= 25 → 不加减"})

    # ROE check
    fin_roe_v = fund_detail.get("latest_roe", 0) or 0
    value_detail["roe"] = fin_roe_v
    if fin_roe_v > 12:
        value_score += 3
        value_breakdown.append({"item": "ROE盈利能力", "change": +3, "score_after": value_score, "detail": f"规则: if ROE>12%: +3分; elif >6%: +1分; elif <=0%: -4分\n当前: ROE={fin_roe_v:.1f}% > 12% → +3分"})
    elif fin_roe_v > 6:
        value_score += 1
        value_breakdown.append({"item": "ROE盈利能力", "change": +1, "score_after": value_score, "detail": f"规则: if ROE>12%: +3分; elif >6%: +1分; elif <=0%: -4分\n当前: ROE={fin_roe_v:.1f}%，6~12% → +1分"})
    elif fin_roe_v is not None and fin_roe_v <= 0:
        value_score -= 4
        value_breakdown.append({"item": "ROE盈利能力", "change": -4, "score_after": value_score, "detail": f"规则: if ROE>12%: +3分; elif >6%: +1分; elif <=0%: -4分\n当前: ROE={fin_roe_v:.1f}% <= 0 → -4分"})

    # PEG estimation (use revenue growth → EPS growth → net profit YoY as fallbacks)
    rev_growth = fund_detail.get("revenue_growth")
    np_yoy = None
    eps_yoy = None
    income_rows_for_peg = financial.get("income", [])
    if income_rows_for_peg and len(income_rows_for_peg) >= 2:
        eps0 = income_rows_for_peg[0].get("基本每股收益")
        eps1 = income_rows_for_peg[1].get("基本每股收益")
        if eps0 is not None and eps1 is not None and float(eps1) != 0:
            eps_yoy = (float(eps0) - float(eps1)) / abs(float(eps1)) * 100
        np_yoy = income_rows_for_peg[0].get("净利同比")

    # Try multiple growth sources, prefer revenue > EPS > net profit
    growth_rate = None
    if rev_growth and rev_growth > 0:
        growth_rate = rev_growth
    elif eps_yoy and eps_yoy > 0:
        growth_rate = eps_yoy
    elif np_yoy and float(np_yoy) > 0:
        growth_rate = float(np_yoy)

    if growth_rate and growth_rate > 0 and pe > 0:
        peg = round(pe / growth_rate, 2)
        value_detail["peg"] = peg
        if peg < 1:
            value_score += 2
            value_detail["peg_assessment"] = "PEG<1，成长性被低估"
            value_breakdown.append({"item": "PEG成长性", "change": +2, "score_after": value_score, "detail": f"规则: if PEG<1: +2分; elif PEG<2: 0分; else: 0分\n当前: PEG={peg:.2f} < 1 → +2分"})
        elif peg < 2:
            value_detail["peg_assessment"] = "PEG处于合理区间"
            value_breakdown.append({"item": "PEG成长性", "change": 0, "score_after": value_score, "detail": f"规则: if PEG<1: +2分; elif PEG<2: 0分; else: 0分\n当前: PEG={peg:.2f}，1~2 → 不加减"})
        else:
            value_detail["peg_assessment"] = "PEG偏高，成长性可能不足以支撑估值"
            value_breakdown.append({"item": "PEG成长性", "change": 0, "score_after": value_score, "detail": f"规则: if PEG<1: +2分; elif PEG<2: 0分; else: 0分\n当前: PEG={peg:.2f} >= 2 → 不加减"})
    else:
        value_breakdown.append({"item": "PEG成长性", "change": 0, "score_after": value_score, "detail": "营收增速数据不完整，无法计算PEG"})

    # PB vs historical (from balance sheet)
    if pb > 0:
        bps = fund_detail.get("latest_report")  # we have BPS from income data
        value_detail["pb"] = pb

    value_score = max(0, min(15, value_score))
    value_breakdown.append({"item": "最终得分", "change": 0, "score_after": value_score, "detail": f"公式: clamp(原始分, 0, 15) = {value_score}/15"})
    value_detail["score_breakdown"] = value_breakdown
    report["scores"]["value"] = {
        "score": value_score, "max": 15, "detail": value_detail,
        "summary": "性价比较高" if value_score >= 10 else "性价比一般" if value_score >= 6 else "性价比较低"
    }

    # ---- Total Score ----
    total = sum(s["score"] for s in report["scores"].values())
    max_total = sum(s["max"] for s in report["scores"].values())
    report["total_score"] = total
    report["max_score"] = max_total

    if total >= 60:
        report["recommendation"] = "持有/增持"
    elif total >= 40:
        report["recommendation"] = "谨慎持有"
    elif total >= 25:
        report["recommendation"] = "建议减仓观望"
    else:
        report["recommendation"] = "不推荐持有，建议换股"

    # ---- Warnings (fallback notices) ----
    report["warnings"] = warnings

    # ---- Raw data for detailed display ----
    report["raw"] = {
        "info": info,
        "financial": {
            "income_count": len(financial.get("income", [])),
            "balance_count": len(financial.get("balance", [])),
            "income": financial.get("income", []),
            "balance": financial.get("balance", []),
            "quarterly": financial.get("quarterly", []),
        },
        "price_days": len(prices_data),
        "money_flow": money_flow,
        "industry": {
            "name": industry_data.get("industry_name", ""),
            "board": industry_data.get("board_name", ""),
            "dividends": industry_data.get("dividends", []),
        },
    }

    return report


# ---------- WeStock CLI helpers ----------
WESTOCK_DATA_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resources", "app.asar.unpacked", "resources", "builtin-skills",
    "westock-data", "scripts", "index.js"
)
# Fallback: try common install paths
if not os.path.exists(WESTOCK_DATA_SCRIPT):
    WESTOCK_DATA_SCRIPT = r"D:\WorkBuddy\resources\app.asar.unpacked\resources\builtin-skills\westock-data\scripts\index.js"

NODE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".workbuddy", "binaries", "node", "versions", "22.22.2", "node.exe"
)
if not os.path.exists(NODE_PATH):
    NODE_PATH = r"C:\Users\jaspe\.workbuddy\binaries\node\versions\22.22.2\node.exe"


@app.route("/")
def index():
    return render_template("index.html")



import re
import json
import requests
from urllib.parse import quote

def _get_industry_peers_by_board(code):
    """Get peer stocks via Eastmoney push2 realtime API using board_code."""
    try:
        ind_data = get_industry_data(code)
        board_code = ind_data.get('board_code', '')
        board_name = ind_data.get('board_name', '')
        if not board_code:
            print(f"[_get_industry_peers_by_board] No board_code for {code}")
            return []

        # Use push2.eastmoney.com realtime clist API
        url = "https://push2.eastmoney.com/api/qt/clist/get"
        params = {
            "pn": 1, "pz": 80,
            "fs": f"b:{board_code}",
            "fields": "f12,f14",
            "fid": "f3", "po": 1,
            "_": "1719900000000",
        }
        em_session = requests.Session()
        em_session.trust_env = False
        resp = em_session.get(
            url, params=params,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"},
            timeout=15
        )
        data = resp.json()

        diff = data.get("data", {}).get("diff", {})
        if not diff:
            print(f"[_get_industry_peers_by_board] Empty diff for board {board_code}")
            return []

        # diff may be a dict {0: {...}, 1: {...}} or a list
        if isinstance(diff, dict):
            items = list(diff.values())
        else:
            items = diff

        symbol = code.replace('.SZ', '').replace('.SH', '').replace('.BJ', '')
        peers = []
        for item in items:
            c = str(item.get('f12', '')).zfill(6)
            if c == symbol:
                continue
            n = item.get('f14', '')
            wc = 'sh' + c if c.startswith(('6', '9')) else 'sz' + c
            peers.append({'code': c, 'name': n, 'wc': wc})

        print(f"[_get_industry_peers_by_board] Found {len(peers)} peers for board {board_name} ({board_code})")
        return peers
    except Exception as e:
        print(f"[_get_industry_peers_by_board] Error: {e}")
        import traceback
        traceback.print_exc()
        return []


def _tencent_search_peers(symbol, keyword):
    """Fallback: use Tencent stock search to find peers by keyword."""
    try:
        url = f"https://smartbox.gtimg.cn/s3/?q={quote(keyword)}&t=gp"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if resp.status_code != 200:
            return []
        text = resp.text.strip()
        # API returns JSONP: callback_func({...})
        m = re.search(r'\((\{.*\})\)', text, re.DOTALL)
        if m:
            data = json.loads(m.group(1))
        else:
            data = resp.json()
        items = data.get('data', [])
        peers = []
        for item in items:
            c = str(item.get('code', '')).zfill(6)
            if c == symbol:
                continue
            n = item.get('name', '')
            wc = 'sh' + c if c.startswith(('6', '9')) else 'sz' + c
            peers.append({'code': c, 'name': n, 'wc': wc})
        return peers[:20]
    except Exception as e:
        print(f"[_tencent_search_peers] Error: {e}")
        return []


def _batch_get_quotes(wc_codes):
    """Batch get quotes from Tencent API. Returns dict of wc -> {pe, pb, price, ...}."""
    result = {}
    if not wc_codes:
        return result
    batch_size = 50
    TENCENT_QT = "https://qt.gtimg.cn/q="
    REQ_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://gu.qq.com/",
    }
    for i in range(0, len(wc_codes), batch_size):
        batch = wc_codes[i:i+batch_size]
        q = ','.join(batch)
        try:
            resp = requests.get(TENCENT_QT + q, headers=REQ_HEADERS, timeout=15)
            if resp.status_code != 200:
                continue
            lines = resp.text.strip().split('\n')
            for line in lines:
                if '=' not in line:
                    continue
                var_name = line.split('=')[0].strip()
                wc = var_name.replace('v_', '').strip()
                data_str = line.split('=', 1)[1].strip()
                if data_str.startswith('"') or data_str.startswith("'"):
                    data_str = data_str[1:-1]
                fields = data_str.split('~')
                if len(fields) < 50:
                    continue
                try:
                    price = float(fields[3]) if fields[3] else 0
                    pe = float(fields[39]) if fields[39] else 0
                    pb = float(fields[46]) if fields[46] else 0
                    change_pct = float(fields[32]) if fields[32] else 0
                    mcap = float(fields[45]) * 1e8 if fields[45] else 0
                    result[wc] = {
                        'price': price, 'pe': pe, 'pb': pb,
                        'change_pct': change_pct, 'mcap': mcap,
                        'name': fields[1] if len(fields) > 1 else '',
                    }
                except (ValueError, IndexError):
                    continue
        except Exception as e:
            print(f"[_batch_get_quotes] Error: {e}")

    # ---- Fallback: Sina batch quotes ----
    # If Tencent returned no results, try Sina
    if not result or len(result) < len(wc_codes) * 0.5:
        try:
            fb_quotes = api_fallback.sina_batch_get_quotes(wc_codes)
            if fb_quotes:
                # Merge Sina data, but don't overwrite existing Tencent data
                for wc, q in fb_quotes.items():
                    if wc not in result:
                        result[wc] = q
        except Exception as fe:
            print(f"[FALLBACK] Sina batch quotes failed: {fe}")

    return result


def find_alternatives(code):
    """Find peer stocks in the same industry. Pure HTTP, Vercel-safe."""
    cache_key = f"alt_{code}"
    cached_val = cached(cache_key, ttl=600)
    if cached_val:
        return cached_val

    market, symbol = detect_market(code)
    if market in ("HK", "US"):
        res = _hk_us_same_industry(code, market, symbol)
        cache_set(cache_key, res)
        return res

    try:
        symbol = code.replace('.SZ', '').replace('.SH', '').replace('.BJ', '')

        # Fetch industry data ONCE (cached internally by get_industry_data)
        ind_data = get_industry_data(code)

        # Method 1: Use board_code from get_industry_data (primary)
        peers = _get_industry_peers_by_board(code)
        print(f"[find_alternatives] Method1 found {len(peers)} peers for {code}")

        # Method 2: Fallback - search by industry keyword
        if not peers:
            board_name = ind_data.get('board_name', '')
            ind_name = ind_data.get('industry_name', '')
            search_kw = board_name or ind_name
            if search_kw:
                clean_kw = search_kw.replace('Ⅰ', '').replace('Ⅱ', '').replace('Ⅲ', '').strip()[:4]
                if clean_kw and len(clean_kw) >= 2:
                    peers = _tencent_search_peers(symbol, clean_kw)
                    print(f"[find_alternatives] Method2 found {len(peers)} peers for {code}")

        # Method 3: Static industry mapping (offline, always available)
        if not peers:
            peers = api_fallback.static_get_industry_peers(
                code,
                ind_data.get('industry_name', ''),
                ind_data.get('board_name', '')
            )
            print(f"[find_alternatives] Method3 (static) found {len(peers)} peers for {code}")

        # Method 4: Last resort — cross-industry fallback from static pool
        if not peers:
            peers = api_fallback.static_get_cross_industry_peers(code, limit=30)
            print(f"[find_alternatives] Method4 (cross-industry) found {len(peers)} peers for {code}")

        if not peers:
            print(f"[find_alternatives] No peers found for {code}")
            cache_set(cache_key, [])
            return []

        # Batch get quotes from Tencent API (with Sina fallback built in)
        peer_codes = [p['wc'] for p in peers[:30]]
        quotes = _batch_get_quotes(peer_codes)

        # Check if quotes are mostly empty (Vercel network issues)
        has_pe_data = any(q.get('pe', 0) for q in quotes.values())

        alternatives = []
        for p in peers[:30]:
            wc = p['wc']
            q = quotes.get(wc, {})
            pe = q.get('pe', 0) or 0
            price = q.get('price', 0) or 0

            # If we have PE data, apply normal PE filter
            if has_pe_data and (pe <= 0 or pe > 200):
                continue
            # If no PE data at all, include stocks with any price data
            if not has_pe_data and price <= 0:
                continue

            pb = q.get('pb', 0) or 0
            change = q.get('change_pct', 0) or 0
            mcap = q.get('mcap', 0) or 0
            alternatives.append({
                'code': p['code'],
                'wc': wc,
                'name': p.get('name', q.get('name', '')),
                'price': price,
                'pe': pe,
                'pb': pb,
                'change': change,
                'market_cap': mcap,
            })

        if not alternatives:
            print(f"[find_alternatives] No valid quotes for peers of {code}")
            cache_set(cache_key, [])
            return []

        # Sort by PE (low = value), fall back to name if no PE
        alternatives.sort(key=lambda x: x['pe'] if x['pe'] > 0 else 999)
        top = alternatives[:6]

        # Run analyze_stock for each top alternative
        result = []
        for a in top:
            wc = a['wc']
            full_code = a['code'] + ('.SH' if wc.startswith('sh') else '.SZ')
            a['code_full'] = full_code
            try:
                analysis = analyze_stock(full_code)
                if analysis and 'error' not in analysis:
                    a['total_score'] = analysis.get('total_score', 0)
                    a['max_score'] = analysis.get('max_score', 100)
                    a['recommendation'] = analysis.get('recommendation', '')
                    a['scores_breakdown'] = analysis.get('scores', {})
                    a['roe'] = (analysis.get('scores', {}).get('fundamental', {}).get('detail', {}) or {}).get('latest_roe', 0)
                    a['dividend_yield'] = (analysis.get('scores', {}).get('fundamental', {}).get('detail', {}) or {}).get('dividend_yield', 0)
                    a['peg'] = (analysis.get('scores', {}).get('value', {}).get('detail', {}) or {}).get('peg', 0)
                    a['industry_board'] = (analysis.get('raw', {}).get('industry', {}) or {}).get('board', '')
                    result.append(a)
            except Exception as e:
                print(f'[find_alternatives] analyze_stock failed for {full_code}: {e}')

        # Sort by total_score desc
        result.sort(key=lambda x: x.get('total_score', 0), reverse=True)
        result = result[:4]

        # Ensure code_full is set
        for a in result:
            if 'code_full' not in a:
                wc = a.get('wc', '')
                a['code_full'] = a['code'] + ('.SH' if wc.startswith('sh') else '.SZ')

        print(f"[find_alternatives] Final result: {len(result)} alternatives for {code}")
        cache_set(cache_key, result)
        return result

    except Exception as e:
        print(f'[find_alternatives] Error: {e}')
        import traceback
        traceback.print_exc()
        return []


# ============================================================
# Lightweight Scoring & Multi-mode Alternatives
# ============================================================

def _get_all_candidate_codes():
    """Return all stock codes from the static industry mapping in api_fallback.
    Returns deduplicated list of (code, name, wc) tuples."""
    seen = set()
    candidates = []
    for cat, stocks in api_fallback.STATIC_PEERS.items():
        for c, n, wc in stocks:
            if c not in seen:
                seen.add(c)
                candidates.append((c, n, wc))
    return candidates


def _lightweight_score(candidate, context=None):
    """
    Pure math scoring — zero API calls, always succeeds.
    
    Dimensions:
      PE score  (0-30): Lower PE = better value
      PB score  (0-20): Lower PB = better value
      MCap score(0-15): Larger market cap = more stable
      Momentum  (0-15): Healthy range preferred
      Quality   (0-20): ROE + stability composite
    
    context: optional dict with extra info (e.g., primary stock price for similarity)
    """
    pe = candidate.get('pe', 0) or 0
    pb = candidate.get('pb', 0) or 0
    mcap = candidate.get('mcap', 0) or candidate.get('market_cap', 0) or 0
    change = candidate.get('change_pct', 0) or candidate.get('change', 0) or 0
    name = candidate.get('name', '')
    
    score = 0
    breakdown = {}
    
    # 1. PE score (0-30)
    if 0 < pe <= 8:
        pe_score = 30
        pe_note = "极低估值"
    elif 8 < pe <= 15:
        pe_score = 25
        pe_note = "低估值"
    elif 15 < pe <= 25:
        pe_score = 20
        pe_note = "合理偏低"
    elif 25 < pe <= 40:
        pe_score = 15
        pe_note = "合理"
    elif 40 < pe <= 80:
        pe_score = 8
        pe_note = "偏高"
    elif pe > 80:
        pe_score = 3
        pe_note = "极高"
    else:
        pe_score = 10
        pe_note = "PE不可用"
    score += pe_score
    breakdown['pe'] = {'score': pe_score, 'note': pe_note, 'value': pe}
    
    # 2. PB score (0-20)
    if 0 < pb <= 1:
        pb_score = 20
        pb_note = "破净/极低"
    elif 1 < pb <= 2:
        pb_score = 16
        pb_note = "低PB"
    elif 2 < pb <= 4:
        pb_score = 12
        pb_note = "合理"
    elif 4 < pb <= 8:
        pb_score = 7
        pb_note = "偏高"
    elif pb > 8:
        pb_score = 3
        pb_note = "极高PB"
    else:
        pb_score = 8
        pb_note = "PB不可用"
    score += pb_score
    breakdown['pb'] = {'score': pb_score, 'note': pb_note, 'value': pb}
    
    # 3. Market cap score (0-15)
    mcap_yi = mcap / 1e8  # convert to 亿
    if mcap_yi > 5000:
        mcap_score = 15
        mcap_note = "超大盘蓝筹"
    elif mcap_yi > 1000:
        mcap_score = 13
        mcap_note = "大盘"
    elif mcap_yi > 300:
        mcap_score = 10
        mcap_note = "中盘"
    elif mcap_yi > 100:
        mcap_score = 7
        mcap_note = "中小盘"
    elif mcap_yi > 0:
        mcap_score = 4
        mcap_note = "小盘"
    else:
        mcap_score = 5
        mcap_note = "市值未知"
    score += mcap_score
    breakdown['mcap'] = {'score': mcap_score, 'note': mcap_note, 'value': round(mcap_yi, 0)}
    
    # 4. Momentum score (0-15)
    if -2 <= change <= 2:
        momentum_score = 12
        momentum_note = "走势平稳"
    elif 2 < change <= 5:
        momentum_score = 14
        momentum_note = "温和上涨"
    elif change > 5:
        momentum_score = 10
        momentum_note = "短期过涨"
    elif -5 <= change < -2:
        momentum_score = 8
        momentum_note = "温和下跌"
    else:
        momentum_score = 5
        momentum_note = "短期急跌"
    score += momentum_score
    breakdown['momentum'] = {'score': momentum_score, 'note': momentum_note, 'value': change}
    
    # 5. Quality / Composite score (0-20)
    quality_score = 10  # baseline
    
    # Blue chip names get bonus
    blue_chip_names = ['贵州茅台', '五粮液', '中国平安', '招商银行', '美的集团',
                       '格力电器', '长江电力', '中国神华', '宁德时代', '比亚迪',
                       '恒瑞医药', '迈瑞医疗', '海天味业', '伊利股份', '中国移动',
                       '工商银行', '建设银行', '农业银行', '中国银行']
    if any(bc in name for bc in blue_chip_names):
        quality_score += 5
        quality_note = "知名蓝筹"
    elif mcap_yi > 500:
        quality_score += 3
        quality_note = "大市值优质"
    elif 100 <= mcap_yi <= 500:
        quality_score += 1
        quality_note = "中等市值"
    else:
        quality_note = "小市值"
    
    score += quality_score
    breakdown['quality'] = {'score': quality_score, 'note': quality_note}
    
    return {
        'total_score': min(100, score),
        'breakdown': breakdown,
        'recommendation': (
            '强烈推荐' if score >= 80 else
            '推荐' if score >= 65 else
            '中性' if score >= 50 else
            '谨慎' if score >= 35 else
            '不推荐'
        ),
    }


# ========== HK / US Alternatives Support ==========
def _to_float(v, default=0.0):
    try:
        if v is None:
            return default
        if isinstance(v, str):
            v = v.strip().replace(",", "").replace("%", "")
            if v in ("", "-", "--", "None", "null"):
                return default
        f = float(v)
        if f != f:  # NaN
            return default
        return f
    except Exception:
        return default


# Coarse sector mapping for HK/US (real industry names + name keywords)
_SECTOR_MAP = {
    "银行": "银行", "保险": "保险", "证券": "非银金融", "信托": "非银金融",
    "软件": "科技", "软件服务": "科技", "互联网": "互联网", "科技": "科技",
    "半导体": "科技", "电子": "科技", "计算机": "科技", "通讯": "通讯", "通信": "通讯",
    "汽车": "汽车", "新能源": "汽车", "电池": "汽车", "电动车": "汽车",
    "医药": "医药", "生物": "医药", "医疗": "医药", "健康": "医药", "制药": "医药",
    "地产": "地产", "房地产": "地产", "置业": "地产",
    "能源": "能源", "石油": "能源", "煤炭": "能源", "电力": "能源", "燃气": "能源", "煤气": "能源",
    "消费": "消费", "零售": "消费", "食品": "消费", "饮料": "消费", "白酒": "消费", "家电": "消费",
    "传媒": "传媒", "教育": "传媒", "旅游": "消费", "航空": "消费", "物流": "消费",
    "化工": "材料", "材料": "材料", "金属": "材料", "钢铁": "材料", "有色金属": "材料",
    "机械": "工业", "工业": "工业", "军工": "工业", "建筑": "工业", "工程": "工业",
    "农业": "农业",
}


def _coarse_sector(industry_name="", name=""):
    text = industry_name or ""
    for k, v in _SECTOR_MAP.items():
        if k in text:
            return v
    nm = name or ""
    for kw, sec in [
        ("银行", "银行"), ("保险", "保险"), ("证券", "非银金融"),
        ("科技", "科技"), ("芯片", "科技"), ("半导体", "科技"), ("电子", "科技"),
        ("软件", "科技"), ("互联网", "互联网"),
        ("腾讯", "互联网"), ("阿里", "互联网"), ("京东", "互联网"), ("美团", "互联网"),
        ("快手", "互联网"), ("网易", "互联网"), ("百度", "互联网"), ("拼多多", "互联网"),
        ("携程", "互联网"), ("小米", "科技"), ("苹果", "科技"), ("微软", "科技"),
        ("谷歌", "科技"), ("英伟达", "科技"), ("英伟", "科技"), ("特斯拉", "汽车"),
        ("亚马逊", "互联网"), ("脸书", "科技"), ("Meta", "科技"), ("英特尔", "科技"),
        ("高通", "科技"), ("美光", "科技"), ("AMD", "科技"), ("超威", "科技"),
        ("台积", "科技"), ("博通", "科技"), ("奈飞", "传媒"), ("迪士尼", "传媒"),
        ("汽车", "汽车"), ("新能源", "汽车"), ("电池", "汽车"), ("电动", "汽车"),
        ("比亚迪", "汽车"), ("蔚来", "汽车"), ("小鹏", "汽车"), ("理想", "汽车"),
        ("医药", "医药"), ("生物", "医药"), ("医疗", "医药"), ("健康", "医药"),
        ("制药", "医药"), ("辉瑞", "医药"), ("强生", "医药"), ("默沙东", "医药"),
        ("礼来", "医药"), ("罗氏", "医药"), ("诺和", "医药"),
        ("地产", "地产"), ("置业", "地产"), ("发展", "地产"), ("恒隆", "地产"),
        ("能源", "能源"), ("石油", "能源"), ("煤炭", "能源"), ("电力", "能源"),
        ("燃气", "能源"), ("煤气", "能源"), ("埃克森", "能源"), ("雪佛龙", "能源"),
        ("消费", "消费"), ("零售", "消费"), ("食品", "消费"), ("饮料", "消费"),
        ("酒", "消费"), ("家电", "消费"), ("沃尔玛", "消费"), ("宝洁", "消费"),
        ("可口", "消费"), ("星巴克", "消费"), ("麦当劳", "消费"), ("耐克", "消费"),
        ("通讯", "通讯"), ("电信", "通讯"), ("移动", "通讯"), ("联通", "通讯"),
        ("传媒", "传媒"), ("教育", "传媒"),
        ("材料", "材料"), ("化工", "材料"), ("金属", "材料"), ("钢铁", "材料"),
        ("机械", "工业"), ("军工", "工业"), ("建筑", "工业"), ("工程", "工业"),
        ("农业", "农业"), ("农", "农业"),
        ("摩根", "非银金融"), ("花旗", "非银金融"), ("高盛", "非银金融"),
        ("富国", "非银金融"), ("贝莱德", "非银金融"),
        # English-name keywords (curated US list uses English names)
        ("Apple", "科技"), ("Microsoft", "科技"), ("Alphabet", "科技"), ("Google", "科技"),
        ("Amazon", "互联网"), ("NVIDIA", "科技"), ("Nvidia", "科技"), ("Meta", "科技"),
        ("Tesla", "汽车"), ("AMD", "科技"), ("Intel", "科技"), ("Qualcomm", "科技"),
        ("Micron", "科技"), ("Broadcom", "科技"), ("Netflix", "传媒"), ("Disney", "传媒"),
        ("Berkshire", "其他"), ("JPMorgan", "非银金融"), ("AT&T", "通讯"), ("Verizon", "通讯"),
        ("Walmart", "消费"), ("P&G", "消费"), ("Coca", "消费"), ("Starbucks", "消费"),
        ("Pfizer", "医药"), ("Johnson", "医药"), ("Merck", "医药"), ("Exxon", "能源"),
        ("Chevron", "能源"), ("Caterpillar", "工业"), ("Boeing", "工业"), ("IBM", "科技"),
        ("Oracle", "科技"), ("Salesforce", "科技"), ("Adobe", "科技"), ("PayPal", "互联网"),
        ("Uber", "互联网"), ("Airbnb", "互联网"), ("Visa", "非银金融"), ("Mastercard", "非银金融"),
        ("Bank", "银行"), ("Goldman", "非银金融"), ("Morgan", "非银金融"),
        ("Citigroup", "非银金融"), ("Wells", "银行"), ("Applied", "科技"), ("Texas", "科技"),
        ("Lam", "科技"), ("TSMC", "科技"), ("Taiwan", "科技"), ("ASML", "科技"),
        ("Nike", "消费"), ("McDonald", "消费"), ("Procter", "消费"), ("Pepsi", "消费"),
        ("Advanced", "科技"), ("Netflix", "传媒"), ("Comcast", "传媒"), ("Fox", "传媒"),
        ("Union", "工业"), ("Home", "消费"), ("Lowe", "消费"), ("Target", "消费"),
        ("Costco", "消费"), ("AbbVie", "医药"), ("Abbott", "医药"), ("Amgen", "医药"),
        ("Gilead", "医药"), ("Moderna", "医药"), ("Novartis", "医药"), ("Roche", "医药"),
        ("Bristol", "医药"), ("CVS", "消费"), ("Walgreens", "消费"), ("Conoco", "能源"),
        ("Marathon", "能源"), ("Phillips", "能源"), ("Schlumberger", "能源"),
        ("Halliburton", "能源"), ("Deere", "工业"), ("Lockheed", "工业"),
        ("General Electric", "工业"), ("Honeywell", "工业"), ("Ford", "汽车"),
        ("General Motors", "汽车"), ("Toyota", "汽车"), ("Honda", "汽车"),
        ("NIO", "汽车"), ("XPeng", "汽车"), ("Li Auto", "汽车"), ("Rivian", "汽车"),
        ("Lucid", "汽车"), ("BYD", "汽车"), ("Intel", "科技"),
    ]:
        if kw in nm:
            return sec
    return "其他"


def _batch_get_quotes_trusted(wc_codes):
    """Batch Tencent quotes via trust_env=False session (bypasses proxy).
    Plain requests.get (used by _batch_get_quotes) fails HK/US batches through
    the local proxy, so HK/US uses this instead. Keys match _batch_get_quotes."""
    result = {}
    if not wc_codes:
        return result
    TENCENT_QT = "https://qt.gtimg.cn/q="
    batch_size = 40
    for i in range(0, len(wc_codes), batch_size):
        batch = wc_codes[i:i + batch_size]
        q = ",".join(batch)
        try:
            resp = _http_session.get(TENCENT_QT + q, headers=REQ_HEADERS, timeout=15)
            if resp.status_code != 200:
                continue
            for line in resp.text.strip().split("\n"):
                if "=" not in line:
                    continue
                var_name = line.split("=")[0].strip()
                wc = var_name[2:] if var_name.startswith("v_") else var_name
                body = line.split("=", 1)[1].strip().strip('"')
                f = body.split("~")
                if len(f) < 30:
                    continue
                try:
                    result[wc] = {
                        "name": f[1],
                        "price": _to_float(f[3]),
                        "pe": _to_float(f[39]),
                        "pb": f[46],   # string (english name) for HK/US -> coerced to 0 later
                        "change_pct": _to_float(f[32]),
                        # Tencent f[45] is market cap in 亿 (e.g. 腾讯=41607). A-share
                        # path multiplies by 1e8 to get 元; do the same here so the
                        # frontend's `/1e8 -> 亿` and the mcap scoring stay consistent.
                        "mcap": _to_float(f[45]) * 1e8 if f[45] else 0,
                    }
                except Exception:
                    continue
        except Exception as e:
            print(f"[_batch_get_quotes_trusted] batch error: {e}")
            continue
    return result


def _get_hk_us_pool(market):
    """Live-priced candidate pool for HK/US from the curated list."""
    try:
        from hk_us_list import HK_US_STOCKS
    except Exception:
        return []
    base = HK_US_STOCKS.get(market, [])
    if not base:
        return []
    wcs = [f"hk{c}" if market == "HK" else f"us{c.upper()}" for c, _ in base]
    quotes = _batch_get_quotes_trusted(wcs)
    pool = []
    for code, name in base:
        wc = f"hk{code}" if market == "HK" else f"us{code.upper()}"
        q = quotes.get(wc, {}) or {}
        pb_raw = q.get("pb")
        pb = _to_float(pb_raw) if isinstance(pb_raw, (int, float)) else 0.0
        pool.append({
            "code": code,
            "name": name,
            "wc": wc,
            "code_full": (code + ".HK") if market == "HK" else (code.upper() + ".US"),
            "sector": _coarse_sector(name=name),
            "price": _to_float(q.get("price")),
            "pe": _to_float(q.get("pe")),
            "pb": pb,
            "change": _to_float(q.get("change_pct")),
            "market_cap": _to_float(q.get("mcap")),
        })
    return pool


def _hk_us_price_similar(code, market, symbol):
    wc = f"hk{symbol}" if market == "HK" else f"us{symbol.upper()}"
    q = _batch_get_quotes_trusted([wc]).get(wc, {}) or {}
    price = _to_float(q.get("price"))
    if price <= 0:
        return []
    pool = _get_hk_us_pool(market)
    lo, hi = price * 0.75, price * 1.25
    in_range = [p for p in pool if p["code"] != symbol and p["price"] > 0 and lo <= p["price"] <= hi]
    if len(in_range) < 4:
        others = [p for p in pool if p["code"] != symbol and p["price"] > 0]
        others.sort(key=lambda x: abs(x["price"] - price))
        seen = {p["code"] for p in in_range}
        for p in others:
            if p["code"] not in seen:
                in_range.append(p)
                seen.add(p["code"])
            if len(in_range) >= 4:
                break
    for c in in_range:
        sr = _lightweight_score(c, context={"primary_price": price})
        c["total_score"] = sr["total_score"]
        c["recommendation"] = sr["recommendation"]
        c["scores_breakdown"] = sr["breakdown"]
    in_range.sort(key=lambda x: x["total_score"], reverse=True)
    return in_range[:4]


def _hk_us_recommended(code, market, symbol):
    pool = _get_hk_us_pool(market)
    primary = next((p for p in pool if p["code"] == symbol), None)
    prim_sector = primary["sector"] if primary else "其他"
    cands = [p for p in pool if p["code"] != symbol and p["price"] > 0]
    for c in cands:
        sr = _lightweight_score(c)
        c["total_score"] = sr["total_score"]
        c["recommendation"] = sr["recommendation"]
        c["scores_breakdown"] = sr["breakdown"]
        if c["sector"] == prim_sector:
            c["total_score"] -= 8  # diversify away from same sector
    cands.sort(key=lambda x: x["total_score"], reverse=True)
    return cands[:4]


def _hk_us_same_industry(code, market, symbol):
    try:
        ind = _get_market_industry(code, market)
        ind_name = (ind.get("industry_name", "") or "")
    except Exception:
        ind_name = ""
    # Primary company name: prefer the curated list (always available, no network
    # call), fall back to get_stock_info only if the list lookup misses.
    primary_name = ""
    try:
        from hk_us_list import HK_US_STOCKS
        for c, n in HK_US_STOCKS.get(market, []):
            if c == symbol:
                primary_name = n
                break
    except Exception:
        pass
    if not primary_name:
        try:
            primary_name = get_stock_info(code, market).get("name", "") or ""
        except Exception:
            pass
    pool = _get_hk_us_pool(market)
    # The pool assigns each stock a sector from its NAME only (see _get_hk_us_pool).
    # The primary's industry_name (e.g. 阿里 "专业零售"->消费) can diverge from its
    # name-based sector (阿里 "互联网"), causing a mismatch. Try the industry_name-
    # derived sector first, then the name-derived one so we align with the pool.
    # Never use the catch-all "其他" as a match sector (it would match the whole
    # pool and defeat the purpose).
    cand_sectors = []
    for s in (_coarse_sector(ind_name, ""), _coarse_sector("", primary_name)):
        if s and s != "其他" and s not in cand_sectors:
            cand_sectors.append(s)
    peers = []
    for sec in cand_sectors:
        peers = [p for p in pool if p["code"] != symbol and p["sector"] == sec and p["price"] > 0]
        if len(peers) >= 3:
            break
    if len(peers) < 3:
        # Generic fallback: lowest-PE liquid names across the market (not pool order)
        peers = sorted(
            [p for p in pool if p["code"] != symbol and p["price"] > 0],
            key=lambda x: x["pe"] if x["pe"] > 0 else 999,
        )[:12]
    peers.sort(key=lambda x: x["pe"] if x["pe"] > 0 else 999)
    return peers[:6]


def find_price_similar(code):
    """
    Find stocks with similar price range (±25%).
    Multi-layer fallback for Vercel resilience:
      1. Batch quotes → filter by price range ±25%
      2. If < 4 results, try individual Sina quotes for known candidates
      3. If still < 4, use price bracket presets from static pool
    """
    cache_key = f"ps_{code}"
    cached_val = cached(cache_key, ttl=600)
    if cached_val:
        return cached_val

    market, symbol = detect_market(code)
    if market in ("HK", "US"):
        res = _hk_us_price_similar(code, market, symbol)
        cache_set(cache_key, res)
        return res

    try:
        # Get primary stock price
        info = get_stock_info(code)
        price = info.get('最新价', 0)
        if not price:
            return []

        symbol = code.replace('.SZ', '').replace('.SH', '').replace('.BJ', '')
        lo = price * 0.75
        hi = price * 1.25

        # ---- Layer 1: Static pool + batch quotes ----
        all_candidates = _get_all_candidate_codes()
        all_candidates = [x for x in all_candidates if x[0] != symbol]

        if not all_candidates:
            return []

        wc_codes = [x[2] for x in all_candidates]
        quotes = _batch_get_quotes(wc_codes)

        # Filter by price range
        in_range = []
        out_of_range = []
        no_data = []
        for c, n, wc in all_candidates:
            q = quotes.get(wc, {})
            p = q.get('price', 0) or 0
            cand = {
                'code': c, 'name': n, 'wc': wc, 'price': p,
                'pe': q.get('pe', 0) or 0, 'pb': q.get('pb', 0) or 0,
                'change': q.get('change_pct', 0) or 0,
                'market_cap': q.get('mcap', 0) or 0,
            }
            if p > 0 and lo <= p <= hi:
                in_range.append(cand)
            elif p > 0:
                out_of_range.append(cand)
            else:
                no_data.append(cand)

        # ---- Layer 2: If not enough in-range, try Sina single quotes for no_data ----
        if len(in_range) < 4 and no_data:
            print(f"[find_price_similar] Layer 1 only {len(in_range)} in range, trying Sina singles for {len(no_data)} without data")
            for cand in no_data[:20]:  # try up to 20
                try:
                    fb = api_fallback.sina_get_stock_info(cand['wc'] if cand['wc'].startswith('sh') else cand['wc'])
                    p = fb.get('最新价', 0)
                    if p > 0:
                        cand['price'] = p
                        cand['name'] = fb.get('股票简称', cand['name'])
                        if lo <= p <= hi:
                            in_range.append(cand)
                        else:
                            out_of_range.append(cand)
                except:
                    pass

        # ---- Layer 3: If still not enough, include out_of_range sorted by price proximity ----
        if len(in_range) < 4 and out_of_range:
            out_of_range.sort(key=lambda x: abs(x['price'] - price))
            needed = 4 - len(in_range)
            print(f"[find_price_similar] Layer 3: adding {needed} closest out-of-range stocks")
            for cand in out_of_range[:needed]:
                in_range.append(cand)

        # ---- Layer 4: Last resort, include no_data as name-only ----
        if len(in_range) < 4 and no_data:
            needed = 4 - len(in_range)
            print(f"[find_price_similar] Layer 4: adding {needed} name-only stocks")
            for cand in no_data[:needed]:
                cand['price'] = 0  # will show as "--" in UI
                in_range.append(cand)

        # Score all candidates
        for cand in in_range:
            score_result = _lightweight_score(cand)
            cand['total_score'] = score_result['total_score']
            cand['recommendation'] = score_result['recommendation']
            cand['scores_breakdown'] = score_result['breakdown']
            cand['code_full'] = cand['code'] + ('.SH' if cand['wc'].startswith('sh') else '.SZ')

        # Sort by score, take top 4
        in_range.sort(key=lambda x: x['total_score'], reverse=True)
        result = in_range[:4]

        print(f"[find_price_similar] Price range ¥{lo:.2f}-{hi:.2f}, returning {len(result)} candidates")
        cache_set(cache_key, result)
        return result

    except Exception as e:
        print(f"[find_price_similar] Error: {e}")
        import traceback
        traceback.print_exc()
        return []
        
        print(f"[find_price_similar] Price range ¥{lo:.2f}-{hi:.2f}, found {len(candidates)} total, returning {len(result)}")
        cache_set(cache_key, result)
        return result
        
    except Exception as e:
        print(f"[find_price_similar] Error: {e}")
        import traceback
        traceback.print_exc()
        return []


def find_recommended(code):
    """
    Find comprehensive recommended stocks across all industries.
    Uses static stock pool + batch quotes + lightweight scoring.
    Excludes stocks from the same industry to avoid overlap with industry tab.
    """
    cache_key = f"rc_{code}"
    cached_val = cached(cache_key, ttl=600)
    if cached_val:
        return cached_val

    market, symbol = detect_market(code)
    if market in ("HK", "US"):
        res = _hk_us_recommended(code, market, symbol)
        cache_set(cache_key, res)
        return res

    try:
        symbol = code.replace('.SZ', '').replace('.SH', '').replace('.BJ', '')
        
        # Try to get industry peers (for exclusion)
        try:
            ind_data = get_industry_data(code)
            peer_codes = _get_industry_peers_by_board(code)
            same_ind_codes = set(p['code'] for p in peer_codes) if peer_codes else set()
            # Try static peers too
            static_peers = api_fallback.static_get_industry_peers(
                code,
                ind_data.get('industry_name', ''),
                ind_data.get('board_name', '')
            )
            if static_peers:
                same_ind_codes.update(p['code'] for p in static_peers)
        except:
            same_ind_codes = set()
        same_ind_codes.add(symbol)
        
        # ---- Robust exclusion: use code-based category lookup ----
        # If API data failed and same_ind_codes is just {symbol},
        # use the static mapping to exclude the entire category
        if len(same_ind_codes) <= 1:
            primary_cat = api_fallback.static_get_industry_for_code(code)
            if primary_cat:
                for cat, stocks in api_fallback.STATIC_PEERS.items():
                    if cat == primary_cat:
                        for c, n, wc in stocks:
                            same_ind_codes.add(c)
                        break
                print(f"[find_recommended] Excluding {len(same_ind_codes)-1} stocks from '{primary_cat}' category")
        
        # Get all candidates from static pool
        all_candidates = _get_all_candidate_codes()
        all_candidates = [x for x in all_candidates if x[0] not in same_ind_codes]
        
        if not all_candidates:
            return []
        
        # Batch get quotes
        wc_codes = [x[2] for x in all_candidates]
        quotes = _batch_get_quotes(wc_codes)
        
        # Score all candidates
        candidates = []
        for c, n, wc in all_candidates:
            q = quotes.get(wc, {})
            p = q.get('price', 0) or 0
            if p <= 0:
                continue
            cand = {
                'code': c,
                'name': n,
                'wc': wc,
                'price': p,
                'pe': q.get('pe', 0) or 0,
                'pb': q.get('pb', 0) or 0,
                'change': q.get('change_pct', 0) or 0,
                'market_cap': q.get('mcap', 0) or 0,
            }
            score_result = _lightweight_score(cand)
            cand['total_score'] = score_result['total_score']
            cand['recommendation'] = score_result['recommendation']
            cand['scores_breakdown'] = score_result['breakdown']
            cand['code_full'] = c + ('.SH' if wc.startswith('sh') else '.SZ')
            candidates.append(cand)
        
        # Sort by score, take top 4
        candidates.sort(key=lambda x: x['total_score'], reverse=True)
        result = candidates[:4]
        
        print(f"[find_recommended] Scored {len(candidates)} candidates, returning {len(result)}")
        cache_set(cache_key, result)
        return result
        
    except Exception as e:
        print(f"[find_recommended] Error: {e}")
        import traceback
        traceback.print_exc()
        return []

@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    if request.method == "GET":
        cfg = load_config()
        if "ai_chat" not in cfg:
            cfg["ai_chat"] = {}
        ak = cfg["ai_chat"].get("api_key", "")
        if len(ak) > 8:
            cfg["ai_chat"]["api_key"] = ak[:8] + "****"
        elif ak:
            cfg["ai_chat"]["api_key"] = "****"
        return jsonify(cfg)
    else:
        data = request.json
        cfg = load_config()
        if "ai_chat" not in cfg:
            cfg["ai_chat"] = {}
        if data.get("ai_chat"):
            for k in ["provider", "api_key", "api_base", "model", "system_prompt"]:
                if k in data["ai_chat"] and data["ai_chat"][k]:
                    cfg["ai_chat"][k] = data["ai_chat"][k]
        save_config(cfg)
        return jsonify({"status": "ok"})
@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.json
    code = data.get("code", "").strip()
    if not code:
        return jsonify({"error": "请输入股票代码"}), 400

    report = analyze_stock(code)
    if "error" in report:
        return jsonify({"error": report["error"]}), 400

    # Attach fallback meta for console debugging
    report["_meta"] = {"fb": api_fallback.get_fallback_log()[-10:]}
    return jsonify(report)


@app.route("/api/alternatives", methods=["POST"])
def alternatives():
    data = request.json
    code = data.get("code", "").strip()
    if not code:
        return jsonify({"error": "请输入股票代码"}), 400

    alts = find_alternatives(code)
    return jsonify({
        "alternatives": alts,
        "_meta": {"fb": api_fallback.get_fallback_log()[-10:]}
    })


# ============================================================
# Progressive Alternatives — Base + Score + Cache
# ============================================================

from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

@app.route("/api/alternatives/base", methods=["POST"])
def alternatives_base():
    """Fast preview: returns candidate lists with basic data + lightweight scores.
    No full analyze_stock calls — responds in < 3s."""
    data = request.json
    code = data.get("code", "").strip()
    if not code:
        return jsonify({"error": "请输入股票代码"}), 400

    cache_key = f"altbase_{code}"
    cached_val = cached(cache_key, ttl=600)
    if cached_val:
        t = cached_val.pop("_cache_time", "")
        cached_val["_cache_meta"] = {"time": t, "from_cache": True, "ttl": 600}
        return jsonify(cached_val)

    import time as _base_time
    _t0 = _base_time.time()
    result = {}

    try:
        result["industry"] = find_alternatives(code)
    except Exception as e:
        print(f"[alt_base] industry error: {e}")
        result["industry"] = []

    try:
        result["price_similar"] = find_price_similar(code)
    except Exception as e:
        print(f"[alt_base] price_sim error: {e}")
        result["price_similar"] = []

    try:
        result["recommended"] = find_recommended(code)
    except Exception as e:
        print(f"[alt_base] recommended error: {e}")
        result["recommended"] = []

    cache_time = now_cn().strftime("%Y-%m-%d %H:%M:%S")
    result["_cache_time"] = cache_time
    result["_cache_meta"] = {"time": cache_time, "from_cache": False, "ttl": 600}
    result["_meta"] = {"fb": api_fallback.get_fallback_log()[-15:]}

    _elapsed = _base_time.time() - _t0
    print(f"[alt_base] Completed in {_elapsed:.2f}s for {code}")

    cache_set(cache_key, result)
    return jsonify(result)


@app.route("/api/alternatives/score", methods=["POST"])
def alternatives_score():
    """Batch full scoring: analyze_stock() for up to 4 codes.
    ThreadPoolExecutor with 10s total timeout."""
    data = request.json
    codes = data.get("codes", [])
    if not codes:
        return jsonify({"error": "请提供股票代码列表"}), 400
    codes = codes[:4]

    cache_key = f"altscore_{'_'.join(sorted(codes))}"
    cached_val = cached(cache_key, ttl=600)
    if cached_val:
        t = cached_val.pop("_cache_time", "")
        cached_val["_cache_meta"] = {"time": t, "from_cache": True, "ttl": 600}
        return jsonify(cached_val)

    import time as _sc_time
    _t0 = _sc_time.time()
    scores = []
    errors = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {executor.submit(analyze_stock, c): c for c in codes}

        try:
            for future in as_completed(future_map, timeout=10):
                stock_code = future_map[future]
                try:
                    analysis = future.result(timeout=8)
                    if analysis and "error" not in analysis:
                        # Extract financial metrics for deep comparison
                        fund_detail = (analysis.get("scores", {}).get("fundamental", {}).get("detail", {}) or {})
                        value_detail = (analysis.get("scores", {}).get("value", {}).get("detail", {}) or {})
                        scores.append({
                            "code": stock_code,
                            "code_full": stock_code,
                            "name": analysis.get("name", ""),
                            "total_score": analysis.get("total_score", 0),
                            "max_score": analysis.get("max_score", 100),
                            "recommendation": analysis.get("recommendation", ""),
                            "scores_breakdown": analysis.get("scores", {}),
                            "pe": analysis.get("pe", 0),
                            "pb": analysis.get("pb", 0),
                            "roe": fund_detail.get("latest_roe") or fund_detail.get("roe", 0),
                            "dividend_yield": fund_detail.get("dividend_yield", 0),
                            "peg": value_detail.get("peg"),  # None if can't calc (negative growth)
                            "source": "full",
                        })
                    else:
                        errors.append({"code": stock_code, "error": str(analysis.get("error", "unknown") if analysis else "no result")})
                except FuturesTimeoutError:
                    errors.append({"code": stock_code, "error": "timeout (8s)"})
                except Exception as e:
                    errors.append({"code": stock_code, "error": str(e)[:80]})
        except FuturesTimeoutError:
            pass  # batch-level timeout

    # Fill in failed codes with empty scores
    completed = {s["code"] for s in scores}
    for c in codes:
        if c not in completed:
            scores.append({"code": c, "code_full": c, "name": "", "total_score": 0, "source": "failed"})

    _elapsed = _sc_time.time() - _t0
    result = {
        "scores": scores,
        "errors": errors,
        "elapsed": round(_elapsed, 2),
        "_cache_time": now_cn().strftime("%Y-%m-%d %H:%M:%S"),
        "_cache_meta": {"time": now_cn().strftime("%Y-%m-%d %H:%M:%S"), "from_cache": False, "ttl": 600},
        "_meta": {"fb": api_fallback.get_fallback_log()[-10:]}
    }
    cache_set(cache_key, result)
    return jsonify(result)


@app.route("/api/alternatives/cache/clear", methods=["POST"])
def alternatives_cache_clear():
    """Clear alternatives-related caches."""
    data = request.json or {}
    code = data.get("code", "").strip()
    cleared = []

    if code and code != "*":
        targets = [f"altbase_{code}", f"ps_{code}", f"rc_{code}", f"alt_{code}"]
        for k in list(CACHE.keys()):
            if k in targets or (k.startswith("altscore_") and code.replace(".SH", "").replace(".SZ", "") in k):
                cleared.append(k)
    else:
        for k in list(CACHE.keys()):
            if any(k.startswith(p) for p in ["altbase_", "altscore_", "alt_", "ps_", "rc_"]):
                cleared.append(k)

    for k in cleared:
        if k in CACHE:
            del CACHE[k]
        if k + "_ttl" in CACHE:
            del CACHE[k + "_ttl"]

    return jsonify({"status": "ok", "cleared": len(cleared), "keys": cleared[:20]})


@app.route("/api/alternatives/all", methods=["POST"])
def alternatives_all():
    """Backward-compatible: all 3 modes with lightweight scores.
    For progressive full scoring, use /api/alternatives/base + /api/alternatives/score."""
    data = request.json
    code = data.get("code", "").strip()
    if not code:
        return jsonify({"error": "请输入股票代码"}), 400

    try:
        industry = find_alternatives(code)
    except Exception as e:
        print(f"[alternatives_all] industry error: {e}")
        industry = []

    try:
        price_similar = find_price_similar(code)
    except Exception as e:
        print(f"[alternatives_all] price_similar error: {e}")
        price_similar = []

    try:
        recommended = find_recommended(code)
    except Exception as e:
        print(f"[alternatives_all] recommended error: {e}")
        recommended = []

    return jsonify({
        "industry": industry,
        "price_similar": price_similar,
        "recommended": recommended,
        "_meta": {"fb": api_fallback.get_fallback_log()[-15:]}
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "").strip()
    stock_context = data.get("stock_context", "")

    if not message:
        return jsonify({"error": "请输入消息"}), 400

    chat_cfg = get_ai_config()

    if not chat_cfg.get("api_key") or chat_cfg["api_key"].startswith("sk-your"):
        return jsonify({
            "reply": "AI 聊天功能尚未配置 API Key。请在页面右上角的「设置」中配置您的 OpenAI 兼容 API Key。\n\n支持的 API 提供商：\n- OpenAI: https://api.openai.com/v1\n- 硅基流动: https://api.siliconflow.cn/v1\n- DeepSeek: https://api.deepseek.com/v1\n- 其他 OpenAI 兼容接口",
            "need_config": True
        })

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=chat_cfg["api_key"],
            base_url=chat_cfg.get("api_base", "https://api.openai.com/v1"),
        )

        system_prompt = chat_cfg.get("system_prompt", "你是一位专业的股票投资分析师。")
        if stock_context:
            system_prompt += f"\n\n当前正在分析的股票信息：\n{stock_context}"

        response = client.chat.completions.create(
            model=chat_cfg.get("model", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            temperature=0.7,
            max_tokens=2000,
        )

        reply = response.choices[0].message.content
        return jsonify({"reply": reply})

    except Exception as e:
        error_msg = str(e)
        return jsonify({
            "reply": f"AI 服务调用失败：{error_msg}\n\n请检查：\n1. API Key 是否正确\n2. API Base URL 是否正确\n3. 网络连接是否正常\n4. 账户余额是否充足",
            "error": True
        })


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


@app.route("/api/test_chat", methods=["POST"])
def test_chat():
    """Test the chat configuration."""
    chat_cfg = get_ai_config()

    if not chat_cfg.get("api_key") or chat_cfg["api_key"].startswith("sk-your"):
        return jsonify({"success": False, "message": "API Key 未配置"})

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=chat_cfg["api_key"],
            base_url=chat_cfg.get("api_base", "https://api.openai.com/v1"),
        )
        response = client.chat.completions.create(
            model=chat_cfg.get("model", "gpt-4o-mini"),
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10,
        )
        return jsonify({"success": True, "message": "连接成功！"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


# ---------- Deep Analysis Cache ----------
DEEP_CACHE = {}  # key: "code:dim" -> (reply, timestamp)
DEEP_CACHE_TTL = 600  # 10 minutes

@app.route("/api/deep_analyze", methods=["POST"])
def deep_analyze():
    """Generate deep analysis for a specific dimension with bull/bear debate."""
    data = request.json
    dim_key = data.get("dim", "")
    stock_data = data.get("stock_data", {})
    force = data.get("force", False)  # force re-analysis, skip cache
    debug_mode = data.get("debug", False)  # return debug info

    if not dim_key:
        return jsonify({"error": "请指定分析维度"}), 400

    stock_code = stock_data.get("code", "")

    # Check cache
    cache_key = f"{stock_code}:{dim_key}"
    if not force and cache_key in DEEP_CACHE:
        cached_reply, cached_ts = DEEP_CACHE[cache_key]
        if time.time() - cached_ts < DEEP_CACHE_TTL:
            return jsonify({
                "reply": cached_reply,
                "dim": dim_key,
                "from_cache": True,
                "cached_at": datetime.fromtimestamp(cached_ts).strftime("%H:%M:%S"),
            })

    chat_cfg = get_ai_config()

    if not chat_cfg.get("api_key") or chat_cfg["api_key"].startswith("sk-your"):
        return jsonify({
            "reply": "⚠️ AI 深度分析需要配置 LLM API Key。请在页面右上角「设置」中配置。",
            "need_config": True
        })

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=chat_cfg["api_key"],
            base_url=chat_cfg.get("api_base", "https://api.openai.com/v1"),
        )

        # Build dimension-specific prompt
        dim_prompts = {
            "fundamental": "基本面（财务状况、盈利能力、成长性、资产负债）",
            "technical": "技术面（K线形态、均线系统、MACD/KDJ/布林带等技术指标）",
            "capital": "资金面（主力资金流向、散户资金动向、量价关系）",
            "events": "事件催化（近期公告、利好利空事件、新闻舆情）",
            "industry": "同业对标（行业竞争格局、估值对比、市场地位）",
            "value": "投资性价比（股息率、估值分位、PEG、风险收益比）",
        }
        dim_name = dim_prompts.get(dim_key, dim_key)

        # Special handling for events: web search + detailed scoring prompt
        if dim_key == "events":
            # Collect event titles
            scores_data = stock_data.get("scores", {})
            events_detail = scores_data.get("events", {}).get("detail", {})
            events_list = events_detail.get("events", [])
            key_events = events_detail.get("key_events", []) or events_list[:5]

            # Build search queries for key events
            stock_name = stock_data.get("name", "")
            search_results = []
            for evt in key_events[:5]:
                evt_title = evt.get("title", "")
                if evt_title and len(evt_title) > 2:
                    query = f"{stock_name} {evt_title}"
                    try:
                        sr = _http_get(
                            "https://www.google.com/search",
                            params={"q": query, "hl": "zh-CN"},
                            timeout=10,
                        )
                        # Extract text snippets from Google results
                        text = sr.text
                        snippets = []
                        # Simple snippet extraction
                        for m in re.finditer(r'<div[^>]*class="[^"]*BNeawe[^"]*"[^>]*>(.*?)</div>', text):
                            snippet = re.sub(r'<[^>]+>', '', m.group(1))
                            if snippet and len(snippet) > 10 and snippet not in snippets:
                                snippets.append(snippet)
                            if len(snippets) >= 3:
                                break
                        if snippets:
                            search_results.append({
                                "event": evt_title,
                                "snippets": snippets[:3],
                            })
                        else:
                            search_results.append({
                                "event": evt_title,
                                "snippets": ["(搜索结果未获取到详细信息)"],
                            })
                    except Exception as e:
                        search_results.append({
                            "event": evt_title,
                            "snippets": [f"(搜索失败: {str(e)})"],
                        })

            # Build the detailed events scoring prompt
            system_prompt = f"""你是一位资深事件驱动分析师。请对以下股票的近期公告和事件进行深度分析，并给出0-10分的评分。

**评分标准（请严格遵守）：**

| 分数 | 描述 | 典型案例 |
|------|------|----------|
| 0分 | 极度利空 | 财务造假曝光、被勒令退市、核心业务毁灭性打击、主要客户永久流失、巨额债务违约 |
| 2分 | 明显偏空 | 大额商誉减值、业绩大幅预亏(>-50%)、大股东大幅减持、被立案调查、核心高管集体离职 |
| 4分 | 偏空/平淡 | 业绩小幅下滑、行业政策收紧、限售股解禁、高管个别离职、增速放缓 |
| 5分 | 中性 | 无重大利好利空、日常经营公告、行业平稳、例行信息披露 |
| 6分 | 温和偏多 | 小额回购/增持、获得小额订单、常规分红公告、签署一般合作协议 |
| 8分 | 明显偏多 | 业绩超预期增长、重大合同中标、新产品获批/重大技术突破、机构密集调研、外资持续增持 |
| 9分 | 强烈利好 | 业绩大幅预增(>50%)、重大战略合作/并购、核心产品供不应求、被纳入重要指数 |
| 10分 | 极度利好 | 行业颠覆性技术突破(本司独有)、获得垄断性牌照、重大资产注入/借壳预期、业绩爆发增长200%+ |

**重要：每个事件都附带了东方财富公告原文链接(url字段)。请尝试访问这些链接获取公告全文，以获得更准确的分析依据。**

**分析要求：**
1. 列出关键的利好事件和利空事件（不超过5个）
2. 重点通过url链接访问公告原文提取关键数据，评估每个事件的影响程度和持续性
3. 综合判断当前事件面是偏多还是偏空
4. 给出最终评分(0-10)并说明理由
5. 指出未来1-3个月最值得关注的事件节点

回复长度控制在 500 字以内。"""

            # Build events list with URLs
            events_with_urls = []
            for evt in (key_events[:8] or events_list[:8]):
                evt_url = evt.get("url", "")
                events_with_urls.append({
                    "title": evt.get("title", ""),
                    "date": evt.get("date", ""),
                    "sentiment": evt.get("sentiment", "neutral"),
                    "url": evt_url,
                    "note": "请访问url获取公告全文" if evt_url else "无原文链接"
                })

            user_context = json.dumps({
                "stock": f"{stock_name}({stock_data.get('code', '')})",
                "events_summary": {
                    "total": events_detail.get("total_count", 0),
                    "positive": events_detail.get("positive_count", 0),
                    "negative": events_detail.get("negative_count", 0),
                    "positive_weight": events_detail.get("positive_weight", 0),
                    "negative_weight": events_detail.get("negative_weight", 0),
                },
                "key_events_with_urls": events_with_urls,
                "search_results": search_results,
                "keyword_score": events_detail.get("score", 0),
            }, ensure_ascii=False)

        else:
            system_prompt = f"""你是一位资深证券分析师，风格类似于通达信诊断师。请对以下股票的{dim_name}进行深度分析。

**回复格式要求（严格遵守）：**

## 多空辩论

### 🔴 空方观点（风险与隐忧）
- 列出 2-3 个具体的看空理由，每个理由包含数据和逻辑支撑
- 说明这些风险可能的严重程度

### 🟢 多方观点（机会与亮点）
- 列出 2-3 个具体的看多理由，每个理由包含数据和逻辑支撑
- 说明这些机会的潜在收益

## 综合研判
- 给出对该维度的综合判断（1-2句话）
- 说明当前市场是否合理定价了这些因素
- 指出最关键的1个观察变量（后续跟踪的核心指标）

回复长度控制在 400 字以内，语言精炼专业。"""

            user_context = json.dumps({
                "name": stock_data.get("name", ""),
                "code": stock_data.get("code", ""),
                "price": stock_data.get("price", 0),
                "pe": stock_data.get("pe", 0),
                "pb": stock_data.get("pb", 0),
                "change_pct": stock_data.get("change_pct", 0),
                "scores": stock_data.get("scores", {}),
                "raw": stock_data.get("raw", {}),
            }, ensure_ascii=False)

        stock_context = json.dumps({
            "name": stock_data.get("name", ""),
            "code": stock_data.get("code", ""),
            "price": stock_data.get("price", 0),
            "pe": stock_data.get("pe", 0),
            "pb": stock_data.get("pb", 0),
            "change_pct": stock_data.get("change_pct", 0),
            "scores": stock_data.get("scores", {}),
            "raw": stock_data.get("raw", {}),
        }, ensure_ascii=False)

        request_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_context},
        ]

        response = client.chat.completions.create(
            model=chat_cfg.get("model", "gpt-4o-mini"),
            messages=request_messages,
            temperature=0.8,
            max_tokens=1500,
        )

        reply = response.choices[0].message.content
        # Save to cache
        DEEP_CACHE[cache_key] = (reply, time.time())

        res = {"reply": reply, "dim": dim_key, "from_cache": False}
        if debug_mode:
            res["debug"] = {
                "model": chat_cfg.get("model", "gpt-4o-mini"),
                "temperature": 0.8,
                "max_tokens": 1500,
                "system_prompt": system_prompt,
                "user_prompt": user_context,
                "response_raw": reply,
                "usage": str(response.usage) if hasattr(response, 'usage') else 'N/A',
            }
        return jsonify(res)

    except Exception as e:
        return jsonify({
            "reply": f"AI 分析服务调用失败：{str(e)}",
            "error": True
        })


@app.route("/api/timing", methods=["POST"])
def timing_analysis():
    """Buy timing analysis — only callable when total_score >= 40."""
    data = request.json
    stock_data = data.get("stock_data", {})

    total_score = stock_data.get("total_score", 0)
    if total_score < 40:
        return jsonify({"reply": "当前评分较低，暂不推荐购入时机分析。", "skip": True})

    chat_cfg = get_ai_config()

    # Fetch market index data for richer context
    mkt = stock_data.get("market", "A")
    index_data = _get_market_indices(mkt)

    if not chat_cfg.get("api_key") or chat_cfg["api_key"].startswith("sk-your"):
        # Rule-based fallback (now much more comprehensive)
        return jsonify({"reply": _rule_based_timing(stock_data), "rule_based": True})

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=chat_cfg["api_key"],
            base_url=chat_cfg.get("api_base", "https://api.openai.com/v1"),
        )

        system_prompt = """你是一位资深多周期交易策略师。请根据提供的股票数据、大盘指数数据，进行多维度买入时机分析。

**分析框架（请严格按照以下结构输出——结论优先，后附分析）：**

### 🎯 核心结论（开门见山，先说建议）
- 用一句话明确给出当前的操作建议：积极买入 / 分批建仓 / 观望等待 / 暂避风险
- 用 1-2 句说明核心依据（例如：估值处于历史低位 + 技术面即将金叉 → 建议分批建仓）

| 策略类型 | 操作建议 | 建议仓位 | 建仓区间 | 止损位 |
|---------|---------|---------|---------|-------|
| 短线(1-5天) | 观望/轻仓/参与 | X% | ¥X~¥Y | ¥Z |
| 波段(1-4周) | 观望/轻仓/参与 | X% | ¥X~¥Y | ¥Z |
| 长线(1月+) | 观望/轻仓/参与 | X% | ¥X~¥Y | ¥Z |

### 📈 一、大盘环境评估
- 上证指数/深证成指当前走势判断（牛市/震荡/熊市）
- 大盘PE估值水平与系统性风险评估
- 个股相对大盘强弱判断

### ⚡ 二、短线趋势分析（1天~2周）
- 均线系统（MA5/10/20）排列及短期方向
- MACD/KDJ/布林带等技术信号
- 近5日资金流向
- 给出短线信号总结

### 🏔️ 三、中长线趋势分析（1个月~1年）
- MA60/MA120/MA250 中长期均线位置与趋势
- 半年涨跌幅表现
- ROE、股息率、资产负债率等基本面长线评估
- PEG等成长性评估
- 给出长线信号总结

### ⚠️ 四、风险提示
- 当前最大的2-3个风险点
- 什么情况下应该止损离场

**回复格式要求：**
- 使用 ## 和 ### 作为标题层级
- 使用 **加粗** 标记关键结论
- 使用 - 列表格式组织要点
- **必须使用 Markdown 表格**展示分策略建仓建议（如上表格式）
- 回复控制在 600 字以内
- 每个信号需给出明确的偏多/偏空判断"""

        stock_context = json.dumps({
            "name": stock_data.get("name", ""),
            "code": stock_data.get("code", ""),
            "price": stock_data.get("price", 0),
            "pe": stock_data.get("pe", 0),
            "pb": stock_data.get("pb", 0),
            "change_pct": stock_data.get("change_pct", 0),
            "total_score": total_score,
            "scores": stock_data.get("scores", {}),
            "raw": stock_data.get("raw", {}),
            "market_indices": index_data,
        }, ensure_ascii=False)

        response = client.chat.completions.create(
            model=chat_cfg.get("model", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"股票数据：\n{stock_context}"},
            ],
            temperature=0.7,
            max_tokens=2000,
        )

        reply = response.choices[0].message.content
        return jsonify({"reply": reply, "rule_based": False})

    except Exception as e:
        return jsonify({"reply": _rule_based_timing(stock_data), "rule_based": True})


def _rule_based_timing(stock_data):
    """Professional multi-dimensional buy timing analysis (no LLM required).
    Covers: short-term trend, long-term trend, market index correlation, 
    multi-timeframe entry strategy.
    """
    scores = stock_data.get("scores", {})
    price = stock_data.get("price", 0)
    pe = stock_data.get("pe", 0)
    pb = stock_data.get("pb", 0)
    total = stock_data.get("total_score", 0)
    name = stock_data.get("name", "")
    code = stock_data.get("code", "")
    change_pct = stock_data.get("change_pct", 0)
    raw = stock_data.get("raw", {})

    tech = scores.get("technical", {})
    fund = scores.get("fundamental", {})
    cap = scores.get("capital", {})
    val = scores.get("value", {})

    tech_detail = tech.get("detail", {})
    cap_detail = cap.get("detail", {})
    fund_detail = fund.get("detail", {})
    val_detail = val.get("detail", {})

    tech_score = tech.get("score", 0) / max(tech.get("max", 20), 1)
    fund_score = fund.get("score", 0) / max(fund.get("max", 25), 1)
    cap_score = cap.get("score", 0) / max(cap.get("max", 15), 1)
    val_score = val.get("score", 0) / max(val.get("max", 15), 1)

    # ---------- 1. Market Index Data ----------
    index_data = _get_market_indices()
    sh_idx = index_data.get("shanghai", {})
    sz_idx = index_data.get("shenzhen", {})

    # ---------- 2. Technical Detail Extraction ----------
    mas = tech_detail.get("mas", {})
    macd = tech_detail.get("macd", {})
    kdj = tech_detail.get("kdj", {})
    boll = tech_detail.get("bollinger", {})

    ma5 = mas.get("MA5", price)
    ma10 = mas.get("MA10", price)
    ma20 = mas.get("MA20", price)
    ma60 = mas.get("MA60", None)
    ma120 = mas.get("MA120", None)
    ma250 = mas.get("MA250", None)

    macd_dif = float(macd.get("DIF", 0)) if macd.get("DIF") else 0
    macd_dea = float(macd.get("DEA", 0)) if macd.get("DEA") else 0
    macd_bar = float(macd.get("MACD", 0)) if macd.get("MACD") else 0

    kdj_k = float(kdj.get("K", 50)) if kdj.get("K") else 50
    kdj_d = float(kdj.get("D", 50)) if kdj.get("D") else 50
    kdj_j = float(kdj.get("J", 50)) if kdj.get("J") else 50

    turnover = tech_detail.get("turnover", 0) or raw.get("turnover", raw.get("换手率", 0))
    hy_return = tech_detail.get("half_year_return", 0)

    boll_upper = float(boll.get("upper", 0)) if boll.get("upper") else 0
    boll_lower = float(boll.get("lower", 0)) if boll.get("lower") else 0
    boll_mid = float(boll.get("mid", 0)) if boll.get("mid") else 0

    # ---------- 3. Capital Flow Detail ----------
    main_5d = cap_detail.get("main_5d_net", 0)
    inflow_days = cap_detail.get("main_inflow_days", 0)
    cap_trend = cap_detail.get("trend", "")

    # ---------- 4. Fundamental Detail ----------
    roe = (fund_detail.get("latest_roe", 0) or fund_detail.get("roe", 0))
    d_yield = fund_detail.get("dividend_yield", 0)
    debt_ratio = fund_detail.get("debt_ratio", 0)

    roe_val = float(roe) if roe else 0
    d_yield_val = float(d_yield) if d_yield else 0
    debt_val = float(debt_ratio) if debt_ratio else 0

    # ---------- 5. Value Detail ----------
    peg = val_detail.get("peg", None)

    # =============================================
    # SHORT-TERM TREND ANALYSIS (短线: 1天~2周)
    # =============================================
    short_signals = []
    short_score = 50  # base neutral

    # MA alignment (short term)
    if price > ma5 > ma10 > ma20:
        short_signals.append(("🟢", "均线多头排列（MA5>MA10>MA20），短线强势"))
        short_score += 15
    elif price < ma5 < ma10 < ma20:
        short_signals.append(("🔴", "均线空头排列（MA5<MA10<MA20），短线弱势"))
        short_score -= 15
    elif price > ma5:
        short_signals.append(("🟡", f"股价站上MA5（¥{ma5:.2f}），短线偏多但均线未完全多头排列"))
        short_score += 5
    else:
        short_signals.append(("🟡", f"股价低于MA5（¥{ma5:.2f}），短线承压"))
        short_score -= 5

    # MACD analysis
    if macd_dif > macd_dea and macd_bar > 0:
        if macd_bar > abs(macd_dif) * 0.3:
            short_signals.append(("🟢", f"MACD金叉后红柱放大（DIF={macd_dif:.3f}, BAR={macd_bar:.3f}），短线动能增强"))
            short_score += 10
        else:
            short_signals.append(("🟢", "MACD金叉状态，短线偏多"))
            short_score += 5
    elif macd_dif < macd_dea and macd_bar < 0:
        short_signals.append(("🔴", f"MACD死叉状态，绿柱（DIF={macd_dif:.3f}），短线偏空"))
        short_score -= 10
    elif macd_bar > 0:
        short_signals.append(("🟡", "MACD红柱但DIF<DEA，多空拉锯中"))
        short_score += 2
    else:
        short_signals.append(("🟡", "MACD绿柱但DIF>DEA，可能即将金叉"))
        short_score += 3

    # KDJ analysis
    if kdj_j < 0:
        short_signals.append(("🟢", f"KDJ超卖（J={kdj_j:.1f}），短线有反弹需求"))
        short_score += 8
    elif kdj_j > 100:
        short_signals.append(("🔴", f"KDJ超买（J={kdj_j:.1f}），短线有回调风险"))
        short_score -= 8
    elif kdj_k > kdj_d and kdj_j < 80:
        short_signals.append(("🟢", "KDJ金叉向上，短线动能偏多"))
        short_score += 5
    elif kdj_k < kdj_d and kdj_j > 20:
        short_signals.append(("🔴", "KDJ死叉向下，短线偏空"))
        short_score -= 5
    else:
        short_signals.append(("🟡", f"KDJ中性区域（K={kdj_k:.1f}, D={kdj_d:.1f}, J={kdj_j:.1f}）"))

    # Bollinger position
    boll_width = None
    if boll_upper > 0 and boll_lower > 0:
        boll_width = (boll_upper - boll_lower) / boll_mid * 100 if boll_mid > 0 else 0
        price_in_boll = (price - boll_lower) / (boll_upper - boll_lower) * 100 if boll_upper > boll_lower else 50

        if price_in_boll > 90:
            short_signals.append(("🔴", f"股价触及布林上轨（¥{boll_upper:.2f}），短线超买，注意回调"))
            short_score -= 8
        elif price_in_boll < 10:
            short_signals.append(("🟢", f"股价触及布林下轨（¥{boll_lower:.2f}），短线超卖，有反弹可能"))
            short_score += 8
        elif price_in_boll > 70:
            short_signals.append(("🟡", "股价位于布林带上半区，短线偏强"))
            short_score += 3
        elif price_in_boll < 30:
            short_signals.append(("🟡", "股价位于布林带下半区，短线偏弱"))
            short_score -= 3
        else:
            short_signals.append(("🟡", "股价位于布林带中轨附近，短线方向不明"))

        if boll_width < 5:
            short_signals.append(("🟡", "布林带收窄（带宽<5%），短线可能变盘"))

    # Volume & turnover
    if turnover > 0:
        if turnover > 15:
            short_signals.append(("🟡", f"换手率极高（{turnover:.1f}%），短线博弈激烈，注意风险"))
        elif turnover > 5:
            short_signals.append(("🟢", f"换手率活跃（{turnover:.1f}%），短线交投活跃"))
        elif turnover < 1:
            short_signals.append(("🟡", f"换手率低迷（{turnover:.1f}%），短线流动性不足"))

    # Recent change
    if change_pct > 5:
        short_signals.append(("🔴", f"单日涨幅{change_pct:.1f}%，短线追高风险较大"))
        short_score -= 5
    elif change_pct < -5:
        short_signals.append(("🟢", f"单日跌幅{change_pct:.1f}%，短线超跌可能有反弹"))
        short_score += 3

    # Capital flow short-term
    if main_5d > 5000:
        short_signals.append(("🟢", f"近5日主力大幅净流入{main_5d/10000:.2f}亿，短线资金面积极"))
        short_score += 10
    elif main_5d > 1000:
        short_signals.append(("🟢", f"近5日主力净流入{main_5d/10000:.2f}亿，短线资金偏多"))
        short_score += 5
    elif main_5d < -5000:
        short_signals.append(("🔴", f"近5日主力大幅净流出{abs(main_5d)/10000:.2f}亿，短线资金出逃"))
        short_score -= 10
    elif main_5d < -1000:
        short_signals.append(("🔴", f"近5日主力净流出{abs(main_5d)/10000:.2f}亿，短线资金偏空"))
        short_score -= 5

    # =============================================
    # LONG-TERM TREND ANALYSIS (长线: 1个月~1年)
    # =============================================
    long_signals = []
    long_score = 50

    # Long MA analysis
    if ma60 is not None:
        if price > ma60:
            long_signals.append(("🟢", f"股价站上MA60（¥{ma60:.2f}），中期趋势偏多"))
            long_score += 8
        else:
            long_signals.append(("🔴", f"股价低于MA60（¥{ma60:.2f}），中期趋势偏空"))
            long_score -= 8

    if ma120 is not None:
        if price > ma120:
            long_signals.append(("🟢", f"股价站上MA120（半年线 ¥{ma120:.2f}），中长期趋势向好"))
            long_score += 10
        else:
            long_signals.append(("🔴", f"股价低于MA120（半年线 ¥{ma120:.2f}），中长期承压"))
            long_score -= 10

    if ma250 is not None:
        if price > ma250:
            long_signals.append(("🟢", f"股价站上MA250（年线 ¥{ma250:.2f}），长期趋势牛市特征"))
            long_score += 12
        else:
            long_signals.append(("🔴", f"股价低于MA250（年线 ¥{ma250:.2f}），长期趋势熊市特征"))
            long_score -= 12

    # All long MAs aligned
    long_mas = [m for m in [ma60, ma120, ma250] if m is not None]
    if len(long_mas) >= 2:
        aligned_up = all(price > m for m in long_mas)
        aligned_down = all(price < m for m in long_mas)
        if aligned_up:
            long_signals.append(("🟢", "股价站上全部中长期均线，长线趋势强劲"))
            long_score += 10
        elif aligned_down:
            long_signals.append(("🔴", "股价低于全部中长期均线，长线趋势偏弱"))
            long_score -= 10

    # Half-year performance
    if hy_return is not None:
        if hy_return > 30:
            long_signals.append(("🟢", f"半年涨幅{hy_return:.1f}%，长期上升趋势明确"))
            long_score += 5
        elif hy_return > 0:
            long_signals.append(("🟢", f"半年涨幅{hy_return:.1f}%，长期趋势偏多"))
            long_score += 3
        elif hy_return > -15:
            long_signals.append(("🟡", f"半年跌幅{abs(hy_return):.1f}%，处于回调中"))
            long_score -= 3
        else:
            long_signals.append(("🔴", f"半年跌幅{abs(hy_return):.1f}%，长期弱势，需警惕"))
            long_score -= 8

    # Fundamental check for long-term
    if roe_val > 0:
        if roe_val > 15:
            long_signals.append(("🟢", f"ROE={roe_val:.1f}%，长期盈利能力优秀"))
            long_score += 8
        elif roe_val > 8:
            long_signals.append(("🟢", f"ROE={roe_val:.1f}%，长期盈利能力良好"))
            long_score += 4
        elif roe_val < 0:
            long_signals.append(("🔴", f"ROE={roe_val:.1f}%，长期盈利能力差"))
            long_score -= 10

    if d_yield_val > 0:
        if d_yield_val > 3:
            long_signals.append(("🟢", f"股息率{d_yield_val:.2f}%，长线持有有较好分红回报"))
            long_score += 5
        elif d_yield_val > 2:
            long_signals.append(("🟡", f"股息率{d_yield_val:.2f}%，长线分红一般"))

    if debt_val > 0:
        if debt_val > 70:
            long_signals.append(("🔴", f"资产负债率{debt_val:.1f}%，长期财务风险偏高"))
            long_score -= 5
        elif debt_val < 40:
            long_signals.append(("🟢", f"资产负债率{debt_val:.1f}%，长期财务稳健"))

    # PEG for growth
    if peg is not None and peg > 0:
        if peg < 0.5:
            long_signals.append(("🟢", f"PEG={peg:.2f}<0.5，成长性被严重低估"))
            long_score += 10
        elif peg < 1:
            long_signals.append(("🟢", f"PEG={peg:.2f}<1，估值匹配成长性"))
            long_score += 5
        elif peg > 2:
            long_signals.append(("🔴", f"PEG={peg:.2f}>2，成长性可能不足以支撑估值"))
            long_score -= 5

    # =============================================
    # MARKET INDEX CORRELATION (大盘环境)
    # =============================================
    market_signals = []
    market_score = 50

    if sh_idx:
        sh_price = sh_idx.get("price", 0)
        sh_change = sh_idx.get("change_pct", 0)
        sh_pe = sh_idx.get("pe", 0)
        sh_name = sh_idx.get("name", "上证指数")

        market_signals.append((None, f"**{sh_name}**：{sh_price:.0f}点 | {sh_change:+.2f}%"))

        if sh_change > 1:
            market_signals.append(("🟢", "大盘强势上攻，个股容易跟随上涨"))
            market_score += 8
        elif sh_change > 0.3:
            market_signals.append(("🟢", "大盘温和上涨，市场情绪偏暖"))
            market_score += 4
        elif sh_change > -0.3:
            market_signals.append(("🟡", "大盘窄幅震荡，个股分化明显"))
        elif sh_change > -1:
            market_signals.append(("🔴", "大盘温和下跌，注意系统性风险"))
            market_score -= 4
        else:
            market_signals.append(("🔴", "大盘大幅下跌，系统性风险较高"))
            market_score -= 8

        # Relative strength
        if abs(sh_change) > 0.3:
            rel_strength = change_pct - sh_change
            if rel_strength > 2:
                market_signals.append(("🟢", f"个股相对大盘强势（{rel_strength:+.2f}%），有独立行情特征"))
                market_score += 5
            elif rel_strength < -2:
                market_signals.append(("🔴", f"个股弱于大盘（{rel_strength:+.2f}%），表现滞后"))
                market_score -= 5

        if sh_pe > 0:
            if sh_pe < 15:
                market_signals.append(("🟢", f"上证PE={sh_pe:.1f}，整体市场估值偏低，长线布局窗口"))
                market_score += 5
            elif sh_pe > 25:
                market_signals.append(("🔴", f"上证PE={sh_pe:.1f}，整体市场估值偏高，注意系统性泡沫"))
                market_score -= 3

    if sz_idx:
        sz_price = sz_idx.get("price", 0)
        sz_change = sz_idx.get("change_pct", 0)
        market_signals.append((None, f"**{sz_idx.get('name', '深证成指')}**：{sz_price:.0f}点 | {sz_change:+.2f}%"))

    # Capital flow trend context
    if cap_trend:
        if "持续流入" in cap_trend or "近10日" in cap_trend and "流入" in cap_trend:
            market_signals.append(("🟢", f"资金面：{cap_trend}"))

    # =============================================
    # COMPREHENSIVE RECOMMENDATIONS
    # =============================================
    short_score = max(0, min(100, short_score))
    long_score = max(0, min(100, long_score))
    market_score = max(0, min(100, market_score))

    def signal_light(score):
        if score >= 65: return "🟢 **偏多**"
        elif score >= 45: return "🟡 **中性**"
        else: return "🔴 **偏空**"

    def recommend_action(score, total):
        if score >= 65:
            if total >= 60: return "✅ **可积极参与** — 技术面与基本面共振，建议分批建仓"
            else: return "🟡 **轻仓试探** — 短线信号偏多，但综合评分偏低，建议控制仓位"
        elif score >= 45:
            return "🟡 **观望为主** — 等待更明确的入场信号，可设提醒"
        else:
            return "❌ **暂时回避** — 短线信号偏空，不建议追入"

    # Build key price levels
    levels = []
    if boll_lower > 0:
        levels.append(f"- 布林下轨支撑：¥{boll_lower:.2f}")
    if boll_mid > 0:
        levels.append(f"- 布林中轨：¥{boll_mid:.2f}")
    if boll_upper > 0:
        levels.append(f"- 布林上轨压力：¥{boll_upper:.2f}")
    if ma60 is not None:
        levels.append(f"- MA60中期支撑/压力：¥{ma60:.2f}")
    if ma120 is not None:
        levels.append(f"- MA120半年线：¥{ma120:.2f}")
    if ma250 is not None:
        levels.append(f"- MA250年线：¥{ma250:.2f}")

    # Entry zones
    short_entry = f"¥{round(price*0.97,2)} ~ ¥{round(price*0.99,2)}"
    safe_entry = f"¥{round(price*0.92,2)} ~ ¥{round(price*0.95,2)}"

    # Build response
    parts = [
        "## 🎯 多维度购入时机分析（专业版）",
        "",
        f"**分析标的**：{name}（{code}）| **现价**：¥{price:.2f} | **综合评分**：{total}/100",
        "",
        "---",
        "",
        "### 📈 一、大盘环境评估",
        "",
        f"**市场信号灯**：{signal_light(market_score)}（{market_score}分）",
        "",
    ]

    for icon, msg in market_signals:
        if icon:
            parts.append(f"{icon} {msg}")
        else:
            parts.append(f"  {msg}")

    parts.extend([
        "",
        "---",
        "",
        "### ⚡ 二、短线趋势分析（1天~2周）",
        "",
        f"**短线信号灯**：{signal_light(short_score)}（{short_score}分）",
        f"**短线建议**：{recommend_action(short_score, total)}",
        "",
    ])

    for icon, msg in short_signals:
        parts.append(f"{icon} {msg}")

    parts.extend([
        "",
        "---",
        "",
        "### 🏔️ 三、中长线趋势分析（1个月~1年）",
        "",
        f"**长线信号灯**：{signal_light(long_score)}（{long_score}分）",
        f"**长线建议**：{recommend_action(long_score, total)}",
        "",
    ])

    for icon, msg in long_signals:
        parts.append(f"{icon} {msg}")

    parts.extend([
        "",
        "---",
        "",
        "### 🎯 四、关键价位与操作策略",
        "",
        "**关键技术位**：",
    ])

    parts.extend(levels)

    parts.extend([
        "",
        "**分策略建仓建议**：",
        "",
        f"| 策略类型 | 信号强度 | 建议仓位 | 买入区间 | 止损位 |",
        f"|---------|---------|---------|---------|--------|",
        f"| 🏃 短线博弈 | {signal_light(short_score)} | ≤20% | {short_entry} | ¥{round(min(price*0.95, boll_lower) if boll_lower>0 else price*0.95, 2):.2f} |",
        f"| 🚶 波段操作 | {signal_light((short_score+long_score)//2)} | ≤30% | {short_entry} | ¥{round(min(price*0.93, boll_lower) if boll_lower>0 else price*0.93, 2):.2f} |",
        f"| 🐢 长线布局 | {signal_light(long_score)} | ≤50% (分批) | {safe_entry} | ¥{round(price*0.85, 2):.2f} |",
        "",
        "---",
        "",
        "### ⚠️ 五、核心风险提示",
        "",
    ])

    # Risk analysis
    risks = []
    if short_score < 45 and long_score < 45:
        risks.append("- 🔴 **双周期共振偏空**：短线与长线信号均偏弱，不建议任何操作")
    if market_score < 45:
        risks.append("- 🔴 **大盘环境偏弱**：系统性风险较高，个股难独立走强")
    if change_pct > 7 and short_score < 50:
        risks.append("- 🔴 **追高风险**：短期涨幅过大，回调概率高")
    if kdj_j > 90:
        risks.append("- 🟡 **超买风险**：KDJ处于超买区，短线有技术回调需求")
    if main_5d < -3000:
        risks.append("- 🟡 **资金出逃**：主力连续流出，需警惕进一步下跌")
    if ma60 and ma120 and price < ma60 < ma120:
        risks.append("- 🔴 **中期均线死叉**：MA60<MA120，中期趋势恶化")
    if boll_width is not None and boll_width < 3:
        risks.append("- 🟡 **变盘在即**：布林带极度收窄，方向性突破即将发生")

    if not risks:
        risks.append("- 当前未检测到明显的极端风险信号")

    parts.extend(risks)

    parts.extend([
        "",
        "---",
        "",
        "> 💡 **总结**：以上分析基于技术指标、资金流向、基本面和大盘环境的综合评估。",
        f"> 短线信号灯 **{signal_light(short_score)}**，长线信号灯 **{signal_light(long_score)}**。",
        "> 请结合自身风险偏好和投资周期选择合适的操作策略。",
        "> *规则引擎自动生成，配置 LLM API 可获得更精准的个性化分析。*",
    ])

    return "\n".join(parts)


def _get_market_indices(market="A"):
    """Fetch major market index data via Tencent real-time API (A/HK/US)."""
    cfg = MARKET_CONFIG.get(market, MARKET_CONFIG["A"])
    cache_key = f"market_indices_{market}"
    cached_val = cached(cache_key, ttl=120)  # 2-minute cache
    if cached_val:
        return cached_val

    indices = cfg["indices"]
    result = {}
    for key, (tcode, name) in indices.items():
        try:
            resp = _http_get(f"{TENCENT_QT}{tcode}")
            text = resp.text
            if len(text) < 10 or "none" in text.lower():
                continue

            start = text.index('"') + 1
            end = text.rindex('"')
            fields = text[start:end].split("~")

            if len(fields) >= 40:
                result[key] = {
                    "name": fields[1] if len(fields) > 1 else name,
                    "price": float(fields[3]) if fields[3] else 0,
                    "change_pct": float(fields[32]) if len(fields) > 32 and fields[32] else 0,
                    "pe": float(fields[39]) if len(fields) > 39 and fields[39] else 0,
                    "volume": float(fields[6]) if len(fields) > 6 and fields[6] else 0,
                    "amount": float(fields[37]) if len(fields) > 37 and fields[37] else 0,
                    "high": float(fields[33]) if len(fields) > 33 and fields[33] else 0,
                    "low": float(fields[34]) if len(fields) > 34 and fields[34] else 0,
                }
        except Exception as e:
            print(f"Error fetching index {name}: {e}")

    # ---- Fallback: Sina indices (A-share only) ----
    if market == "A" and (not result or len(result) < 2):
        try:
            fb_result = api_fallback.sina_get_market_indices()
            if fb_result:
                for key, val in fb_result.items():
                    if key not in result:
                        result[key] = val
        except Exception as fe:
            print(f"[FALLBACK] Sina indices failed: {fe}")

    cache_set(cache_key, result)
    return result


# ========== Cross-Market Fallback Helpers (Beta) ==========
def _compute_volume_flow(prices_data, info=None):
    """Volume-price divergence analysis — market-agnostic money flow proxy.
    Computes: volume trend, price-volume correlation, net pressure score.
    Returns A-share-compatible money flow dict."""
    if not prices_data or len(prices_data) < 5:
        return {
            "records": [], "last5": [], "latest": None,
            "summary": {"main_5d_net": 0, "super_large_5d_net": 0, "large_5d_net": 0,
                         "retail_5d_net": 0, "main_inflow_days": 0, "main_outflow_days": 0,
                         "trend": "数据不足"},
            "note": "数据不足，无法进行量价分析"
        }

    # Last 20 trading days
    recent = prices_data[-20:] if len(prices_data) >= 20 else prices_data
    records = []
    for d in recent:
        close = d.get("收盘", 0) or 0
        vol = d.get("成交量", 0) or 0
        records.append({"date": d.get("日期", ""), "close": close, "volume": vol})

    # Compute simple metrics
    vols = [r["volume"] for r in records]
    closes = [r["close"] for r in records]
    avg_vol = sum(vols) / len(vols) if vols else 1

    # 5-day slice
    last5 = records[-5:] if len(records) >= 5 else records
    last5_vols = [r["volume"] for r in last5]
    last5_closes = [r["close"] for r in last5]

    # Volume trend: rising volume on up days = buying pressure; rising volume on down days = selling pressure
    up_vol = 0
    down_vol = 0
    for i in range(1, len(last5)):
        if last5_closes[i] > last5_closes[i-1]:
            up_vol += last5_vols[i]
        elif last5_closes[i] < last5_closes[i-1]:
            down_vol += last5_vols[i]

    # Net flow score: normalized to similar range as A-share main_net (万元)
    total_vol_5d = sum(last5_vols)
    if total_vol_5d > 0:
        net_pressure = (up_vol - down_vol) / total_vol_5d * 10000  # scale to ~万元
    else:
        net_pressure = 0

    # Trend determination
    if up_vol > down_vol * 1.5:
        trend = "量价配合 — 放量上涨为主"
    elif down_vol > up_vol * 1.5:
        trend = "量价背离 — 放量下跌为主"
    elif up_vol > down_vol:
        trend = "量能偏多 — 买入量略大"
    elif down_vol > up_vol:
        trend = "量能偏空 — 卖出量略大"
    else:
        trend = "量能均衡"

    # Volume vs average
    avg_5d_vol = sum(last5_vols) / len(last5_vols) if last5_vols else 0
    vol_ratio = avg_5d_vol / avg_vol if avg_vol else 1
    vol_note = "放量" if vol_ratio > 1.3 else "缩量" if vol_ratio < 0.7 else "正常"

    # Build flow records — distribute net_pressure across days proportionally
    # A-share format: daily main_net in 元, main_5d_net in 万元 (= sum_daily / 10000)
    flow_records = []
    # Compute daily contributions: volume * direction for each day
    daily_contribs = []
    for i in range(1, len(last5)):
        if last5_closes[i] > last5_closes[i-1]:
            daily_contribs.append(last5_vols[i])  # up day → buying
        elif last5_closes[i] < last5_closes[i-1]:
            daily_contribs.append(-last5_vols[i])  # down day → selling
        else:
            daily_contribs.append(0)
    total_contrib = sum(abs(c) for c in daily_contribs) or 1

    # net_pressure is a score in ~万元; scale to 元 for daily records
    daily_total_yuan = 0.0
    for i, r in enumerate(last5):
        if i == 0:
            main_net = 0.0
        else:
            # Scale: proportion * net_pressure_score * 10000 (万元→元)
            main_net = round(daily_contribs[i-1] / total_contrib * net_pressure * 10000, 0)
        daily_total_yuan += main_net
        flow_records.append({
            "date": r["date"],
            "main_net": main_net,
            "retail_net": 0, "medium_net": 0, "large_net": 0, "super_large_net": 0,
        })

    # main_5d_net in 万元 (A-share convention: daily sum / 10000)
    main_5d_wan = round(daily_total_yuan / 10000, 2)

    result = {
        "records": flow_records,
        "last5": flow_records,
        "latest": flow_records[-1] if flow_records else None,
        "summary": {
            "main_5d_net": main_5d_wan,
            "super_large_5d_net": 0,
            "large_5d_net": 0,
            "retail_5d_net": 0,
            "main_inflow_days": sum(1 for i in range(1, len(last5)) if last5_closes[i] > last5_closes[i-1]),
            "main_outflow_days": sum(1 for i in range(1, len(last5)) if last5_closes[i] < last5_closes[i-1]),
            "trend": f"{trend} ({vol_note})",
        },
        "note": "量价分析（基于成交量和价格方向估算资金流向）"
    }
    return result


def _get_em_money_flow(code, market):
    """EastMoney 分笔资金流 (fflow) for HK (116.xxxxx) / US (105.TICKER).
    Returns real daily 主力/超大单/大单/中单/小单 net inflow (元).
    Falls back to None if the API yields no data."""
    if market == "HK":
        em_symbol = code.replace(".HK", "").zfill(5)
        secid = f"116.{em_symbol}"
    elif market == "US":
        em_symbol = code.replace(".US", "").replace(".OQ", "").replace(".N", "")
        secid = f"105.{em_symbol}"
    else:
        return None

    url = "http://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
    params = {
        "lmt": "10", "klt": "101", "secid": secid,
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
    }
    try:
        r = _http_get(url, params=params, timeout=10)
        j = r.json()
        d = j.get("data")
        if not d or not d.get("klines"):
            return None
        records = []
        for line in d["klines"][-10:]:
            p = line.split(",")
            if len(p) < 6:
                continue
            def _f(i):
                try:
                    return float(p[i])
                except Exception:
                    return 0.0
            records.append({
                "date": p[0],
                "main_net": _f(1),
                "retail_net": _f(2),
                "medium_net": _f(3),
                "large_net": _f(4),
                "super_large_net": _f(5),
            })
        if not records:
            return None

        last5 = records[-5:]
        def _sum5d(key):
            return round(sum(r.get(key, 0) for r in last5) / 1e4, 2)
        main_5d = _sum5d("main_net")
        super_5d = _sum5d("super_large_net")
        large_5d = _sum5d("large_net")
        medium_5d = _sum5d("medium_net")
        retail_5d = _sum5d("retail_net")

        inflow_days = sum(1 for r in last5 if r["main_net"] > 0)
        outflow_days = sum(1 for r in last5 if r["main_net"] < 0)

        if main_5d > 0 and (super_5d + large_5d) > 0:
            trend = "主力与机构资金净流入（偏多）"
        elif main_5d > 0:
            trend = "主力资金净流入（偏多）"
        elif main_5d < 0 and (super_5d + large_5d) < 0:
            trend = "主力与机构资金净流出（偏空）"
        elif main_5d < 0:
            trend = "主力资金净流出（偏空）"
        else:
            trend = "主力资金多空均衡"

        return {
            "records": records,
            "last5": last5,
            "latest": records[-1],
            "summary": {
                "main_5d_net": main_5d,
                "super_large_5d_net": super_5d,
                "large_5d_net": large_5d,
                "medium_5d_net": medium_5d,
                "retail_5d_net": retail_5d,
                "main_inflow_days": inflow_days,
                "main_outflow_days": outflow_days,
                "trend": trend,
            },
            "note": "东方财富分笔资金流（主力/超大单/大单/中单/小单，单位：万元）",
            "source": "eastmoney_fflow",
        }
    except Exception as e:
        print(f"[_get_em_money_flow] {code} ({market}) failed: {e}")
        return None


def _get_market_money_flow(code, market, prices_data=None):
    """Money flow for HK/US: volume-price divergence analysis.
    Also attempts HK southbound flow for HK market."""
    cfg = MARKET_CONFIG.get(market, {})
    symbol = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "").replace(".HK", "").replace(".US", "")

    # ---- Primary: EastMoney fflow (real 主力/超大单/大单/中单/小单) ----
    em_flow = _get_em_money_flow(code, market)
    if em_flow and em_flow.get("records"):
        em_flow["note"] = f"{cfg.get('name','')}东方财富资金流（真实分笔数据）"
        return em_flow

    # ---- Fallback: volume-price divergence analysis ----
    if prices_data and len(prices_data) >= 5:
        result = _compute_volume_flow(prices_data)
    else:
        # No price data — fallback to basic quote info
        try:
            tc = _tencent_code(code, market)
            resp = _http_get(f"{TENCENT_QT}{tc}")
            text = resp.text
            if "v_" in text and len(text) >= 10:
                start = text.index('"') + 1
                end = text.rindex('"')
                fields = text[start:end].split("~")
                vol = int(float(fields[6])) if len(fields) > 6 and fields[6] else 0
                amount = float(fields[37]) if len(fields) > 37 and fields[37] else 0
                result = {
                    "records": [], "last5": [],
                    "latest": {"date": now_cn().strftime("%Y-%m-%d"), "main_net": 0},
                    "summary": {"main_5d_net": 0, "super_large_5d_net": 0, "large_5d_net": 0,
                                 "retail_5d_net": 0, "main_inflow_days": 0, "main_outflow_days": 0,
                                 "trend": f"当日成交: {vol}手, {amount:.0f}万{cfg['currency_label']}"},
                    "note": "仅有当日数据，历史K线不足无法量价分析"
                }
            else:
                raise ValueError("No quote data")
        except Exception:
            result = {"error": "资金流数据不可用", "records": [], "note": "历史K线数据不足"}

    # ---- HK: try southbound flow enhancement ----
    if market == "HK":
        try:
            # EastMoney southbound flow: secid for 00700 is 116.xxxxx
            em_symbol = symbol  # keep full code: 00700, 09988 etc. (EastMoney southbound uses full format)
            flow_url = "http://push2.eastmoney.com/api/qt/stock/fflow/daykline/get"
            params = {"lmt": "5", "klt": "1", "secid": f"116.{em_symbol}",
                      "fields1": "f1,f2,f3,f7",
                      "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65"}
            flow_resp = _http_get(flow_url, params=params, timeout=10)
            flow_data = flow_resp.json()
            if flow_data.get("data") and flow_data["data"].get("klines"):
                # Successfully got eastmoney flow data for HK
                klines = flow_data["data"]["klines"]
                records = []
                for line in klines[-5:]:
                    parts = line.split(",")
                    if len(parts) >= 7:
                        records.append({
                            "date": parts[0],
                            "main_net": float(parts[1]) if parts[1] != "-" else 0,
                            "retail_net": float(parts[2]) if parts[2] != "-" else 0,
                            "medium_net": float(parts[3]) if parts[3] != "-" else 0,
                            "large_net": float(parts[4]) if parts[4] != "-" else 0,
                            "super_large_net": float(parts[5]) if parts[5] != "-" else 0,
                        })
                if records:
                    # Enrich summary with real southbound data, keep volume analysis records
                    result["summary"]["super_large_5d_net"] = round(sum(r["super_large_net"] for r in records) / 1e4, 2)
                    result["summary"]["large_5d_net"] = round(sum(r.get("large_net", 0) for r in records) / 1e4, 2)
                    retail_sum = sum(r.get("retail_net", 0) for r in records)
                    if retail_sum != 0:
                        result["summary"]["retail_5d_net"] = round(retail_sum / 1e4, 2)
                    result["note"] = "港股通资金流 (东方财富) 增强 + 量价分析"
        except Exception as e:
            pass  # EM flow failed, keep volume analysis result

    return result


# ---- Shared sentiment keyword libraries (HK/US news) ----
POS_KW = [
    ("大涨", 5), ("增持", 3), ("回购", 3), ("超预期", 3), ("利好", 3), ("中标", 3),
    ("合作", 2), ("突破", 3), ("获批", 3), ("增长", 2), ("创新高", 3), ("净流入", 2),
    ("买入评级", 3), ("上调目标价", 3), ("收购", 2), ("分红", 2), ("派息", 2), ("业绩预增", 4),
    ("扭亏", 4), ("扭亏为盈", 4), ("盈利", 1), ("收益", 1), ("上升", 1), ("反弹", 2),
    ("战略合作", 3), ("签署协议", 2), ("获得订单", 3), ("产能扩张", 2), ("产品获批", 3),
    ("政策利好", 3), ("补贴", 2), ("研发突破", 3), ("专利", 2),
    ("机构增持", 3), ("外资流入", 2), ("南下资金", 2),
    ("ETF纳入", 2), ("指数纳入", 2),
]
NEG_KW = [
    ("大跌", 5), ("减持", 3), ("亏损", 4), ("暴雷", 5), ("处罚", 4), ("罚款", 3),
    ("立案调查", 5), ("退市", 5), ("违约", 5), ("诉讼", 3), ("仲裁", 3),
    ("业绩预亏", 4), ("下滑", 2), ("不及预期", 3), ("下调目标价", 3), ("卖出评级", 3),
    ("监管", 3), ("停牌", 3), ("重组失败", 4), ("终止", 3),
    ("裁员", 2), ("关闭", 2), ("召回", 3), ("事故", 4),
]


def _em_news_search(keyword):
    """EastMoney news search (JSONP) by keyword. Returns raw item list or []."""
    if not keyword:
        return []
    try:
        url = "https://search-api-web.eastmoney.com/search/jsonp"
        inner = {
            "uid": "", "keyword": keyword,
            "type": ["cmsArticleWebOld"],
            "client": "web", "clientType": "web", "clientVersion": "curr",
            "param": {"cmsArticleWebOld": {
                "searchScope": "default", "sort": "default",
                "pageIndex": 1, "pageSize": 12,
                "preTag": "<em>", "postTag": "</em>",
            }},
        }
        params = {
            "cb": "jQuery35101792940631092459_1764599530165",
            "param": json.dumps(inner, ensure_ascii=False),
            "_": str(int(time.time() * 1000)),
        }
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://so.eastmoney.com/news/s",
        }
        resp = _http_get(url, params=params, headers_extra=headers, timeout=15)
        text = resp.text
        m = re.search(r"jQuery\d+_\d+\((.*)\)\s*$", text, re.S)
        if not m:
            return []
        data = json.loads(m.group(1))
        return data.get("result", {}).get("cmsArticleWebOld", []) or []
    except Exception as e:
        print(f"[news_search] keyword={keyword} failed: {e}")
        return []


def _em_announcement(symbol, market):
    """EastMoney announcement API for HK/US. Returns raw item list or []."""
    try:
        secid = f"116.{symbol.zfill(5)}" if market == "HK" else f"105.{symbol}"
        url = "https://np-anotice-stock.eastmoney.com/api/v1/notice/get"
        params = {
            "sr": "-1", "page_size": "12", "page_index": "1",
            "ann_type": "0,1,2,3", "client_source": "web", "secid": secid,
        }
        resp = _http_get(url, params=params,
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        data = resp.json()
        return data.get("data", {}).get("list", []) or []
    except Exception as e:
        print(f"[news_ann] {symbol} failed: {e}")
        return []


def _parse_em_news_item(item):
    """Map a raw EastMoney news/announcement item to the event dict shape."""
    title = re.sub(r"</?em>", "", item.get("title", "") or "")
    pub_time = str(item.get("date", "") or "")[:10]
    source = item.get("mediaName", "") or item.get("source", "") or ""
    code = item.get("code", "") or ""
    url = f"http://finance.eastmoney.com/a/{code}.html" if code else ""
    pos_score = sum(w for kw, w in POS_KW if kw in title)
    neg_score = sum(w for kw, w in NEG_KW if kw in title)
    if pos_score > neg_score:
        sent, ss = "positive", min(pos_score - neg_score, 5)
    elif neg_score > pos_score:
        sent, ss = "negative", min(neg_score - pos_score, 5)
    else:
        sent, ss = "neutral", 0
    return {"date": pub_time, "title": title, "type": source or "新闻",
            "url": url, "sentiment": sent, "sentiment_score": ss}


def _flag_buyback_anomaly(events):
    """Flag buyback events whose amount is an order of magnitude off from peers.

    Upstream announcement text sometimes carries a unit/scale data-entry error
    (e.g. a HK$12.73M line among a series of HK$500M buybacks). We can't fix the
    source, so we parse the amount out of each buyback title, compare against the
    peer median, and attach an 'anomaly_note' to any record that is <=1/10 of it.
    Mutates events in place. Pure text parsing -> Vercel-safe, no network.
    """
    if not events:
        return events
    import re as _re

    def _amount_wan(title):
        """Parse a buyback cash amount from the title, normalized to 万 (0.01M)."""
        # 亿 first (斥资/耗资/涉资/动用 约 X 亿 港/美/元)
        m = _re.search(r'(?:耗资|涉资|斥资|动用|共|合计)?\s*(?:约)?\s*'
                       r'([\d,.]+)\s*亿\s*(?:港|美|元)', title)
        if m:
            try:
                return float(m.group(1).replace(",", "")) * 10000.0
            except ValueError:
                return None
        # 万 (note: "X万股" won't match because we require a currency word after 万)
        m = _re.search(r'(?:耗资|涉资|斥资|动用|共|合计)?\s*(?:约)?\s*'
                       r'([\d,.]+)\s*万\s*(?:港|美|元)', title)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                return None
        return None

    buybacks = []
    for ev in events:
        title = ev.get("title", "") or ""
        if "回购" not in title and "购回" not in title:
            continue
        amt = _amount_wan(title)
        if amt and amt > 0:
            buybacks.append((ev, amt))

    if len(buybacks) < 3:
        return events  # too few peers to judge a magnitude outlier reliably

    amts = sorted(a for _, a in buybacks)
    n = len(amts)
    median = amts[n // 2] if n % 2 else (amts[n // 2 - 1] + amts[n // 2]) / 2.0
    if median <= 0:
        return events

    for ev, amt in buybacks:
        if amt < median * 0.1:  # an order of magnitude below the peer median
            ev["anomaly_note"] = (
                f"回购金额量级异常：本次约{amt:.0f}万，同期多为{median:.0f}万级，"
                "疑似上游公告单位/量级录入异常，建议核对公告原文"
            )
    return events


def _get_market_news(code, market):
    """News/events for HK/US via EastMoney HTTP (no akshare, Vercel-safe).
    Chain: 1 EM search by code -> 2 EM search by name -> 3 EM announcement."""
    symbol = code.replace(".HK", "").replace(".US", "")
    info = get_stock_info(code, market) or {}
    name = info.get("股票简称", "") or ""

    items = _em_news_search(symbol)            # 1 by code
    if not items and name:
        items = _em_news_search(name)          # 2 by name
    if not items:
        items = _em_announcement(symbol, market)  # 3 announcement

    if not items:
        return []

    events = []
    seen = set()
    for it in items[:15]:
        ev = _parse_em_news_item(it)
        key = ev["title"][:30]
        if key in seen:
            continue
        seen.add(key)
        events.append(ev)
    return events




# HK industry keyword rules -> industry_name (aligned with _industry_pe_range_intl keys)
# Order matters: more specific categories (and full company names) come FIRST so a
# name like "平安好医生" hits 医药健康 before 金融/保险 grabs "中国平安".
_HK_INDUSTRY_RULES = [
    # --- 医药健康 ---
    ((u"药明", u"信达生物", u"石药", u"中国生物制药", u"百济神州", u"翰森", u"京东健康",
      u"阿里健康", u"平安好医生", u"微创", u"威高", u"联邦制药", u"中国中药", u"复星医药",
      u"国药", u"康方", u"再鼎", u"启明医疗", u"沛嘉", u"锦欣", u"海吉亚", u"医渡", u"医脉通",
      u"云顶新耀", u"诺诚健华", u"腾盛博药"), u"医药健康"),
    # --- 汽车/新能源 ---
    ((u"比亚迪", u"吉利", u"长城汽车", u"蔚来", u"小鹏", u"理想", u"广汽", u"北京汽车",
      u"零跑", u"恒大汽车", u"雅迪", u"耐世特", u"福耀", u"敏实", u"赣锋锂业", u"天齐锂业", u"锂业"),
     u"汽车/新能源"),
    # --- 半导体 ---
    ((u"中芯", u"华虹", u"上海复旦", u"晶门", u"芯智"), u"半导体"),
    # --- 科技硬件 ---
    ((u"小米", u"联想", u"舜宇", u"瑞声", u"比亚迪电子", u"鸿腾", u"中兴", u"丘钛", u"通达",
      u"信利", u"ASM", u"雷蛇", u"高伟", u"建滔"), u"科技硬件"),
    # --- 传媒娱乐 (specific before 互联网) ---
    ((u"腾讯音乐", u"网易云", u"猫眼", u"阿里影业", u"电视广播", u"凤凰", u"卫视", u"IMAX", u"星空",
      u"英皇", u"寰宇"), u"传媒娱乐"),
    # --- 互联网 ---
    ((u"腾讯", u"阿里", u"京东", u"美团", u"快手", u"网易", u"百度", u"携程", u"阅文",
      u"哔哩", u"同程", u"唯品会", u"汽车之家", u"声网", u"知乎", u"微博", u"B站", u"BILI",
      u"金山软件", u"猎聘", u"宝尊"), u"互联网"),
    # --- 金融/保险 ---
    ((u"友邦", u"中国平安", u"中国人寿", u"中国太保", u"中国财险", u"众安", u"保诚", u"宏利",
      u"太平"), u"金融/保险"),
    # --- 金融/银行 ---
    ((u"建设银行", u"工商银行", u"中国银行", u"农业银行", u"汇丰", u"渣打", u"招商银行", u"邮储",
      u"交通银行", u"民生银行", u"中信银行", u"恒生银行", u"东亚银行", u"中银香港", u"银行"),
     u"金融/银行"),
    # --- 金融 (券商/交易所) ---
    ((u"港交所", u"中信证券", u"中金", u"国泰君安", u"海通", u"华泰", u"广发", u"申万", u"招商证券",
      u"证券", u"期货", u"交易所"), u"金融"),
    # --- 能源 ---
    ((u"海洋石油", u"中国石油", u"中国神华", u"中煤", u"兖矿", u"昆仑能源", u"新奥", u"华润电力",
      u"龙源", u"大唐", u"华能", u"华电", u"中国燃气", u"煤气", u"电能", u"中电", u"港灯",
      u"北控水务", u"燃气", u"石油", u"煤炭", u"电力"), u"能源"),
    # --- 房地产 ---
    ((u"长江实业", u"新鸿基", u"恒基", u"华润置地", u"中国海外", u"龙湖", u"碧桂园", u"万科",
      u"九龙仓", u"太古", u"恒隆", u"嘉里", u"新世界", u"融创", u"世茂", u"置业", u"地产", u"发展"),
     u"房地产"),
    # --- 通信 ---
    ((u"中国移动", u"中国联通", u"中国电信", u"中国铁塔", u"铁塔", u"电信"), u"通信"),
    # --- 消费品 ---
    ((u"安踏", u"李宁", u"蒙牛", u"农夫山泉", u"华润啤酒", u"海底捞", u"蓝月亮", u"维达", u"申洲",
      u"波司登", u"周大福", u"泡泡玛特", u"名创", u"呷哺", u"九毛九", u"百胜", u"统一", u"康师傅",
      u"颐海", u"敏华", u"普拉达", u"安踏体育", u"体育", u"啤酒", u"乳业", u"食品", u"饮料", u"零售",
      u"珠宝"), u"消费品"),
    # --- 工业制造 ---
    ((u"中车", u"三一", u"中联重科", u"潍柴", u"中国重汽", u"东方电气", u"中国中车", u"中国中铁",
      u"中国铁建", u"中国建筑", u"中交建", u"机械", u"重工", u"电气"), u"工业制造"),
    # --- 软件服务 ---
    ((u"金蝶", u"微盟", u"有赞", u"明源", u"软件", u"云"), u"软件服务"),
    # --- 电商/云计算 (fallback) ---
    ((u"云计算", u"数据中心", u"SaaS"), u"电商/云计算"),
]


def _infer_hk_industry(name):
    """Infer HK industry from the Chinese company name (no network/akshare)."""
    if not name:
        return "其他"
    for kws, ind in _HK_INDUSTRY_RULES:
        for kw in kws:
            if kw in name:
                return ind
    return "其他"


def _em_hk_dividend(symbol):
    """Fetch latest HK cash dividend per share from EastMoney F10 (Vercel-safe).

    The Tencent quote feed does not expose dividend yield for HK stocks, so the
    report used to show 0.00% even for high-yield names like Tencent. This hits
    EastMoney's datacenter API (no akshare dependency) and parses the cash amount
    out of the PLAN_EXPLAIN text, e.g. "每股派港币5.3元" -> 5.3.
    Returns {"cash_per_share": float, "ex_date": "YYYY-MM-DD"|"", "year": ""} or None.
    """
    try:
        import re as _re
        params = {
            "reportName": "RPT_HKF10_MAIN_DIVBASIC",
            "columns": "SECURITY_CODE,UPDATE_DATE,REPORT_TYPE,EX_DIVIDEND_DATE,DIVIDEND_DATE,"
                       "TRANSFER_END_DATE,YEAR,PLAN_EXPLAIN,IS_BFP",
            "filter": f'(SECURITY_CODE="{symbol}")(IS_BFP="0")',
            "pageNumber": "1", "pageSize": "10",
            "sortTypes": "-1,-1", "sortColumns": "NOTICE_DATE,EX_DIVIDEND_DATE",
            "source": "F10", "client": "PC", "v": "035584639294227527",
        }
        resp = _http_get(EM_DATACENTER, params=params, timeout=15)
        data = resp.json()
        rows = (data.get("result") or {}).get("data") or []

        def _parse_cash(plan):
            if not plan:
                return 0.0
            # 1) Prefer the explicit HKD-equivalent note, e.g.
            #    "每股派美元0.13125元(相当于港币1.027545元(计算值))"
            m = _re.search(r'相当于港币\s*([\d.]+)\s*元', plan)
            if m:
                return float(m.group(1))
            # 2) Direct per-share HKD/RMB, e.g. "每股派港币5.3元" /
            #    "每股派人民币0.5元". The currency word is 1-2 chars -> use +.
            m = _re.search(r'每股派[港币人民]+\s*([\d.]+)\s*元', plan)
            if m:
                return float(m.group(1))
            # 3) 10-share variant, e.g. "每10股派港币3.4元"
            m = _re.search(r'每10股派[港币人民]+\s*([\d.]+)\s*港元', plan)
            if m:
                return float(m.group(1)) / 10.0
            return 0.0

        for r in rows:
            cash = _parse_cash(r.get("PLAN_EXPLAIN", ""))
            if cash > 0:
                return {
                    "cash_per_share": cash,
                    "ex_date": (r.get("EX_DIVIDEND_DATE") or "")[:10],
                    "year": r.get("YEAR", ""),
                }
        return None
    except Exception as e:
        print(f"[hk_dividend] fail {e}", flush=True)
        return None


def _get_market_industry(code, market):
    """Industry/peers/dividends for HK/US. Name-inference based (no akshare; Vercel-safe).

    The previous version imported akshare for HK company profile + dividends. akshare
    is heavy and not in requirements.txt, so it fails on Vercel (ImportError -> empty
    industry). We now infer the industry from the curated stock name (always available),
    which is also what the candidate pool uses for sector grouping, so peers still align.
    HK cash dividend (per share) is fetched from EastMoney F10 so the report's
    dividend yield is real instead of 0.00%. US keeps the soft-gap (no US ticker
    match) so latest_dividend stays None and yield reads 0 gracefully.
    """
    result = {
        "industry_name": "", "board_name": "", "board_code": "",
        "dividends": [],
        "latest_dividend": None,
    }
    symbol = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "").replace(".HK", "").replace(".US", "")

    # 1) Prefer the curated-list name (always available, no network call)
    name = ""
    try:
        from hk_us_list import HK_US_STOCKS
        for c, n in HK_US_STOCKS.get(market, []):
            if c == symbol:
                name = n
                break
    except Exception:
        pass
    # 2) Fall back to Tencent spot name if the curated list missed
    if not name:
        try:
            info = get_stock_info(code, market)
            name = (info.get("股票简称", "") or info.get("name", "") or "")
        except Exception:
            pass

    # 3) Infer industry
    try:
        if market == "HK":
            result["industry_name"] = _infer_hk_industry(name)
        else:
            result["industry_name"] = _infer_us_industry(name)
        result["board_name"] = result["industry_name"]
    except Exception as e:
        print(f"[industry] infer failed for {code}: {e}", flush=True)

    # 4) Dividends: HK cash dividend per share via EastMoney F10 (Vercel-safe,
    #    no akshare). US keeps the soft-gap (EastMoney HK report won't match US
    #    tickers) so latest_dividend stays None and downstream yields 0 gracefully.
    if market == "HK" and symbol:
        try:
            div = _em_hk_dividend(symbol)
            if div and div.get("cash_per_share", 0) > 0:
                result["latest_dividend"] = {
                    "cash_per_share": div["cash_per_share"],
                    "ex_date": div.get("ex_date", ""),
                    "year": div.get("year", ""),
                }
                result["dividends"] = [result["latest_dividend"]]
                print(f"[industry] {code}: dividend HK$ {div['cash_per_share']} ex {div.get('ex_date','')}", flush=True)
        except Exception as e:
            print(f"[industry] {code}: dividend fetch skipped: {e}", flush=True)
    print(f"[industry] {code}: industry={result['industry_name']}, name={name}", flush=True)
    return result


# US industry name → PE/PB benchmark mapping
_US_INDUSTRY_MAP = {
    "苹果": "科技硬件", "apple": "科技硬件",
    "微软": "软件服务", "microsoft": "软件服务",
    "谷歌": "互联网", "alphabet": "互联网", "google": "互联网",
    "亚马逊": "电商/云计算", "amazon": "电商/云计算",
    "特斯拉": "汽车/新能源", "tesla": "汽车/新能源",
    "英伟达": "半导体", "nvidia": "半导体",
    "meta": "互联网", "facebook": "互联网",
    "台积电": "半导体", "tsm": "半导体",
    "伯克希尔": "金融/保险", "berkshire": "金融/保险",
    "摩根": "金融/银行", "jpmorgan": "金融/银行", "jp morgan": "金融/银行",
    "强生": "医药健康", "johnson": "医药健康",
    "辉瑞": "医药健康", "pfizer": "医药健康",
    "默克": "医药健康", "merck": "医药健康",
    "可口可乐": "消费品", "coca": "消费品",
    "宝洁": "消费品", "procter": "消费品",
    "耐克": "消费品", "nike": "消费品",
    "波音": "工业制造", "boeing": "工业制造",
    "卡特彼勒": "工业制造", "caterpillar": "工业制造",
    "埃克森": "能源", "exxon": "能源",
    "雪佛龙": "能源", "chevron": "能源",
    "迪士尼": "传媒娱乐", "disney": "传媒娱乐",
    "奈飞": "传媒娱乐", "netflix": "传媒娱乐",
}
_US_GENERIC_INDUSTRY = "科技"  # default for US

def _infer_us_industry(name):
    """Infer US stock industry from Chinese name."""
    if not name: return _US_GENERIC_INDUSTRY
    name_lower = name.lower()
    for key, ind in _US_INDUSTRY_MAP.items():
        if key in name_lower or key in name:
            return ind
    return _US_GENERIC_INDUSTRY


# Add international sector PE benchmarks
def _industry_pe_range_intl(industry_name):
    """PE/PB/ROE benchmarks for international (HK/US) sectors."""
    mapping = {
        "软件服务": {"low": 15, "high": 40, "pb_low": 3, "pb_high": 10, "roe_avg": 12, "roe_good": 20},
        "互联网": {"low": 12, "high": 35, "pb_low": 2, "pb_high": 8, "roe_avg": 10, "roe_good": 18},
        "科技硬件": {"low": 12, "high": 30, "pb_low": 2, "pb_high": 7, "roe_avg": 15, "roe_good": 25},
        "半导体": {"low": 15, "high": 45, "pb_low": 2, "pb_high": 8, "roe_avg": 12, "roe_good": 20},
        "电商/云计算": {"low": 15, "high": 40, "pb_low": 3, "pb_high": 10, "roe_avg": 12, "roe_good": 20},
        "金融/保险": {"low": 5, "high": 15, "pb_low": 0.5, "pb_high": 2, "roe_avg": 8, "roe_good": 15},
        "金融/银行": {"low": 4, "high": 12, "pb_low": 0.3, "pb_high": 1.5, "roe_avg": 8, "roe_good": 12},
        "金融": {"low": 8, "high": 20, "pb_low": 0.8, "pb_high": 2.5, "roe_avg": 8, "roe_good": 15},
        "汽车/新能源": {"low": 10, "high": 30, "pb_low": 1, "pb_high": 5, "roe_avg": 8, "roe_good": 15},
        "医药健康": {"low": 12, "high": 35, "pb_low": 1.5, "pb_high": 6, "roe_avg": 10, "roe_good": 18},
        "消费品": {"low": 12, "high": 28, "pb_low": 1.5, "pb_high": 6, "roe_avg": 10, "roe_good": 18},
        "工业制造": {"low": 8, "high": 22, "pb_low": 1, "pb_high": 4, "roe_avg": 8, "roe_good": 15},
        "能源": {"low": 5, "high": 18, "pb_low": 0.6, "pb_high": 2.5, "roe_avg": 8, "roe_good": 12},
        "传媒娱乐": {"low": 12, "high": 30, "pb_low": 1.5, "pb_high": 5, "roe_avg": 8, "roe_good": 15},
        "科技": {"low": 15, "high": 35, "pb_low": 2, "pb_high": 8, "roe_avg": 12, "roe_good": 20},
        "房地产": {"low": 5, "high": 15, "pb_low": 0.3, "pb_high": 1.5, "roe_avg": 5, "roe_good": 10},
        "通信": {"low": 10, "high": 25, "pb_low": 1, "pb_high": 3.5, "roe_avg": 8, "roe_good": 15},
        "其他金融": {"low": 8, "high": 25, "pb_low": 1, "pb_high": 5, "roe_avg": 10, "roe_good": 20},
    }
    # Exact match first
    if industry_name in mapping:
        return mapping[industry_name]
    # Keyword partial match
    for key, val in mapping.items():
        if key in industry_name or industry_name in key:
            return val
    # Default: technology sector
    return mapping["科技"]


# ========== Market Indices API Endpoint ==========
@app.route("/api/market_indices", methods=["POST"])
def api_market_indices():
    """Get market indices for a given market."""
    data = request.json or {}
    mkt = data.get("market", "A")
    result = _get_market_indices(mkt)
    return jsonify({"indices": result, "market": mkt, "market_name": MARKET_CONFIG.get(mkt, {}).get("name", "")})


def deep_cache_clear():
    """Clear deep analysis cache for a specific stock or all."""
    data = request.json
    code = data.get("code", "")
    if code:
        # Clear all dimensions for this stock
        keys_to_del = [k for k in DEEP_CACHE if k.startswith(f"{code}:")]
        for k in keys_to_del:
            del DEEP_CACHE[k]
        return jsonify({"status": "ok", "cleared": len(keys_to_del)})
    else:
        DEEP_CACHE.clear()
        return jsonify({"status": "ok", "cleared": "all"})


# ---------- Stock Search / Autocomplete ----------

# Global stock list cache with TTL
_STOCK_LIST_CACHE = None
_STOCK_LIST_CACHE_TIME = 0

# Comprehensive pinyin mapping for A-share stock name characters
# Each entry: char → (initial, full_pinyin)
_PINYIN_FULL = {
    # --- A -- -
    '安': ('a', 'an'), '爱': ('a', 'ai'), '奥': ('a', 'ao'),
    # --- B -- -
    '白': ('b', 'bai'), '宝': ('b', 'bao'), '北': ('b', 'bei'), '本': ('b', 'ben'),
    '比': ('b', 'bi'), '博': ('b', 'bo'), '邦': ('b', 'bang'), '步': ('b', 'bu'),
    '百': ('b', 'bai'), '玻': ('b', 'bo'), '滨': ('b', 'bin'), '保': ('b', 'bao'),
    '贝': ('b', 'bei'), '碧': ('b', 'bi'), '八': ('b', 'ba'), '巴': ('b', 'ba'),
    '波': ('b', 'bo'), '榜': ('b', 'bang'), '包': ('b', 'bao'), '标': ('b', 'biao'),
    # --- C -- -
    '长': ('c', 'chang'), '成': ('c', 'cheng'), '城': ('c', 'cheng'), '创': ('c', 'chuang'),
    '川': ('c', 'chuan'), '传': ('c', 'chuan'), '船': ('c', 'chuan'), '材': ('c', 'cai'),
    '产': ('c', 'chan'), '车': ('c', 'che'), '晨': ('c', 'chen'), '辰': ('c', 'chen'),
    '财': ('c', 'cai'), '储': ('c', 'chu'), '慈': ('c', 'ci'), '磁': ('c', 'ci'),
    # --- D -- -
    '大': ('d', 'da'), '电': ('d', 'dian'), '东': ('d', 'dong'), '地': ('d', 'di'),
    '达': ('d', 'da'), '迪': ('d', 'di'), '德': ('d', 'de'), '动': ('d', 'dong'),
    '道': ('d', 'dao'), '第': ('d', 'di'), '鼎': ('d', 'ding'), '端': ('d', 'duan'),
    '豆': ('d', 'dou'), '丹': ('d', 'dan'), '盾': ('d', 'dun'), '戴': ('d', 'dai'),
    # --- E -- -
    '鄂': ('e', 'e'), '恩': ('e', 'en'), '尔': ('e', 'er'),
    # --- F -- -
    '方': ('f', 'fang'), '风': ('f', 'feng'), '福': ('f', 'fu'), '富': ('f', 'fu'),
    '飞': ('f', 'fei'), '复': ('f', 'fu'), '发': ('f', 'fa'), '纺': ('f', 'fang'),
    '服': ('f', 'fu'), '房': ('f', 'fang'), '分': ('f', 'fen'), '峰': ('f', 'feng'),
    # --- G -- -
    '工': ('g', 'gong'), '国': ('g', 'guo'), '广': ('g', 'guang'), '高': ('g', 'gao'),
    '格': ('g', 'ge'), '贵': ('g', 'gui'), '公': ('g', 'gong'), '股': ('g', 'gu'),
    '钢': ('g', 'gang'), '光': ('g', 'guang'), '港': ('g', 'gang'), '古': ('g', 'gu'),
    '关': ('g', 'guan'), '冠': ('g', 'guan'), '观': ('g', 'guan'), '管': ('g', 'guan'),
    '硅': ('g', 'gui'), '轨': ('g', 'gui'), '歌': ('g', 'ge'), '谷': ('g', 'gu'),
    '供': ('g', 'gong'), '甘': ('g', 'gan'), '干': ('g', 'gan'), '赣': ('g', 'gan'),
    # --- H -- -
    '海': ('h', 'hai'), '华': ('h', 'hua'), '恒': ('h', 'heng'), '航': ('h', 'hang'),
    '合': ('h', 'he'), '河': ('h', 'he'), '化': ('h', 'hua'), '惠': ('h', 'hui'),
    '湖': ('h', 'hu'), '好': ('h', 'hao'), '环': ('h', 'huan'), '沪': ('h', 'hu'),
    '鸿': ('h', 'hong'), '宏': ('h', 'hong'), '红': ('h', 'hong'), '回': ('h', 'hui'),
    '互': ('h', 'hu'), '花': ('h', 'hua'), '欢': ('h', 'huan'), '汉': ('h', 'han'),
    '杭': ('h', 'hang'), '皇': ('h', 'huang'), '禾': ('h', 'he'), '汇': ('h', 'hui'),
    '和': ('h', 'he'), '浩': ('h', 'hao'), '黑': ('h', 'hei'), '哈': ('h', 'ha'),
    '辉': ('h', 'hui'), '火': ('h', 'huo'),
    # --- J -- -
    '机': ('j', 'ji'), '建': ('j', 'jian'), '金': ('j', 'jin'), '交': ('j', 'jiao'),
    '家': ('j', 'jia'), '京': ('j', 'jing'), '酒': ('j', 'jiu'), '军': ('j', 'jun'),
    '技': ('j', 'ji'), '集': ('j', 'ji'), '健': ('j', 'jian'), '江': ('j', 'jiang'),
    '九': ('j', 'jiu'), '精': ('j', 'jing'), '晶': ('j', 'jing'), '君': ('j', 'jun'),
    '景': ('j', 'jing'), '炬': ('j', 'ju'), '加': ('j', 'jia'), '嘉': ('j', 'jia'),
    '节': ('j', 'jie'), '杰': ('j', 'jie'), '教': ('j', 'jiao'), '基': ('j', 'ji'),
    '洁': ('j', 'jie'), '锦': ('j', 'jin'), '井': ('j', 'jing'), '进': ('j', 'jin'),
    '吉': ('j', 'ji'), '均': ('j', 'jun'), '巨': ('j', 'ju'), '经': ('j', 'jing'),
    # --- K -- -
    '科': ('k', 'ke'), '康': ('k', 'kang'), '凯': ('k', 'kai'), '开': ('k', 'kai'),
    '控': ('k', 'kong'), '口': ('k', 'kou'), '矿': ('k', 'kuang'), '空': ('k', 'kong'),
    '扩': ('k', 'kuo'), '垦': ('k', 'ken'), '酷': ('k', 'ku'), '快': ('k', 'kuai'),
    # --- L -- -
    '联': ('l', 'lian'), '老': ('l', 'lao'), '龙': ('l', 'long'), '利': ('l', 'li'),
    '林': ('l', 'lin'), '泸': ('l', 'lu'), '路': ('l', 'lu'), '绿': ('l', 'lü'),
    '蓝': ('l', 'lan'), '领': ('l', 'ling'), '隆': ('l', 'long'), '洛': ('l', 'luo'),
    '鲁': ('l', 'lu'), '力': ('l', 'li'), '流': ('l', 'liu'), '拉': ('l', 'la'),
    '来': ('l', 'lai'), '乐': ('l', 'le'), '量': ('l', 'liang'), '零': ('l', 'ling'),
    '六': ('l', 'liu'), '临': ('l', 'lin'), '理': ('l', 'li'), '丽': ('l', 'li'),
    '锂': ('l', 'li'), '铝': ('l', 'lü'), '兰': ('l', 'lan'), '良': ('l', 'liang'),
    # --- M -- -
    '美': ('m', 'mei'), '民': ('m', 'min'), '明': ('m', 'ming'), '牡': ('m', 'mu'),
    '煤': ('m', 'mei'), '蒙': ('m', 'meng'), '曼': ('m', 'man'), '名': ('m', 'ming'),
    '模': ('m', 'mo'), '摩': ('m', 'mo'), '幕': ('m', 'mu'), '迈': ('m', 'mai'),
    '码': ('m', 'ma'), '密': ('m', 'mi'), '茅': ('m', 'mao'),
    # --- N -- -
    '宁': ('n', 'ning'), '南': ('n', 'nan'), '能': ('n', 'neng'), '农': ('n', 'nong'),
    '内': ('n', 'nei'), '纳': ('n', 'na'), '男': ('n', 'nan'),
    # --- P -- -
    '平': ('p', 'ping'), '浦': ('p', 'pu'), '普': ('p', 'pu'), '品': ('p', 'pin'),
    '片': ('p', 'pian'), '牌': ('p', 'pai'),
    # --- Q -- -
    '汽': ('q', 'qi'), '青': ('q', 'qing'), '全': ('q', 'quan'), '券': ('q', 'quan'),
    '泉': ('q', 'quan'), '区': ('q', 'qu'), '旗': ('q', 'qi'), '强': ('q', 'qiang'),
    '轻': ('q', 'qing'), '铅': ('q', 'qian'), '奇': ('q', 'qi'), '球': ('q', 'qiu'),
    '氢': ('q', 'qing'), '前': ('q', 'qian'), '企': ('q', 'qi'), '千': ('q', 'qian'),
    # --- R -- -
    '人': ('r', 'ren'), '日': ('r', 'ri'), '瑞': ('r', 'rui'), '软': ('r', 'ruan'),
    '荣': ('r', 'rong'), '融': ('r', 'rong'), '润': ('r', 'run'), '燃': ('r', 'ran'),
    # --- S -- -
    '上': ('s', 'shang'), '深': ('s', 'shen'), '石': ('s', 'shi'), '世': ('s', 'shi'),
    '三': ('s', 'san'), '生': ('s', 'sheng'), '数': ('s', 'shu'), '水': ('s', 'shui'),
    '商': ('s', 'shang'), '实': ('s', 'shi'), '设': ('s', 'she'), '山': ('s', 'shan'),
    '苏': ('s', 'su'), '神': ('s', 'shen'), '沙': ('s', 'sha'), '食': ('s', 'shi'),
    '双': ('s', 'shuang'), '塑': ('s', 'su'), '首': ('s', 'shou'), '顺': ('s', 'shun'),
    '申': ('s', 'shen'), '省': ('s', 'sheng'), '盛': ('s', 'sheng'), '胜': ('s', 'sheng'),
    '四': ('s', 'si'), '松': ('s', 'song'), '丝': ('s', 'si'), '穗': ('s', 'sui'),
    # --- T -- -
    '天': ('t', 'tian'), '通': ('t', 'tong'), '太': ('t', 'tai'), '台': ('t', 'tai'),
    '泰': ('t', 'tai'), '铁': ('t', 'tie'), '投': ('t', 'tou'), '同': ('t', 'tong'),
    '拓': ('t', 'tuo'), '团': ('t', 'tuan'), '特': ('t', 'te'), '塔': ('t', 'ta'),
    '唐': ('t', 'tang'), '太': ('t', 'tai'), '铜': ('t', 'tong'), '腾': ('t', 'teng'),
    '太': ('t', 'tai'),
    # --- W -- -
    '万': ('w', 'wan'), '五': ('w', 'wu'), '物': ('w', 'wu'), '芜': ('w', 'wu'),
    '网': ('w', 'wang'), '潍': ('w', 'wei'), '文': ('w', 'wen'), '维': ('w', 'wei'),
    '微': ('w', 'wei'), '无': ('w', 'wu'), '武': ('w', 'wu'), '伟': ('w', 'wei'),
    '旺': ('w', 'wang'), '威': ('w', 'wei'), '卫': ('w', 'wei'), '温': ('w', 'wen'),
    '唯': ('w', 'wei'), '外': ('w', 'wai'), '玩': ('w', 'wan'),
    # --- X -- -
    '新': ('x', 'xin'), '西': ('x', 'xi'), '兴': ('x', 'xing'), '信': ('x', 'xin'),
    '星': ('x', 'xing'), '小': ('x', 'xiao'), '厦': ('x', 'xia'), '协': ('x', 'xie'),
    '学': ('x', 'xue'), '希': ('x', 'xi'), '秀': ('x', 'xiu'), '芯': ('x', 'xin'),
    '选': ('x', 'xuan'), '销': ('x', 'xiao'), '新': ('x', 'xin'), '许': ('x', 'xu'),
    '锡': ('x', 'xi'), '祥': ('x', 'xiang'), '先': ('x', 'xian'), '雪': ('x', 'xue'),
    '现': ('x', 'xian'), '香': ('x', 'xiang'), '湘': ('x', 'xiang'), '翔': ('x', 'xiang'),
    '效': ('x', 'xiao'), '消': ('x', 'xiao'),
    # --- Y -- -
    '一': ('y', 'yi'), '银': ('y', 'yin'), '洋': ('y', 'yang'), '医': ('y', 'yi'),
    '云': ('y', 'yun'), '运': ('y', 'yun'), '有': ('y', 'you'), '药': ('y', 'yao'),
    '易': ('y', 'yi'), '阳': ('y', 'yang'), '亚': ('y', 'ya'), '因': ('y', 'yin'),
    '园': ('y', 'yuan'), '亿': ('y', 'yi'), '永': ('y', 'yong'), '烟': ('y', 'yan'),
    '医': ('y', 'yi'), '延': ('y', 'yan'), '英': ('y', 'ying'), '娱': ('y', 'yu'),
    '元': ('y', 'yuan'), '远': ('y', 'yuan'), '粤': ('y', 'yue'), '越': ('y', 'yue'),
    '沿': ('y', 'yan'), '优': ('y', 'you'), '盈': ('y', 'ying'), '宇': ('y', 'yu'),
    '饮': ('y', 'yin'), '液': ('y', 'ye'), '业': ('y', 'ye'), '研': ('y', 'yan'),
    # --- Z -- -
    '中': ('z', 'zhong'), '招': ('z', 'zhao'), '重': ('z', 'zhong'), '证': ('z', 'zheng'),
    '智': ('z', 'zhi'), '紫': ('z', 'zi'), '正': ('z', 'zheng'), '张': ('z', 'zhang'),
    '制': ('z', 'zhi'), '站': ('z', 'zhan'), '展': ('z', 'zhan'), '振': ('z', 'zhen'),
    '住': ('z', 'zhu'), '装': ('z', 'zhuang'), '浙': ('z', 'zhe'), '珠': ('z', 'zhu'),
    '自': ('z', 'zi'), '总': ('z', 'zong'), '知': ('z', 'zhi'), '兆': ('z', 'zhao'),
    '卓': ('z', 'zhuo'), '尊': ('z', 'zun'), '庄': ('z', 'zhuang'), '洲': ('z', 'zhou'),
    '志': ('z', 'zhi'), '纸': ('z', 'zhi'), '之': ('z', 'zhi'), '轴': ('z', 'zhou'),
    '众': ('z', 'zhong'), '泽': ('z', 'ze'), '筑': ('z', 'zhu'), '周': ('z', 'zhou'),
    '专': ('z', 'zhuan'), '子': ('z', 'zi'), '资': ('z', 'zi'),
    # --- Additional common stock name characters ---
    '州': ('z', 'zhou'), '的': ('d', 'de'), '行': ('h', 'hang'),
    '视': ('s', 'shi'), '伊': ('y', 'yi'), '井': ('j', 'jing'),
    '窖': ('j', 'jiao'), '田': ('t', 'tian'), '贡': ('g', 'gong'),
    '司': ('s', 'si'), '味': ('w', 'wei'), '份': ('f', 'fen'),
    '业': ('y', 'ye'), '黄': ('h', 'huang'), '移': ('y', 'yi'),
    '讯': ('x', 'xun'), '秦': ('q', 'qin'), '介': ('j', 'jie'),
    '源': ('y', 'yuan'), '药': ('y', 'yao'), '牧': ('m', 'mu'),
    '峰': ('f', 'feng'), '锋': ('f', 'feng'), '望': ('w', 'wang'),
    '超': ('c', 'chao'), '夏': ('x', 'xia'), '媒': ('m', 'mei'),
    '场': ('c', 'chang'), '果': ('g', 'guo'), '时': ('s', 'shi'),
    '境': ('j', 'jing'), '解': ('j', 'jie'), '粮': ('l', 'liang'),
    '络': ('l', 'luo'), '线': ('x', 'xian'), '酒': ('j', 'jiu'),
    '稀': ('x', 'xi'), '土': ('t', 'tu'), '原': ('y', 'yuan'),
    '石': ('s', 'shi'), '油': ('y', 'you'), '气': ('q', 'qi'),
    '焦': ('j', 'jiao'), '炭': ('t', 'tan'), '纸': ('z', 'zhi'),
    '饮': ('y', 'yin'), '料': ('l', 'liao'), '食': ('s', 'shi'),
    '品': ('p', 'pin'), '房': ('f', 'fang'), '地': ('d', 'di'),
    '产': ('c', 'chan'), '建': ('j', 'jian'), '筑': ('z', 'zhu'),
    '寿': ('s', 'shou'), '险': ('x', 'xian'), '秦': ('q', 'qin'),
    '鞍': ('a', 'an'), '钢': ('g', 'gang'), '铁': ('t', 'tie'),
    '煤': ('m', 'mei'), '炭': ('t', 'tan'), '烯': ('x', 'xi'),
    '顶': ('d', 'ding'), '集': ('j', 'ji'), '团': ('t', 'tuan'),
    '控': ('k', 'kong'), '股': ('g', 'gu'), '有': ('y', 'you'),
    '限': ('x', 'xian'), '公': ('g', 'gong'), '司': ('s', 'si'),
    '馆': ('g', 'guan'), '园': ('y', 'yuan'), '地': ('d', 'di'),
}


def _build_pinyin_info(name):
    """Build pinyin data for a Chinese stock name.
    Returns dict with:
      - initials: concatenated first letters, lowercased (e.g., '*ST贵州茅台' → '*stgzmt')
      - full: concatenated full pinyin, lowercased (e.g., '*ST贵州茅台' → '*stguizhoumaotai')
    Uses pypinyin library if available (covers all Chinese characters),
    falls back to _PINYIN_FULL manual dictionary otherwise.
    IMPORTANT: Both initials and full are lowercased so that case-insensitive
    matching works correctly (e.g. user types '*stx' → matches init='*stxl').
    """
    if _HAS_PYPINYIN:
        try:
            initials_list = _pypinyin_func(name, style=_PinyinStyle.FIRST_LETTER, errors='default')
            full_list = _pypinyin_func(name, style=_PinyinStyle.NORMAL, errors='default')
            initials = ''.join([p[0] for p in initials_list]).lower()
            full = ''.join([''.join(p) for p in full_list]).lower()
            return {'initials': initials, 'full': full}
        except Exception:
            pass  # fall through to manual method

    # Fallback: manual dictionary
    initials = []
    full = []
    for ch in name:
        p = _PINYIN_FULL.get(ch)
        if p:
            initials.append(p[0])
            full.append(p[1])
        else:
            # Unknown character: just use the character itself for full, skip initial
            full.append(ch)
    return {
        'initials': ''.join(initials).lower(),
        'full': ''.join(full).lower(),
    }


def _get_stock_list():
    """Get full A-share stock list (always available).
    Uses embedded static list (~5500 stocks) instantly.
    Optionally refreshes from EastMoney in background."""
    global _STOCK_LIST_CACHE, _STOCK_LIST_CACHE_TIME

    now = time.time()
    if _STOCK_LIST_CACHE and (now - _STOCK_LIST_CACHE_TIME) < 3600:
        return _STOCK_LIST_CACHE

    # Load static list first (instant, always available)
    try:
        import stock_list_full
        result = []
        for code, name, suffix in stock_list_full.ALL_STOCKS:
            pinyin_info = _build_pinyin_info(name)
            result.append({
                "code": code, "name": name, "code_full": code + suffix,
                "pinyin": pinyin_info["initials"], "pinyin_full": pinyin_info["full"],
            })
        pinyin_source = "pypinyin" if _HAS_PYPINYIN else "manual"
        print(f"[stock_list] Static embedded: {len(result)} stocks (pinyin: {pinyin_source})")
        _STOCK_LIST_CACHE = result
        _STOCK_LIST_CACHE_TIME = now
        return result
    except ImportError:
        print("[stock_list] stock_list_full.py not found, trying alternatives...")

    # Fallback: STATIC_PEERS (185 stocks)
    result = []
    seen = set()
    for cat, stocks in api_fallback.STATIC_PEERS.items():
        for c, n, wc in stocks:
            if c not in seen:
                seen.add(c)
                suffix = ".SH" if c.startswith("6") else ".SZ"
                pinyin_info = _build_pinyin_info(n)
                result.append({
                    "code": c, "name": n, "code_full": c + suffix,
                    "pinyin": pinyin_info["initials"], "pinyin_full": pinyin_info["full"],
                })
    print(f"[stock_list] STATIC_PEERS fallback: {len(result)} stocks")
    _STOCK_LIST_CACHE = result
    _STOCK_LIST_CACHE_TIME = now
    return result


def _score_and_rank(candidates, q, top_n):
    """Unified multi-level fuzzy scorer shared by A / HK / US search.

    Each candidate: dict with keys code, name, pinyin(initials), pinyin_full(full).
    Matching levels (same as A-share):
      - exact/prefix/substring code
      - exact/prefix/substring name
      - non-contiguous ordered name chars (e.g. '光股' → '阳光股份')
      - pinyin initials (e.g. 'tx' → '腾讯控股')
      - full pinyin (e.g. 'txkg' → '腾讯控股')
      - compound pinyin prefix
    Returns top_n ranked dicts (without the score tuple)."""
    q_lower = q.lower()
    scored = []
    for s in candidates:
        score = 0
        code = s.get("code", "")
        name = s.get("name", "")
        name_lower = name.lower()
        pin = s.get("pinyin", "") or ""
        pin_full = s.get("pinyin_full", "") or ""

        # --- Code matching ---
        if code.lower() == q_lower:
            score = max(score, 1000)
        elif code.lower().startswith(q_lower):
            score = max(score, 800)
        elif q_lower in code.lower():
            score = max(score, 600)

        # --- Name matching ---
        if name == q:
            score = max(score, 950)
        elif name.startswith(q):
            score = max(score, 700)
        elif q_lower in name_lower:
            score = max(score, 500)

        # --- Non-contiguous ordered name matching (e.g. '光股' → '阳光股份') ---
        if len(q) >= 2 and score < 500:
            pos = 0
            match = True
            for ch in q_lower:
                idx = name_lower.find(ch, pos)
                if idx == -1:
                    match = False
                    break
                pos = idx + 1
            if match and q_lower not in name_lower:
                score = max(score, 350)

        # --- Pinyin initials matching (e.g. 'tx' → '腾讯控股') ---
        if pin and q_lower in pin:
            score = max(score, 400)

        # --- Full pinyin matching (e.g. 'txkg' → '腾讯控股') ---
        if pin_full and q_lower in pin_full:
            score = max(score, 420)

        # --- Compound pinyin prefix matching ---
        if pin_full and len(q_lower) >= 3 and q_lower not in pin_full and q_lower not in (pin or ''):
            test_pos = 0
            for ch in pin_full:
                if test_pos < len(q_lower) and ch == q_lower[test_pos]:
                    test_pos += 1
            if test_pos >= len(q_lower) * 0.7:
                score = max(score, 300)

        if score > 0:
            scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:top_n]]


_HK_US_LIST_CACHE = {}


def _get_hk_us_list(mkt):
    """Curated popular HK/US stocks with pinyin (cached). Enables A-share-like
    initials / full-pinyin / fuzzy matching for the most-searched names."""
    global _HK_US_LIST_CACHE
    if _HK_US_LIST_CACHE.get(mkt):
        return _HK_US_LIST_CACHE[mkt]
    data = HK_US_STOCKS.get(mkt, [])
    result = []
    for code, name in data:
        pin = _build_pinyin_info(name)
        result.append({
            "code": code,
            "code_full": f"{code}.{mkt}",
            "name": name,
            "market": mkt,
            "pinyin": pin["initials"],
            "pinyin_full": pin["full"],
        })
    _HK_US_LIST_CACHE[mkt] = result
    print(f"[stock_list] HK/US curated {mkt}: {len(result)} stocks")
    return result


def _smartbox_search(q, mkt):
    """Tencent smartbox for HK/US (long-tail coverage beyond the curated list).
    Field format: hk~00700~腾讯控股~txkg~GP  (field[3] = pinyin initials)."""
    cfg = MARKET_CONFIG.get(mkt, {})
    t_code = cfg.get("search_t_code")
    if not t_code:
        return []
    try:
        url = f"https://smartbox.gtimg.cn/s3/?q={quote(q)}&t={t_code}"
        resp = _http_get(url, timeout=8)
        text = resp.text.strip()
        if not text or text == "None" or "=" not in text:
            return []
        parts = text.split('"')
        if len(parts) < 2:
            return []
        raw = parts[1]
        results = []
        for item in raw.split("^"):
            fields = item.split("~")
            if len(fields) < 3:
                continue
            raw_code = fields[1]
            name = fields[2]
            # Tencent smartbox embeds literal \uXXXX escapes inside the JS
            # string (e.g. "\u817e\u8baf\u63a7\u80a1"); decode them so the
            # names render as real Chinese instead of "\u817e..." in the UI.
            if "\\u" in name:
                try:
                    name = re.sub(r"\\u([0-9a-fA-F]{4})",
                                  lambda m: chr(int(m.group(1), 16)), name)
                except Exception:
                    pass
            smart_pinyin = fields[3] if len(fields) >= 4 else ""
            clean_code = raw_code.split(".")[0] if mkt == "US" else raw_code
            # Build full pinyin from the name; reuse smartbox initials if present.
            pin_full_info = _build_pinyin_info(name)
            results.append({
                "code": clean_code,
                "code_full": f"{clean_code}.{mkt}",
                "name": name,
                "market": mkt,
                "pinyin": smart_pinyin or pin_full_info["initials"],
                "pinyin_full": pin_full_info["full"],
            })
        return results
    except Exception as e:
        print(f"[Search] smartbox {mkt} failed: {e}")
        return []


@app.route("/api/search")
def search_stocks():
    """Fuzzy search stocks. Supports A-share / HK / US via market filter.
    All three markets share the same multi-level fuzzy scorer (code / name /
    pinyin initials / full pinyin / non-contiguous). HK/US additionally merge
    Tencent smartbox results for long-tail coverage."""
    q = request.args.get("q", "").strip()
    mkt = request.args.get("market", "A")
    if not q or len(q) < 1:
        return jsonify({"results": []})

    if mkt == "A":
        stock_list = _get_stock_list()
        results = _score_and_rank(stock_list, q, 6)
        for r in results:
            r["market"] = "A"
        return jsonify({"results": results, "market": "A"})

    # ---- HK / US: curated local list (pinyin + fuzzy) + smartbox long-tail ----
    candidates = list(_get_hk_us_list(mkt))
    local_codes = {c["code"] for c in candidates}
    for s in _smartbox_search(q, mkt):
        if s["code"] not in local_codes:
            candidates.append(s)

    results = _score_and_rank(candidates, q, 12)
    for r in results:
        r["market"] = mkt
    return jsonify({"results": results, "market": mkt})


@app.route("/api/search/refresh")
def refresh_stock_list():
    """Force refresh the stock list cache."""
    global _STOCK_LIST_CACHE, _STOCK_LIST_CACHE_TIME
    _STOCK_LIST_CACHE = None
    _STOCK_LIST_CACHE_TIME = 0
    new_list = _get_stock_list()
    return jsonify({"status": "ok", "count": len(new_list)})


# ---------- Health Check ----------

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "data_source": "Tencent + EastMoney + Sina(fallback)",
        "time": now_cn().isoformat(),
    })


@app.route("/api/fallback_status")
def fallback_status():
    """Debug endpoint: show recent API fallback events."""
    since = request.args.get("since", type=int)
    log_entries = api_fallback.get_fallback_log(since)
    primary_count = 0
    fallback_count = 0
    fail_count = 0
    sources = {}
    for entry in log_entries:
        if entry["source"] == "primary":
            primary_count += 1
        elif entry["ok"]:
            fallback_count += 1
        else:
            fail_count += 1
        sources[entry["source"]] = sources.get(entry["source"], 0) + 1

    return jsonify({
        "summary": {
            "total_events": len(log_entries),
            "primary_ok": primary_count,
            "fallback_ok": fallback_count,
            "fallback_fail": fail_count,
        },
        "sources": sources,
        "recent": log_entries[-20:],
        "_next_since": len(api_fallback.FALLBACK_LOG),
    })


# ---------- Alternative Deep Compare (SSE streaming) ----------

@app.route("/api/alt_deep_compare", methods=["POST"])
def alt_deep_compare():
    """Stream LLM-generated deep comparison analysis for alternative stocks."""
    data = request.json
    alt_stock = data.get("alt_stock", {})
    current_stock = data.get("current_stock", {})

    chat_cfg = get_ai_config()

    # Build context for the LLM
    alt_name = alt_stock.get("name", "替代标的")
    cur_name = current_stock.get("name", "当前股票")

    alt_pe = alt_stock.get("pe", 0)
    cur_pe = current_stock.get("pe", 0)
    alt_pb = alt_stock.get("pb", 0)
    cur_pb = current_stock.get("pb", 0)
    alt_roe = alt_stock.get("roe", 0)
    cur_roe = current_stock.get("scores", {}).get("fundamental", {}).get("detail", {}).get("latest_roe", 0) or 0
    alt_div = alt_stock.get("dividend_yield", 0) or 0
    cur_div = current_stock.get("scores", {}).get("fundamental", {}).get("detail", {}).get("dividend_yield", 0) or 0
    alt_mcap = alt_stock.get("market_cap", 0) or 0
    cur_mcap = current_stock.get("total_mv", 0) or 0
    alt_peg = alt_stock.get("peg", 0) or 0
    cur_peg = current_stock.get("scores", {}).get("value", {}).get("detail", {}).get("peg", 0) or 0
    alt_price = alt_stock.get("price", 0)
    cur_price = current_stock.get("price", 0)
    alt_score = alt_stock.get("total_score", 0)
    cur_score = current_stock.get("total_score", 0)
    alt_change = alt_stock.get("change", 0) or 0
    cur_change = current_stock.get("change_pct", 0) or 0

    # Build scores comparison
    alt_scores = alt_stock.get("scores_breakdown", {})
    cur_scores = current_stock.get("scores", {})
    dims = ["fundamental", "technical", "capital", "events", "industry", "value"]
    score_text = ""
    dim_labels_map = {
        "fundamental": "基本面(25分)", "technical": "技术面(20分)",
        "capital": "资金面(15分)", "events": "事件催化(10分)",
        "industry": "同业对标(15分)", "value": "投资性价比(15分)"
    }
    for dim in dims:
        alt_s = alt_scores.get(dim, {}).get("score", 0) if isinstance(alt_scores, dict) else 0
        cur_s = cur_scores.get(dim, {}).get("score", 0) if isinstance(cur_scores, dict) else 0
        score_text += f"  {dim_labels_map[dim]}: {alt_name}={alt_s}分, {cur_name}={cur_s}分\n"

    context = f"""请你作为资深证券分析师，对以下两只同行业股票进行专业的深度对比分析。

【当前股票】{cur_name}
  价格: ¥{cur_price:.2f}  涨跌幅: {cur_change:+.2f}%
  PE: {'亏损' if cur_pe <= 0 else f'{cur_pe:.1f}'}  PB: {cur_pb:.2f}
  ROE: {cur_roe:.1f}%  股息率: {cur_div:.2f}%
  市值: {cur_mcap:.0f}亿  PEG: {'无' if cur_peg <= 0 else f'{cur_peg:.2f}'}
  综合评分: {cur_score}/100

【替代标的】{alt_name}
  价格: ¥{alt_price:.2f}  涨跌幅: {alt_change:+.2f}%
  PE: {'亏损' if alt_pe <= 0 else f'{alt_pe:.1f}'}  PB: {alt_pb:.2f}
  ROE: {alt_roe:.1f}%  股息率: {alt_div:.2f}%
  市值: {'无' if alt_mcap <= 0 else f'{alt_mcap/1e8:.0f}亿'}  PEG: {'无' if alt_peg <= 0 else f'{alt_peg:.2f}'}
  综合评分: {alt_score}/100

【六维度评分对比】
{score_text}

请严格按以下四个阶段输出（每个阶段以 [SECTION:xxx] 开始标记）：

[SECTION:score_analysis]
针对六维度评分对比进行深度解读（150字左右）。分析为什么某些维度某只股票占优，评分差异背后的逻辑是什么，哪个维度差距最大意味着什么。

[SECTION:financial_analysis]
针对财务指标对比进行深度解读（150字左右）。对比 PE/PB/ROE/股息率/PEG/市值等关键指标，分析谁的估值更合理、谁的盈利能力更强、谁的成长性更好。

[SECTION:debate_analysis]
用辩论式口吻进行综合评价（200字左右）。分别阐述{alt_name}相对{cur_name}的3个核心优势，和{alt_name}相对{cur_name}的3个潜在风险或不足。

[SECTION:verdict]
给出最终综合研判（100字左右）。明确说明{alt_name}是否值得作为{cur_name}的补充或替代配置，适合什么类型的投资者。"""

    def generate():
        """SSE generator - collects full LLM response, then sends per-section."""
        if not chat_cfg.get("api_key") or chat_cfg["api_key"].startswith("sk-your"):
            yield f"data: {json.dumps({'section': 'error', 'content': 'AI 分析功能尚未配置 API Key'}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=chat_cfg["api_key"],
                base_url=chat_cfg.get("api_base", "https://api.openai.com/v1"),
            )

            system_prompt = "你是一位资深证券分析师，擅长同行业股票对比分析。请严格按照用户要求的格式输出，不要遗漏任何标记。"

            # Send progress indicator immediately
            yield f"data: {json.dumps({'section': 'progress', 'content': 'start'}, ensure_ascii=False)}\n\n"

            response = client.chat.completions.create(
                model=chat_cfg.get("model", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context},
                ],
                temperature=0.7,
                max_tokens=1500,
                stream=True,
            )

            # Collect full response with progress updates
            full_text = ""
            token_count = 0
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content is None:
                    continue
                full_text += delta.content
                token_count += 1
                # Send progress every ~25 tokens
                if token_count % 25 == 0:
                    yield f"data: {json.dumps({'section': 'progress', 'content': f'token_{token_count}'}, ensure_ascii=False)}\n\n"

            # Parse sections: split by [SECTION:xxx]
            import re as _re
            parts = _re.split(r'\[SECTION:(\w+)\]\s*', full_text)

            section_name = None
            for i in range(1, len(parts), 2):
                try:
                    section_name = parts[i]
                    section_content = parts[i + 1].strip() if i + 1 < len(parts) else ""
                    if section_content:
                        yield f"data: {json.dumps({'section': section_name, 'content': section_content}, ensure_ascii=False)}\n\n"
                except Exception:
                    continue

            yield "data: [DONE]\n\n"

        except Exception as e:
            error_content = f"AI 分析生成失败: {str(e)}\\n\\n请检查 API Key 和网络连接"
            yield f"data: {json.dumps({'section': 'error', 'content': error_content}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


if __name__ == "__main__":
    import time as _time
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # 禁用静态文件缓存
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.jinja_env.auto_reload = True

    # Cache-busting helper
    @app.context_processor
    def inject_cache_buster():
        return {'cache_buster': int(_time.time())}

    app.run(host="0.0.0.0", port=8888, debug=True)
