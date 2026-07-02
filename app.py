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
from datetime import datetime, timedelta
from functools import lru_cache

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
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory, Response

# ----- Trust no proxy: create a dedicated session that never touches system proxy settings -----
_http_session = requests.Session()
_http_session.trust_env = False

app = Flask(__name__)

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

def _http_get(url, params=None, timeout=15, headers_extra=None):
    """HTTP GET using trust_env=False session — never touches system proxy."""
    headers = dict(REQ_HEADERS)
    if headers_extra:
        headers.update(headers_extra)
    return _http_session.get(url, params=params, headers=headers, timeout=timeout)

def _tencent_code(code):
    """Convert standard code to Tencent format: sz000001, sh600519"""
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

def get_stock_info(code):
    """Get stock basic info via Tencent real-time quote API."""
    cache_key = f"info_{code}"
    cached_val = cached(cache_key)
    if cached_val:
        return cached_val

    try:
        tc = _tencent_code(code)
        symbol = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "")
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

        # Tencent real-time quote field mapping (index-based):
        # 0: unknown, 1: name, 2: code, 3: current price, 4: prev close,
        # 5: open, 6: volume(lots), 7: outer, 8: inner,
        # 30: date, 31: change, 32: change%, 33: high, 34: low,
        # 35: price/volume/amount, 36: volume, 37: amount(wan),
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
            "总市值": float(fields[44]) if len(fields) > 44 and fields[44] else 0,
            "流通市值": float(fields[45]) if len(fields) > 45 and fields[45] else 0,
            "成交量": int(float(fields[6])) if len(fields) > 6 and fields[6] else 0,
            "成交额": float(fields[37]) if len(fields) > 37 and fields[37] else 0,
        }

        cache_set(cache_key, info)
        return info
    except Exception as e:
        print(f"Error fetching stock info for {code}: {e}")
        return {}


def get_price_history(code, days=250):
    """Get historical K-line data via Tencent API."""
    cache_key = f"price_{code}"
    cached_val = cached(cache_key, ttl=600)
    if cached_val:
        return cached_val

    try:
        tc = _tencent_code(code)
        resp = _http_get(f"{TENCENT_QT}{tc}", params={"q": "jk", "fmt": "json"})
        if resp.status_code != 200 or not resp.text:
            cache_set(cache_key, [])
            return []

        # Tencent K-line API
        kline_url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        params = {
            "param": f"{tc},day,,,{days + 30},qfq",
            "_": str(int(time.time() * 1000)),
        }
        resp2 = _http_get(kline_url, params=params, timeout=15)
        data = resp2.json()

        # Navigate the nested structure
        stock_data = data.get("data", {}).get(tc, {})
        klines = stock_data.get("qfqday", stock_data.get("day", []))

        if not klines:
            cache_set(cache_key, [])
            return []

        result = []
        for k in klines[-days:]:
            result.append({
                "日期": k[0],
                "开盘": float(k[1]),
                "收盘": float(k[2]),
                "最高": float(k[3]),
                "最低": float(k[4]),
                "成交量": int(float(k[5])),
            })

        cache_set(cache_key, result)
        return result
    except Exception as e:
        print(f"Error fetching price history for {code}: {e}")
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
        if report_name == "RPT_DMSK_FN_BALANCE":
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


def get_financial_data(code):
    """Fetch financial statements from EastMoney Datacenter API."""
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
                "加权ROE": r.get("WEIGHTAVG_ROE"),
            })

    return {
        "income": income,
        "balance": balance,
        "quarterly": quarterly_income,
    }


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

        url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
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
        return {"error": f"资金流向获取失败: {str(e)}"}


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
            ("增持", 3), ("回购", 3), ("股权激励", 3), ("中标", 3),
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
def _industry_pe_range(industry_name):
    """Return (low, high, pb_low, pb_high, roe_avg, roe_good) for given industry."""
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
    # Match by keyword
    for key, val in mapping.items():
        if key in industry_name:
            return val
    # Default: general industry
    return {"low": 10, "high": 30, "pb_low": 1.0, "pb_high": 4.0, "roe_avg": 8, "roe_good": 15}


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
    """Main analysis function - generates comprehensive report."""
    # Clean code
    code = code.strip().upper()
    if not code.endswith((".SZ", ".SH", ".BJ")):
        if code.startswith(("6", "9")):
            code = f"{code}.SH"
        elif code.startswith(("0", "3")):
            code = f"{code}.SZ"
        elif code.startswith(("4", "8")):
            code = f"{code}.BJ"
        else:
            return {"error": "无法识别股票代码，请输入如 000607.SZ 或 600968.SH 的格式"}

    symbol = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "")

    # ---- Init warnings list for fallback tracking ----
    warnings = []

    # Fetch all data
    info = get_stock_info(code)
    financial = get_financial_data(code)
    prices_data = get_price_history(code, days=250)
    money_flow = get_money_flow(code)
    news = get_news_events(code)
    industry_data = get_industry_data(code)

    # Check for data fetch warnings
    if not financial.get("income"):
        warnings.append({"dim": "基本面", "msg": "利润表数据获取失败，基本面分析可能不完整"})
    if money_flow.get("error"):
        warnings.append({"dim": "资金面", "msg": money_flow["error"]})
    if not news:
        warnings.append({"dim": "事件催化", "msg": "新闻公告数据获取失败"})
    if not industry_data.get("industry_name"):
        warnings.append({"dim": "同业对标", "msg": "行业分类数据获取失败，同业对比可能不完整"})
    if not prices_data:
        warnings.append({"dim": "技术面", "msg": "历史K线数据获取失败，技术分析不完整"})

    if not info or not info.get("最新价"):
        return {"error": f"无法获取股票 {code} 的数据，请检查代码是否正确或稍后重试"}

    # ---- Basic Info ----
    price = info.get("最新价", 0)
    pe = info.get("市盈率-动态", 0)
    pb = info.get("市净率", 0)
    total_mv = info.get("总市值", 0)
    circ_mv = info.get("流通市值", 0)
    change_pct = info.get("涨跌幅", 0)

    report = {
        "code": code,
        "symbol": symbol,
        "name": info.get("股票简称", code),
        "price": price,
        "pe": pe,
        "pb": pb,
        "total_mv": total_mv,
        "circ_mv": circ_mv,
        "change_pct": change_pct,
        "report_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
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
        flow_detail["retail_5d_net"] = summary["retail_5d_net"]
        flow_detail["main_inflow_days"] = summary["main_inflow_days"]
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
        pe_range = _industry_pe_range(ind_name)
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
            value_detail["dividend_note"] = "无分红记录"
        value_breakdown.append({"item": "股息率", "change": 0, "score_after": value_score, "detail": "无分红数据或每股分红为0"})

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

    # PEG estimation (if revenue growth available)
    rev_growth = fund_detail.get("revenue_growth")
    if rev_growth and rev_growth > 0 and pe > 0:
        peg = pe / rev_growth
        value_detail["peg"] = round(peg, 2)
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
    return result


def find_alternatives(code):
    """Find peer stocks in the same industry. Pure HTTP, Vercel-safe."""
    cache_key = f"alt_{code}"
    cached_val = cached(cache_key, ttl=600)
    if cached_val:
        return cached_val

    try:
        symbol = code.replace('.SZ', '').replace('.SH', '').replace('.BJ', '')

        # Method 1: Use board_code from get_industry_data (primary)
        peers = _get_industry_peers_by_board(code)
        print(f"[find_alternatives] Method1 found {len(peers)} peers for {code}")

        # Method 2: Fallback - search by industry keyword
        if not peers:
            ind_data = get_industry_data(code)
            board_name = ind_data.get('board_name', '')
            ind_name = ind_data.get('industry_name', '')
            search_kw = board_name or ind_name
            if search_kw:
                clean_kw = search_kw.replace('Ⅰ', '').replace('Ⅱ', '').replace('Ⅲ', '').strip()[:4]
                if clean_kw and len(clean_kw) >= 2:
                    peers = _tencent_search_peers(symbol, clean_kw)
                    print(f"[find_alternatives] Method2 found {len(peers)} peers for {code}")

        if not peers:
            print(f"[find_alternatives] No peers found for {code}")
            cache_set(cache_key, [])
            return []

        # Batch get quotes from Tencent API
        peer_codes = [p['wc'] for p in peers[:30]]
        quotes = _batch_get_quotes(peer_codes)

        alternatives = []
        for p in peers[:30]:
            wc = p['wc']
            q = quotes.get(wc, {})
            pe = q.get('pe', 0) or 0
            if pe <= 0 or pe > 200:
                continue
            pb = q.get('pb', 0) or 0
            price = q.get('price', 0) or 0
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

        # Sort by PE (low = value)
        alternatives.sort(key=lambda x: x['pe'])
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

    return jsonify(report)


@app.route("/api/alternatives", methods=["POST"])
def alternatives():
    data = request.json
    code = data.get("code", "").strip()
    if not code:
        return jsonify({"error": "请输入股票代码"}), 400

    alts = find_alternatives(code)
    return jsonify({"alternatives": alts})


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
    index_data = _get_market_indices()

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


def _get_market_indices():
    """Fetch major market index data via Tencent real-time API."""
    cache_key = "market_indices"
    cached_val = cached(cache_key, ttl=120)  # 2-minute cache
    if cached_val:
        return cached_val

    indices = {
        "shanghai": ("sh000001", "上证指数"),
        "shenzhen": ("sz399001", "深证成指"),
        "chinext": ("sz399006", "创业板指"),
    }

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

    cache_set(cache_key, result)
    return result


@app.route("/api/deep_cache_clear", methods=["POST"])
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


# ---------- Health Check ----------

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "data_source": "westock-data + EastMoney",
        "time": datetime.now().isoformat(),
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
