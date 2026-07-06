#!/usr/bin/env python3
"""
API Fallback Module for stock-analyzer
=======================================
Provides alternative implementations for all major external APIs.
Each function mirrors the return structure of the primary API caller.

Fallback sources:
  - Sina Finance (hq.sinajs.cn)        → real-time quotes, market indices
  - EastMoney K-line API               → historical price data
  - Static industry mapping             → industry peers
  - Graceful degradation               → money flow, financials, news

All functions are designed to be called as drop-in fallbacks:
  try: result = primary_api(...)
  except / empty: result = api_fallback.xxx(...)

A global FALLBACK_LOG tracks all fallback events for the /api/fallback_status endpoint.
"""

import requests
import json
from datetime import datetime

# ==============================
# Global fallback log
# ==============================
FALLBACK_LOG = []   # list of {time, func, source, ok, detail}
MAX_LOG = 300

def _log(func_name, source, success, detail=""):
    """Record a fallback event."""
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "func": func_name,
        "source": source,
        "ok": success,
        "detail": detail,
    }
    FALLBACK_LOG.append(entry)
    if len(FALLBACK_LOG) > MAX_LOG:
        FALLBACK_LOG[:] = FALLBACK_LOG[-MAX_LOG:]
    status = "OK" if success else "FAIL"
    print(f"[FALLBACK] {func_name} -> {source}: {status} | {detail}")

def get_fallback_log(since=None):
    """Return recent fallback events, optionally since a given index."""
    if since is not None and since < len(FALLBACK_LOG):
        return FALLBACK_LOG[since:]
    return FALLBACK_LOG[-50:]

def reset_log():
    """Clear the fallback log (for testing)."""
    FALLBACK_LOG.clear()


# ==============================
# Sina Finance helpers
# ==============================

SINA_QT_URL = "http://hq.sinajs.cn/list="
SINA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://finance.sina.com.cn/",
}

# Sina field mapping: https://blog.csdn.net/qq_26948675/article/details/79746487
# 0:name, 1:今开, 2:昨收, 3:当前价, 4:最高, 5:最低,
# 6:竞买价, 7:竞卖价, 8:成交量(手), 9:成交额(万),
# 10:买1量, 11:买1价, ... 20:卖1量, 21:卖1价,
# 22:日期, 23:时间, 24:--, ...
# 30:--, 31:--, 32:--, 33:--, ...

def _parse_sina_quote(text, scode):
    """Parse a single Sina quote line into a dict matching get_stock_info() structure."""
    for line in text.split("\n"):
        if scode not in line or "=" not in line:
            continue
        data_str = line.split("=", 1)[1].strip().strip('"').strip("'")
        if not data_str or data_str == "":
            continue
        fields = data_str.split(",")
        if len(fields) < 33:
            continue
        try:
            price = float(fields[3]) if fields[3] else 0
            prev_close = float(fields[2]) if fields[2] else 0
            change_pct = 0.0
            if prev_close > 0:
                change_pct = round((price - prev_close) / prev_close * 100, 2)

            return {
                "股票简称": fields[0] if fields[0] else "",
                "最新价": price,
                "昨收": prev_close,
                "今开": float(fields[1]) if fields[1] else 0,
                "最高价": float(fields[4]) if fields[4] else 0,
                "最低价": float(fields[5]) if fields[5] else 0,
                "涨跌幅": change_pct,
                "换手率": 0,           # Sina basic quote doesn't include turnover
                "市盈率-动态": 0,      # Not in basic quote
                "市净率": 0,           # Not in basic quote
                "总市值": 0,           # Not in basic quote
                "流通市值": 0,         # Not in basic quote
                "成交量": int(float(fields[8]) * 100) if len(fields) > 8 and fields[8] else 0,
                "成交额": float(fields[9]) * 10000 if len(fields) > 9 and fields[9] else 0,
            }
        except (ValueError, IndexError):
            continue
    return {}


# ==============================
# 1. Sina fallback for get_stock_info
# ==============================

def sina_get_stock_info(code):
    """
    Get stock info from Sina Finance.
    code: '600519.SH', '000651.SZ', etc.

    Returns: dict matching get_stock_info() structure, or {} on failure.
    """
    symbol = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "")
    if ".SZ" in code or (not code.startswith(("6", "9")) and not "." in code and symbol.startswith(("0", "3"))):
        scode = f"sz{symbol}"
    else:
        scode = f"sh{symbol}"

    try:
        url = f"{SINA_QT_URL}{scode}"
        resp = requests.get(url, headers=SINA_HEADERS, timeout=10)
        if resp.status_code != 200:
            _log("sina_get_stock_info", "sina", False, f"HTTP {resp.status_code}")
            return {}

        result = _parse_sina_quote(resp.text, scode)
        if result:
            _log("sina_get_stock_info", "sina", True, f"{code} -> {result.get('股票简称', '?')}")
        else:
            _log("sina_get_stock_info", "sina", False, "Empty parse result")
        return result
    except Exception as e:
        _log("sina_get_stock_info", "sina", False, str(e)[:80])
        return {}


# ==============================
# 2. Sina fallback for _batch_get_quotes
# ==============================

def sina_batch_get_quotes(wc_codes):
    """
    Batch get quotes from Sina Finance.
    wc_codes: ['sh600519', 'sz000651', ...]

    Returns: dict of wc -> {price, pe, pb, change_pct, mcap, name}
             (PE/PB/mcap are 0 since Sina basic quote doesn't include them)
    """
    result = {}
    if not wc_codes:
        return result

    batch_size = 80  # Sina supports up to ~100 codes per request
    for i in range(0, len(wc_codes), batch_size):
        batch = wc_codes[i : i + batch_size]
        try:
            url = f"{SINA_QT_URL}{','.join(batch)}"
            resp = requests.get(url, headers=SINA_HEADERS, timeout=15)
            if resp.status_code != 200:
                _log("sina_batch_get_quotes", "sina", False, f"HTTP {resp.status_code} (batch {i // batch_size})")
                continue

            text = resp.text
            parsed_count = 0
            for wc in batch:
                quote = _parse_sina_quote(text, wc)
                if quote:
                    result[wc] = {
                        "price": quote["最新价"],
                        "pe": 0,          # Not available in basic Sina quote
                        "pb": 0,
                        "change_pct": quote["涨跌幅"],
                        "mcap": 0,
                        "name": quote["股票简称"],
                    }
                    parsed_count += 1

            _log("sina_batch_get_quotes", "sina", True,
                 f"Batch {i // batch_size}: {parsed_count}/{len(batch)} parsed")
        except Exception as e:
            _log("sina_batch_get_quotes", "sina", False, str(e)[:80])

    return result


# ==============================
# 3. Sina fallback for _get_market_indices
# ==============================

SINA_INDEX_MAP = {
    "shanghai": ("s_sh000001", "sh000001"),  # short + long format
    "shenzhen": ("s_sz399001", "sz399001"),
    "chinext": ("s_sz399006", "sz399006"),
}

_INDEX_NAMES = {
    "shanghai": "上证指数",
    "shenzhen": "深证成指",
    "chinext": "创业板指",
}

def sina_get_market_indices():
    """
    Get market indices from Sina Finance.
    Sina has 2 formats for indices:
    - Short (s_sh000001): 7 fields (name,price,change,change_pct,volume,amount)
    - Long (sh000001): 33+ fields (like regular stocks)
    We try short first, fall back to long format.

    Returns: dict matching _get_market_indices() structure.
    """
    result = {}
    # Collect all candidate codes
    all_codes = []
    code_pairs = {}
    for key, (short_code, long_code) in SINA_INDEX_MAP.items():
        all_codes.extend([short_code, long_code])
        code_pairs[key] = (short_code, long_code)

    try:
        url = f"{SINA_QT_URL}{','.join(all_codes)}"
        resp = requests.get(url, headers=SINA_HEADERS, timeout=10)
        if resp.status_code != 200:
            _log("sina_get_market_indices", "sina", False, f"HTTP {resp.status_code}")
            return result

        text = resp.text
        parsed = 0

        for key, (short_code, long_code) in code_pairs.items():
            price = 0
            change_pct = 0
            name = _INDEX_NAMES.get(key, "")
            volume = 0
            amount = 0
            high = 0
            low = 0

            # Try short format first (s_ prefix)
            short_data = None
            for line in text.split("\n"):
                if short_code in line and "=" in line:
                    data_str = line.split("=", 1)[1].strip().strip('"').strip("'")
                    fields = data_str.split(",")
                    if len(fields) >= 6:
                        short_data = fields
                    break  # found our line, stop searching

            if short_data:
                try:
                    name = short_data[0] if short_data[0] else name
                    price = float(short_data[1]) if short_data[1] else 0
                    change_raw = float(short_data[2]) if short_data[2] else 0
                    change_pct_raw = float(short_data[3]) if short_data[3] else 0
                    # Use change_pct directly when available, otherwise calculate
                    if change_pct_raw:
                        change_pct = change_pct_raw
                    elif price > 0:
                        prev = price - change_raw
                        change_pct = round(change_raw / prev * 100, 2) if prev else 0
                    volume = float(short_data[4]) if len(short_data) > 4 and short_data[4] else 0
                    amount = float(short_data[5]) if len(short_data) > 5 and short_data[5] else 0
                    parsed += 1
                except (ValueError, IndexError):
                    pass

            # Also try long format for high/low (only available in long format)
            for line in text.split("\n"):
                if long_code in line and "=" in line:
                    data_str = line.split("=", 1)[1].strip().strip('"').strip("'")
                    fields = data_str.split(",")
                    if len(fields) >= 6:
                        try:
                            if not name or name == _INDEX_NAMES.get(key, ""):
                                name = fields[0] if fields[0] else name
                            high = float(fields[4]) if len(fields) > 4 and fields[4] else high
                            low = float(fields[5]) if len(fields) > 5 and fields[5] else low
                        except (ValueError, IndexError):
                            pass

            if price > 0:
                result[key] = {
                    "name": name,
                    "price": price,
                    "change_pct": round(change_pct, 2),
                    "pe": 0,
                    "volume": volume,
                    "amount": amount,
                    "high": high,
                    "low": low,
                }

        _log("sina_get_market_indices", "sina", len(result) > 0,
             f"Parsed {len(result)}/{len(code_pairs)} indices")
    except Exception as e:
        _log("sina_get_market_indices", "sina", False, str(e)[:80])
    return result


# ==============================
# 4. EastMoney K-line fallback for get_price_history
# ==============================

EM_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

def em_get_price_history(code, days=250):
    """
    Get K-line data from EastMoney API.
    Returns: list of {日期, 开盘, 收盘, 最高, 最低, 成交量},
             same structure as get_price_history().
    """
    try:
        symbol = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "")
        if code.endswith(".SZ"):
            secid = f"0.{symbol}"
        else:
            secid = f"1.{symbol}"

        params = {
            "secid": secid,
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": "101",        # daily K-line
            "fqt": "1",          # 前复权 (forward-adjusted)
            "end": "20500101",
            "lmt": days + 30,
        }
        resp = requests.get(
            EM_KLINE_URL,
            params=params,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://quote.eastmoney.com/",
            },
            timeout=15,
        )
        data = resp.json()
        klines = data.get("data", {}).get("klines", [])
        if not klines:
            _log("em_get_price_history", "eastmoney_kline", False, "No klines in response")
            return []

        result = []
        for line in klines[-days:]:
            parts = line.split(",")
            if len(parts) >= 6:
                result.append({
                    "日期": parts[0],
                    "开盘": float(parts[1]),
                    "收盘": float(parts[2]),
                    "最高": float(parts[3]),
                    "最低": float(parts[4]),
                    "成交量": int(float(parts[5])),
                })

        _log("em_get_price_history", "eastmoney_kline", True,
             f"Got {len(result)} K-lines for {code}")
        return result
    except Exception as e:
        _log("em_get_price_history", "eastmoney_kline", False, str(e)[:80])
        return []


# ==============================
# 5. Static industry mapping fallback for peer search
# ==============================

# Comprehensive static mapping: industry category -> list of (code, name, wc)
# Organized by recognizable Chinese industry categories.
STATIC_PEERS = {
    "白酒": [
        ("600519", "贵州茅台", "sh600519"), ("000858", "五粮液", "sz000858"),
        ("000568", "泸州老窖", "sz000568"), ("002304", "洋河股份", "sz002304"),
        ("000596", "古井贡酒", "sz000596"), ("600809", "山西汾酒", "sh600809"),
        ("000799", "酒鬼酒", "sz000799"), ("600702", "舍得酒业", "sh600702"),
        ("603369", "今世缘", "sh603369"), ("600559", "老白干酒", "sh600559"),
    ],
    "家电": [
        ("000651", "格力电器", "sz000651"), ("000333", "美的集团", "sz000333"),
        ("600690", "海尔智家", "sh600690"), ("000921", "海信家电", "sz000921"),
        ("002508", "老板电器", "sz002508"), ("002032", "苏泊尔", "sz002032"),
        ("603486", "科沃斯", "sh603486"), ("000100", "TCL科技", "sz000100"),
        ("600060", "海信视像", "sh600060"), ("002677", "浙江美大", "sz002677"),
    ],
    "银行": [
        ("600036", "招商银行", "sh600036"), ("601398", "工商银行", "sh601398"),
        ("601288", "农业银行", "sh601288"), ("601939", "建设银行", "sh601939"),
        ("601988", "中国银行", "sh601988"), ("600016", "民生银行", "sh600016"),
        ("600000", "浦发银行", "sh600000"), ("601166", "兴业银行", "sh601166"),
        ("000001", "平安银行", "sz000001"), ("002142", "宁波银行", "sz002142"),
    ],
    "证券": [
        ("600030", "中信证券", "sh600030"), ("601211", "国泰君安", "sh601211"),
        ("600837", "海通证券", "sh600837"), ("601688", "华泰证券", "sh601688"),
        ("000776", "广发证券", "sz000776"), ("600999", "招商证券", "sh600999"),
        ("601066", "中信建投", "sh601066"), ("600958", "东方证券", "sh600958"),
        ("300059", "东方财富", "sz300059"), ("002736", "国信证券", "sz002736"),
    ],
    "保险": [
        ("601318", "中国平安", "sh601318"), ("601628", "中国人寿", "sh601628"),
        ("601601", "中国太保", "sh601601"), ("601336", "新华保险", "sh601336"),
        ("601319", "中国人保", "sh601319"),
    ],
    "医药": [
        ("600276", "恒瑞医药", "sh600276"), ("000538", "云南白药", "sz000538"),
        ("300760", "迈瑞医疗", "sz300760"), ("000963", "华东医药", "sz000963"),
        ("600085", "同仁堂", "sh600085"), ("002001", "新和成", "sz002001"),
        ("300015", "爱尔眼科", "sz300015"), ("000661", "长春高新", "sz000661"),
        ("002007", "华兰生物", "sz002007"), ("600196", "复星医药", "sh600196"),
    ],
    "电子": [
        ("002475", "立讯精密", "sz002475"), ("000725", "京东方A", "sz000725"),
        ("002415", "海康威视", "sz002415"), ("603501", "韦尔股份", "sh603501"),
        ("300433", "蓝思科技", "sz300433"), ("002241", "歌尔股份", "sz002241"),
        ("600703", "三安光电", "sh600703"), ("300408", "三环集团", "sz300408"),
    ],
    "半导体": [
        ("688981", "中芯国际", "sh688981"), ("002049", "紫光国微", "sz002049"),
        ("603986", "兆易创新", "sh603986"), ("300782", "卓胜微", "sz300782"),
        ("688012", "中微公司", "sh688012"), ("002371", "北方华创", "sz002371"),
        ("688008", "澜起科技", "sh688008"), ("603160", "汇顶科技", "sh603160"),
    ],
    "汽车": [
        ("600104", "上汽集团", "sh600104"), ("000625", "长安汽车", "sz000625"),
        ("002594", "比亚迪", "sz002594"), ("601238", "广汽集团", "sh601238"),
        ("600741", "华域汽车", "sh600741"), ("601633", "长城汽车", "sh601633"),
        ("000800", "一汽解放", "sz000800"), ("600733", "北汽蓝谷", "sh600733"),
    ],
    "新能源": [
        ("300750", "宁德时代", "sz300750"), ("601012", "隆基绿能", "sh601012"),
        ("002129", "TCL中环", "sz002129"), ("300274", "阳光电源", "sz300274"),
        ("600438", "通威股份", "sh600438"), ("002459", "晶澳科技", "sz002459"),
        ("688599", "天合光能", "sh688599"), ("300763", "锦浪科技", "sz300763"),
    ],
    "电力": [
        ("600900", "长江电力", "sh600900"), ("600011", "华能国际", "sh600011"),
        ("600023", "浙能电力", "sh600023"), ("601985", "中国核电", "sh601985"),
        ("600886", "国投电力", "sh600886"), ("600025", "华能水电", "sh600025"),
        ("003816", "中国广核", "sz003816"), ("600795", "国电电力", "sh600795"),
    ],
    "食品": [
        ("600887", "伊利股份", "sh600887"), ("002714", "牧原股份", "sz002714"),
        ("000895", "双汇发展", "sz000895"), ("603288", "海天味业", "sh603288"),
        ("603345", "安井食品", "sh603345"), ("002557", "洽洽食品", "sz002557"),
        ("000876", "新希望", "sz000876"), ("600882", "妙可蓝多", "sh600882"),
    ],
    "房地产": [
        ("000002", "万科A", "sz000002"), ("600048", "保利发展", "sh600048"),
        ("001979", "招商蛇口", "sz001979"), ("600383", "金地集团", "sh600383"),
        ("600325", "华发股份", "sh600325"), ("600340", "华夏幸福", "sh600340"),
    ],
    "通信": [
        ("600050", "中国联通", "sh600050"), ("601728", "中国电信", "sh601728"),
        ("600941", "中国移动", "sh600941"), ("000063", "中兴通讯", "sz000063"),
        ("300628", "亿联网络", "sz300628"), ("002396", "星网锐捷", "sz002396"),
    ],
    "计算机": [
        ("002230", "科大讯飞", "sz002230"), ("600570", "恒生电子", "sh600570"),
        ("300033", "同花顺", "sz300033"), ("002410", "广联达", "sz002410"),
        ("300454", "深信服", "sz300454"), ("688111", "金山办公", "sh688111"),
    ],
    "化工": [
        ("600309", "万华化学", "sh600309"), ("002601", "龙佰集团", "sz002601"),
        ("600346", "恒力石化", "sh600346"), ("000301", "东方盛虹", "sz000301"),
        ("600426", "华鲁恒升", "sh600426"), ("600989", "宝丰能源", "sh600989"),
    ],
    "钢铁": [
        ("600019", "宝钢股份", "sh600019"), ("000898", "鞍钢股份", "sz000898"),
        ("000932", "华菱钢铁", "sz000932"), ("600010", "包钢股份", "sh600010"),
        ("000709", "河钢股份", "sz000709"), ("600022", "山东钢铁", "sh600022"),
    ],
    "有色金属": [
        ("601899", "紫金矿业", "sh601899"), ("600547", "山东黄金", "sh600547"),
        ("603799", "华友钴业", "sh603799"), ("002460", "赣锋锂业", "sz002460"),
        ("600111", "北方稀土", "sh600111"), ("000630", "铜陵有色", "sz000630"),
    ],
    "煤炭": [
        ("601088", "中国神华", "sh601088"), ("600188", "兖矿能源", "sh600188"),
        ("601225", "陕西煤业", "sh601225"), ("000983", "山西焦煤", "sz000983"),
        ("601898", "中煤能源", "sh601898"), ("600348", "华阳股份", "sh600348"),
    ],
    "建筑": [
        ("601668", "中国建筑", "sh601668"), ("601390", "中国中铁", "sh601390"),
        ("601800", "中国交建", "sh601800"), ("601186", "中国铁建", "sh601186"),
        ("600170", "上海建工", "sh600170"), ("600039", "四川路桥", "sh600039"),
    ],
    "交通运输": [
        ("601111", "中国国航", "sh601111"), ("600029", "南方航空", "sh600029"),
        ("600115", "中国东航", "sh600115"), ("601816", "京沪高铁", "sh601816"),
        ("600009", "上海机场", "sh600009"), ("601006", "大秦铁路", "sh601006"),
    ],
    "传媒": [
        ("300413", "芒果超媒", "sz300413"), ("002624", "完美世界", "sz002624"),
        ("300251", "光线传媒", "sz300251"), ("002739", "万达电影", "sz002739"),
        ("000156", "华数传媒", "sz000156"), ("600637", "东方明珠", "sh600637"),
    ],
    "军工": [
        ("600893", "航发动力", "sh600893"), ("600760", "中航沈飞", "sh600760"),
        ("600862", "中航高科", "sh600862"), ("002013", "中航机电", "sz002013"),
        ("600685", "中船防务", "sh600685"), ("000768", "中航西飞", "sz000768"),
    ],
    "石油": [
        ("601857", "中国石油", "sh601857"), ("600028", "中国石化", "sh600028"),
        ("600938", "中国海油", "sh600938"), ("002207", "准油股份", "sz002207"),
    ],
    "环保": [
        ("300070", "碧水源", "sz300070"), ("000826", "启迪环境", "sz000826"),
        ("603588", "高能环境", "sh603588"), ("300187", "永清环保", "sz300187"),
        ("300422", "博世科", "sz300422"), ("300815", "玉禾田", "sz300815"),
    ],
}

# Keyword-to-category fuzzy matching
_PARTIAL_MAP = {
    "白酒": "白酒", "家电": "家电", "家用电器": "家电", "银行": "银行",
    "证券": "证券", "券商": "证券", "保险": "保险",
    "医药": "医药", "医疗": "医药", "制药": "医药",
    "电子": "电子", "半导体": "半导体", "芯片": "半导体",
    "汽车": "汽车", "新能源": "新能源", "电池": "新能源",
    "光伏": "新能源", "锂电": "新能源", "风能": "新能源",
    "电力": "电力", "食品": "食品", "饮料": "食品",
    "乳品": "食品", "农业": "食品", "农林牧渔": "食品",
    "房地产": "房地产", "地产": "房地产",
    "通信": "通信", "计算机": "计算机", "软件": "计算机",
    "化工": "化工", "钢铁": "钢铁", "有色": "有色金属",
    "黄金": "有色金属", "稀土": "有色金属", "贵金属": "有色金属",
    "煤炭": "煤炭", "建筑": "建筑", "建材": "建筑",
    "交通运输": "交通运输", "航空": "交通运输", "铁路": "交通运输",
    "机场": "交通运输", "航运": "交通运输",
    "传媒": "传媒", "游戏": "传媒", "影视": "传媒",
    "军工": "军工", "航空航天": "军工", "船舶": "军工",
    "石油": "石油", "石化": "石油", "环保": "环保",
}


def _match_industry_category(industry_name, board_name):
    """Find best matching industry category from keyword fuzzy match."""
    search_text = (industry_name or "") + " " + (board_name or "")

    # Direct match first
    for cat_name in STATIC_PEERS:
        if cat_name in search_text:
            return cat_name

    # Fuzzy match
    for kw, cat in _PARTIAL_MAP.items():
        if kw in search_text:
            return cat

    return None


def static_get_industry_peers(code, industry_name, board_name):
    """
    Get industry peers from static mapping.
    Used when push2.eastmoney.com and smartbox.gtimg.cn both fail.

    Args:
        code: stock code like '000651.SZ'
        industry_name: from get_industry_data like '电气机械和器材制造业'
        board_name: from get_industry_data like '家电行业'

    Returns: list of {code, name, wc}
    """
    cat = _match_industry_category(industry_name, board_name)

    # Fallback: if API data is empty, look up by code directly
    if not cat:
        cat = static_get_industry_for_code(code)

    if not cat:
        _log("static_get_industry_peers", "static", False,
             f"No category match for '{industry_name}' / '{board_name}' / code={code}")
        return []

    peers_list = STATIC_PEERS.get(cat, [])
    symbol = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "")

    result = []
    for c, n, wc in peers_list:
        if c == symbol:
            continue
        result.append({"code": c, "name": n, "wc": wc})

    _log("static_get_industry_peers", "static", True,
         f"Category='{cat}', {len(result)} peers for {code}")
    return result


def static_get_industry_for_code(code):
    """
    Reverse lookup: find which industry category a stock code belongs to.
    Returns category name string or None.
    """
    symbol = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "")
    for cat, stocks in STATIC_PEERS.items():
        for c, n, wc in stocks:
            if c == symbol:
                return cat
    return None


# ==============================
# 6. Graceful degradation for money flow
# ==============================

def graceful_money_flow(code):
    """
    Return a degraded money flow result when EastMoney API fails.
    This prevents the entire report from failing just because money flow is unavailable.
    """
    _log("graceful_money_flow", "degraded", True, f"No money flow data for {code}")
    return {
        "error": "资金流向数据暂不可用（API 降级）",
        "records": [],
        "last5": [],
        "latest": None,
        "summary": {
            "main_5d_net": 0,
            "super_large_5d_net": 0,
            "large_5d_net": 0,
            "retail_5d_net": 0,
            "main_inflow_days": 0,
            "main_outflow_days": 0,
            "trend": "数据不可用",
        },
    }


# ==============================
# 7. Graceful degradation for financial data
# ==============================

def graceful_financial_data(code):
    """
    Return empty financial data structure when EastMoney datacenter fails.
    """
    _log("graceful_financial_data", "degraded", True, f"No financial data for {code}")
    return {"income": [], "balance": [], "quarterly": []}


# ==============================
# 8. Graceful degradation for news events
# ==============================

def graceful_news_events(code):
    """
    Return empty news list when EastMoney announcement API fails.
    """
    _log("graceful_news_events", "degraded", True, f"No news data for {code}")
    return []


# ==============================
# 9. Graceful degradation for industry data
# ==============================

def graceful_industry_data(code):
    """
    Return minimal industry data when EastMoney datacenter fails.
    """
    _log("graceful_industry_data", "degraded", True, f"No industry data for {code}")
    return {
        "industry_name": "",
        "board_name": "",
        "board_code": "",
        "dividends": [],
        "latest_dividend": None,
    }
