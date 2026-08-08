from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Tuple
from urllib.parse import quote
from urllib.request import Request, urlopen
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="한미 성장주 퀀트", layout="wide")

DEFAULT_TICKERS = [
    "NVDA", "MSFT", "META", "AMZN", "GOOGL", "AVGO", "PLTR",
    "AMD", "TSLA", "CRWD", "ORCL", "RKLB", "JOBY", "UEC", "RCAT",
]

MY_HOLDINGS = ["JOBY", "ONDS", "RKLB", "UEC", "IONQ", "RCAT"]

KR_NAME_MAP = {
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "에스케이하이닉스": "000660.KS",
    "한화에어로스페이스": "012450.KS",
    "LIG넥스원": "079550.KS",
    "엘아이지넥스원": "079550.KS",
    "LG에너지솔루션": "373220.KS",
    "엘지에너지솔루션": "373220.KS",
    "레인보우로보틱스": "277810.KQ",
    "한화시스템": "272210.KS",
    "한화엔진": "082740.KS",
    "남광토건": "001260.KS",
}

POSITIVE_WORDS = [
    "beat", "beats", "surge", "surges", "record", "growth", "profit", "profits",
    "approval", "approved", "contract", "award", "wins", "partnership", "upgrade",
    "outperform", "raises", "raised", "strong", "expansion", "breakthrough",
    "호실적", "수주", "계약", "승인", "흑자", "성장", "상향", "신고가", "급증",
]
NEGATIVE_WORDS = [
    "miss", "misses", "plunge", "plunges", "loss", "losses", "downgrade",
    "underperform", "cuts", "cut", "lawsuit", "investigation", "offering",
    "dilution", "delay", "delays", "weak", "warning", "recall",
    "적자", "하향", "유상증자", "희석", "소송", "조사", "리콜", "지연", "급락",
]

FORECAST_HORIZONS = {
    "1개월": 21,
    "3개월": 63,
    "6개월": 126,
    "1년": 252,
    "3년": 756,
    "5년": 1260,
}


@dataclass
class StrategyConfig:
    ma_short: int = 20
    ma_long: int = 120
    rsi_period: int = 14
    momentum_days: int = 63
    rebalance_days: int = 21
    max_positions: int = 5
    fee_rate: float = 0.001
    buy_score_min: float = 50.0


# =========================
# 종목 입력 / 검색
# =========================
def normalize_ticker(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    if raw in KR_NAME_MAP:
        return KR_NAME_MAP[raw]
    upper = raw.upper()
    if upper in KR_NAME_MAP:
        return KR_NAME_MAP[upper]
    if raw.isdigit() and len(raw) == 6:
        return f"{raw}.KS"
    return upper


def parse_tickers(raw: str) -> List[str]:
    values = [normalize_ticker(x) for x in raw.replace("\n", ",").split(",")]
    return list(dict.fromkeys([x for x in values if x]))


@st.cache_data(ttl=3600, show_spinner=False)
def search_us_stocks(query: str) -> List[dict]:
    """
    Yahoo Finance 검색 결과에서 미국 상장 주식 위주로 반환.
    나스닥(NMS/NCM/NGM)을 우선 정렬합니다.
    """
    q = query.strip()
    if not q:
        return []

    url = (
        "https://query1.finance.yahoo.com/v1/finance/search"
        f"?q={quote(q)}&quotesCount=20&newsCount=0"
    )
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urlopen(req, timeout=6) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []

    rows = []
    for item in payload.get("quotes", []):
        symbol = str(item.get("symbol", "")).upper()
        quote_type = str(item.get("quoteType", "")).upper()
        exchange = str(item.get("exchange", "")).upper()
        name = item.get("shortname") or item.get("longname") or symbol

        if not symbol or quote_type not in {"EQUITY", "ETF"}:
            continue

        is_nasdaq = exchange in {"NMS", "NCM", "NGM", "NASDAQ"}
        is_us = exchange in {
            "NMS", "NCM", "NGM", "NASDAQ", "NYQ", "ASE", "PCX", "BTS",
        }
        if not is_us:
            continue

        rows.append({
            "symbol": symbol,
            "name": str(name),
            "exchange": exchange,
            "nasdaq": is_nasdaq,
        })

    rows.sort(key=lambda x: (not x["nasdaq"], x["symbol"]))
    return rows[:10]


# =========================
# 데이터 다운로드
# =========================
@st.cache_data(ttl=300, show_spinner=False)
def download_market_data(
    tickers: Tuple[str, ...],
    start: str,
    end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    end_inclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    data = yf.download(
        list(tickers),
        start=start,
        end=end_inclusive,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    empty = pd.DataFrame()
    if data.empty:
        return empty, empty, empty, empty

    def extract(field: str) -> pd.DataFrame:
        if isinstance(data.columns, pd.MultiIndex):
            if field not in data.columns.get_level_values(0):
                return pd.DataFrame()
            out = data[field].copy()
        else:
            if field not in data.columns:
                return pd.DataFrame()
            out = data[[field]].copy()
            out.columns = [tickers[0]]

        if isinstance(out, pd.Series):
            out = out.to_frame(name=tickers[0])
        return out

    close = extract("Close")
    high = extract("High")
    low = extract("Low")
    volume = extract("Volume")

    if close.empty:
        return empty, empty, empty, empty

    close = close.sort_index().dropna(how="all")
    high = high.reindex(index=close.index, columns=close.columns)
    low = low.reindex(index=close.index, columns=close.columns)
    volume = volume.reindex(index=close.index, columns=close.columns)

    return close, high, low, volume


@st.cache_data(ttl=60, show_spinner=False)
def fetch_latest_quotes(tickers: Tuple[str, ...]) -> Dict[str, float]:
    quotes: Dict[str, float] = {}

    for ticker in tickers:
        try:
            obj = yf.Ticker(ticker)
            price = np.nan

            try:
                info = obj.fast_info
                price = getattr(info, "last_price", np.nan)
                if pd.isna(price):
                    price = info.get("last_price", np.nan)
            except Exception:
                pass

            if pd.isna(price) or float(price) <= 0:
                hist = obj.history(period="5d", auto_adjust=True)
                if not hist.empty:
                    price = float(hist["Close"].dropna().iloc[-1])

            if pd.notna(price) and float(price) > 0:
                quotes[ticker] = float(price)

        except Exception:
            pass

    return quotes


def _news_title(item: dict) -> str:
    if isinstance(item.get("title"), str):
        return item["title"]

    content = item.get("content", {})
    if isinstance(content, dict):
        title = content.get("title")
        if isinstance(title, str):
            return title

    return ""


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news_sentiment(tickers: Tuple[str, ...]) -> Dict[str, dict]:
    result: Dict[str, dict] = {}

    for ticker in tickers:
        titles: List[str] = []

        try:
            for item in (yf.Ticker(ticker).news or [])[:8]:
                title = _news_title(item)
                if title:
                    titles.append(title)
        except Exception:
            pass

        if not titles:
            result[ticker] = {
                "adjustment": 0.0,
                "label": "뉴스 없음",
                "summary": "최근 뉴스 제목을 불러오지 못함",
            }
            continue

        raw_score = 0
        for title in titles:
            lower = title.lower()
            raw_score += sum(1 for w in POSITIVE_WORDS if w.lower() in lower)
            raw_score -= sum(1 for w in NEGATIVE_WORDS if w.lower() in lower)

        adjustment = float(np.clip(raw_score * 2.0, -10.0, 10.0))

        if adjustment >= 4:
            label = "긍정"
        elif adjustment <= -4:
            label = "부정"
        else:
            label = "중립"

        summary = " | ".join(titles[:2])
        if len(summary) > 180:
            summary = summary[:177] + "..."

        result[ticker] = {
            "adjustment": adjustment,
            "label": label,
            "summary": summary,
        }

    return result


# =========================
# 지표 / 점수
# =========================
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def percentile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(0.0, index=series.index)

    ranked = valid.rank(pct=True, method="average")
    if not higher_is_better:
        ranked = 1 - ranked + 1 / len(valid)

    out = pd.Series(0.0, index=series.index)
    out.loc[ranked.index] = ranked.clip(0, 1)
    return out


def build_indicators(
    prices: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    cfg: StrategyConfig,
) -> Dict[str, pd.DataFrame]:
    ma_short = prices.rolling(cfg.ma_short).mean()
    ma_long = prices.rolling(cfg.ma_long).mean()
    momentum = prices.pct_change(cfg.momentum_days)
    volatility = prices.pct_change().rolling(63).std() * np.sqrt(252)

    high_52w = prices.rolling(252, min_periods=120).max()
    high_proximity = prices / high_52w

    avg_volume20 = volume.rolling(20).mean()
    volume_ratio = volume / avg_volume20.replace(0, np.nan)

    rsi = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for col in prices.columns:
        rsi[col] = compute_rsi(prices[col], cfg.rsi_period)

    prev_close = prices.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.DataFrame(
        np.maximum.reduce([tr1.values, tr2.values, tr3.values]),
        index=prices.index,
        columns=prices.columns,
    )

    atr14 = true_range.rolling(14, min_periods=5).mean()
    support20 = low.rolling(20, min_periods=5).min()
    support60 = low.rolling(60, min_periods=20).min()

    return {
        "ma_short": ma_short,
        "ma_long": ma_long,
        "momentum": momentum,
        "volatility": volatility,
        "high_proximity": high_proximity,
        "volume_ratio": volume_ratio,
        "rsi": rsi,
        "atr14": atr14,
        "support20": support20,
        "support60": support60,
    }


def opinion_from_score(score: float) -> str:
    if score >= 85:
        return "적극 매수 후보"
    if score >= 75:
        return "매수 후보"
    if score >= 65:
        return "관심"
    if score >= 50:
        return "중립"
    return "관망"


def latest_screen(
    prices: pd.DataFrame,
    indicators: Dict[str, pd.DataFrame],
    cfg: StrategyConfig,
) -> pd.DataFrame:
    date = prices.index[-1]

    raw = pd.DataFrame(index=prices.columns)
    raw["종가"] = prices.loc[date]
    raw[f"{cfg.ma_short}일선"] = indicators["ma_short"].loc[date]
    raw[f"{cfg.ma_long}일선"] = indicators["ma_long"].loc[date]
    raw[f"{cfg.momentum_days}일 모멘텀"] = indicators["momentum"].loc[date]
    raw["RSI"] = indicators["rsi"].loc[date]
    raw["연환산 변동성"] = indicators["volatility"].loc[date]
    raw["거래량 배수"] = indicators["volume_ratio"].loc[date]
    raw["52주 고점 근접도"] = indicators["high_proximity"].loc[date]
    raw["ATR14"] = indicators["atr14"].loc[date]
    raw["20일 지지"] = indicators["support20"].loc[date]
    raw["60일 지지"] = indicators["support60"].loc[date]

    momentum_pct = percentile_score(raw[f"{cfg.momentum_days}일 모멘텀"], True)
    volatility_pct = percentile_score(raw["연환산 변동성"], False)
    volume_pct = percentile_score(raw["거래량 배수"], True)
    high_pct = percentile_score(raw["52주 고점 근접도"], True)

    rows = []

    for ticker, row in raw.iterrows():
        price = row["종가"]
        ma_s = row[f"{cfg.ma_short}일선"]
        ma_l = row[f"{cfg.ma_long}일선"]
        mom = row[f"{cfg.momentum_days}일 모멘텀"]
        rsi = row["RSI"]
        vol = row["연환산 변동성"]
        vol_ratio = row["거래량 배수"]
        high_near = row["52주 고점 근접도"]

        trend_score = 0.0
        if pd.notna(price) and pd.notna(ma_l) and price > ma_l:
            trend_score += 15.0
        if pd.notna(ma_s) and pd.notna(ma_l) and ma_s > ma_l:
            trend_score += 15.0

        momentum_score = float(momentum_pct.loc[ticker] * 25.0)

        if pd.isna(rsi):
            rsi_score = 0.0
        elif 55 <= rsi <= 65:
            rsi_score = 15.0
        elif 45 <= rsi < 55 or 65 < rsi <= 72:
            rsi_score = 12.0
        elif 35 <= rsi < 45 or 72 < rsi <= 78:
            rsi_score = 7.0
        else:
            rsi_score = 2.0

        volatility_score = float(volatility_pct.loc[ticker] * 10.0)
        volume_score = float(volume_pct.loc[ticker] * 10.0)
        high_score = float(high_pct.loc[ticker] * 10.0)

        technical_total = (
            trend_score
            + momentum_score
            + rsi_score
            + volatility_score
            + volume_score
            + high_score
        )

        reasons = []

        if trend_score == 30:
            reasons.append("추세 강함")
        elif trend_score == 15:
            reasons.append("추세 혼조")
        else:
            reasons.append("추세 약함")

        if momentum_score >= 20:
            reasons.append("모멘텀 상위")
        elif momentum_score >= 12:
            reasons.append("모멘텀 중간")
        else:
            reasons.append("모멘텀 하위")

        if rsi_score >= 12:
            reasons.append("RSI 양호")
        elif rsi_score >= 7:
            reasons.append("RSI 주의")
        else:
            reasons.append("RSI 과열·침체")

        if volatility_score >= 7:
            reasons.append("변동성 양호")
        elif volatility_score <= 3:
            reasons.append("변동성 높음")

        if volume_score >= 7:
            reasons.append("거래량 강함")

        if high_score >= 7:
            reasons.append("52주 고점 근접")

        rows.append({
            "종목": ticker,
            "기술 점수": round(technical_total, 1),
            "뉴스 보정": 0.0,
            "종합 점수": round(technical_total, 1),
            "의견": opinion_from_score(technical_total),
            "추세 점수": round(trend_score, 1),
            "모멘텀 점수": round(momentum_score, 1),
            "RSI 점수": round(rsi_score, 1),
            "변동성 점수": round(volatility_score, 1),
            "거래량 점수": round(volume_score, 1),
            "52주 고점 점수": round(high_score, 1),
            "종가": price,
            f"{cfg.ma_short}일선": ma_s,
            f"{cfg.ma_long}일선": ma_l,
            f"{cfg.momentum_days}일 모멘텀": mom,
            "RSI": rsi,
            "연환산 변동성": vol,
            "거래량 배수": vol_ratio,
            "52주 고점 근접도": high_near,
            "ATR14": row["ATR14"],
            "20일 지지": row["20일 지지"],
            "60일 지지": row["60일 지지"],
            "뉴스 판단": "미반영",
            "뉴스 요약": "",
            "판단 근거": " / ".join(reasons),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("종합 점수", ascending=False)
        .reset_index(drop=True)
    )


def apply_live_quotes(screen: pd.DataFrame, quotes: Dict[str, float]) -> pd.DataFrame:
    out = screen.copy()
    for idx, row in out.iterrows():
        ticker = row["종목"]
        if ticker in quotes:
            out.at[idx, "종가"] = quotes[ticker]
    return out


def apply_news_adjustment(screen: pd.DataFrame, news_data: Dict[str, dict]) -> pd.DataFrame:
    out = screen.copy()

    for idx, row in out.iterrows():
        ticker = row["종목"]
        item = news_data.get(ticker, {})
        adj = float(item.get("adjustment", 0.0))
        final = float(np.clip(row["기술 점수"] + adj, 0, 100))

        out.at[idx, "뉴스 보정"] = adj
        out.at[idx, "종합 점수"] = round(final, 1)
        out.at[idx, "의견"] = opinion_from_score(final)
        out.at[idx, "뉴스 판단"] = item.get("label", "중립")
        out.at[idx, "뉴스 요약"] = item.get("summary", "")

    return out.sort_values("종합 점수", ascending=False).reset_index(drop=True)


# =========================
# 1·2·3차 매수지점
# =========================
def calc_buy_points(
    current: float,
    ma_short: float,
    ma_long: float,
    atr: float,
    support20: float,
    support60: float,
) -> Tuple[float, float, float]:
    if pd.isna(current) or current <= 0:
        return np.nan, np.nan, np.nan

    if pd.isna(atr) or atr <= 0:
        atr = current * 0.03

    raw1 = current - max(0.75 * atr, current * 0.02)
    candidates1 = [raw1]
    if pd.notna(ma_short) and 0 < ma_short < current:
        candidates1.append(ma_short)
    buy1 = min(max(candidates1), current * 0.995)

    raw2 = current - max(1.50 * atr, current * 0.05)
    candidates2 = [raw2]
    if pd.notna(support20) and 0 < support20 < buy1:
        candidates2.append(support20)
    buy2 = min(max(candidates2), buy1 * 0.98)

    raw3 = current - max(2.50 * atr, current * 0.08)
    candidates3 = [raw3]
    if pd.notna(ma_long) and 0 < ma_long < buy2:
        candidates3.append(ma_long)
    if pd.notna(support60) and 0 < support60 < buy2:
        candidates3.append(support60)
    buy3 = min(max(candidates3), buy2 * 0.97)

    return float(buy1), float(buy2), float(buy3)


def add_buy_points(screen: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    out = screen.copy()

    b1s, b2s, b3s = [], [], []

    for _, row in out.iterrows():
        if row["종합 점수"] < cfg.buy_score_min:
            b1s.append(np.nan)
            b2s.append(np.nan)
            b3s.append(np.nan)
            continue

        b1, b2, b3 = calc_buy_points(
            row["종가"],
            row[f"{cfg.ma_short}일선"],
            row[f"{cfg.ma_long}일선"],
            row["ATR14"],
            row["20일 지지"],
            row["60일 지지"],
        )

        b1s.append(b1)
        b2s.append(b2)
        b3s.append(b3)

    out["1차 매수가"] = b1s
    out["2차 매수가"] = b2s
    out["3차 매수가"] = b3s

    for label in ["1차", "2차", "3차"]:
        out[f"{label} 하락률"] = out[f"{label} 매수가"] / out["종가"] - 1

    return out


# =========================
# 현재가 매수 판단
# =========================
def add_current_entry_judgement(screen: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    out = screen.copy()

    suitability, judgements = [], []
    current_allocs, b1_allocs, b2_allocs, b3_allocs = [], [], [], []
    reasons_out = []

    for _, row in out.iterrows():
        total = float(row.get("종합 점수", 0) or 0)
        current = row.get("종가", np.nan)
        ma_s = row.get(f"{cfg.ma_short}일선", np.nan)
        ma_l = row.get(f"{cfg.ma_long}일선", np.nan)
        rsi = row.get("RSI", np.nan)
        momentum = row.get(f"{cfg.momentum_days}일 모멘텀", np.nan)
        news_adj = float(row.get("뉴스 보정", 0) or 0)
        buy1 = row.get("1차 매수가", np.nan)

        score = total * 0.60
        reasons = []

        if pd.notna(current) and pd.notna(ma_l) and current > ma_l:
            score += 10
            reasons.append("장기 추세 위")
        else:
            reasons.append("장기 추세 약함")

        if pd.notna(ma_s) and pd.notna(ma_l) and ma_s > ma_l:
            score += 5
            reasons.append("단기 추세 우위")

        if pd.notna(rsi):
            if 45 <= rsi <= 65:
                score += 10
                reasons.append("RSI 적정")
            elif 35 <= rsi < 45 or 65 < rsi <= 72:
                score += 5
                reasons.append("RSI 허용")
            elif rsi > 78:
                score -= 10
                reasons.append("RSI 과열")
            elif rsi < 30:
                score -= 5
                reasons.append("RSI 침체")

        if pd.notna(current) and pd.notna(buy1) and buy1 > 0:
            gap = current / buy1 - 1

            if gap <= 0.02:
                score += 15
                reasons.append("1차 매수가와 매우 근접")
            elif gap <= 0.05:
                score += 10
                reasons.append("1차 매수가와 근접")
            elif gap <= 0.08:
                score += 5
                reasons.append("1차 매수가보다 다소 높음")
            else:
                reasons.append("1차 매수가보다 높음")

        if pd.notna(momentum) and momentum > 0:
            score += 5
            reasons.append("모멘텀 양호")

        if news_adj <= -4:
            score -= 5
            reasons.append("부정 뉴스 주의")

        score = float(np.clip(score, 0, 100))

        if score >= 80:
            judgement = "🟢 현재가 매수 가능"
            alloc = (0.25, 0.25, 0.30, 0.20)
        elif score >= 70:
            judgement = "🟡 소액 진입 가능"
            alloc = (0.15, 0.30, 0.35, 0.20)
        elif score >= 55:
            judgement = "⚪ 눌림 대기"
            alloc = (0.00, 0.35, 0.40, 0.25)
        else:
            judgement = "🔴 현재 매수 비추천"
            alloc = (0.00, 0.00, 0.00, 0.00)

        suitability.append(round(score, 1))
        judgements.append(judgement)
        current_allocs.append(alloc[0])
        b1_allocs.append(alloc[1])
        b2_allocs.append(alloc[2])
        b3_allocs.append(alloc[3])
        reasons_out.append(" / ".join(reasons))

    out["현재가 적정도"] = suitability
    out["현재가 판단"] = judgements
    out["현재가 비중"] = current_allocs
    out["1차 비중"] = b1_allocs
    out["2차 비중"] = b2_allocs
    out["3차 비중"] = b3_allocs
    out["현재가 판단 근거"] = reasons_out

    return out


# =========================
# 미래 예상수익률
# =========================
def bootstrap_forecast(
    series: pd.Series,
    horizon_days: int,
    simulations: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """
    과거 월간(21거래일) 수익률 블록을 복원추출하여 미래 경로를 시뮬레이션.
    중앙값 / 10% 분위 / 90% 분위 / 상승확률 반환.
    """
    s = series.dropna().astype(float)
    if len(s) < 126:
        return np.nan, np.nan, np.nan, np.nan

    block = 21
    block_returns = s.pct_change(block).dropna()

    # 최근 10년 정도만 사용
    if len(block_returns) > 2520:
        block_returns = block_returns.iloc[-2520:]

    if len(block_returns) < 60:
        return np.nan, np.nan, np.nan, np.nan

    values = np.clip(block_returns.values, -0.999, None)
    blocks_needed = int(np.ceil(horizon_days / block))

    rng = np.random.default_rng(seed + horizon_days)
    sampled = rng.choice(values, size=(simulations, blocks_needed), replace=True)

    remainder = horizon_days % block
    if remainder:
        frac = remainder / block
        sampled[:, -1] = np.expm1(np.log1p(sampled[:, -1]) * frac)

    terminal = np.prod(1 + sampled, axis=1) - 1

    return (
        float(np.median(terminal)),
        float(np.quantile(terminal, 0.10)),
        float(np.quantile(terminal, 0.90)),
        float((terminal > 0).mean()),
    )


def add_forecasts(screen: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    out = screen.copy()

    for label, days in FORECAST_HORIZONS.items():
        out[f"{label} 예상"] = np.nan
        out[f"{label} 범위"] = ""
        out[f"{label} 상승확률"] = np.nan

    for idx, row in out.iterrows():
        ticker = row["종목"]
        if ticker not in prices.columns:
            continue

        history = prices[ticker]

        for label, days in FORECAST_HORIZONS.items():
            expected, low, high, up_prob = bootstrap_forecast(history, days)

            out.at[idx, f"{label} 예상"] = expected
            out.at[idx, f"{label} 상승확률"] = up_prob
            out.at[idx, f"{label} 범위"] = (
                f"{low:.1%} ~ {high:.1%}"
                if pd.notna(low) and pd.notna(high)
                else "데이터 부족"
            )

    return out


# =========================
# 백테스트 / 추천 비중
# =========================
def generate_weights(
    prices: pd.DataFrame,
    indicators: Dict[str, pd.DataFrame],
    cfg: StrategyConfig,
) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    first_valid = max(cfg.ma_long, cfg.momentum_days, cfg.rsi_period, 120) + 1

    for i in range(first_valid, len(prices), cfg.rebalance_days):
        date = prices.index[i]

        snapshot_prices = prices.iloc[: i + 1]
        snapshot_indicators = {
            key: value.iloc[: i + 1]
            for key, value in indicators.items()
        }

        screen = latest_screen(snapshot_prices, snapshot_indicators, cfg)
        selected = screen[screen["기술 점수"] >= 65].head(cfg.max_positions)

        next_i = min(i + cfg.rebalance_days, len(prices))

        if not selected.empty:
            selected_tickers = selected["종목"].tolist()
            selected_vol = (
                indicators["volatility"]
                .loc[date, selected_tickers]
                .replace(0, np.nan)
            )

            inv_vol = 1 / selected_vol
            chosen = inv_vol / inv_vol.sum()
            weights.loc[prices.index[i:next_i], selected_tickers] = chosen.values

    return weights


def current_recommended_weights(screen: pd.DataFrame, max_positions: int) -> pd.DataFrame:
    selected = screen[screen["종합 점수"] >= 65].head(max_positions).copy()

    if selected.empty:
        return pd.DataFrame(columns=["종목", "추천 비중"])

    vol = selected.set_index("종목")["연환산 변동성"].replace(0, np.nan)
    inv_vol = 1 / vol
    w = inv_vol / inv_vol.sum()

    return (
        w.rename("추천 비중")
        .reset_index()
        .sort_values("추천 비중", ascending=False)
        .reset_index(drop=True)
    )


def backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    cfg: StrategyConfig,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    returns = prices.pct_change().fillna(0.0)
    shifted_weights = weights.shift(1).fillna(0.0)

    gross_return = (shifted_weights * returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    net_return = gross_return - turnover * cfg.fee_rate

    strategy_curve = (1 + net_return).cumprod()
    benchmark_return = returns.mean(axis=1)
    benchmark_curve = (1 + benchmark_return).cumprod()

    result = pd.DataFrame({
        "전략 일수익률": net_return,
        "전략 누적": strategy_curve,
        "동일비중 일수익률": benchmark_return,
        "동일비중 누적": benchmark_curve,
        "회전율": turnover,
    })

    years = max((result.index[-1] - result.index[0]).days / 365.25, 1 / 365.25)

    total_return = strategy_curve.iloc[-1] - 1
    cagr = strategy_curve.iloc[-1] ** (1 / years) - 1
    vol = net_return.std() * np.sqrt(252)
    sharpe = (net_return.mean() * 252) / vol if vol > 0 else np.nan

    rolling_max = strategy_curve.cummax()
    drawdown = strategy_curve / rolling_max - 1
    mdd = drawdown.min()

    win_rate = (
        (net_return[net_return != 0] > 0).mean()
        if (net_return != 0).any()
        else np.nan
    )

    stats = {
        "누적수익률": total_return,
        "CAGR": cagr,
        "연환산 변동성": vol,
        "샤프지수": sharpe,
        "MDD": mdd,
        "일간 승률": win_rate,
    }

    return result, stats


# =========================
# 표시 / 엑셀
# =========================
def format_price(ticker: str, price: float) -> str:
    if pd.isna(price):
        return "-"
    if ticker.endswith(".KS") or ticker.endswith(".KQ"):
        return f"{price:,.0f}원"
    return f"${price:,.2f}"


def format_buy_level(ticker: str, price: float, pct: float) -> str:
    if pd.isna(price):
        return "-"
    return f"{format_price(ticker, price)} ({pct:.1%})"


def to_excel_bytes(
    screen: pd.DataFrame,
    forecast: pd.DataFrame,
    recommended: pd.DataFrame,
    bt: pd.DataFrame,
) -> bytes:
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        screen.to_excel(writer, sheet_name="현재종목선별", index=False)
        forecast.to_excel(writer, sheet_name="예상수익률", index=False)
        recommended.to_excel(writer, sheet_name="현재추천비중", index=False)
        bt.to_excel(writer, sheet_name="백테스트")

    return output.getvalue()


# =========================
# 세션 상태
# =========================
if "selected_tickers" not in st.session_state:
    st.session_state.selected_tickers = MY_HOLDINGS.copy()

if "search_results" not in st.session_state:
    st.session_state.search_results = []


# =========================
# UI
# =========================
st.title("한미 성장주 퀀트 스크리너")
st.caption(
    "현재가 매수판단 + 1·2·3차 분할매수 + 기술점수/뉴스 보정 + "
    "1개월·3개월·6개월·1년·3년·5년 통계적 예상수익률"
)

with st.sidebar:
    st.header("종목 선택")

    preset = st.selectbox(
        "빠른 목록",
        ["보유종목", "기본 성장주", "보유종목 + 기본 성장주", "빈 목록"],
    )

    if st.button("이 목록 적용", use_container_width=True):
        if preset == "보유종목":
            st.session_state.selected_tickers = MY_HOLDINGS.copy()
        elif preset == "기본 성장주":
            st.session_state.selected_tickers = DEFAULT_TICKERS.copy()
        elif preset == "보유종목 + 기본 성장주":
            st.session_state.selected_tickers = list(
                dict.fromkeys(MY_HOLDINGS + DEFAULT_TICKERS)
            )
        else:
            st.session_state.selected_tickers = []

    st.divider()
    st.subheader("🔎 미국/나스닥 종목 검색")

    search_query = st.text_input(
        "회사명 또는 티커",
        placeholder="예: NVIDIA, NVDA, Rocket Lab",
    )

    if st.button("검색", use_container_width=True):
        st.session_state.search_results = search_us_stocks(search_query)

    if st.session_state.search_results:
        for item in st.session_state.search_results:
            label = (
                f"➕ {item['symbol']} | {item['name']} "
                f"{'(NASDAQ)' if item['nasdaq'] else ''}"
            )
            if st.button(label, key=f"add_{item['symbol']}", use_container_width=True):
                if item["symbol"] not in st.session_state.selected_tickers:
                    st.session_state.selected_tickers.append(item["symbol"])

    st.divider()
    manual = st.text_input(
        "직접 추가",
        placeholder="PLTR 또는 삼성전자 또는 005930.KS",
    )

    if st.button("직접 추가", use_container_width=True):
        ticker = normalize_ticker(manual)
        if ticker and ticker not in st.session_state.selected_tickers:
            st.session_state.selected_tickers.append(ticker)

    if st.session_state.selected_tickers:
        st.caption("현재 분석 목록")
        st.code(", ".join(st.session_state.selected_tickers), language=None)

    if st.button("선택 목록 비우기", use_container_width=True):
        st.session_state.selected_tickers = []

    st.divider()
    st.header("전략 설정")

    start_date = st.date_input(
        "시작일",
        value=pd.Timestamp.today() - pd.DateOffset(years=10),
    )
    end_date = st.date_input("종료일", value=pd.Timestamp.today())

    ma_short = st.number_input(
        "단기 이동평균",
        min_value=5,
        max_value=150,
        value=20,
    )
    ma_long = st.number_input(
        "장기 이동평균",
        min_value=50,
        max_value=400,
        value=120,
    )
    momentum_days = st.number_input(
        "모멘텀 기간(거래일)",
        min_value=20,
        max_value=252,
        value=63,
    )
    rebalance_days = st.number_input(
        "리밸런싱 주기(거래일)",
        min_value=5,
        max_value=63,
        value=21,
    )
    max_positions = st.number_input(
        "최대 추천 종목",
        min_value=1,
        max_value=20,
        value=5,
    )
    fee_rate = st.number_input(
        "매매비용률",
        min_value=0.0,
        max_value=0.02,
        value=0.001,
        step=0.0001,
        format="%.4f",
    )
    buy_score_min = st.number_input(
        "매수지점 표시 최소점수",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=5.0,
    )

    use_live_quote = st.checkbox("최신가 보정 사용", value=True)
    use_news = st.checkbox("뉴스 제목 보정 사용", value=True)

    run = st.button(
        "분석 실행",
        type="primary",
        use_container_width=True,
    )


if run:
    tickers = list(dict.fromkeys(st.session_state.selected_tickers))

    if not tickers:
        st.error("분석할 종목을 하나 이상 추가하세요.")
        st.stop()

    if start_date >= end_date:
        st.error("시작일은 종료일보다 빨라야 합니다.")
        st.stop()

    cfg = StrategyConfig(
        ma_short=int(ma_short),
        ma_long=int(ma_long),
        momentum_days=int(momentum_days),
        rebalance_days=int(rebalance_days),
        max_positions=int(max_positions),
        fee_rate=float(fee_rate),
        buy_score_min=float(buy_score_min),
    )

    with st.spinner("시장 데이터를 불러오는 중입니다."):
        prices, high, low, volume = download_market_data(
            tuple(tickers),
            str(start_date),
            str(end_date),
        )

    if prices.empty:
        st.error("시장 데이터를 불러오지 못했습니다.")
        st.stop()

    missing = sorted(set(tickers) - set(prices.columns))
    if missing:
        st.warning("데이터 누락 종목: " + ", ".join(missing))

    indicators = build_indicators(prices, high, low, volume, cfg)

    # 과거 백테스트는 기술점수만 사용
    weights = generate_weights(prices, indicators, cfg)
    bt, stats = backtest(prices, weights, cfg)

    # 현재 분석
    screen = latest_screen(prices, indicators, cfg)

    if use_live_quote:
        with st.spinner("최신 가격 확인 중입니다."):
            quotes = fetch_latest_quotes(tuple(screen["종목"].tolist()))
        screen = apply_live_quotes(screen, quotes)

    if use_news:
        with st.spinner("최근 뉴스 제목 확인 중입니다."):
            news_data = fetch_news_sentiment(tuple(screen["종목"].tolist()))
        screen = apply_news_adjustment(screen, news_data)

    screen = add_buy_points(screen, cfg)
    screen = add_current_entry_judgement(screen, cfg)
    screen = add_forecasts(screen, prices)

    recommended = current_recommended_weights(screen, cfg.max_positions)

    # ---------- 상단 요약 ----------
    st.subheader("현재 종목 선별")

    summary_cols = st.columns(4)

    if not screen.empty:
        top = screen.iloc[0]
        summary_cols[0].metric("1위 종목", str(top["종목"]))
        summary_cols[1].metric("최고 점수", f'{top["종합 점수"]:.1f}점')
        summary_cols[2].metric("현재가 판단", str(top["현재가 판단"]))
        summary_cols[3].metric(
            "65점 이상",
            f'{int((screen["종합 점수"] >= 65).sum())}개',
        )

    # ---------- 메인 표 ----------
    display = screen.copy()

    display["현재가"] = display.apply(
        lambda r: format_price(r["종목"], r["종가"]),
        axis=1,
    )
    display["1차 매수"] = display.apply(
        lambda r: format_buy_level(
            r["종목"], r["1차 매수가"], r["1차 하락률"]
        ),
        axis=1,
    )
    display["2차 매수"] = display.apply(
        lambda r: format_buy_level(
            r["종목"], r["2차 매수가"], r["2차 하락률"]
        ),
        axis=1,
    )
    display["3차 매수"] = display.apply(
        lambda r: format_buy_level(
            r["종목"], r["3차 매수가"], r["3차 하락률"]
        ),
        axis=1,
    )

    for label in FORECAST_HORIZONS:
        display[f"{label} 예상"] = display[f"{label} 예상"].map(
            lambda x: "" if pd.isna(x) else f"{x:.1%}"
        )

    for col in ["현재가 비중", "1차 비중", "2차 비중", "3차 비중"]:
        display[col] = display[col].map(
            lambda x: "" if pd.isna(x) else f"{x:.0%}"
        )

    st.dataframe(
        display[
            [
                "종목",
                "종합 점수",
                "의견",
                "현재가 적정도",
                "현재가 판단",
                "현재가",
                "1차 매수",
                "2차 매수",
                "3차 매수",
                "현재가 비중",
                "1차 비중",
                "2차 비중",
                "3차 비중",
                "1개월 예상",
                "3개월 예상",
                "6개월 예상",
                "1년 예상",
                "3년 예상",
                "5년 예상",
                "뉴스 판단",
                "판단 근거",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    # ---------- 현재가 진입 판단 ----------
    st.subheader("현재가 진입 판단")
    entry_view = screen[
        [
            "종목",
            "종합 점수",
            "현재가 적정도",
            "현재가 판단",
            "종가",
            "1차 매수가",
            "2차 매수가",
            "3차 매수가",
            "현재가 비중",
            "1차 비중",
            "2차 비중",
            "3차 비중",
            "현재가 판단 근거",
        ]
    ].copy()

    entry_view["현재가"] = entry_view.apply(
        lambda r: format_price(r["종목"], r["종가"]),
        axis=1,
    )
    entry_view["1차"] = entry_view.apply(
        lambda r: format_price(r["종목"], r["1차 매수가"]),
        axis=1,
    )
    entry_view["2차"] = entry_view.apply(
        lambda r: format_price(r["종목"], r["2차 매수가"]),
        axis=1,
    )
    entry_view["3차"] = entry_view.apply(
        lambda r: format_price(r["종목"], r["3차 매수가"]),
        axis=1,
    )

    st.dataframe(
        entry_view[
            [
                "종목",
                "종합 점수",
                "현재가 적정도",
                "현재가 판단",
                "현재가",
                "1차",
                "2차",
                "3차",
                "현재가 비중",
                "1차 비중",
                "2차 비중",
                "3차 비중",
                "현재가 판단 근거",
            ]
        ].style.format(
            {
                "현재가 비중": "{:.0%}",
                "1차 비중": "{:.0%}",
                "2차 비중": "{:.0%}",
                "3차 비중": "{:.0%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "분할비중은 해당 종목에 투자할 예정금액을 100%로 놓고 계산합니다."
    )

    # ---------- 미래 예상수익률 ----------
    st.subheader("기간별 미래 예상수익률")

    forecast_rows = []

    for _, row in screen.iterrows():
        for label in FORECAST_HORIZONS:
            forecast_rows.append({
                "종목": row["종목"],
                "기간": label,
                "예상 수익률": row[f"{label} 예상"],
                "10~90% 예상 범위": row[f"{label} 범위"],
                "상승 확률": row[f"{label} 상승확률"],
            })

    forecast_df = pd.DataFrame(forecast_rows)

    st.dataframe(
        forecast_df.style.format(
            {
                "예상 수익률": "{:.1%}",
                "상승 확률": "{:.1%}",
            },
            na_rep="-",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "예상수익률은 과거 월간 수익률 분포를 반복 복원추출한 통계적 시나리오의 중앙값입니다. "
        "특히 3년·5년은 불확실성이 매우 크므로 예상 범위와 상승확률을 함께 보세요."
    )

    # ---------- 추천 비중 ----------
    st.subheader("현재 추천 비중")

    if recommended.empty:
        st.info("현재 65점 이상 종목이 없어 추천 비중을 산출하지 않습니다.")
    else:
        st.dataframe(
            recommended.style.format({"추천 비중": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
        )

    # ---------- 세부 분석 ----------
    with st.expander("점수 구성 / 뉴스 / 세부 지표 보기"):
        detail = screen.copy()

        for col in [
            f"{cfg.momentum_days}일 모멘텀",
            "연환산 변동성",
            "52주 고점 근접도",
        ]:
            detail[col] = detail[col].map(
                lambda x: "" if pd.isna(x) else f"{x:.2%}"
            )

        st.dataframe(
            detail[
                [
                    "종목",
                    "기술 점수",
                    "뉴스 보정",
                    "종합 점수",
                    "추세 점수",
                    "모멘텀 점수",
                    "RSI 점수",
                    "변동성 점수",
                    "거래량 점수",
                    "52주 고점 점수",
                    "RSI",
                    f"{cfg.momentum_days}일 모멘텀",
                    "연환산 변동성",
                    "거래량 배수",
                    "52주 고점 근접도",
                    "뉴스 판단",
                    "뉴스 요약",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    # ---------- 백테스트 ----------
    with st.expander("과거 전략 백테스트 보기"):
        cols = st.columns(len(stats))

        for col, (name, value) in zip(cols, stats.items()):
            if name == "샤프지수":
                col.metric(
                    name,
                    "N/A" if pd.isna(value) else f"{value:.2f}",
                )
            else:
                col.metric(
                    name,
                    "N/A" if pd.isna(value) else f"{value:.2%}",
                )

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(bt.index, bt["전략 누적"], label="Quant strategy")
        ax.plot(bt.index, bt["동일비중 누적"], label="Equal weight")
        ax.set_ylabel("Cumulative wealth")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig, use_container_width=True)

        st.caption(
            "백테스트는 과거 뉴스 원문 데이터가 없으므로 기술점수만 사용합니다."
        )

    # ---------- 엑셀 ----------
    excel_bytes = to_excel_bytes(screen, forecast_df, recommended, bt)

    st.download_button(
        "결과 엑셀 다운로드",
        data=excel_bytes,
        file_name="quant_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.info(
        "이 앱은 투자판단 보조 및 전략 검증용입니다. 최신가는 데이터 제공처 상황에 따라 지연될 수 있고, "
        "뉴스 보정은 기사 제목 키워드 기반이며 기사 본문 AI 분석이 아닙니다. "
        "예상수익률과 매수지점은 미래 수익을 보장하지 않습니다."
    )

else:
    st.markdown(
        """
        ### 이 앱에서 할 수 있는 것

        1. **미국/나스닥 종목 검색 후 바로 추가**
        2. 한국 종목 코드(.KS/.KQ) 또는 일부 한국 종목명 추가
        3. 기술적 100점 점수 + 최근 뉴스 제목 보정
        4. **현재가에서 지금 사도 되는지 판단**
        5. 현재가 / 1차 / 2차 / 3차 분할매수 비중 제안
        6. ATR·이동평균·최근 지지선을 이용한 1·2·3차 매수지점
        7. **1개월·3개월·6개월·1년·3년·5년 예상수익률**
        8. 각 기간 10~90% 예상범위와 과거 분포 기반 상승확률
        9. 필요할 때만 과거 백테스트 확인

        ### 기술 점수 구성
        - 추세 30점
        - 모멘텀 25점
        - RSI 15점
        - 변동성 10점
        - 거래량 10점
        - 52주 고점 근접도 10점
        """
    )
