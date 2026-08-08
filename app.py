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

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1280px;
        padding-top: 1.1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        padding-top: 0.1rem;
        padding-bottom: 0.1rem;
    }

    div[data-testid="stDataFrame"] {
        margin-top: 0.2rem;
        margin-bottom: 0.8rem;
    }

    h1, h2, h3 {
        margin-top: 0.7rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
US_KR_ALIAS = {
    "엔비디아": ("NVDA", "NVIDIA"),
    "마이크로소프트": ("MSFT", "Microsoft"),
    "애플": ("AAPL", "Apple"),
    "아마존": ("AMZN", "Amazon"),
    "메타": ("META", "Meta Platforms"),
    "구글": ("GOOGL", "Alphabet"),
    "알파벳": ("GOOGL", "Alphabet"),
    "브로드컴": ("AVGO", "Broadcom"),
    "팔란티어": ("PLTR", "Palantir Technologies"),
    "테슬라": ("TSLA", "Tesla"),
    "에이엠디": ("AMD", "Advanced Micro Devices"),
    "크라우드스트라이크": ("CRWD", "CrowdStrike"),
    "오라클": ("ORCL", "Oracle"),
    "로켓랩": ("RKLB", "Rocket Lab USA"),
    "조비": ("JOBY", "Joby Aviation"),
    "조비에비에이션": ("JOBY", "Joby Aviation"),
    "온다스": ("ONDS", "Ondas Holdings"),
    "우라늄에너지": ("UEC", "Uranium Energy"),
    "아이온큐": ("IONQ", "IonQ"),
    "레드캣": ("RCAT", "Red Cat Holdings"),
    "리게티": ("RGTI", "Rigetti Computing"),
    "디웨이브": ("QBTS", "D-Wave Quantum"),
    "퀀텀컴퓨팅": ("QUBT", "Quantum Computing"),
    "오클로": ("OKLO", "Oklo"),
    "뉴스케일": ("SMR", "NuScale Power"),
    "아처": ("ACHR", "Archer Aviation"),
    "루시드": ("LCID", "Lucid Group"),
    "리비안": ("RIVN", "Rivian Automotive"),
    "코인베이스": ("COIN", "Coinbase"),
    "넷플릭스": ("NFLX", "Netflix"),
    "인텔": ("INTC", "Intel"),
    "퀄컴": ("QCOM", "Qualcomm"),
    "마이크론": ("MU", "Micron Technology"),
    "슈퍼마이크로": ("SMCI", "Super Micro Computer"),
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


@st.cache_data(ttl=21600, show_spinner=False)
def load_nasdaq_universe() -> pd.DataFrame:
    """NASDAQ Trader 공식 심볼 파일에서 현재 NASDAQ 상장 종목 목록을 가져옵니다."""
    url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urlopen(req, timeout=10) as response:
            raw = response.read().decode("utf-8", errors="ignore")

        df = pd.read_csv(BytesIO(raw.encode("utf-8")), sep="|")

        # 마지막 File Creation Time 행 제거
        df = df[df["Symbol"].notna()].copy()
        df = df[~df["Symbol"].astype(str).str.startswith("File Creation Time")]

        # 테스트 종목 제외
        if "Test Issue" in df.columns:
            df = df[df["Test Issue"].astype(str).str.upper() != "Y"]

        df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
        df["Security Name"] = df["Security Name"].astype(str).str.strip()

        return df[["Symbol", "Security Name"]].drop_duplicates("Symbol")

    except Exception:
        return pd.DataFrame(columns=["Symbol", "Security Name"])


@st.cache_data(ttl=3600, show_spinner=False)
def search_stocks(query: str) -> List[dict]:
    """
    1) 한글 별칭/한국 종목명 검색
    2) NASDAQ Trader에서 자동으로 가져온 전체 NASDAQ 목록 검색
    3) 필요 시 Yahoo Finance 검색으로 보완
    """
    q = query.strip()
    if not q:
        return []

    q_clean = q.lower().replace(" ", "")
    results: List[dict] = []
    seen = set()

    # 한국 종목 한글 검색
    for name, symbol in KR_NAME_MAP.items():
        name_clean = name.lower().replace(" ", "")
        if q_clean in name_clean or name_clean in q_clean:
            if symbol not in seen:
                seen.add(symbol)
                results.append({
                    "symbol": symbol,
                    "name": name,
                    "exchange": "KRX",
                    "nasdaq": False,
                })

    # 미국 종목 한글 별칭 검색
    for alias, (symbol, english_name) in US_KR_ALIAS.items():
        alias_clean = alias.lower().replace(" ", "")
        if q_clean in alias_clean or alias_clean in q_clean:
            if symbol not in seen:
                seen.add(symbol)
                results.append({
                    "symbol": symbol,
                    "name": f"{alias} · {english_name}",
                    "exchange": "NASDAQ/US",
                    "nasdaq": True,
                })

    # NASDAQ 전체 상장 목록에서 티커/영문 회사명 검색
    nasdaq = load_nasdaq_universe()
    if not nasdaq.empty:
        symbol_match = nasdaq["Symbol"].str.contains(q.upper(), regex=False, na=False)
        name_match = nasdaq["Security Name"].str.contains(q, case=False, regex=False, na=False)
        matched = nasdaq[symbol_match | name_match].copy()

        # 정확한 티커 일치 -> 티커 시작 -> 이름 포함 순으로 정렬
        matched["_exact"] = (matched["Symbol"] == q.upper()).astype(int)
        matched["_prefix"] = matched["Symbol"].str.startswith(q.upper()).astype(int)
        matched = matched.sort_values(["_exact", "_prefix", "Symbol"], ascending=[False, False, True])

        for _, item in matched.head(25).iterrows():
            symbol = str(item["Symbol"]).upper()
            if symbol in seen:
                continue
            seen.add(symbol)
            results.append({
                "symbol": symbol,
                "name": str(item["Security Name"]),
                "exchange": "NASDAQ",
                "nasdaq": True,
            })

    # NASDAQ 목록에서 안 잡히는 미국/한국 종목을 Yahoo로 보완
    if len(results) < 15:
        url = (
            "https://query1.finance.yahoo.com/v1/finance/search"
            f"?q={quote(q)}&quotesCount=20&newsCount=0"
        )
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})

        try:
            with urlopen(req, timeout=6) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            payload = {}

        for item in payload.get("quotes", []):
            symbol = str(item.get("symbol", "")).upper()
            quote_type = str(item.get("quoteType", "")).upper()
            exchange = str(item.get("exchange", "")).upper()
            name = item.get("shortname") or item.get("longname") or symbol

            if not symbol or quote_type not in {"EQUITY", "ETF"} or symbol in seen:
                continue

            is_nasdaq = exchange in {"NMS", "NCM", "NGM", "NASDAQ"}
            is_us = exchange in {"NMS", "NCM", "NGM", "NASDAQ", "NYQ", "ASE", "PCX", "BTS"}
            is_korea = symbol.endswith(".KS") or symbol.endswith(".KQ")

            if not (is_us or is_korea):
                continue

            seen.add(symbol)
            results.append({
                "symbol": symbol,
                "name": str(name),
                "exchange": exchange,
                "nasdaq": is_nasdaq,
            })

    return results[:30]




# =========================
# 2차 종합 분석: 기술 + 실적 + 가치 + 뉴스 + 시장기대 + 통계
# =========================

def _safe_float(value, default=np.nan):
    try:
        if value is None:
            return default
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return default


def _clip_score(value: float, max_score: float) -> float:
    return float(np.clip(value, 0.0, max_score))


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_company_snapshot(ticker: str) -> dict:
    """yfinance에서 실적·밸류·애널리스트 핵심값을 한 번에 가져옵니다."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}

    return {
        "revenueGrowth": _safe_float(info.get("revenueGrowth")),
        "earningsGrowth": _safe_float(
            info.get("earningsGrowth", info.get("earningsQuarterlyGrowth"))
        ),
        "profitMargins": _safe_float(info.get("profitMargins")),
        "operatingMargins": _safe_float(info.get("operatingMargins")),
        "freeCashflow": _safe_float(info.get("freeCashflow")),
        "totalCash": _safe_float(info.get("totalCash")),
        "totalDebt": _safe_float(info.get("totalDebt")),
        "trailingPE": _safe_float(info.get("trailingPE")),
        "forwardPE": _safe_float(info.get("forwardPE")),
        "priceToSalesTrailing12Months": _safe_float(
            info.get("priceToSalesTrailing12Months")
        ),
        "pegRatio": _safe_float(info.get("pegRatio")),
        "targetMeanPrice": _safe_float(info.get("targetMeanPrice")),
        "currentPrice": _safe_float(
            info.get("currentPrice", info.get("regularMarketPrice"))
        ),
        "recommendationMean": _safe_float(info.get("recommendationMean")),
        "numberOfAnalystOpinions": _safe_float(info.get("numberOfAnalystOpinions")),
    }


def score_fundamentals(info: dict) -> Tuple[float, str]:
    """기업 실적/재무 25점."""
    rev = info.get("revenueGrowth", np.nan)
    earn = info.get("earningsGrowth", np.nan)
    pm = info.get("profitMargins", np.nan)
    om = info.get("operatingMargins", np.nan)
    cash = info.get("totalCash", np.nan)
    debt = info.get("totalDebt", np.nan)
    fcf = info.get("freeCashflow", np.nan)

    score = 0.0
    parts = 0.0
    reasons = []

    # 매출 성장 8점
    if pd.notna(rev):
        if rev >= 0.30: s = 8
        elif rev >= 0.20: s = 7
        elif rev >= 0.10: s = 6
        elif rev >= 0.05: s = 4.5
        elif rev >= 0: s = 3
        else: s = 1
        score += s
        parts += 8
        reasons.append(f"매출성장 {rev:.1%}")

    # 이익 성장 7점
    if pd.notna(earn):
        if earn >= 0.40: s = 7
        elif earn >= 0.25: s = 6
        elif earn >= 0.10: s = 5
        elif earn >= 0: s = 3.5
        else: s = 1
        score += s
        parts += 7
        reasons.append(f"이익성장 {earn:.1%}")

    # 수익성 6점
    if pd.notna(pm):
        s = 4 if pm >= 0.20 else 3 if pm >= 0.10 else 2 if pm >= 0 else 0.5
        score += s
        parts += 4
        reasons.append(f"순이익률 {pm:.1%}")
    if pd.notna(om):
        s = 2 if om >= 0.20 else 1.5 if om >= 0.10 else 1 if om >= 0 else 0.25
        score += s
        parts += 2

    # 현금흐름/재무건전성 4점
    if pd.notna(fcf):
        score += 2 if fcf > 0 else 0.5
        parts += 2
        reasons.append("FCF 흑자" if fcf > 0 else "FCF 적자")

    if pd.notna(cash) and pd.notna(debt):
        if debt <= 0:
            s = 2
        else:
            ratio = cash / debt
            s = 2 if ratio >= 1 else 1.5 if ratio >= 0.5 else 0.75
        score += s
        parts += 2

    # 누락 데이터는 중립 50%로 채움
    if parts < 25:
        score += (25 - parts) * 0.5

    return round(_clip_score(score, 25), 1), " / ".join(reasons[:4]) or "재무데이터 제한"


def score_valuation(info: dict) -> Tuple[float, str]:
    """밸류에이션 15점. 성장주 특성상 완전 저PER만 우대하지 않습니다."""
    pe = info.get("forwardPE", np.nan)
    if pd.isna(pe):
        pe = info.get("trailingPE", np.nan)
    ps = info.get("priceToSalesTrailing12Months", np.nan)
    peg = info.get("pegRatio", np.nan)

    available = []
    reasons = []

    if pd.notna(pe) and pe > 0:
        s = 5 if pe <= 20 else 4.2 if pe <= 30 else 3.2 if pe <= 45 else 2 if pe <= 70 else 0.8
        available.append((s, 5))
        reasons.append(f"PER {pe:.1f}")

    if pd.notna(ps) and ps > 0:
        s = 5 if ps <= 3 else 4 if ps <= 6 else 3 if ps <= 10 else 1.8 if ps <= 20 else 0.7
        available.append((s, 5))
        reasons.append(f"PS {ps:.1f}")

    if pd.notna(peg) and peg > 0:
        s = 5 if peg <= 1 else 4 if peg <= 1.5 else 3 if peg <= 2.5 else 1.5 if peg <= 4 else 0.5
        available.append((s, 5))
        reasons.append(f"PEG {peg:.2f}")

    if not available:
        return 7.5, "가치평가 데이터 제한"

    score = sum(s for s, _ in available)
    max_seen = sum(m for _, m in available)
    # 없는 항목은 중립점수로 채움
    score += (15 - max_seen) * 0.5
    return round(_clip_score(score, 15), 1), " / ".join(reasons)


def score_analyst(info: dict, fallback_price=np.nan) -> Tuple[float, str]:
    """애널리스트/시장 기대 10점."""
    target = info.get("targetMeanPrice", np.nan)
    current = info.get("currentPrice", np.nan)
    if pd.isna(current):
        current = fallback_price
    rec = info.get("recommendationMean", np.nan)
    count = info.get("numberOfAnalystOpinions", np.nan)

    score = 5.0  # 기본 중립
    reasons = []

    if pd.notna(target) and pd.notna(current) and current > 0:
        upside = target / current - 1
        target_score = 6 if upside >= 0.30 else 5 if upside >= 0.15 else 4 if upside >= 0.05 else 3 if upside >= -0.05 else 1.5
        reasons.append(f"목표가 여력 {upside:.1%}")
    else:
        target_score = 3

    if pd.notna(rec) and rec > 0:
        # Yahoo recommendationMean: 1 강매수 ~ 5 매도
        rec_score = 4 if rec <= 1.7 else 3.3 if rec <= 2.2 else 2.5 if rec <= 2.8 else 1.5 if rec <= 3.5 else 0.5
        reasons.append(f"애널리스트 {rec:.2f}")
    else:
        rec_score = 2

    score = target_score + rec_score

    # 의견 수가 너무 적으면 과신 방지
    if pd.notna(count) and count < 3:
        score *= 0.85
        reasons.append("의견수 적음")

    return round(_clip_score(score, 10), 1), " / ".join(reasons) or "시장기대 데이터 제한"


def score_news(news_item: dict) -> Tuple[float, str]:
    """뉴스 15점: 기존 -10~+10 보정을 0~15점으로 매핑."""
    adj = float(news_item.get("adjustment", 0.0) or 0.0)
    score = 7.5 + adj * 0.75
    label = news_item.get("label", "중립")
    return round(_clip_score(score, 15), 1), f"{label} ({adj:+.1f})"


def statistical_score_from_history(series: pd.Series) -> Tuple[float, str]:
    """
    현재와 비슷한 추세/RSI/모멘텀 구간 이후 63거래일 수익률을 찾아 10점 평가.
    표본이 적으면 전체 3개월 분포로 대체합니다.
    """
    s = series.dropna().astype(float)
    if len(s) < 190:
        return 5.0, "통계 표본 부족"

    ma20 = s.rolling(20).mean()
    ma120 = s.rolling(120).mean()
    mom63 = s.pct_change(63)
    rsi = compute_rsi(s, 14)
    fwd63 = s.shift(-63) / s - 1

    cur_rsi = rsi.iloc[-1]
    cur_mom = mom63.iloc[-1]
    cur_trend1 = s.iloc[-1] > ma120.iloc[-1]
    cur_trend2 = ma20.iloc[-1] > ma120.iloc[-1]

    mask = (
        (rsi.sub(cur_rsi).abs() <= 8)
        & (mom63.sub(cur_mom).abs() <= 0.12)
        & ((s > ma120) == cur_trend1)
        & ((ma20 > ma120) == cur_trend2)
    )

    samples = fwd63[mask].dropna()
    if len(samples) < 12:
        samples = fwd63.dropna()

    if len(samples) < 12:
        return 5.0, "통계 표본 부족"

    up_prob = float((samples > 0).mean())
    median = float(samples.median())

    prob_score = np.interp(up_prob, [0.35, 0.50, 0.65, 0.80], [0.5, 2.5, 5.5, 7.0])
    return_score = np.interp(median, [-0.10, 0.00, 0.10, 0.25], [0.0, 1.0, 2.0, 3.0])
    score = float(np.clip(prob_score + return_score, 0, 10))

    return round(score, 1), f"유사구간 {len(samples)}회 · 3개월 상승 {up_prob:.0%} · 중앙값 {median:.1%}"


def add_multifactor_scores(
    screen: pd.DataFrame,
    prices: pd.DataFrame,
    news_data: Dict[str, dict] | None = None,
) -> pd.DataFrame:
    """
    기존 기술점수를 25점으로 축소하고,
    실적25 + 가치15 + 뉴스15 + 시장기대10 + 통계10 = 총 100점으로 재평가합니다.
    """
    out = screen.copy()
    news_data = news_data or {}

    for idx, row in out.iterrows():
        ticker = row["종목"]
        technical_25 = round(float(row.get("기술 점수", 0.0)) * 0.25, 1)

        info = fetch_company_snapshot(ticker)
        fundamental, fundamental_reason = score_fundamentals(info)
        valuation, valuation_reason = score_valuation(info)
        news_score, news_reason = score_news(news_data.get(ticker, {}))
        analyst, analyst_reason = score_analyst(info, row.get("종가", np.nan))

        if ticker in prices.columns:
            stat_score, stat_reason = statistical_score_from_history(prices[ticker])
        else:
            stat_score, stat_reason = 5.0, "가격 이력 부족"

        total = (
            technical_25
            + fundamental
            + valuation
            + news_score
            + analyst
            + stat_score
        )

        out.at[idx, "기술/차트 점수"] = technical_25
        out.at[idx, "기업실적 점수"] = fundamental
        out.at[idx, "밸류에이션 점수"] = valuation
        out.at[idx, "뉴스/이벤트 점수"] = news_score
        out.at[idx, "시장기대 점수"] = analyst
        out.at[idx, "통계/확률 점수"] = stat_score
        out.at[idx, "종합 점수"] = round(float(np.clip(total, 0, 100)), 1)
        out.at[idx, "의견"] = opinion_from_score(out.at[idx, "종합 점수"])
        out.at[idx, "종합 판단 근거"] = (
            f"기술 {technical_25:.1f}/25 · 실적 {fundamental:.1f}/25 · "
            f"가치 {valuation:.1f}/15 · 뉴스 {news_score:.1f}/15 · "
            f"시장기대 {analyst:.1f}/10 · 통계 {stat_score:.1f}/10"
        )
        out.at[idx, "실적 근거"] = fundamental_reason
        out.at[idx, "가치 근거"] = valuation_reason
        out.at[idx, "뉴스 근거"] = news_reason
        out.at[idx, "시장기대 근거"] = analyst_reason
        out.at[idx, "통계 근거"] = stat_reason

    return out.sort_values("종합 점수", ascending=False).reset_index(drop=True)


# =========================
# NASDAQ 추천 후보 자동 탐색
# =========================

NASDAQ100_FALLBACK = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN",
    "AMZN", "ANSS", "APP", "ARM", "ASML", "AVGO", "AXON", "AZN", "BIIB", "BKNG",
    "BKR", "CCEP", "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD",
    "CSCO", "CSGP", "CSX", "CTAS", "CTSH", "DASH", "DDOG", "DXCM", "EA", "EXC",
    "FANG", "FAST", "FTNT", "GEHC", "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX",
    "INTC", "INTU", "ISRG", "KDP", "KHC", "KLAC", "LIN", "LRCX", "MAR", "MCHP",
    "MDB", "MDLZ", "MELI", "META", "MNST", "MRVL", "MSFT", "MSTR", "MU", "NFLX",
    "NVDA", "NXPI", "ODFL", "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PEP",
    "PLTR", "PYPL", "QCOM", "REGN", "ROP", "ROST", "SBUX", "SNPS", "TEAM", "TMUS",
    "TSLA", "TTD", "TTWO", "TXN", "VRSK", "VRTX", "WBD", "WDAY", "XEL", "ZS",
]


@st.cache_data(ttl=86400, show_spinner=False)
def load_nasdaq100_universe() -> List[str]:
    """
    Nasdaq-100 구성 종목을 웹에서 가져오고, 실패하면 내장 목록으로 대체합니다.
    """
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for table in tables:
            cols = [str(c).strip().lower() for c in table.columns]
            if "ticker" in cols:
                ticker_col = table.columns[cols.index("ticker")]
                tickers = (
                    table[ticker_col]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .str.replace(".", "-", regex=False)
                    .tolist()
                )
                tickers = [x for x in tickers if x and x != "NAN"]
                if len(tickers) >= 80:
                    return list(dict.fromkeys(tickers))
    except Exception:
        pass

    return NASDAQ100_FALLBACK.copy()


@st.cache_data(ttl=900, show_spinner=False)
def scan_nasdaq_candidates(limit: int = 15) -> pd.DataFrame:
    """
    1차: Nasdaq-100 전체를 기술점수로 빠르게 스캔
    2차: 상위 후보만 실적·밸류·뉴스·애널리스트·통계까지 심층 분석
    """
    tickers = load_nasdaq100_universe()
    if not tickers:
        return pd.DataFrame()

    try:
        data = yf.download(
            tickers,
            period="3y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    def get_field(field: str) -> pd.DataFrame:
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
            out = out.to_frame()
        return out

    close = get_field("Close")
    volume = get_field("Volume")

    if close.empty:
        return pd.DataFrame()

    close = close.dropna(how="all")
    volume = volume.reindex(index=close.index, columns=close.columns)

    rows = []
    for ticker in close.columns:
        s = close[ticker].dropna()
        if len(s) < 130:
            continue

        price = float(s.iloc[-1])
        ma20 = float(s.rolling(20).mean().iloc[-1])
        ma120 = float(s.rolling(120).mean().iloc[-1])
        mom63 = float(s.pct_change(63).iloc[-1])
        vol63 = float(s.pct_change().rolling(63).std().iloc[-1] * np.sqrt(252))
        high52 = float(s.rolling(252, min_periods=120).max().iloc[-1])
        high_near = price / high52 if high52 > 0 else np.nan
        rsi = float(compute_rsi(s, 14).iloc[-1])

        vol_ratio = np.nan
        if ticker in volume.columns:
            v = volume[ticker].dropna()
            if len(v) >= 20:
                avg20 = float(v.rolling(20).mean().iloc[-1])
                if avg20 > 0:
                    vol_ratio = float(v.iloc[-1] / avg20)

        rows.append({
            "종목": ticker,
            "종가": price,
            "20일선": ma20,
            "120일선": ma120,
            "63일 모멘텀": mom63,
            "RSI": rsi,
            "연환산 변동성": vol63,
            "거래량 배수": vol_ratio,
            "52주 고점 근접도": high_near,
        })

    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw

    raw = raw.set_index("종목")
    mom_pct = percentile_score(raw["63일 모멘텀"], True)
    volatility_pct = percentile_score(raw["연환산 변동성"], False)
    volume_pct = percentile_score(raw["거래량 배수"], True)
    high_pct = percentile_score(raw["52주 고점 근접도"], True)

    technical_rows = []

    for ticker, row in raw.iterrows():
        trend_score = 0.0
        if row["종가"] > row["120일선"]:
            trend_score += 15.0
        if row["20일선"] > row["120일선"]:
            trend_score += 15.0

        momentum_score = float(mom_pct.loc[ticker] * 25.0)

        rsi = row["RSI"]
        if 55 <= rsi <= 65:
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

        tech100 = (
            trend_score
            + momentum_score
            + rsi_score
            + volatility_score
            + volume_score
            + high_score
        )

        technical_rows.append({
            "종목": ticker,
            "기술원점수": round(tech100, 1),
            "현재가": row["종가"],
            "RSI": round(rsi, 1),
        })

    # 1차 필터: 기술 상위 20개만 심층분석
    pre = (
        pd.DataFrame(technical_rows)
        .sort_values("기술원점수", ascending=False)
        .head(max(20, limit))
        .reset_index(drop=True)
    )

    news_map = fetch_news_sentiment(tuple(pre["종목"].tolist()))
    final_rows = []

    for _, row in pre.iterrows():
        ticker = row["종목"]
        tech25 = round(float(row["기술원점수"]) * 0.25, 1)

        info = fetch_company_snapshot(ticker)
        fundamental, fundamental_reason = score_fundamentals(info)
        valuation, valuation_reason = score_valuation(info)
        news_score, news_reason = score_news(news_map.get(ticker, {}))
        analyst, analyst_reason = score_analyst(info, row["현재가"])
        stat_score, stat_reason = statistical_score_from_history(close[ticker])

        total = tech25 + fundamental + valuation + news_score + analyst + stat_score

        final_rows.append({
            "종목": ticker,
            "점수": round(float(np.clip(total, 0, 100)), 1),
            "의견": opinion_from_score(total),
            "기술": tech25,
            "실적": fundamental,
            "가치": valuation,
            "뉴스": news_score,
            "시장기대": analyst,
            "통계": stat_score,
            "현재가": row["현재가"],
            "RSI": row["RSI"],
            "판단 근거": (
                f"기술 {tech25:.1f}/25 · 실적 {fundamental:.1f}/25 · "
                f"가치 {valuation:.1f}/15 · 뉴스 {news_score:.1f}/15 · "
                f"시장 {analyst:.1f}/10 · 통계 {stat_score:.1f}/10"
            ),
            "세부 근거": (
                f"{fundamental_reason} | {valuation_reason} | "
                f"{news_reason} | {analyst_reason} | {stat_reason}"
            ),
        })

    return (
        pd.DataFrame(final_rows)
        .sort_values("점수", ascending=False)
        .head(int(limit))
        .reset_index(drop=True)
    )

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
TICKER_KR_NAME = {
    "NVDA": "엔비디아",
    "MSFT": "마이크로소프트",
    "AAPL": "애플",
    "AMZN": "아마존",
    "META": "메타",
    "GOOGL": "알파벳",
    "AVGO": "브로드컴",
    "PLTR": "팔란티어",
    "AMD": "AMD",
    "TSLA": "테슬라",
    "CRWD": "크라우드스트라이크",
    "ORCL": "오라클",

    "RKLB": "로켓랩",
    "JOBY": "조비 에비에이션",
    "ONDS": "온다스 홀딩스",
    "UEC": "우라늄 에너지",
    "IONQ": "아이온큐",
    "RCAT": "레드캣 홀딩스",

    "RGTI": "리게티 컴퓨팅",
    "QBTS": "디웨이브 퀀텀",
    "QUBT": "퀀텀 컴퓨팅",
    "OKLO": "오클로",
    "SMR": "뉴스케일 파워",
    "ACHR": "아처 에비에이션",

    "005930.KS": "삼성전자",
    "000660.KS": "SK하이닉스",
    "012450.KS": "한화에어로스페이스",
    "079550.KS": "LIG넥스원",
    "373220.KS": "LG에너지솔루션",
    "277810.KQ": "레인보우로보틱스",
    "272210.KS": "한화시스템",
    "082740.KS": "한화엔진",
    "001260.KS": "남광토건",

    "SKHY": "SK하이닉스 ADR",
}


def korean_ticker_name(ticker):
    name = TICKER_KR_NAME.get(ticker)

    if name:
        return f"{name} ({ticker})"

    return ticker
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


if "favorites" not in st.session_state:
    st.session_state.favorites = []


if "nasdaq_recommendations" not in st.session_state:
    st.session_state.nasdaq_recommendations = pd.DataFrame()


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

    # =========================
    # 즐겨찾기
    # =========================
    st.subheader(f"⭐ 즐겨찾기 ({len(st.session_state.favorites)}개)")

    if st.session_state.favorites:
        favorite_to_remove = None

        for i, ticker in enumerate(list(st.session_state.favorites)):
            c1, c2, c3 = st.columns([5, 1, 1])

            c1.write(korean_ticker_name(ticker))

            if c2.button(
                "＋",
                key=f"favorite_add_{i}_{ticker}",
                help="현재 분석 목록에 추가",
            ):
                if ticker not in st.session_state.selected_tickers:
                    st.session_state.selected_tickers.append(ticker)
                st.rerun()

            if c3.button(
                "✕",
                key=f"favorite_remove_{i}_{ticker}",
                help="즐겨찾기에서 삭제",
            ):
                favorite_to_remove = ticker

        if favorite_to_remove:
            st.session_state.favorites = [
                x for x in st.session_state.favorites
                if x != favorite_to_remove
            ]
            st.rerun()
    else:
        st.caption("검색 결과의 ⭐ 버튼으로 종목을 저장할 수 있습니다.")

    st.divider()

    # =========================
    # 현재 분석 종목
    # =========================

    st.subheader(
        f"📌 현재 분석 목록 "
        f"({len(st.session_state.selected_tickers)}개)"
    )

    if st.session_state.selected_tickers:

        remove_ticker = None

        for i, ticker in enumerate(
            list(st.session_state.selected_tickers)
        ):

            col1, col2 = st.columns([4, 1])

            korean_name = ""

            # 미국 한글 이름 찾기
            for alias, (
                symbol,
                english_name,
            ) in US_KR_ALIAS.items():

                if symbol == ticker:
                    korean_name = alias
                    break

            # 한국 종목 이름 찾기
            if not korean_name:

                for name, symbol in KR_NAME_MAP.items():

                    if symbol == ticker:
                        korean_name = name
                        break

            label = ticker

            if korean_name:
                label += f" · {korean_name}"

            col1.write(label)

            if col2.button(
                "✕",
                key=f"remove_{i}_{ticker}",
            ):
                remove_ticker = ticker

        if remove_ticker:

            st.session_state.selected_tickers = [
                x
                for x in st.session_state.selected_tickers
                if x != remove_ticker
            ]

            st.rerun()

    else:

        st.caption(
            "현재 선택된 종목이 없습니다."
        )

    col1, col2 = st.columns(2)

    if col1.button(
        "보유종목 불러오기",
        use_container_width=True,
    ):

        st.session_state.selected_tickers = (
            MY_HOLDINGS.copy()
        )

        st.rerun()

    if col2.button(
        "전체 비우기",
        use_container_width=True,
    ):

        st.session_state.selected_tickers = []

        st.rerun()

    st.divider()

    # =========================
    # NASDAQ 추천 후보 자동 탐색
    # =========================
    st.subheader("🚀 NASDAQ 추천 후보")

    st.caption(
        "1차 기술 필터 후 실적·가치·뉴스·애널리스트·통계까지 2차 분석합니다."
    )

    if st.button(
        "NASDAQ 추천 후보 찾기",
        use_container_width=True,
        key="scan_nasdaq_recommendations",
    ):
        with st.spinner("NASDAQ 종목을 스캔하는 중입니다..."):
            st.session_state.nasdaq_recommendations = scan_nasdaq_candidates(15)

    if not st.session_state.nasdaq_recommendations.empty:
        for i, row in st.session_state.nasdaq_recommendations.head(10).iterrows():
            ticker = str(row["종목"])
            c1, c2, c3 = st.columns([5, 1, 1])

            c1.write(
                f"**{korean_ticker_name(ticker)}** · {float(row['점수']):.1f}점"
            )
            c1.caption(
                f"{row['의견']} · 기술 {float(row['기술']):.1f}/25 · 실적 {float(row['실적']):.1f}/25"
            )

            if c2.button(
                "＋",
                key=f"nasdaq_rec_add_{i}_{ticker}",
                help="현재 분석 목록에 추가",
            ):
                if ticker not in st.session_state.selected_tickers:
                    st.session_state.selected_tickers.append(ticker)
                st.rerun()

            if c3.button(
                "⭐",
                key=f"nasdaq_rec_fav_{i}_{ticker}",
                help="즐겨찾기에 추가",
            ):
                if ticker not in st.session_state.favorites:
                    st.session_state.favorites.append(ticker)
                st.rerun()


        with st.expander("추천 점수 근거 보기"):
            reason_view = st.session_state.nasdaq_recommendations[
                ["종목", "점수", "기술", "실적", "가치", "뉴스", "시장기대", "통계", "판단 근거"]
            ].copy()
            reason_view["종목"] = reason_view["종목"].map(korean_ticker_name)
            st.dataframe(reason_view, hide_index=True, use_container_width=True)

    st.divider()

    # =========================
    # 종목 검색
    # =========================

    st.subheader("🔎 종목 검색")

    nasdaq_universe = load_nasdaq_universe()
    if not nasdaq_universe.empty:
        st.caption(
            f"NASDAQ 상장 종목 {len(nasdaq_universe):,}개 자동 연결 · "
            "한글명/영문 회사명/티커 검색 가능"
        )
    else:
        st.caption(
            "NASDAQ 목록 연결 실패 · 한글 별칭/영문 회사명/티커 검색 가능"
        )

    search_query = st.text_input(
        "검색어",
        placeholder=(
            "엔비디아, NVIDIA, "
            "삼성전자, NVDA"
        ),
    )

    if st.button(
        "검색",
        use_container_width=True,
    ):

        st.session_state.search_results = (
            search_stocks(search_query)
        )

    if st.session_state.search_results:

        st.caption("검색 결과")

        for i, item in enumerate(
            st.session_state.search_results
        ):

            col1, col2, col3 = st.columns([4, 1, 1])

            exchange_text = ""

            if item["nasdaq"]:

                exchange_text = " · NASDAQ"

            elif (
                item["symbol"].endswith(".KS")
                or item["symbol"].endswith(".KQ")
            ):

                exchange_text = " · 한국"

            col1.write(
                f"**{item['symbol']}**"
                f"{exchange_text}"
            )

            col1.caption(
                item["name"]
            )

            if col2.button(
                "＋",
                key=f"add_{i}_{item['symbol']}",
            ):

                if (
                    item["symbol"]
                    not in st.session_state.selected_tickers
                ):

                    st.session_state.selected_tickers.append(
                        item["symbol"]
                    )

                st.rerun()


            if col3.button(
                "⭐",
                key=f"favorite_{i}_{item['symbol']}",
                help="즐겨찾기에 추가",
            ):
                if item["symbol"] not in st.session_state.favorites:
                    st.session_state.favorites.append(item["symbol"])
                st.rerun()

    st.divider()

    # =========================
    # 직접 추가
    # =========================

    manual = st.text_input(
        "직접 추가",
        placeholder=(
            "PLTR / 삼성전자 / "
            "005930.KS"
        ),
    )

    if st.button(
        "직접 추가",
        use_container_width=True,
    ):

        ticker = normalize_ticker(
            manual
        )

        if (
            ticker
            and ticker
            not in st.session_state.selected_tickers
        ):

            st.session_state.selected_tickers.append(
                ticker
            )

            st.rerun()

    st.divider()

    # 여기 아래부터 기존 전략 설정 코드 계속

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

    news_data = {}
    if use_news:
        with st.spinner("최근 뉴스 제목 확인 중입니다."):
            news_data = fetch_news_sentiment(tuple(screen["종목"].tolist()))
        # 뉴스 제목/라벨은 유지하되 최종 종합점수는 아래 6요소 모델에서 다시 계산합니다.
        screen = apply_news_adjustment(screen, news_data)

    # 통계 점수에 미래 분포를 활용할 수 있도록 예상수익률을 먼저 계산
    screen = add_forecasts(screen, prices)

    with st.spinner("실적·밸류·뉴스·시장기대·통계까지 종합 분석 중입니다."):
        screen = add_multifactor_scores(screen, prices, news_data)

    screen = add_buy_points(screen, cfg)
    screen = add_current_entry_judgement(screen, cfg)

    recommended = current_recommended_weights(screen, cfg.max_positions)

    # ---------- 상단 요약 ----------
    st.subheader("현재 종목 선별")

    summary_cols = st.columns(4)

    if not screen.empty:
        top = screen.iloc[0]
        summary_cols[0].metric("1위 종목", korean_ticker_name(str(top["종목"])))
        summary_cols[1].metric("최고 점수", f'{top["종합 점수"]:.1f}점')
        summary_cols[2].metric("현재가 판단", str(top["현재가 판단"]))
        summary_cols[3].metric(
            "65점 이상",
            f'{int((screen["종합 점수"] >= 65).sum())}개',
        )

    # ---------- 메인 표 ----------
    display = screen.copy()

    # 가격 문자열은 원래 티커를 사용해 먼저 생성해야 한국 종목이 원화로 표시됨
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

    # 화면에 보이는 종목명만 한글명 + 티커로 변경
    display["종목"] = display["종목"].map(korean_ticker_name)

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
                "기술/차트 점수",
                "기업실적 점수",
                "밸류에이션 점수",
                "뉴스/이벤트 점수",
                "시장기대 점수",
                "통계/확률 점수",
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
        column_config={
            "종목": st.column_config.TextColumn("종목", width="large"),
            "종합 점수": st.column_config.NumberColumn(
                "종합 점수", format="%.1f", width="small"
            ),
            "현재가 적정도": st.column_config.NumberColumn(
                "현재가 적정도", format="%.1f", width="small"
            ),
            "현재가 판단": st.column_config.TextColumn(
                "현재가 판단", width="medium"
            ),
            "현재가": st.column_config.TextColumn("현재가", width="medium"),
            "1차 매수": st.column_config.TextColumn("1차 매수", width="medium"),
            "2차 매수": st.column_config.TextColumn("2차 매수", width="medium"),
            "3차 매수": st.column_config.TextColumn("3차 매수", width="medium"),
        },
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("왜 이 점수인지 쉽게 보기"):
        explain_view = screen[
            [
                "종목",
                "종합 점수",
                "종합 판단 근거",
                "실적 근거",
                "가치 근거",
                "뉴스 근거",
                "시장기대 근거",
                "통계 근거",
            ]
        ].copy()
        explain_view["종목"] = explain_view["종목"].map(korean_ticker_name)
        st.dataframe(explain_view, use_container_width=True, hide_index=True)

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

    # 가격 표시 후 종목명을 한글로 변경
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
    entry_view["종목"] = entry_view["종목"].map(korean_ticker_name)

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
        ],
        column_config={
            "종목": st.column_config.TextColumn("종목", width="large"),
            "종합 점수": st.column_config.NumberColumn(
                "종합 점수", format="%.1f", width="small"
            ),
            "현재가 적정도": st.column_config.NumberColumn(
                "현재가 적정도", format="%.1f", width="small"
            ),
            "현재가 판단": st.column_config.TextColumn(
                "현재가 판단", width="medium"
            ),
            "현재가": st.column_config.TextColumn("현재가", width="medium"),
            "1차": st.column_config.TextColumn("1차 매수", width="medium"),
            "2차": st.column_config.TextColumn("2차 매수", width="medium"),
            "3차": st.column_config.TextColumn("3차 매수", width="medium"),
            "현재가 비중": st.column_config.NumberColumn(
                "현재가 비중", format="%.0f%%"
            ),
            "1차 비중": st.column_config.NumberColumn(
                "1차 비중", format="%.0f%%"
            ),
            "2차 비중": st.column_config.NumberColumn(
                "2차 비중", format="%.0f%%"
            ),
            "3차 비중": st.column_config.NumberColumn(
                "3차 비중", format="%.0f%%"
            ),
        },
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

    if not forecast_df.empty:
        forecast_df["종목"] = forecast_df["종목"].map(korean_ticker_name)

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
        recommended_display = recommended.copy()
        recommended_display["종목"] = recommended_display["종목"].map(korean_ticker_name)

        st.dataframe(
            recommended_display.style.format({"추천 비중": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
        )

    # ---------- 세부 분석 ----------
    with st.expander("점수 구성 / 뉴스 / 세부 지표 보기"):
        detail = screen.copy()
        detail["종목"] = detail["종목"].map(korean_ticker_name)

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

        ### 최종 종합 점수 구성
        - 기술/차트 25점
        - 기업 실적·재무 25점
        - 밸류에이션 15점
        - 뉴스·이벤트 15점
        - 애널리스트·시장 기대 10점
        - 과거 유사구간 통계·확률 10점

        **NASDAQ 추천은 1차로 차트가 강한 종목을 걸러낸 뒤, 위 6개 요소로 2차 심층 분석합니다.**
        """
    )
