from __future__ import annotations
import os
import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---- Inlined from indicators.py ----

def rsi(close: pd.Series, period: int=14) -> pd.Series:
    delta=close.diff()
    gain=delta.clip(lower=0).ewm(alpha=1/period,min_periods=period,adjust=False).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/period,min_periods=period,adjust=False).mean()
    rs=gain/loss.replace(0,float("nan"))
    return 100-100/(1+rs)

def atr(df: pd.DataFrame, period: int=14) -> pd.Series:
    prev=df.Close.shift(1)
    tr=pd.concat([df.High-df.Low,(df.High-prev).abs(),(df.Low-prev).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/period,min_periods=period,adjust=False).mean()

def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df=data.copy()
    for p in (20,50,200):
        df[f"EMA{p}"]=df.Close.ewm(span=p,adjust=False,min_periods=p).mean()
    df["RSI"]=rsi(df.Close)
    e12=df.Close.ewm(span=12,adjust=False).mean()
    e26=df.Close.ewm(span=26,adjust=False).mean()
    df["MACD"]=e12-e26
    df["MACDSignal"]=df.MACD.ewm(span=9,adjust=False).mean()
    df["ATR"]=atr(df)
    plus_move=df.High.diff()
    minus_move=-df.Low.diff()
    plus=plus_move.where((plus_move>minus_move)&(plus_move>0),0.0).ewm(alpha=1/14,adjust=False).mean()
    minus=minus_move.where((minus_move>plus_move)&(minus_move>0),0.0).ewm(alpha=1/14,adjust=False).mean()
    safe_atr=df.ATR.replace(0,float("nan"))
    pdi=100*plus/safe_atr
    mdi=100*minus/safe_atr
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,float("nan"))
    df["ADX"]=dx.ewm(alpha=1/14,adjust=False).mean()
    return df

# ---- Inlined from structure.py ----

def add_market_structure(data: pd.DataFrame, strength: int=3) -> pd.DataFrame:
    """Pivot values are exposed on the confirmation bar, never backdated to pivot bar."""
    if strength<1:
        raise ValueError("strength must be at least 1")
    df=data.copy()
    n=len(df)
    highs=df.High.to_numpy()
    lows=df.Low.to_numpy()
    closes=df.Close.to_numpy()
    pivot_high=np.full(n,np.nan)
    pivot_low=np.full(n,np.nan)
    sweep_high=np.zeros(n,dtype=bool)
    sweep_low=np.zeros(n,dtype=bool)
    events=[""]*n
    last_high=np.nan
    last_low=np.nan
    trend=0
    for i in range(n):
        p=i-strength
        if p>=strength:
            wh=highs[p-strength:p+strength+1]
            wl=lows[p-strength:p+strength+1]
            if highs[p]==np.max(wh) and np.sum(wh==highs[p])==1:
                pivot_high[i]=highs[p]
                last_high=highs[p]
            if lows[p]==np.min(wl) and np.sum(wl==lows[p])==1:
                pivot_low[i]=lows[p]
                last_low=lows[p]
        # The current candle may sweep only a swing already confirmed before it.
        prior_high=last_high
        prior_low=last_low
        if np.isfinite(prior_high) and highs[i]>prior_high and closes[i]<prior_high:
            sweep_high[i]=True
        if np.isfinite(prior_low) and lows[i]<prior_low and closes[i]>prior_low:
            sweep_low[i]=True
        if np.isfinite(prior_high) and closes[i]>prior_high:
            events[i]="BOS↑" if trend>=0 else "CHoCH↑"
            trend=1
        elif np.isfinite(prior_low) and closes[i]<prior_low:
            events[i]="BOS↓" if trend<=0 else "CHoCH↓"
            trend=-1
    df["PivotHigh"]=pivot_high
    df["PivotLow"]=pivot_low
    df["LiquiditySweepHigh"]=sweep_high
    df["LiquiditySweepLow"]=sweep_low
    df["StructureEvent"]=events
    df["LastSwingHigh"]=pd.Series(pivot_high,index=df.index).ffill()
    df["LastSwingLow"]=pd.Series(pivot_low,index=df.index).ffill()
    return df

# ---- Inlined from strategy.py ----

def _bias(frame):
    row=frame.iloc[-1]
    if pd.isna(row.EMA50) or pd.isna(row.EMA200):
        return "NEUTRAL"
    if row.Close>row.EMA50>row.EMA200: return "BULLISH"
    if row.Close<row.EMA50<row.EMA200: return "BEARISH"
    return "NEUTRAL"

def analyse_market(base, higher, profile):
    r=base.iloc[-1]
    htf_bias=_bias(higher)
    votes=[]
    reasons=[]
    def vote(name,observation,value,weight):
        votes.append((name,observation,"Bullish" if value>0 else "Bearish" if value<0 else "Neutral",weight))
    ema_vote=1 if r.Close>r.EMA20>r.EMA50 else -1 if r.Close<r.EMA20<r.EMA50 else 0
    vote("EMA structure","EMA20 / EMA50 alignment",ema_vote,1.5)
    higher_vote=1 if htf_bias=="BULLISH" else -1 if htf_bias=="BEARISH" else 0
    vote("Higher timeframe",htf_bias,higher_vote,2.5)
    if profile=="Reversal / liquidity sweep":
        momentum=1 if r.LiquiditySweepLow else -1 if r.LiquiditySweepHigh else 0
        momentum_read="Sweep reversal filter"
    else:
        momentum=1 if pd.notna(r.RSI) and 50<r.RSI<70 else -1 if pd.notna(r.RSI) and 30<r.RSI<50 else 0
        momentum_read=f"RSI {r.RSI:.1f}" if pd.notna(r.RSI) else "RSI unavailable"
    vote("Momentum",momentum_read,momentum,1.0)
    macd_vote=1 if r.MACD>r.MACDSignal else -1 if r.MACD<r.MACDSignal else 0
    vote("MACD","Above signal" if macd_vote>0 else "Below signal" if macd_vote<0 else "Flat",macd_vote,.75)
    sweep=1 if r.LiquiditySweepLow else -1 if r.LiquiditySweepHigh else 0
    vote("Liquidity sweep","Sell-side sweep" if sweep>0 else "Buy-side sweep" if sweep<0 else "No confirmed sweep",sweep,1.5)
    event=str(r.StructureEvent)
    struct_vote=1 if "↑" in event else -1 if "↓" in event else 0
    vote("Structure",event or "No fresh break",struct_vote,1.5)
    if profile=="Trend continuation":
        votes.append(("Trend regime","ADX "+(f"{r.ADX:.1f}" if pd.notna(r.ADX) else "n/a"),
                      "Bullish" if ema_vote>0 and pd.notna(r.ADX) and r.ADX>=20 else "Bearish" if ema_vote<0 and pd.notna(r.ADX) and r.ADX>=20 else "Neutral",1.0))
    raw=sum((1 if v[2]=="Bullish" else -1 if v[2]=="Bearish" else 0)*v[3] for v in votes)
    denom=sum(v[3] for v in votes)
    score=100*raw/denom if denom else 0
    direction="BULLISH" if score>=25 else "BEARISH" if score<=-25 else "NEUTRAL"
    aligned=sum(1 for v in votes if (v[2]=="Bullish" and score>0) or (v[2]=="Bearish" and score<0))
    confidence=round(100*aligned/len(votes))
    if htf_bias!="NEUTRAL": reasons.append(f"Higher timeframe EMA regime is {htf_bias.lower()}.")
    if ema_vote: reasons.append("Execution timeframe EMA structure is aligned.")
    if bool(r.LiquiditySweepHigh): reasons.append("Price swept above a confirmed swing high and closed back below it.")
    if bool(r.LiquiditySweepLow): reasons.append("Price swept below a confirmed swing low and closed back above it.")
    if event: reasons.append(f"Latest structure break: {event}.")
    if direction=="NEUTRAL": reasons.append("Confluence is below threshold. Waiting is a valid outcome.")
    if not reasons: reasons.append("Evidence is mixed; no strong directional edge is present.")
    return {"direction":direction,"score":round(score),"confidence":confidence,"htf_bias":htf_bias,
            "structure":event or "No fresh break","components":votes,"reasons":reasons[:6]}

def build_trade_plan(frame, analysis, rr=2.5):
    row=frame.iloc[-1]
    side=analysis["direction"]
    if side=="NEUTRAL" or pd.isna(row.ATR) or row.ATR<=0:
        return {"entry":None,"stop":None,"target":None,"distance":None,"side":side,
                "status":"No plan: directional agreement or ATR data is insufficient."}
    entry=float(row.Close)
    atr=float(row.ATR)
    if side=="BULLISH":
        swing=float(row.LastSwingLow) if pd.notna(row.LastSwingLow) else entry-1.5*atr
        stop=min(swing,entry-.5*atr)
        if entry-stop>3*atr: stop=entry-1.5*atr
        distance=entry-stop
        target=entry+rr*distance
    else:
        swing=float(row.LastSwingHigh) if pd.notna(row.LastSwingHigh) else entry+1.5*atr
        stop=max(swing,entry+.5*atr)
        if stop-entry>3*atr: stop=entry+1.5*atr
        distance=stop-entry
        target=entry-rr*distance
    return {"entry":entry,"stop":float(stop),"target":float(target),"distance":float(distance),
            "side":"LONG" if side=="BULLISH" else "SHORT",
            "status":"Reference levels from latest candle close; require a separate trigger and broker quote check."}

# ---- Inlined from backtest.py ----

def run_backtest(base,higher,profile,risk_fraction,rr,starting_balance,cost_fraction=0.0005):
    """Simplified OHLC backtest. Signals at close, entry next open, one position, stop-first ambiguity."""
    if not 0<risk_fraction<=0.05:
        raise ValueError("risk_fraction must be greater than 0 and no more than 5%.")
    if starting_balance<=0:
        raise ValueError("starting_balance must be positive.")
    if rr<=0 or cost_fraction<0:
        raise ValueError("rr must be positive and cost_fraction non-negative.")
    cash=float(starting_balance)
    position=None
    trades=[]
    equity=[]
    higher= higher.sort_index()
    for i in range(205,len(base)-1):
        signal_time=base.index[i]
        h_available=higher.loc[:signal_time]
        if len(h_available)<205:
            equity.append((base.index[i+1],cash))
            continue
        signal=analyse_market(base.iloc[:i+1],h_available,profile)["direction"]
        current=base.iloc[i]
        nxt=base.iloc[i+1]
        exited_this_bar = False
        if position is not None:
            exit_price=None
            reason=None
            if position["side"]=="LONG":
                if nxt.Low<=position["stop"]:
                    exit_price=position["stop"]; reason="Stop"
                elif nxt.High>=position["target"]:
                    exit_price=position["target"]; reason="Target"
            else:
                if nxt.High>=position["stop"]:
                    exit_price=position["stop"]; reason="Stop"
                elif nxt.Low<=position["target"]:
                    exit_price=position["target"]; reason="Target"
            if exit_price is not None:
                direction=1 if position["side"]=="LONG" else -1
                gross=(exit_price-position["entry"])*position["qty"]*direction
                costs=(abs(position["entry"]*position["qty"])+abs(exit_price*position["qty"]))*cost_fraction
                pnl=gross-costs
                cash+=pnl
                trades.append({"Entry time":position["time"],"Exit time":base.index[i+1],"Side":position["side"],
                    "Entry":position["entry"],"Exit":exit_price,"Stop":position["stop"],"Target":position["target"],
                    "P&L":pnl,"R multiple":pnl/max(position["risk_cash"],1e-12),"Exit reason":reason})
                position=None
                exited_this_bar = True
        if position is None and not exited_this_bar and signal in ("BULLISH","BEARISH"):
            atr=current.ATR
            if pd.notna(atr) and atr>0:
                entry=float(nxt.Open)
                long=signal=="BULLISH"
                if long:
                    swing=float(current.LastSwingLow) if pd.notna(current.LastSwingLow) else entry-1.5*atr
                    stop=min(swing,entry-.5*atr)
                    if entry-stop>3*atr: stop=entry-1.5*atr
                    distance=entry-stop
                    target=entry+rr*distance
                else:
                    swing=float(current.LastSwingHigh) if pd.notna(current.LastSwingHigh) else entry+1.5*atr
                    stop=max(swing,entry+.5*atr)
                    if stop-entry>3*atr: stop=entry+1.5*atr
                    distance=stop-entry
                    target=entry-rr*distance
                risk_cash=max(cash,0)*risk_fraction
                qty=risk_cash/max(distance,1e-12)
                position={"side":"LONG" if long else "SHORT","entry":entry,"stop":float(stop),
                          "target":float(target),"qty":qty,"risk_cash":risk_cash,"time":base.index[i+1]}
        equity.append((base.index[i+1],cash))
    # Do not silently count an open trade as closed: report it separately.
    open_trade=None
    if position:
        mark=float(base.iloc[-1].Close)
        direction=1 if position["side"]=="LONG" else -1
        unrealised=(mark-position["entry"])*position["qty"]*direction
        open_trade={"side":position["side"],"entry":position["entry"],"mark":mark,"unrealised_pnl":unrealised}
    t=pd.DataFrame(trades)
    eq=pd.Series(dict(equity),dtype=float).sort_index()
    if eq.empty: eq=pd.Series([starting_balance],index=[base.index[-1]])
    if not t.empty:
        wins=t.loc[t["P&L"]>0,"P&L"].sum()
        losses=abs(t.loc[t["P&L"]<0,"P&L"].sum())
        pf=wins/losses if losses>0 else (float("inf") if wins>0 else 0.0)
        win_rate=100*(t["P&L"]>0).mean()
    else:
        pf=0.0; win_rate=0.0
    peak=eq.cummax()
    drawdown=(eq/peak-1)*100
    stats={"return_pct":(cash/starting_balance-1)*100,"win_rate":win_rate,
           "profit_factor":pf,"max_drawdown":abs(float(drawdown.min())) if len(drawdown) else 0.0,
           "trades":len(t),"ending_balance":cash}
    return {"stats":stats,"trades":t,"equity":eq,"open_trade":open_trade}

# ---- Inlined from market_data.py ----

def fetch_market_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    import yfinance as yf  # Lazy import keeps indicator/backtest tests independent of network-feed package.
    frame=yf.Ticker(symbol).history(period=period,interval=interval,auto_adjust=False,actions=False)
    if frame is None or frame.empty:
        raise ValueError("No candles returned by Yahoo Finance.")
    frame=frame.rename(columns=lambda c:str(c).title())
    required=["Open","High","Low","Close","Volume"]
    missing=[c for c in required if c not in frame.columns]
    if missing:
        raise ValueError("Missing market columns: "+", ".join(missing))
    frame=frame[required].copy()
    frame=frame.dropna(subset=["Open","High","Low","Close"])
    frame=frame[~frame.index.duplicated(keep="last")].sort_index()
    if len(frame)<5:
        raise ValueError("Market provider returned too few candles.")
    return frame

def resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    return frame.resample(rule).agg({
        "Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"
    }).dropna(subset=["Open","High","Low","Close"])

def feed_label(symbol: str) -> str:
    return "Yahoo Finance · gold spot reference" if symbol=="XAUUSD=X" else "Yahoo Finance · COMEX futures" if symbol=="GC=F" else "Yahoo Finance"

# ---- Inlined from events.py ----

def get_event_calendar():
    """
    Optional Trading Economics calendar.
    Configure TRADING_ECONOMICS_API_KEY in environment or Streamlit secrets.
    A missing key returns an empty DataFrame and is explicitly shown as unavailable in UI.
    """
    key=os.getenv("TRADING_ECONOMICS_API_KEY","")
    try:
        import streamlit as st
        key=st.secrets.get("TRADING_ECONOMICS_API_KEY",key)
    except Exception:
        pass
    if not key:
        return pd.DataFrame()
    try:
        import requests
        url="https://api.tradingeconomics.com/calendar"
        response=requests.get(url,params={"c":key,"country":"united states","importance":3},timeout=12)
        response.raise_for_status()
        payload=response.json()
        df=pd.DataFrame(payload)
        if df.empty:
            return df
        keep=[c for c in ["Date","Country","Category","Event","Actual","Previous","Forecast","Importance"] if c in df.columns]
        df=df[keep].copy()
        if "Date" in df:
            df["Date"]=pd.to_datetime(df["Date"],errors="coerce",utc=True)
            df=df.sort_values("Date")
        return df
    except Exception:
        return pd.DataFrame()

def event_risk_summary(events):
    if events.empty:
        return "Economic calendar unavailable — no API key or no provider response. Do not interpret this as no scheduled risk."
    return f"{len(events)} calendar entries returned. Review event times and impact before trading."

# ---- Inlined from risk_tools.py ----


def calculate_position_size(
    balance: float,
    risk_percent: float,
    entry: float,
    stop: float,
    account_currency_per_dollar_per_lot: float,
) -> dict[str, float]:
    """Calculate an illustrative lot size using a broker-supplied P/L multiplier.

    The multiplier is the account-currency P/L for a $1 move in the instrument
    at 1.00 lot. It must come from the user's broker contract specification.
    """
    values = [balance, risk_percent, entry, stop, account_currency_per_dollar_per_lot]
    if any(v is None for v in values):
        raise ValueError("All position-sizing inputs are required.")
    if balance <= 0:
        raise ValueError("Balance must be positive.")
    if not 0 < risk_percent <= 5:
        raise ValueError("Risk must be greater than 0% and no more than 5%.")
    if entry <= 0 or stop <= 0:
        raise ValueError("Entry and stop must be positive prices.")
    if entry == stop:
        raise ValueError("Entry and stop cannot be the same price.")
    if account_currency_per_dollar_per_lot <= 0:
        raise ValueError("Broker P/L multiplier must be positive.")
    risk_cash = balance * risk_percent / 100.0
    stop_distance = abs(entry - stop)
    lots = risk_cash / (stop_distance * account_currency_per_dollar_per_lot)
    return {
        "risk_cash": risk_cash,
        "stop_distance": stop_distance,
        "lots": lots,
        "daily_cash_at_2pct": balance * 0.02,
    }

# ---- CART3R user interface ----

from datetime import datetime, timezone
import math
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


st.set_page_config(
    page_title="CART3R | Gold Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--bg:#080b10;--panel:#111720;--line:#263140;--muted:#94a0b1;--red:#ef586b;--green:#3bc995;--amber:#e7bd70}
html,body,[class*="css"]{font-family:'Manrope',sans-serif}
.stApp{background:var(--bg);color:#eaf0f7}
.block-container{padding-top:.8rem;padding-bottom:2rem;max-width:1450px}
section[data-testid="stSidebar"]{background:#0d121a;border-right:1px solid var(--line)}
.metric{background:linear-gradient(145deg,#151d28,#10151d);border:1px solid var(--line);border-radius:13px;padding:12px 14px;min-height:86px}
.metric .k{font-size:10px;color:var(--muted);letter-spacing:.11em;text-transform:uppercase}
.metric .v{font-family:'DM Mono',monospace;font-size:clamp(17px,2vw,24px);margin-top:7px;overflow-wrap:anywhere}
.metric .s{font-size:11px;color:var(--muted);margin-top:3px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:15px}
.up{color:var(--green);font-weight:800}.down{color:var(--red);font-weight:800}.flat{color:var(--amber);font-weight:800}
.muted{font-size:11px;color:var(--muted)}
.stButton button{border-radius:10px;min-height:44px}
[data-testid="stMetric"]{background:#111720;border:1px solid #263140;padding:10px;border-radius:10px}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:9px}
div[data-testid="stAlert"]{border-radius:12px}
@media(max-width:700px){
 .block-container{padding-left:.65rem;padding-right:.65rem}
 h1{font-size:1.42rem!important} h2{font-size:1.2rem!important} h3{font-size:1.03rem!important}
 .metric{padding:9px;min-height:82px}.metric .k{font-size:9px}.metric .v{font-size:17px}
 [data-testid="stHorizontalBlock"]{gap:.4rem}
 button{width:100%}
 [data-testid="stPlotlyChart"]{overflow:hidden}
}
</style>
""", unsafe_allow_html=True)


def fmt(value, places=2):
    try:
        return "—" if pd.isna(value) else f"{float(value):,.{places}f}"
    except (TypeError, ValueError):
        return "—"


def metric(label, value, sub=""):
    st.markdown(
        f'<div class="metric"><div class="k">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>',
        unsafe_allow_html=True,
    )


st.markdown("## ◈ CART3R <span style='color:#ef586b'>/</span> GOLD INTELLIGENCE", unsafe_allow_html=True)
st.caption("FINAL BUILD · PHONE-FIRST · MULTI-TIMEFRAME · RISK TOOLS · BACKTEST · JOURNAL")
st.info("Start with **OVERVIEW**. Check the market feed, read the bias, then inspect the chart and risk before considering any trade. CART3R never places orders.")

# Keep widget values in session state so quick-pick buttons work reliably on mobile.
st.session_state.setdefault("cart3r_symbol", "XAUUSD=X")
st.session_state.setdefault("cart3r_profile", "Hybrid institutional")
st.session_state.setdefault("cart3r_execution_tf", "15m")
st.session_state.setdefault("cart3r_confirmation_tf", "4h")
st.session_state.setdefault("cart3r_rr", 2.5)
st.session_state.setdefault("cart3r_pivot", 3)
st.session_state.setdefault("cart3r_risk", 0.5)
st.session_state.setdefault("cart3r_journal", [])

st.markdown("**Quick market**")
quick_cols = st.columns(4)
for col, (label, symbol_value) in zip(quick_cols, [("GOLD", "XAUUSD=X"), ("GOLD FUTURES", "GC=F"), ("EUR/USD", "EURUSD=X"), ("BITCOIN", "BTC-USD")]):
    with col:
        if st.button(label, key=f"pick_{symbol_value}", use_container_width=True):
            st.session_state.cart3r_symbol = symbol_value
            st.rerun()

with st.expander("⚙️ MARKET SETTINGS", expanded=False):
    st.text_input("Market symbol", key="cart3r_symbol", help="Examples: XAUUSD=X (gold reference), GC=F (COMEX gold futures), EURUSD=X, GBPUSD=X, BTC-USD.")
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Execution timeframe", ["5m", "15m", "30m", "1h", "4h", "1d"], key="cart3r_execution_tf")
        st.selectbox("Analysis profile", ["Hybrid institutional", "Trend continuation", "Reversal / liquidity sweep"], key="cart3r_profile")
    with c2:
        st.selectbox("Confirmation timeframe", ["15m", "30m", "1h", "4h", "1d"], key="cart3r_confirmation_tf")
        st.selectbox("Target multiple", [1.5, 2.0, 2.5, 3.0, 4.0], key="cart3r_rr")
    st.slider("Swing pivot sensitivity", 2, 8, key="cart3r_pivot", help="Higher values identify larger, slower swings.")
    st.number_input("Default risk per trade (%)", min_value=0.1, max_value=5.0, step=0.1, key="cart3r_risk")
    refresh = st.button("↻ Refresh market data", type="primary", use_container_width=True)

symbol = st.session_state.cart3r_symbol.strip().upper()
execution_tf = st.session_state.cart3r_execution_tf
confirmation_tf = st.session_state.cart3r_confirmation_tf
profile = st.session_state.cart3r_profile
rr = float(st.session_state.cart3r_rr)
pivot_strength = int(st.session_state.cart3r_pivot)
risk_pct = float(st.session_state.cart3r_risk)

periods = {"5m": "5d", "15m": "10d", "30m": "30d", "1h": "60d", "4h": "730d", "1d": "5y"}
fetch_intervals = {"5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "1h", "1d": "1d"}

@st.cache_data(ttl=45, show_spinner=False)
def load_data(sym, exec_tf, conf_tf, force_key):
    base = fetch_market_data(sym, periods[exec_tf], fetch_intervals[exec_tf])
    higher = fetch_market_data(sym, periods[conf_tf], fetch_intervals[conf_tf])
    if exec_tf == "4h":
        base = resample_ohlcv(base, "4h")
    if conf_tf == "4h":
        higher = resample_ohlcv(higher, "4h")
    return base, higher

try:
    if "refresh" in locals() and refresh:
        load_data.clear()
    with st.spinner(f"Getting {symbol} candles…"):
        raw, higher_raw = load_data(symbol, execution_tf, confirmation_tf, datetime.now(timezone.utc).isoformat() if "refresh" in locals() and refresh else "cached")
        base = add_market_structure(add_indicators(raw), pivot_strength)
        higher = add_market_structure(add_indicators(higher_raw), pivot_strength)
except Exception as exc:
    st.error("The market feed did not return usable candles.")
    st.write(f"Technical detail: `{type(exc).__name__}: {exc}`")
    st.markdown("Try **GOLD** or **GOLD FUTURES**, select 15m / 1h / 4h, then refresh. Yahoo Finance can rate-limit or temporarily stop returning intraday data.")
    st.stop()

if len(base) < 220 or len(higher) < 205:
    st.warning("Not enough history for the full EMA200 and structure model. Choose a higher timeframe or another symbol.")
    st.stop()

analysis = analyse_market(base, higher, profile)
plan = build_trade_plan(base, analysis, rr)
last = base.iloc[-1]
previous = base.iloc[-2]
change = (float(last.Close) / float(previous.Close) - 1) * 100 if previous.Close else 0.0
bias_class = "up" if analysis["direction"] == "BULLISH" else "down" if analysis["direction"] == "BEARISH" else "flat"
event_df = get_event_calendar()
event_summary = event_risk_summary(event_df)

st.markdown(f"### {symbol} <span class='muted'>· {execution_tf} / {confirmation_tf}</span>", unsafe_allow_html=True)
st.markdown(f"<span class='muted'>Latest provider candle: {base.index[-1]} · {feed_label(symbol)} · Auto-refresh cache: 45 sec</span>", unsafe_allow_html=True)

cards = st.columns(2)
with cards[0]: metric("Last price", fmt(last.Close), f"{change:+.2f}% vs previous candle")
with cards[1]: metric("Model bias", analysis["direction"], f"Agreement score {analysis['confidence']}/100")
cards2 = st.columns(2)
with cards2[0]: metric("Higher timeframe", analysis["htf_bias"], "Trend filter")
with cards2[1]: metric("RSI / ADX", f"{fmt(last.RSI, 1)} / {fmt(last.ADX, 1)}", "Momentum / trend strength")

section = st.selectbox(
    "GO TO SECTION",
    ["OVERVIEW", "CHART", "SIGNAL & PLAN", "RISK CALCULATOR", "BACKTEST", "NEWS & EVENTS", "TRADE JOURNAL", "LEARN / HOW TO USE"],
    key="cart3r_section",
    help="Use this menu to move around. It is easier than narrow tabs on an iPhone.",
)

if section == "OVERVIEW":
    st.markdown("### Your market snapshot")
    st.markdown(f"<div class='{bias_class}' style='font-size:27px'>{analysis['direction']}</div>", unsafe_allow_html=True)
    st.progress(analysis["confidence"] / 100)
    st.caption("Agreement score = how many model components point in the same direction. It is NOT the chance of winning a trade.")
    st.markdown("**What to check next**")
    st.write("1. Open **CHART** and look at the most recent price structure.")
    st.write("2. Open **SIGNAL & PLAN** and read why the model has this bias.")
    st.write("3. Open **RISK CALCULATOR** and check the size before risking money.")
    if analysis["direction"] == "NEUTRAL":
        st.warning("No clear directional agreement. Waiting is a valid decision; the model does not need to produce a trade every time.")
    elif plan["entry"] is not None:
        st.info(f"Reference levels only · Entry {fmt(plan['entry'])} · Stop {fmt(plan['stop'])} · Target {fmt(plan['target'])}. Wait for your own entry trigger and check your broker quote.")
    st.markdown("**Current model reasons**")
    for reason in analysis["reasons"]:
        st.write("• " + reason)
    st.caption("Data can be delayed, incomplete or different from your broker. The latest intraday candle may still be forming.")

elif section == "CHART":
    st.markdown("### Price action & indicators")
    candle_count = st.select_slider("Candles to display", options=[80, 120, 180, 240, 320, 450], value=180)
    view = base.tail(candle_count)
    view_plan = plan if plan["entry"] is not None else None
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=view.index, open=view.Open, high=view.High, low=view.Low, close=view.Close, name="Price", increasing_line_color="#3bc995", decreasing_line_color="#ef586b"))
    for column, color in [("EMA20", "#e7bd70"), ("EMA50", "#82aaff"), ("EMA200", "#bd91ef")]:
        if view[column].notna().any():
            fig.add_trace(go.Scatter(x=view.index, y=view[column], name=column, line=dict(color=color, width=1.35)))
    highs = view[view.PivotHigh.notna()]
    lows = view[view.PivotLow.notna()]
    fig.add_trace(go.Scatter(x=highs.index, y=highs.PivotHigh, mode="markers", name="Confirmed swing high", marker=dict(color="#ef586b", size=7, symbol="triangle-down")))
    fig.add_trace(go.Scatter(x=lows.index, y=lows.PivotLow, mode="markers", name="Confirmed swing low", marker=dict(color="#3bc995", size=7, symbol="triangle-up")))
    if view_plan:
        for value, label, color in [(plan["entry"], "Reference entry", "#e7bd70"), (plan["stop"], "Stop", "#ef586b"), (plan["target"], "Target", "#3bc995")]:
            fig.add_hline(y=value, line_dash="dot", line_color=color, annotation_text=label)
    fig.update_layout(height=470, template="plotly_dark", paper_bgcolor="#111720", plot_bgcolor="#111720", margin=dict(l=4, r=4, t=20, b=5), xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.02, x=0), font=dict(family="DM Mono", size=10, color="#aab4c4"), autosize=True)
    fig.update_xaxes(gridcolor="#263140")
    fig.update_yaxes(gridcolor="#263140")
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True, "responsive": True})
    st.caption("Pinch to zoom where supported. Swipe/drag to move through the chart. EMA = exponential moving average.")
    vol = go.Figure(go.Bar(x=view.index, y=view.Volume, name="Volume", marker_color="#52647d"))
    vol.update_layout(height=170, template="plotly_dark", paper_bgcolor="#111720", plot_bgcolor="#111720", margin=dict(l=4, r=4, t=8, b=5))
    st.plotly_chart(vol, use_container_width=True, config={"displaylogo": False, "responsive": True})
    st.caption("Yahoo Finance reference data is not your broker's executable bid/ask feed.")

elif section == "SIGNAL & PLAN":
    st.markdown("### Directional analysis")
    st.markdown(f"<div class='{bias_class}' style='font-size:30px'>{analysis['direction']}</div>", unsafe_allow_html=True)
    st.markdown(f"**Component agreement: {analysis['confidence']} / 100**")
    st.progress(analysis["confidence"] / 100)
    st.caption("This is a model agreement score, not a probability, forecast guarantee or trade signal by itself.")
    st.markdown("#### Why it leans this way")
    for reason in analysis["reasons"]:
        st.write("• " + reason)
    st.markdown("#### Reference trade plan")
    if plan["entry"] is None:
        st.warning(plan["status"])
    else:
        pcols = st.columns(2)
        with pcols[0]:
            metric("Side", plan["side"])
            metric("Reference entry", fmt(plan["entry"]))
        with pcols[1]:
            metric("Stop / invalidation", fmt(plan["stop"]))
            metric(f"Target ({rr}R)", fmt(plan["target"]))
        st.info(plan["status"])
    st.markdown("#### Model components")
    components = pd.DataFrame(analysis["components"], columns=["Factor", "What it sees", "Vote", "Weight"])
    st.dataframe(components, use_container_width=True, hide_index=True)

elif section == "RISK CALCULATOR":
    st.markdown("### Position size & daily risk")
    st.write("This calculator does not know your broker's contract size. You must enter the cash value your broker gives for a $1 move in gold at 1.00 lot.")
    defaults_entry = float(plan["entry"]) if plan["entry"] is not None else float(last.Close)
    defaults_stop = float(plan["stop"]) if plan["stop"] is not None else defaults_entry - float(last.ATR if pd.notna(last.ATR) else 1.0)
    r1, r2 = st.columns(2)
    with r1:
        balance = st.number_input("Account balance (account currency)", min_value=1.0, value=1000.0, step=100.0)
        risk_input = st.number_input("Risk per trade (%)", min_value=0.1, max_value=5.0, value=risk_pct, step=0.1)
        daily_limit = st.number_input("Maximum daily loss (%)", min_value=0.1, max_value=20.0, value=2.0, step=0.1)
    with r2:
        entry_input = st.number_input("Entry price", min_value=0.000001, value=max(defaults_entry, 0.000001), format="%.5f")
        stop_input = st.number_input("Stop price", min_value=0.000001, value=max(defaults_stop, 0.000001), format="%.5f")
        cash_per_dollar = st.number_input("Account-currency P/L for a $1 gold move at 1.00 lot", min_value=0.0, value=0.0, step=1.0, help="Enter the exact value from your broker's XAUUSD contract specification. Zero means it has not been confirmed yet.")
    try:
        sizing = calculate_position_size(balance, risk_input, entry_input, stop_input, cash_per_dollar)
        risk_cash = sizing["risk_cash"]
        distance = sizing["stop_distance"]
        lots = sizing["lots"]
        daily_cash = balance * daily_limit / 100
        st.markdown("#### Your sizing result")
        out = st.columns(2)
        with out[0]: metric("Cash at risk", f"{risk_cash:,.2f}", f"{risk_input:.2f}% of balance")
        with out[1]: metric("Illustrative position size", f"{lots:.3f} lots", "Depends on the contract multiplier you entered")
        out2 = st.columns(2)
        with out2[0]: metric("Stop distance", fmt(distance, 3), "Gold price units")
        with out2[1]: metric("Daily loss limit", f"{daily_cash:,.2f}", f"{daily_limit:.2f}% of balance")
        st.caption(f"At this risk setting, the daily limit equals approximately {daily_cash / risk_cash:.1f} full-risk losses. Real losses can exceed the estimate due to slippage, gaps and fees.")
        st.warning("Verify the lot calculation with your broker's contract specification and demo account before using it live. The displayed size is not an instruction to trade.")
    except ValueError as exc:
        st.error(str(exc))

elif section == "BACKTEST":
    st.markdown("### Historical strategy test")
    st.caption("Completed-bar signal → next-bar-open entry. If stop and target are both touched in one candle, the test assumes the stop happened first.")
    a, b, c = st.columns(2)
    with a:
        starting_balance = st.number_input("Starting balance", min_value=100.0, value=10000.0, step=500.0, key="bt_balance")
        backtest_risk = st.number_input("Risk per trade (%)", min_value=0.1, max_value=5.0, value=risk_pct, step=0.1, key="bt_risk")
    with b:
        costs_bps = st.number_input("Estimated round-trip cost (basis points)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
        st.write(f"Test sample: {len(base):,} execution candles")
    if st.button("▶ RUN BACKTEST", type="primary", use_container_width=True):
        try:
            st.session_state.cart3r_backtest = run_backtest(base, higher, profile, backtest_risk / 100, rr, starting_balance, costs_bps / 10000)
            st.session_state.cart3r_backtest_config = {"symbol": symbol, "timeframe": execution_tf, "confirmation": confirmation_tf, "profile": profile, "rr": rr, "balance": starting_balance, "risk": backtest_risk, "costs": costs_bps}
        except Exception as exc:
            st.error(f"Backtest could not run: {exc}")
    expected_config = {"symbol": symbol, "timeframe": execution_tf, "confirmation": confirmation_tf, "profile": profile, "rr": rr, "balance": starting_balance, "risk": backtest_risk, "costs": costs_bps}
    result = st.session_state.get("cart3r_backtest")
    if result and st.session_state.get("cart3r_backtest_config") != expected_config:
        result = None
        st.info("Your market or test settings have changed since the previous run. Press RUN BACKTEST again to refresh the results.")
    if result:
        stats = result["stats"]
        st.markdown("#### Results")
        s1, s2 = st.columns(2)
        with s1: metric("Net return", f"{stats['return_pct']:+.2f}%", f"Ending balance {stats['ending_balance']:,.2f}")
        with s2: metric("Win rate", f"{stats['win_rate']:.1f}%", f"{stats['trades']} closed trades")
        s3, s4 = st.columns(2)
        with s3: metric("Profit factor", fmt(stats["profit_factor"], 2), "Gross wins / gross losses")
        with s4: metric("Maximum drawdown", f"{stats['max_drawdown']:.2f}%", "Peak-to-trough equity decline")
        if result["open_trade"]:
            st.info(f"One trade remains open at sample end: {result['open_trade']['side']} · unrealised P/L estimate {result['open_trade']['unrealised_pnl']:.2f}.")
        if not result["trades"].empty:
            st.dataframe(result["trades"].tail(100).iloc[::-1], use_container_width=True, hide_index=True)
            csv = result["trades"].to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download trades CSV", csv, file_name="cart3r_backtest_trades.csv", mime="text/csv", use_container_width=True)
            equity_fig = go.Figure(go.Scatter(x=result["equity"].index, y=result["equity"], line=dict(color="#3bc995"), name="Equity"))
            equity_fig.update_layout(height=280, template="plotly_dark", paper_bgcolor="#111720", plot_bgcolor="#111720", title="Equity curve")
            st.plotly_chart(equity_fig, use_container_width=True, config={"responsive": True})
        else:
            st.info("No completed trades for this sample and configuration.")
    st.caption("This is a simplified OHLC simulation. It is not proof of a live edge and does not model real broker execution, spread changes, slippage, swaps or margin calls.")

elif section == "NEWS & EVENTS":
    st.markdown("### Economic event watch")
    st.write(event_summary)
    if event_df.empty:
        st.warning("Calendar feed is not connected. This does NOT mean there are no important events today.")
        st.write("The optional Trading Economics integration needs an API key added by the app owner in the hosting provider's secrets settings. You can still use the app without it, but check a separate economic calendar before trading gold.")
    else:
        st.dataframe(event_df, use_container_width=True, hide_index=True)
    st.caption("Confirm event times, revisions and impact with the calendar provider. Do not trade solely from this panel.")

elif section == "TRADE JOURNAL":
    st.markdown("### Simple trade journal")
    st.caption("Entries are held in this app session. Download the CSV regularly because session data may reset when the app restarts.")
    with st.form("journal_form", clear_on_submit=True):
        j1, j2 = st.columns(2)
        with j1:
            journal_symbol = st.text_input("Symbol", value=symbol)
            journal_side = st.selectbox("Side", ["LONG", "SHORT"])
            journal_result = st.selectbox("Outcome", ["Win", "Loss", "Break-even", "Skipped / observation"])
        with j2:
            journal_entry = st.number_input("Entry price", min_value=0.0, value=0.0, format="%.5f")
            journal_exit = st.number_input("Exit price", min_value=0.0, value=0.0, format="%.5f")
            journal_r = st.number_input("Result (R multiple)", min_value=-20.0, max_value=20.0, value=0.0, step=0.1)
        journal_notes = st.text_area("Why did you take it? What did you learn?", max_chars=1000)
        submitted = st.form_submit_button("＋ SAVE JOURNAL ENTRY", use_container_width=True)
    if submitted:
        st.session_state.cart3r_journal.append({"Logged UTC": datetime.now(timezone.utc).isoformat(), "Symbol": journal_symbol, "Side": journal_side, "Outcome": journal_result, "Entry": journal_entry, "Exit": journal_exit, "R": journal_r, "Notes": journal_notes})
        st.success("Entry added to this session's journal.")
    if st.session_state.cart3r_journal:
        journal_df = pd.DataFrame(st.session_state.cart3r_journal)
        st.dataframe(journal_df.iloc[::-1], use_container_width=True, hide_index=True)
        st.download_button("⬇ DOWNLOAD JOURNAL CSV", journal_df.to_csv(index=False).encode("utf-8"), file_name="cart3r_trade_journal.csv", mime="text/csv", use_container_width=True)
        if st.button("Clear session journal", type="secondary"):
            st.session_state.cart3r_journal = []
            st.rerun()
    else:
        st.info("Your journal is empty. Add a trade or a skipped setup above.")

elif section == "LEARN / HOW TO USE":
    st.markdown("### Plain-English guide")
    with st.expander("1. What does BULLISH / BEARISH / NEUTRAL mean?", expanded=True):
        st.write("Bullish means more of the model's weighted checks point up. Bearish means more point down. Neutral means the checks are mixed or too weak. It is a directional summary, not a command to buy or sell.")
    with st.expander("2. What is the agreement score?"):
        st.write("It counts how many model components agree with the current direction. It is not a win probability. A score of 80 does not mean an 80% chance of profit.")
    with st.expander("3. What are EMA, RSI, MACD and ADX?"):
        st.write("EMA smooths price to show trend. RSI estimates momentum. MACD compares moving averages. ADX estimates trend strength, not direction. No indicator predicts the future on its own.")
    with st.expander("4. What is BOS / CHoCH / liquidity sweep?"):
        st.write("BOS means a close beyond a previously confirmed swing level. CHoCH is a break against the model's previous structure direction. A liquidity sweep is a candle moving beyond a confirmed swing and closing back through it. These are OHLC-based approximations, not actual order-book or institutional-flow data.")
    with st.expander("5. How do I use the risk calculator?"):
        st.write("Enter your balance, risk percentage, entry, stop and your broker's cash P/L for a $1 gold move at 1.00 lot. The app calculates an illustrative size. Contract specifications differ between brokers, so verify it on demo first.")
    with st.expander("6. What does the backtest prove?"):
        st.write("Nothing about future performance. It tests historical rules with simplified candle data. It cannot reproduce tick-by-tick order execution, changing spreads, slippage, swaps, commissions, leverage or all broker rules. Test other periods and demo trade before drawing conclusions.")
    with st.expander("7. Does CART3R place trades or connect to MT5?"):
        st.write("No. It does not connect to your broker, access your account, store passwords or place orders. You manually verify everything in your own platform.")
    st.warning("Gold is volatile. You can lose more than the planned amount because of gaps, slippage, leverage and broker rules. Never risk money you cannot afford to lose.")

st.markdown("---")
st.caption(f"CART3R GOLD INTELLIGENCE · FINAL BUILD 4.0 · {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · Research / education only")