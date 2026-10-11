"""
本地Web应用 - A股超跌反弹分析
浏览器访问 http://localhost:8888
纯本地运行，数据不上传
"""
import sys
import json
import warnings
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime

warnings.filterwarnings('ignore')
sys.path.insert(0, r'c:\Users\cks\Documents\trae_projects\stock_analyzer')

from data_fetcher import DataFetcher
from factors import FactorEngine
from backtest import calc_rsi
import numpy as np

fetcher = DataFetcher()
factor_engine = FactorEngine()

# 缓存股票列表
_stocks_cache = None
# 缓存大盘数据
_market_df = None


def get_stocks():
    global _stocks_cache
    if _stocks_cache is None:
        try:
            _stocks_cache = fetcher.get_all_stocks()
        except:
            _stocks_cache = None
    return _stocks_cache


def get_market_df():
    global _market_df
    if _market_df is None:
        try:
            _market_df = fetcher.get_daily_history('000001',
                start_date=(datetime.now() - timedelta(days=500)).strftime('%Y%m%d'))
        except:
            _market_df = pd.DataFrame()
    return _market_df


def is_market_uptrend(date):
    """大盘是否在MA20上方（上升趋势）"""
    mdf = get_market_df()
    if mdf.empty:
        return True
    m = mdf[mdf['date'] <= date]
    if len(m) < 20:
        return True
    c = m['close'].values
    return c[-1] > np.mean(c[-20:])


def analyze_stock(symbol):
    """分析单只股票，返回字典（含综合评分、买卖价建议）"""
    result = {'symbol': symbol, 'name': symbol, 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    df = fetcher.get_daily_history(symbol)
    if df.empty:
        result['error'] = '无法获取行情数据'
        return result

    stocks = get_stocks()
    if stocks is not None and not stocks.empty:
        name_row = stocks[stocks['代码'] == symbol]
        if not name_row.empty:
            result['name'] = name_row['名称'].values[0]

    latest = df.iloc[-1]
    result['date'] = latest['date'].strftime('%Y-%m-%d')
    close = float(latest['close'])
    result['close'] = round(close, 2)
    result['pct_change'] = round(float(latest['pct_change']), 2)

    technical = factor_engine.calc_technical_factors(df)

    closes = df['close'].values
    lows = df['low'].values
    volumes = df['volume'].values
    ret5 = (closes[-1] / closes[-6] - 1) * 100 if len(closes) >= 6 else 0
    rsi6 = calc_rsi(closes, 6)
    rsi14 = calc_rsi(closes, 14)

    result['ret5'] = round(ret5, 2)
    result['rsi6'] = round(rsi6, 1)
    result['rsi14'] = round(rsi14, 1)

    # 近5日最低价（支撑位）
    low_5d = float(np.min(lows[-5:])) if len(lows) >= 5 else close
    low_10d = float(np.min(lows[-10:])) if len(lows) >= 10 else close
    result['low_5d'] = round(low_5d, 2)
    result['low_10d'] = round(low_10d, 2)

    # 量比
    vol_ratio = float(technical.get('volume_ratio', 1))
    result['volume_ratio'] = round(vol_ratio, 2)

    # ===== 过滤条件检查 =====
    ma20 = float(technical.get('ma20', close))
    ma60 = float(technical.get('ma60', close))
    above_ma60 = close > ma60
    market_ok = is_market_uptrend(latest['date'])

    # 20日波动率
    if len(closes) >= 21:
        daily_ret = np.diff(closes[-21:]) / closes[-21:-1]
        volatility = float(np.std(daily_ret))
    else:
        volatility = 0.02
    result['volatility'] = round(volatility * 100, 2)

    # MA20趋势
    if len(closes) >= 25:
        ma20_now = np.mean(closes[-20:])
        ma20_prev = np.mean(closes[-21:-1])
        ma20_5d_ago = np.mean(closes[-25:-5])
    else:
        ma20_now = ma20
        ma20_prev = ma20
        ma20_5d_ago = ma20
    ma20_down = ma20_now < ma20_prev

    # 连续下跌天数
    consecutive_down = 0
    for j in range(len(closes)-1, max(len(closes)-11, 0), -1):
        if closes[j] < closes[j-1]:
            consecutive_down += 1
        else:
            break

    # 10日涨跌幅
    close_10d_ago = closes[-11] if len(closes) >= 11 else closes[-1]
    ret10 = (close / close_10d_ago - 1) * 100

    # ===== 信号判定 =====
    signal = 'none'
    win_rate = 0
    avg_return = 0
    if ret5 <= -12 and rsi6 < 35:
        signal = 'strong'
        win_rate = 78.9  # 改进后强信号胜率
        avg_return = 6.92  # 改进后强信号收益
    elif ret5 <= -10 and rsi6 < 35:
        signal = 'mild'
        win_rate = 63.2
        avg_return = 2.92
    result['signal'] = signal

    # ===== 综合评分（改进版 v4）=====
    score = 50

    # 超跌反弹信号
    if signal == 'strong':
        score += 35
    elif signal == 'mild':
        score += 22

    # RSI超卖
    if rsi6 < 25:
        score += 12
    elif rsi6 < 30:
        score += 8
    elif rsi6 < 35:
        score += 4
    elif rsi6 > 75:
        score -= 15
    elif rsi6 > 70:
        score -= 10

    # 5日跌幅
    if ret5 <= -15:
        score += 12
    elif ret5 <= -12:
        score += 10
    elif ret5 <= -10:
        score += 7
    elif ret5 <= -8:
        score += 3
    elif ret5 > 5:
        score -= 8

    # 缩量
    if vol_ratio < 0.7:
        score += 8
    elif vol_ratio < 0.9:
        score += 4
    elif vol_ratio > 2.0 and ret5 < 0:
        score -= 8

    # 均线支撑
    if above_ma60:
        score += 8  # 中期趋势向上
    if close < ma20 and above_ma60:
        score += 3

    # 当日大跌
    if result['pct_change'] <= -7:
        score -= 5

    # ===== 趋势过滤减分（核心改进）=====
    if ma20_down:
        score -= 8  # MA20拐头向下
    if ma20_now < ma20_5d_ago:
        score -= 5  # MA20持续下行
    if consecutive_down >= 4:
        score -= 10  # 连续跌4天以上
    elif consecutive_down >= 3:
        score -= 5
    if ret10 <= -20:
        score -= 12  # 10天跌超20%，崩盘趋势
    elif ret10 <= -15:
        score -= 6
    if close < ma20:
        dist_from_ma20 = (ma20 / close - 1) * 100
        if dist_from_ma20 > 15:
            score -= 8
        elif dist_from_ma20 > 10:
            score -= 4

    score = max(0, min(100, score))
    result['score'] = round(score, 1)

    # ===== 过滤状态展示 =====
    filters = []
    filters.append(('大盘趋势', '✓ 上升' if market_ok else '✗ 下跌'))
    filters.append(('个股中期趋势', '✓ MA60上方' if above_ma60 else '✗ MA60下方'))
    filters.append(('波动率', f'{volatility*100:.1f}% {"✓" if volatility >= 0.025 else "✗ 偏低"}'))
    result['filters'] = filters

    # ===== 买入建议（基于改进后回测的阈值）=====
    # 只有大盘上升 + 个股MA60上方 + 波动率足够 + 评分够高才建议买入
    can_buy = market_ok and above_ma60 and volatility >= 0.02

    if not can_buy:
        if not market_ok:
            recommendation = '大盘弱势，不建议买入'
        elif not above_ma60:
            recommendation = '中期趋势向下，不建议买入'
        else:
            recommendation = '波动率不足，不建议买入'
        rec_level = 'avoid'
    elif score >= 70:
        recommendation = '强烈建议买入'
        rec_level = 'strong'
    elif score >= 60:
        recommendation = '建议买入'
        rec_level = 'buy'
    elif score >= 50:
        recommendation = '可轻仓试探'
        rec_level = 'light'
    else:
        recommendation = '观望为主'
        rec_level = 'wait'
    result['recommendation'] = recommendation
    result['rec_level'] = rec_level

    # ===== 建议买入价 =====
    # 激进价：当前价（信号强时直接买）
    # 稳健价：回调到MA5或前低附近
    ma5 = float(technical.get('ma5', close))
    buy_aggressive = close
    # 稳健买入价：取当前价下方的支撑位（MA5或5日低点，取较高者）
    support = max(low_5d, ma5 * 0.99) if ma5 < close else low_5d
    buy_safe = min(close * 0.98, support)  # 至少回调2%
    if buy_safe >= close:
        buy_safe = close * 0.97
    result['buy_aggressive'] = round(buy_aggressive, 2)
    result['buy_safe'] = round(buy_safe, 2)

    # ===== 止损价 =====
    # 强信号：-6% 或 跌破10日低点
    # 温和信号：-8% 或 跌破10日低点
    if signal == 'strong':
        stop_loss_pct = 0.06
    elif signal == 'mild':
        stop_loss_pct = 0.08
    else:
        stop_loss_pct = 0.07
    stop_by_pct = buy_aggressive * (1 - stop_loss_pct)
    stop_by_low = low_10d * 0.99  # 跌破10日低点1%
    stop_loss = min(stop_by_pct, stop_by_low)
    result['stop_loss'] = round(stop_loss, 2)
    result['stop_loss_pct'] = round((1 - stop_loss / buy_aggressive) * 100, 1)

    # ===== 目标价 =====
    if signal == 'strong':
        target_pct = avg_return / 100
    elif signal == 'mild':
        target_pct = avg_return / 100
    else:
        target_pct = 0.03  # 默认3%
    target_price = buy_aggressive * (1 + target_pct)
    # 目标价不超过MA20压力位太多
    if ma20 > close and ma20 < target_price * 1.02:
        target_price = ma20 * 0.99  # 受MA20压制
    result['target_price'] = round(target_price, 2)
    result['target_pct'] = round(target_pct * 100, 1)

    # ===== 风险收益比 =====
    risk = buy_aggressive - stop_loss
    reward = target_price - buy_aggressive
    rr_ratio = reward / risk if risk > 0 else 0
    result['rr_ratio'] = round(rr_ratio, 2)

    # ===== 持有天数建议 =====
    result['hold_days'] = 5 if signal != 'none' else 3

    # 技术指标
    result['ma5'] = round(ma5, 2)
    result['ma10'] = round(float(technical.get('ma10', 0)), 2)
    result['ma20'] = round(ma20, 2)
    result['ma60'] = round(ma60, 2)
    result['macd_golden'] = bool(technical.get('macd_golden_cross', False))
    result['ma_bullish'] = bool(technical.get('ma_bullish', False))
    result['break_high'] = bool(technical.get('break_20d_high', False))

    # 信号文案
    signal_text_map = {
        'strong': '强超跌反弹',
        'mild': '温和超跌反弹',
        'none': '未触发',
    }
    result['signal_text'] = signal_text_map.get(signal, '未触发')
    result['win_rate'] = win_rate
    result['avg_return'] = avg_return

    return result


HTML_PAGE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>A股超跌反弹分析</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    min-height: 100vh;
    color: #e2e8f0;
    padding: 20px;
  }
  .container { max-width: 640px; margin: 0 auto; }
  .header {
    text-align: center;
    padding: 30px 0 20px;
  }
  .header h1 {
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
  }
  .header p { color: #94a3b8; font-size: 13px; }
  .search-box {
    background: rgba(30, 41, 59, 0.8);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
  }
  .search-row { display: flex; gap: 10px; }
  .search-input {
    flex: 1;
    padding: 14px 18px;
    font-size: 16px;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.3);
    border-radius: 12px;
    color: #f1f5f9;
    outline: none;
    transition: all 0.3s;
  }
  .search-input:focus {
    border-color: #38bdf8;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15);
  }
  .search-btn {
    padding: 14px 24px;
    font-size: 15px;
    font-weight: 600;
    background: linear-gradient(135deg, #38bdf8, #6366f1);
    border: none;
    border-radius: 12px;
    color: white;
    cursor: pointer;
    transition: all 0.3s;
    white-space: nowrap;
  }
  .search-btn:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(56,189,248,0.4); }
  .search-btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
  .auto-refresh-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 12px;
    font-size: 13px;
    color: #94a3b8;
  }
  .auto-refresh-label {
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    user-select: none;
  }
  .auto-refresh-label input { cursor: pointer; width: 16px; height: 16px; }
  .refresh-select {
    background: rgba(15,23,42,0.6);
    border: 1px solid rgba(148,163,184,0.3);
    border-radius: 8px;
    padding: 6px 10px;
    color: #e2e8f0;
    font-size: 13px;
    cursor: pointer;
    outline: none;
  }
  .refresh-status { font-size: 12px; color: #4ade80; }
  .loading { text-align: center; padding: 40px; color: #94a3b8; }
  .spinner {
    width: 40px; height: 40px;
    border: 3px solid rgba(148,163,184,0.2);
    border-top-color: #38bdf8;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: 0 auto 12px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .result-card {
    background: rgba(30, 41, 59, 0.8);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    animation: fadeIn 0.4s ease;
  }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
  .stock-title {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    padding-bottom: 16px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.15);
  }
  .stock-name { font-size: 22px; font-weight: 700; color: #f1f5f9; }
  .stock-code { font-size: 14px; color: #94a3b8; margin-left: 8px; }
  .stock-price { text-align: right; }
  .price { font-size: 24px; font-weight: 700; }
  .price.up { color: #ef4444; }
  .price.down { color: #22c55e; }
  .pct { font-size: 13px; margin-top: 2px; }
  .signal-banner {
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    text-align: center;
  }
  .signal-strong {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(249, 115, 22, 0.2));
    border: 1px solid rgba(239, 68, 68, 0.4);
  }
  .signal-mild {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(234, 179, 8, 0.2));
    border: 1px solid rgba(245, 158, 11, 0.4);
  }
  .signal-none {
    background: rgba(100, 116, 139, 0.15);
    border: 1px solid rgba(100, 116, 139, 0.3);
  }
  .signal-label {
    font-size: 13px;
    color: #cbd5e1;
    margin-bottom: 6px;
  }
  .signal-text {
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 8px;
  }
  .signal-strong .signal-text { color: #fca5a5; }
  .signal-mild .signal-text { color: #fcd34d; }
  .signal-none .signal-text { color: #94a3b8; }
  .signal-stats {
    display: flex;
    justify-content: center;
    gap: 24px;
    margin-top: 12px;
  }
  .stat { text-align: center; }
  .stat-value { font-size: 18px; font-weight: 700; color: #f1f5f9; }
  .stat-label { font-size: 12px; color: #94a3b8; margin-top: 2px; }
  .advice {
    margin-top: 14px;
    font-size: 15px;
    font-weight: 600;
  }
  .signal-strong .advice { color: #fca5a5; }
  .signal-mild .advice { color: #fcd34d; }
  .section-title {
    font-size: 14px;
    font-weight: 600;
    color: #94a3b8;
    margin: 16px 0 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .indicator-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
  }
  .indicator-item {
    background: rgba(15, 23, 42, 0.5);
    border-radius: 10px;
    padding: 12px 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .indicator-label { font-size: 13px; color: #94a3b8; }
  .indicator-value { font-size: 14px; font-weight: 600; color: #f1f5f9; }
  .indicator-value.up { color: #ef4444; }
  .indicator-value.down { color: #22c55e; }
  .tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
  }
  .tag-yes { background: rgba(34, 197, 94, 0.2); color: #4ade80; }
  .tag-no { background: rgba(100, 116, 139, 0.2); color: #94a3b8; }
  .risk-box {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 10px;
    padding: 14px;
    margin-top: 16px;
    font-size: 13px;
    color: #fcd34d;
    line-height: 1.6;
  }
  .footer {
    text-align: center;
    padding: 20px;
    color: #64748b;
    font-size: 12px;
  }
  /* 综合评分 + 操作建议横幅 */
  .rec-banner {
    display: flex;
    align-items: center;
    gap: 20px;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 16px;
  }
  .rec-strong { background: linear-gradient(135deg, rgba(239,68,68,0.25), rgba(249,115,22,0.2)); border: 1px solid rgba(239,68,68,0.5); }
  .rec-buy { background: linear-gradient(135deg, rgba(245,158,11,0.2), rgba(234,179,8,0.15)); border: 1px solid rgba(245,158,11,0.5); }
  .rec-light { background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3); }
  .rec-wait { background: rgba(100,116,139,0.15); border: 1px solid rgba(100,116,139,0.3); }
  .rec-avoid { background: rgba(100,116,139,0.08); border: 1px solid rgba(100,116,139,0.2); }
  .rec-left { flex-shrink: 0; text-align: center; min-width: 90px; }
  .rec-label { font-size: 12px; color: #94a3b8; margin-bottom: 4px; }
  .rec-score { font-size: 32px; font-weight: 800; color: #f1f5f9; line-height: 1; }
  .rec-strong .rec-score { color: #fca5a5; }
  .rec-buy .rec-score { color: #fcd34d; }
  .rec-score-bar { width: 80px; height: 4px; background: rgba(148,163,184,0.2); border-radius: 2px; margin-top: 8px; overflow: hidden; }
  .rec-score-fill { height: 100%; background: linear-gradient(90deg, #38bdf8, #818cf8); border-radius: 2px; transition: width 0.5s; }
  .rec-right { flex: 1; }
  .rec-action { font-size: 20px; font-weight: 700; color: #f1f5f9; }
  .rec-strong .rec-action { color: #fca5a5; }
  .rec-buy .rec-action { color: #fcd34d; }
  .rec-hold { font-size: 13px; color: #94a3b8; margin-top: 4px; }
  /* 交易计划价格网格 */
  .price-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
  }
  .price-item {
    border-radius: 10px;
    padding: 14px;
    text-align: center;
  }
  .price-buy { background: rgba(56,189,248,0.1); border: 1px solid rgba(56,189,248,0.3); }
  .price-safe { background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.3); }
  .price-stop { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); }
  .price-target { background: rgba(168,85,247,0.1); border: 1px solid rgba(168,85,247,0.3); }
  .price-label { font-size: 12px; color: #94a3b8; margin-bottom: 6px; }
  .price-value { font-size: 22px; font-weight: 700; }
  .price-buy .price-value { color: #38bdf8; }
  .price-safe .price-value { color: #4ade80; }
  .price-stop .price-value { color: #f87171; }
  .price-target .price-value { color: #c084fc; }
  .price-sub { font-size: 11px; color: #64748b; margin-top: 4px; }
  /* 风险收益比 */
  .rr-box {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    background: rgba(15,23,42,0.5);
    border-radius: 10px;
    padding: 12px;
    margin-top: 12px;
  }
  .rr-label { font-size: 13px; color: #94a3b8; }
  .rr-value { font-size: 18px; font-weight: 700; color: #f1f5f9; }
  .rr-hint { font-size: 13px; color: #4ade80; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>A股超跌反弹分析</h1>
    <p>历史回测胜率 72.5% · 纯本地运行 · 数据不上传</p>
  </div>

  <div class="search-box">
    <div class="search-row">
      <input type="text" id="symbol" class="search-input" placeholder="输入股票代码，如 002579" maxlength="6">
      <button class="search-btn" id="btn" onclick="analyze()">分析</button>
    </div>
    <div class="auto-refresh-row">
      <label class="auto-refresh-label">
        <input type="checkbox" id="autoRefresh" onchange="toggleAutoRefresh()">
        <span>盘中自动刷新</span>
      </label>
      <select id="refreshInterval" class="refresh-select">
        <option value="5">每5秒</option>
        <option value="10" selected>每10秒</option>
        <option value="30">每30秒</option>
        <option value="60">每60秒</option>
      </select>
      <span id="refreshStatus" class="refresh-status"></span>
    </div>
  </div>

  <div id="result"></div>

  <div class="footer">
    仅供学习参考，不构成投资建议 · 股市有风险，入市需谨慎
  </div>
</div>

<script>
const btn = document.getElementById('btn');
const input = document.getElementById('symbol');
input.addEventListener('keypress', e => { if (e.key === 'Enter') analyze(); });

let autoTimer = null;
let lastSymbol = '';

function toggleAutoRefresh() {
  const cb = document.getElementById('autoRefresh');
  const status = document.getElementById('refreshStatus');
  if (cb.checked) {
    if (!lastSymbol) {
      alert('请先输入股票代码并分析一次');
      cb.checked = false;
      return;
    }
    const interval = parseInt(document.getElementById('refreshInterval').value) * 1000;
    status.textContent = '● 自动刷新中';
    status.style.color = '#4ade80';
    autoTimer = setInterval(() => {
      if (lastSymbol) analyze(true);
    }, interval);
  } else {
    clearInterval(autoTimer);
    autoTimer = null;
    status.textContent = '';
  }
}

async function analyze(silent) {
  const symbol = input.value.trim();
  if (!symbol) { alert('请输入股票代码'); return; }
  if (!/^\\d{6}$/.test(symbol)) { alert('请输入6位股票代码'); return; }

  lastSymbol = symbol;
  if (!silent) {
    btn.disabled = true;
    btn.textContent = '分析中...';
    document.getElementById('result').innerHTML = '<div class="loading"><div class="spinner"></div>正在获取数据...</div>';
  }

  try {
    const res = await fetch('/api/analyze?symbol=' + symbol);
    const data = await res.json();
    render(data);
    // 更新刷新时间
    const status = document.getElementById('refreshStatus');
    if (document.getElementById('autoRefresh').checked) {
      const now = new Date();
      status.textContent = '● 自动刷新中 · ' + now.toLocaleTimeString();
    }
  } catch (e) {
    if (!silent) {
      document.getElementById('result').innerHTML = '<div class="loading">请求失败，请重试</div>';
    }
  }
  if (!silent) {
    btn.disabled = false;
    btn.textContent = '分析';
  }
}

function render(d) {
  if (d.error) {
    document.getElementById('result').innerHTML = `<div class="result-card" style="text-align:center;color:#fca5a5">${d.error}</div>`;
    return;
  }
  const priceCls = d.pct_change >= 0 ? 'up' : 'down';
  const pctSign = d.pct_change >= 0 ? '+' : '';
  let signalCls = 'signal-none';
  if (d.signal === 'strong') signalCls = 'signal-strong';
  else if (d.signal === 'mild') signalCls = 'signal-mild';

  const recCls = 'rec-' + d.rec_level;

  const ret5Cls = d.ret5 >= 0 ? 'up' : 'down';
  const rsiCls = d.rsi6 < 35 ? 'down' : '';

  document.getElementById('result').innerHTML = `
    <div class="result-card">
      <div class="stock-title">
        <div><span class="stock-name">${d.name}</span><span class="stock-code">${d.symbol}</span></div>
        <div class="stock-price">
          <div class="price ${priceCls}">${d.close}</div>
          <div class="pct ${priceCls}">${pctSign}${d.pct_change}% · ${d.date}</div>
        </div>
      </div>

      <!-- 综合评分 + 买入建议 -->
      <div class="rec-banner ${recCls}">
        <div class="rec-left">
          <div class="rec-label">综合评分</div>
          <div class="rec-score">${d.score}</div>
          <div class="rec-score-bar"><div class="rec-score-fill" style="width:${d.score}%"></div></div>
        </div>
        <div class="rec-right">
          <div class="rec-label">操作建议</div>
          <div class="rec-action">${d.recommendation}</div>
          <div class="rec-hold">建议持有 ${d.hold_days} 天</div>
        </div>
      </div>

      <!-- 买卖价格建议 -->
      <div class="section-title">交易计划</div>
      <div class="price-grid">
        <div class="price-item price-buy">
          <div class="price-label">激进买入价</div>
          <div class="price-value">${d.buy_aggressive}</div>
          <div class="price-sub">当前价直接买</div>
        </div>
        <div class="price-item price-safe">
          <div class="price-label">稳健买入价</div>
          <div class="price-value">${d.buy_safe}</div>
          <div class="price-sub">回调到支撑位</div>
        </div>
        <div class="price-item price-stop">
          <div class="price-label">止损价</div>
          <div class="price-value">${d.stop_loss}</div>
          <div class="price-sub">亏 ${d.stop_loss_pct}% 离场</div>
        </div>
        <div class="price-item price-target">
          <div class="price-label">目标价</div>
          <div class="price-value">${d.target_price}</div>
          <div class="price-sub">盈利 +${d.target_pct}%</div>
        </div>
      </div>

      <!-- 风险收益比 -->
      <div class="rr-box">
        <span class="rr-label">风险收益比</span>
        <span class="rr-value">1 : ${d.rr_ratio}</span>
        <span class="rr-hint">${d.rr_ratio >= 1.5 ? '✓ 性价比高' : d.rr_ratio >= 1 ? '○ 尚可' : '△ 偏低'}</span>
      </div>

      <!-- 过滤条件 -->
      <div class="filter-box">
        ${d.filters.map(f => `<div class="filter-item"><span class="filter-label">${f[0]}</span><span class="filter-value ${f[1].includes('✓') ? 'ok' : 'no'}">${f[1]}</span></div>`).join('')}
      </div>

      <!-- 超跌反弹信号 -->
      <div class="signal-banner ${signalCls}" style="margin-top:16px">
        <div class="signal-label">超跌反弹信号</div>
        <div class="signal-text">${d.signal_text}</div>
        ${d.signal !== 'none' ? `
          <div class="signal-stats">
            <div class="stat"><div class="stat-value">${d.win_rate}%</div><div class="stat-label">历史胜率</div></div>
            <div class="stat"><div class="stat-value">+${d.avg_return}%</div><div class="stat-label">平均收益</div></div>
          </div>
        ` : ''}
      </div>

      <div class="section-title">核心指标</div>
      <div class="indicator-grid">
        <div class="indicator-item"><span class="indicator-label">5日涨跌幅</span><span class="indicator-value ${ret5Cls}">${d.ret5 >= 0 ? '+' : ''}${d.ret5}%</span></div>
        <div class="indicator-item"><span class="indicator-label">RSI6</span><span class="indicator-value ${rsiCls}">${d.rsi6}</span></div>
        <div class="indicator-item"><span class="indicator-label">RSI14</span><span class="indicator-value">${d.rsi14}</span></div>
        <div class="indicator-item"><span class="indicator-label">量比</span><span class="indicator-value">${d.volume_ratio}</span></div>
        <div class="indicator-item"><span class="indicator-label">5日低点</span><span class="indicator-value">${d.low_5d}</span></div>
        <div class="indicator-item"><span class="indicator-label">10日低点</span><span class="indicator-value">${d.low_10d}</span></div>
      </div>

      <div class="section-title">均线系统</div>
      <div class="indicator-grid">
        <div class="indicator-item"><span class="indicator-label">MA5</span><span class="indicator-value">${d.ma5}</span></div>
        <div class="indicator-item"><span class="indicator-label">MA10</span><span class="indicator-value">${d.ma10}</span></div>
        <div class="indicator-item"><span class="indicator-label">MA20</span><span class="indicator-value">${d.ma20}</span></div>
        <div class="indicator-item"><span class="indicator-label">MA60</span><span class="indicator-value">${d.ma60}</span></div>
      </div>

      <div class="section-title">技术信号</div>
      <div class="indicator-grid">
        <div class="indicator-item"><span class="indicator-label">MACD金叉</span><span class="tag ${d.macd_golden ? 'tag-yes' : 'tag-no'}">${d.macd_golden ? '是' : '否'}</span></div>
        <div class="indicator-item"><span class="indicator-label">均线多头</span><span class="tag ${d.ma_bullish ? 'tag-yes' : 'tag-no'}">${d.ma_bullish ? '是' : '否'}</span></div>
        <div class="indicator-item"><span class="indicator-label">突破20日新高</span><span class="tag ${d.break_high ? 'tag-yes' : 'tag-no'}">${d.break_high ? '是' : '否'}</span></div>
        <div class="indicator-item"><span class="indicator-label">信号强度</span><span class="tag ${d.signal !== 'none' ? 'tag-yes' : 'tag-no'}">${d.signal === 'strong' ? '强' : d.signal === 'mild' ? '中' : '无'}</span></div>
      </div>

      <div class="risk-box">
        ⚠️ 风险提示：以上建议基于历史数据统计，不保证未来收益。急跌可能伴随基本面利空，请先了解下跌原因。严格执行止损，单只仓位控制在10-20%以内。
      </div>
    </div>
  `;
}
</script>
</body>
</html>'''


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # 静默日志

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        elif parsed.path == '/api/analyze':
            params = parse_qs(parsed.query)
            symbol = params.get('symbol', [''])[0]
            try:
                result = analyze_stock(symbol)
            except Exception as e:
                result = {'error': f'分析出错: {str(e)}'}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()


def main():
    port = 8888
    print("正在加载股票列表...")
    get_stocks()
    print("正在加载大盘数据...")
    get_market_df()
    print("加载完成")
    server = HTTPServer(('127.0.0.1', port), Handler)
    url = f'http://localhost:{port}'
    print(f"\n服务已启动: {url}")
    print(f"按 Ctrl+C 停止服务\n")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
        server.server_close()


if __name__ == '__main__':
    main()
