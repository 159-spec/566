"""多股票验证改进策略的稳定性"""
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, r'c:\Users\cks\Documents\trae_projects\stock_analyzer')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from data_fetcher import DataFetcher
from factors import FactorEngine
from backtest import calc_rsi

fetcher = DataFetcher()
factor_engine = FactorEngine()


def calc_score_v2(ret5, rsi6, vol_ratio, close, ma20, ma60, pct_change,
                  ma20_prev, ma20_5d_ago, close_10d_ago, consecutive_down):
    score = 50
    if ret5 <= -12 and rsi6 < 35:
        score += 35
    elif ret5 <= -10 and rsi6 < 35:
        score += 22
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
    if vol_ratio < 0.7:
        score += 8
    elif vol_ratio < 0.9:
        score += 4
    elif vol_ratio > 2.0 and ret5 < 0:
        score -= 8
    if close > ma60:
        score += 5
    if close < ma20 and close > ma60:
        score += 3
    if pct_change <= -7:
        score -= 5

    # 趋势过滤
    if ma20 < ma20_prev:
        score -= 8
    if ma20 < ma20_5d_ago:
        score -= 5
    if consecutive_down >= 4:
        score -= 10
    elif consecutive_down >= 3:
        score -= 5
    ret10 = (close / close_10d_ago - 1) * 100 if close_10d_ago > 0 else 0
    if ret10 <= -20:
        score -= 12
    elif ret10 <= -15:
        score -= 6
    if close < ma20:
        dist = (ma20 / close - 1) * 100
        if dist > 15:
            score -= 8
        elif dist > 10:
            score -= 4

    return max(0, min(100, score))


def backtest_one(symbol, days=400, hold_days=5):
    df = fetcher.get_daily_history(symbol,
        start_date=(datetime.now() - timedelta(days=days+60)).strftime('%Y%m%d'))
    if df.empty or len(df) < 65:
        return None

    closes = df['close'].values
    lows = df['low'].values
    signals = []

    for i in range(65, len(df)):
        window = df.iloc[:i+1].reset_index(drop=True)
        technical = factor_engine.calc_technical_factors(window)

        ret5 = (closes[i] / closes[i-5] - 1) * 100
        rsi6 = calc_rsi(closes[:i+1], 6)
        vol_ratio = float(technical.get('volume_ratio', 1))
        ma20 = float(technical.get('ma20', closes[i]))
        ma60 = float(technical.get('ma60', closes[i]))
        pct = float(df.iloc[i]['pct_change'])

        ma20_prev = np.mean(closes[i-20:i]) if i >= 20 else ma20
        ma20_5d_ago = np.mean(closes[i-25:i-5]) if i >= 25 else ma20
        close_10d_ago = closes[i-10] if i >= 10 else closes[i]

        consecutive_down = 0
        for j in range(i, max(i-10, 0), -1):
            if closes[j] < closes[j-1]:
                consecutive_down += 1
            else:
                break

        score = calc_score_v2(ret5, rsi6, vol_ratio, closes[i], ma20, ma60, pct,
                              ma20_prev, ma20_5d_ago, close_10d_ago, consecutive_down)

        if score >= 55 and i + hold_days < len(df):
            buy = closes[i]
            sell = closes[i + hold_days]
            ret = (sell / buy - 1) * 100
            signals.append({'score': score, 'ret': ret, 'win': ret > 0,
                           'signal': 'strong' if score >= 70 else 'buy'})

    if not signals:
        return {'symbol': symbol, 'total': 0, 'wr': 0, 'ar': 0}

    s_df = pd.DataFrame(signals)
    return {
        'symbol': symbol,
        'total': len(s_df),
        'wr': round(len(s_df[s_df['win']]) / len(s_df) * 100, 1),
        'ar': round(s_df['ret'].mean(), 2),
        'strong_wr': round(len(s_df[(s_df['signal']=='strong') & (s_df['win'])]) / max(len(s_df[s_df['signal']=='strong']),1) * 100, 1),
        'strong_ar': round(s_df[s_df['signal']=='strong']['ret'].mean() if len(s_df[s_df['signal']=='strong'])>0 else 0, 2),
        'strong_n': len(s_df[s_df['signal']=='strong']),
    }


symbols = ['002579', '600519', '000858', '601318', '000001', '600036',
           '002594', '300750', '600900', '601899', '000333', '600276',
           '000651', '601166', '600887', '002475', '600030', '000725',
           '000568', '600809', '002415', '300760', '600031', '000063',
           '601398', '601288']

print(f"\n{'='*80}")
print(f"  改进策略(v2) 多股票验证 | 持有5天 | 评分>=55")
print(f"{'='*80}")
print(f"  {'代码':<8} {'信号数':>6} {'胜率':>7} {'平均收益':>9} {'强信号数':>8} {'强信号胜率':>10} {'强信号收益':>10}")
print(f"  {'-'*70}")

all_signals = []
strong_signals = []
for sym in symbols:
    r = backtest_one(sym)
    if r and r['total'] > 0:
        all_signals.append(r)
        strong_signals.extend([r] * r['strong_n'] if r['strong_n'] > 0 else [])
        print(f"  {sym:<8} {r['total']:>6} {r['wr']:>6.1f}% {r['ar']:>+8.2f}% {r['strong_n']:>8} {r['strong_wr']:>9.1f}% {r['strong_ar']:>+9.2f}%")

if all_signals:
    total_s = sum(r['total'] for r in all_signals)
    weighted_wr = sum(r['wr'] * r['total'] for r in all_signals) / total_s
    weighted_ar = sum(r['ar'] * r['total'] for r in all_signals) / total_s
    strong_total = sum(r['strong_n'] for r in all_signals)
    strong_wr = sum(r['strong_wr'] * r['strong_n'] for r in all_signals) / max(strong_total, 1)
    strong_ar = sum(r['strong_ar'] * r['strong_n'] for r in all_signals) / max(strong_total, 1)

    print(f"  {'-'*70}")
    print(f"  {'汇总':<8} {total_s:>6} {weighted_wr:>6.1f}% {weighted_ar:>+8.2f}% {strong_total:>8} {strong_wr:>9.1f}% {strong_ar:>+9.2f}%")
    print(f"\n  改进策略整体胜率: {weighted_wr:.1f}%, 平均收益: {weighted_ar:+.2f}%")
    print(f"  强信号胜率: {strong_wr:.1f}%, 强信号收益: {strong_ar:+.2f}%")
print(f"{'='*80}\n")
