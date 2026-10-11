"""v3策略 - 加入大盘趋势+个股中期趋势过滤"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, r'c:\Users\cks\Documents\trae_projects\stock_analyzer')
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from data_fetcher import DataFetcher
from factors import FactorEngine
from backtest import calc_rsi

fetcher = DataFetcher()
factor_engine = FactorEngine()

# 大盘数据
print("加载大盘数据...")
market_df = fetcher.get_daily_history('000001',
    start_date=(datetime.now() - timedelta(days=500)).strftime('%Y%m%d'))
market_closes = market_df['close'].values

def market_trend(date):
    m = market_df[market_df['date'] <= date]
    if len(m) < 20: return True
    c = m['close'].values
    return c[-1] > np.mean(c[-20:])


def calc_score_v3(ret5, rsi6, vol_ratio, close, ma20, ma60, pct,
                  ma20_prev, ma20_5d_ago, close_10d_ago, cd, above_ma60):
    score = 50
    if ret5 <= -12 and rsi6 < 35: score += 35
    elif ret5 <= -10 and rsi6 < 35: score += 22
    if rsi6 < 25: score += 12
    elif rsi6 < 30: score += 8
    elif rsi6 < 35: score += 4
    elif rsi6 > 75: score -= 15
    elif rsi6 > 70: score -= 10
    if ret5 <= -15: score += 12
    elif ret5 <= -12: score += 10
    elif ret5 <= -10: score += 7
    elif ret5 <= -8: score += 3
    elif ret5 > 5: score -= 8
    if vol_ratio < 0.7: score += 8
    elif vol_ratio < 0.9: score += 4
    elif vol_ratio > 2.0 and ret5 < 0: score -= 8
    if above_ma60: score += 8  # 中期趋势向上加分
    if close < ma20 and above_ma60: score += 3
    if pct <= -7: score -= 5
    # 趋势过滤
    if ma20 < ma20_prev: score -= 8
    if ma20 < ma20_5d_ago: score -= 5
    if cd >= 4: score -= 10
    elif cd >= 3: score -= 5
    ret10 = (close / close_10d_ago - 1) * 100 if close_10d_ago > 0 else 0
    if ret10 <= -20: score -= 12
    elif ret10 <= -15: score -= 6
    if close < ma20:
        dist = (ma20 / close - 1) * 100
        if dist > 15: score -= 8
        elif dist > 10: score -= 4
    return max(0, min(100, score))


def backtest_one(symbol, days=400, hold_days=5, use_market_filter=True, use_trend_filter=True):
    df = fetcher.get_daily_history(symbol,
        start_date=(datetime.now() - timedelta(days=days+60)).strftime('%Y%m%d'))
    if df.empty or len(df) < 65: return None
    closes = df['close'].values; lows = df['low'].values
    signals = []
    for i in range(65, len(df)):
        window = df.iloc[:i+1].reset_index(drop=True)
        tech = factor_engine.calc_technical_factors(window)
        ret5 = (closes[i]/closes[i-5]-1)*100
        rsi6 = calc_rsi(closes[:i+1], 6)
        vr = float(tech.get('volume_ratio', 1))
        ma20 = float(tech.get('ma20', closes[i]))
        ma60 = float(tech.get('ma60', closes[i]))
        pct = float(df.iloc[i]['pct_change'])
        ma20_prev = np.mean(closes[i-20:i]) if i>=20 else ma20
        ma20_5d = np.mean(closes[i-25:i-5]) if i>=25 else ma20
        c10 = closes[i-10] if i>=10 else closes[i]
        cd = 0
        for j in range(i, max(i-10,0), -1):
            if closes[j] < closes[j-1]: cd += 1
            else: break
        above_ma60 = closes[i] > ma60

        # 大盘过滤
        if use_market_filter and not market_trend(df.iloc[i]['date']):
            continue
        # 个股中期趋势过滤
        if use_trend_filter and not above_ma60:
            continue

        score = calc_score_v3(ret5, rsi6, vr, closes[i], ma20, ma60, pct,
                              ma20_prev, ma20_5d, c10, cd, above_ma60)
        if score >= 55 and i + hold_days < len(df):
            buy = closes[i]; sell = closes[i+hold_days]
            ret = (sell/buy-1)*100
            signals.append({'score': score, 'ret': ret, 'win': ret>0,
                           'signal': 'strong' if score>=70 else 'buy'})
    if not signals:
        return {'symbol': symbol, 'total': 0, 'wr': 0, 'ar': 0, 'strong_n': 0, 'strong_wr': 0, 'strong_ar': 0}
    s = pd.DataFrame(signals)
    ss = s[s['signal']=='strong']
    return {'symbol': symbol, 'total': len(s),
            'wr': round(len(s[s['win']])/len(s)*100, 1),
            'ar': round(s['ret'].mean(), 2),
            'strong_n': len(ss),
            'strong_wr': round(len(ss[ss['win']])/max(len(ss),1)*100, 1) if len(ss)>0 else 0,
            'strong_ar': round(ss['ret'].mean(), 2) if len(ss)>0 else 0}


symbols = ['002579','600519','000858','601318','000001','600036','002594','300750',
           '600900','601899','000333','600276','000651','601166','600887','002475',
           '600030','000725','000568','600809','002415','300760','600031','000063',
           '601398','601288']

configs = [
    (False, False, "v2 无过滤"),
    (True, False, "v3 仅大盘过滤"),
    (False, True, "v3 仅个股趋势过滤"),
    (True, True, "v3 大盘+个股趋势过滤"),
]

for mf, tf, name in configs:
    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"{'='*80}")
    print(f"  {'代码':<8} {'信号':>5} {'胜率':>7} {'收益':>8} {'强信号':>7} {'强胜率':>8} {'强收益':>8}")
    all_r = []
    for sym in symbols:
        r = backtest_one(sym, use_market_filter=mf, use_trend_filter=tf)
        if r and r['total'] > 0:
            all_r.append(r)
            print(f"  {sym:<8} {r['total']:>5} {r['wr']:>6.1f}% {r['ar']:>+7.2f}% {r['strong_n']:>7} {r['strong_wr']:>7.1f}% {r['strong_ar']:>+7.2f}%")
    if all_r:
        tot = sum(r['total'] for r in all_r)
        wr = sum(r['wr']*r['total'] for r in all_r)/tot
        ar = sum(r['ar']*r['total'] for r in all_r)/tot
        st = sum(r['strong_n'] for r in all_r)
        swr = sum(r['strong_wr']*r['strong_n'] for r in all_r)/max(st,1)
        sar = sum(r['strong_ar']*r['strong_n'] for r in all_r)/max(st,1)
        print(f"  {'汇总':<8} {tot:>5} {wr:>6.1f}% {ar:>+7.2f}% {st:>7} {swr:>7.1f}% {sar:>+7.2f}%")
