"""
改进版回测 - 加入趋势过滤，对比改进前后
核心改进：避免下跌趋势中"超跌之后更超跌"的连续失败
"""
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


def calc_score_v1(ret5, rsi6, vol_ratio, close, ma20, ma60, pct_change):
    """原始评分逻辑（网页当前用的）"""
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
    return max(0, min(100, score))


def calc_score_v2(ret5, rsi6, vol_ratio, close, ma20, ma60, pct_change,
                  ma20_prev, ma20_5d_ago, close_10d_ago, consecutive_down):
    """改进版评分 - 加入趋势过滤"""
    score = calc_score_v1(ret5, rsi6, vol_ratio, close, ma20, ma60, pct_change)

    # 改进1：下跌趋势过滤 - MA20向下倾斜时减分
    if ma20 < ma20_prev:
        score -= 8  # MA20拐头向下
    if ma20 < ma20_5d_ago:
        score -= 5  # MA20持续下行

    # 改进2：连续下跌过滤 - 已连续跌多天，可能继续跌
    if consecutive_down >= 4:
        score -= 10
    elif consecutive_down >= 3:
        score -= 5

    # 改进3：相对10天前跌幅过大（>20%）说明处于暴跌趋势，减分
    ret10 = (close / close_10d_ago - 1) * 100 if close_10d_ago > 0 else 0
    if ret10 <= -20:
        score -= 12
    elif ret10 <= -15:
        score -= 6

    # 改进4：MA20下方且远离MA20（>10%）说明趋势性下跌，减分
    if close < ma20:
        dist_from_ma20 = (ma20 / close - 1) * 100
        if dist_from_ma20 > 15:
            score -= 8
        elif dist_from_ma20 > 10:
            score -= 4

    return max(0, min(100, score))


def run_backtest(symbol, days=400, hold_days=5):
    print(f"\n{'='*75}")
    print(f"  回测: {symbol} | {days}天 | 持有{hold_days}天")
    print(f"{'='*75}")

    df = fetcher.get_daily_history(symbol,
        start_date=(datetime.now() - timedelta(days=days+60)).strftime('%Y%m%d'))
    if df.empty or len(df) < 60:
        print("数据不足")
        return

    closes = df['close'].values
    lows = df['low'].values
    volumes = df['volume'].values

    v1_signals = []
    v2_signals = []

    for i in range(65, len(df)):
        window = df.iloc[:i+1].reset_index(drop=True)
        technical = factor_engine.calc_technical_factors(window)

        ret5 = (closes[i] / closes[i-5] - 1) * 100
        rsi6 = calc_rsi(closes[:i+1], 6)
        vol_ratio = float(technical.get('volume_ratio', 1))
        ma20 = float(technical.get('ma20', closes[i]))
        ma60 = float(technical.get('ma60', closes[i]))
        pct = float(df.iloc[i]['pct_change'])

        # 趋势相关数据
        ma20_prev = float(technical.get('ma20', closes[i-1])) if i > 0 else ma20
        # 用简化方式计算MA20的变化
        if i >= 25:
            ma20_prev = np.mean(closes[i-20:i])
            ma20_5d_ago = np.mean(closes[i-25:i-5])
        else:
            ma20_prev = ma20
            ma20_5d_ago = ma20

        close_10d_ago = closes[i-10] if i >= 10 else closes[i]

        # 连续下跌天数
        consecutive_down = 0
        for j in range(i, max(i-10, 0), -1):
            if closes[j] < closes[j-1]:
                consecutive_down += 1
            else:
                break

        score_v1 = calc_score_v1(ret5, rsi6, vol_ratio, closes[i], ma20, ma60, pct)
        score_v2 = calc_score_v2(ret5, rsi6, vol_ratio, closes[i], ma20, ma60, pct,
                                 ma20_prev, ma20_5d_ago, close_10d_ago, consecutive_down)

        if i + hold_days < len(df):
            buy_price = closes[i]
            sell_price = closes[i + hold_days]
            actual_ret = (sell_price / buy_price - 1) * 100
            period_high = np.max(closes[i+1:i+hold_days+1])
            max_gain = (period_high / buy_price - 1) * 100
            period_low = np.min(lows[i+1:i+hold_days+1])
            max_loss = (period_low / buy_price - 1) * 100

            base = {
                'date': df.iloc[i]['date'].strftime('%Y-%m-%d'),
                'close': round(buy_price, 2),
                'ret5': round(ret5, 2),
                'rsi6': round(rsi6, 1),
                'actual_ret': round(actual_ret, 2),
                'max_gain': round(max_gain, 2),
                'max_loss': round(max_loss, 2),
                'win': actual_ret > 0,
                'consecutive_down': consecutive_down,
            }

            if score_v1 >= 55:
                s = dict(base)
                s['score'] = round(score_v1, 1)
                s['signal'] = 'strong' if score_v1 >= 70 else 'buy'
                v1_signals.append(s)

            if score_v2 >= 55:
                s = dict(base)
                s['score'] = round(score_v2, 1)
                s['signal'] = 'strong' if score_v2 >= 70 else 'buy'
                v2_signals.append(s)

    def print_stats(name, signals):
        if not signals:
            print(f"\n  [{name}] 无信号")
            return
        s_df = pd.DataFrame(signals)
        total = len(s_df)
        wins = len(s_df[s_df['win']])
        wr = wins / total * 100
        ar = s_df['actual_ret'].mean()
        mg = s_df['max_gain'].mean()
        ml = s_df['max_loss'].mean()

        print(f"\n  [{name}]")
        print(f"    信号数: {total}")
        print(f"    胜率: {wr:.1f}% ({wins}/{total})")
        print(f"    平均收益: {ar:+.2f}%")
        print(f"    平均最大涨: {mg:+.2f}%")
        print(f"    平均最大跌: {ml:+.2f}%")

        for stype, label in [('strong', '强信号(>=70)'), ('buy', '普通(55-70)')]:
            sub = s_df[s_df['signal'] == stype]
            if len(sub) > 0:
                sw = len(sub[sub['win']]) / len(sub) * 100
                sa = sub['actual_ret'].mean()
                print(f"    {label}: {len(sub)}次, 胜率{sw:.1f}%, 收益{sa:+.2f}%")

    print_stats("v1 原始策略", v1_signals)
    print_stats("v2 改进策略(+趋势过滤)", v2_signals)

    # 对比
    print(f"\n  --- 对比总结 ---")
    if v1_signals and v2_signals:
        v1_wr = len([s for s in v1_signals if s['win']]) / len(v1_signals) * 100
        v2_wr = len([s for s in v2_signals if s['win']]) / len(v2_signals) * 100
        v1_ar = np.mean([s['actual_ret'] for s in v1_signals])
        v2_ar = np.mean([s['actual_ret'] for s in v2_signals])
        print(f"    胜率: v1={v1_wr:.1f}% → v2={v2_wr:.1f}%  变化: {v2_wr-v1_wr:+.1f}%")
        print(f"    收益: v1={v1_ar:+.2f}% → v2={v2_ar:+.2f}%  变化: {v2_ar-v1_ar:+.2f}%")
        print(f"    信号数: v1={len(v1_signals)} → v2={len(v2_signals)}  (过滤掉{len(v1_signals)-len(v2_signals)}个)")

    print(f"\n{'='*75}")
    return v1_signals, v2_signals


if __name__ == '__main__':
    symbol = sys.argv[1] if len(sys.argv) > 1 else '002579'
    run_backtest(symbol)
