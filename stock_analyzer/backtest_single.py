"""
单股回测 - 用网页相同的因子逻辑回测
验证策略信号在历史上的真实表现
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


def calc_score(ret5, rsi6, vol_ratio, close, ma20, ma60, pct_change):
    """和网页完全一致的综合评分逻辑"""
    score = 50
    # 超跌反弹信号
    if ret5 <= -12 and rsi6 < 35:
        score += 35
    elif ret5 <= -10 and rsi6 < 35:
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
    if close > ma60:
        score += 5
    if close < ma20 and close > ma60:
        score += 3
    # 单日暴跌
    if pct_change <= -7:
        score -= 5
    return max(0, min(100, score))


def backtest_single(symbol, days=400, hold_days=5):
    """单股回测"""
    print(f"\n{'='*70}")
    print(f"  回测股票: {symbol}")
    print(f"  回测周期: 最近{days}天，持有{hold_days}天")
    print(f"{'='*70}")

    df = fetcher.get_daily_history(symbol,
        start_date=(datetime.now() - timedelta(days=days+60)).strftime('%Y%m%d'))
    if df.empty or len(df) < 60:
        print("数据不足")
        return

    closes = df['close'].values
    lows = df['low'].values
    volumes = df['volume'].values

    signals = []
    for i in range(60, len(df)):
        window = df.iloc[:i+1].reset_index(drop=True)
        technical = factor_engine.calc_technical_factors(window)

        ret5 = (closes[i] / closes[i-5] - 1) * 100
        rsi6 = calc_rsi(closes[:i+1], 6)
        vol_ratio = float(technical.get('volume_ratio', 1))
        ma20 = float(technical.get('ma20', closes[i]))
        ma60 = float(technical.get('ma60', closes[i]))
        pct = float(df.iloc[i]['pct_change'])

        score = calc_score(ret5, rsi6, vol_ratio, closes[i], ma20, ma60, pct)

        # 只记录评分>=55的信号（建议买入/强烈建议买入）
        if score >= 55:
            signal_type = 'strong' if score >= 70 else 'buy'
            if i + hold_days < len(df):
                buy_price = closes[i]
                sell_price = closes[i + hold_days]
                actual_ret = (sell_price / buy_price - 1) * 100
                period_high = np.max(closes[i+1:i+hold_days+1])
                max_gain = (period_high / buy_price - 1) * 100
                period_low = np.min(lows[i+1:i+hold_days+1])
                max_loss = (period_low / buy_price - 1) * 100

                signals.append({
                    'date': df.iloc[i]['date'].strftime('%Y-%m-%d'),
                    'close': round(buy_price, 2),
                    'score': round(score, 1),
                    'signal': signal_type,
                    'ret5': round(ret5, 2),
                    'rsi6': round(rsi6, 1),
                    'actual_ret': round(actual_ret, 2),
                    'max_gain': round(max_gain, 2),
                    'max_loss': round(max_loss, 2),
                    'win': actual_ret > 0,
                })

    if not signals:
        print("  历史上无买入信号（评分>=55）")
        return

    df_sig = pd.DataFrame(signals)
    total = len(df_sig)
    wins = len(df_sig[df_sig['win']])
    win_rate = wins / total * 100
    avg_ret = df_sig['actual_ret'].mean()
    avg_max_gain = df_sig['max_gain'].mean()
    avg_max_loss = df_sig['max_loss'].mean()

    print(f"\n  总信号数: {total}")
    print(f"  胜率: {win_rate:.1f}% ({wins}/{total})")
    print(f"  平均实际收益: {avg_ret:+.2f}%")
    print(f"  平均期间最大涨幅: {avg_max_gain:+.2f}%")
    print(f"  平均期间最大跌幅: {avg_max_loss:+.2f}%")

    # 按信号类型统计
    print(f"\n  --- 按信号类型 ---")
    for stype, label in [('strong', '强烈建议买入(>=70分)'), ('buy', '建议买入(55-70分)')]:
        sub = df_sig[df_sig['signal'] == stype]
        if len(sub) > 0:
            wr = len(sub[sub['win']]) / len(sub) * 100
            ar = sub['actual_ret'].mean()
            print(f"  {label}: {len(sub)}次, 胜率{wr:.1f}%, 平均收益{ar:+.2f}%")

    # 策略预期 vs 实际
    print(f"\n  --- 策略预期 vs 实际 ---")
    strong_sigs = df_sig[df_sig['signal'] == 'strong']
    if len(strong_sigs) > 0:
        print(f"  强信号预期收益: +4.90%")
        print(f"  强信号实际收益: {strong_sigs['actual_ret'].mean():+.2f}%")
        print(f"  差别: {strong_sigs['actual_ret'].mean() - 4.90:+.2f}%")

    # 明细
    print(f"\n  --- 信号明细 ---")
    print(f"  {'日期':<12} {'买入价':>8} {'评分':>6} {'信号':>6} {'5日跌':>8} {'RSI6':>6} {'实际收益':>9} {'最大涨':>8} {'最大跌':>8}")
    print(f"  {'-'*70}")
    for _, r in df_sig.iterrows():
        win_mark = '✓' if r['win'] else '✗'
        print(f"  {r['date']:<12} {r['close']:>8.2f} {r['score']:>6.1f} {r['signal']:>6} {r['ret5']:>+7.2f}% {r['rsi6']:>6.1f} {r['actual_ret']:>+8.2f}%{win_mark} {r['max_gain']:>+7.2f}% {r['max_loss']:>+7.2f}%")

    print(f"\n{'='*70}")
    print(f"  结论: 策略{'有效' if win_rate >= 55 else '一般'}，实际胜率{win_rate:.1f}%{'高于' if avg_ret > 0 else '低于'}预期")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    else:
        symbol = '002579'
    backtest_single(symbol)
