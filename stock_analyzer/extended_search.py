"""
扩展搜索 - 寻找信号数>=30且胜率>=70%的配置
放宽部分过滤条件，扩大信号池
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, r'c:\Users\cks\Documents\trae_projects\stock_analyzer')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from data_fetcher import DataFetcher
from factors import FactorEngine
from backtest import calc_rsi

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fetcher = DataFetcher()
factor_engine = FactorEngine()
OUTPUT_DIR = r'c:\Users\cks\Documents\Trae-CN\default'

SYMBOLS = [
    '002579', '600519', '000858', '601318', '000001', '600036', '002594', '300750',
    '600900', '601899', '000333', '600276', '000651', '601166', '600887', '002475',
    '600030', '000725', '000568', '600809', '002415', '300760', '600031', '000063',
    '601398', '601288', '000002', '600009', '601888', '300059', '002230', '603259',
    '600438', '601012',
]


def load_market(days=600):
    print("加载大盘数据...")
    return fetcher.get_daily_history('000001',
        start_date=(datetime.now() - timedelta(days=days)).strftime('%Y%m%d'))


def market_trend(market_df, date):
    m = market_df[market_df['date'] <= date]
    if len(m) < 20: return True
    c = m['close'].values
    return c[-1] > np.mean(c[-20:])


def collect_all_signals(symbols, market_df, days=500, hold_days=5, min_vol=0.02, require_ma60=True):
    """可调整过滤严格度的信号收集"""
    all_signals = []
    for idx, sym in enumerate(symbols):
        print(f"  收集 {sym} ({idx+1}/{len(symbols)})...", end='\r')
        df = fetcher.get_daily_history(sym,
            start_date=(datetime.now() - timedelta(days=days+60)).strftime('%Y%m%d'))
        if df.empty or len(df) < 65: continue
        closes = df['close'].values
        volumes = df['volume'].values
        highs = df['high'].values
        lows = df['low'].values
        opens = df['open'].values
        pcts = df['pct_change'].values

        for i in range(65, len(df) - hold_days):
            window = df.iloc[:i+1].reset_index(drop=True)
            tech = factor_engine.calc_technical_factors(window)
            ret5 = (closes[i] / closes[i-5] - 1) * 100
            ret10 = (closes[i] / closes[i-10] - 1) * 100 if i >= 10 else 0
            rsi6 = calc_rsi(closes[:i+1], 6)
            vr = float(tech.get('volume_ratio', 1))
            ma20 = float(tech.get('ma20', closes[i]))
            ma60 = float(tech.get('ma60', closes[i])) if not np.isnan(tech.get('ma60', np.nan)) else closes[i]
            cd = sum(1 for j in range(i, max(i-10,0),-1) if closes[j] < closes[j-1])
            dist_ma20 = (closes[i] / ma20 - 1) * 100 if ma20 > 0 else 0
            volatility = np.std(np.diff(closes[i-20:i]) / closes[i-20:i-1]) if i >= 20 else 0
            amplitude = (highs[i] - lows[i]) / closes[i-1] * 100 if closes[i-1] > 0 else 0
            vol_ma20 = np.mean(volumes[i-20:i]) if i >= 20 else volumes[i]
            vol_shrink = volumes[i] / vol_ma20 if vol_ma20 > 0 else 1
            kdj_j = float(tech.get('kdj_j', 50))
            kdj_golden = tech.get('kdj_golden_cross', 0)
            macd_hist = float(tech.get('macd_hist', 0))
            near_boll = closes[i] < float(tech.get('boll_lower', closes[i])) * 1.02
            body = abs(closes[i] - opens[i])
            lower_shadow = min(closes[i], opens[i]) - lows[i]
            upper_shadow = highs[i] - max(closes[i], opens[i])
            hammer = int(lower_shadow > 2 * body and upper_shadow < body) if body > 0 else 0

            if not market_trend(market_df, df.iloc[i]['date']): continue
            if require_ma60 and closes[i] <= ma60: continue
            if volatility < min_vol: continue

            buy = closes[i]
            sell = closes[i + hold_days]
            hold_ret = (sell / buy - 1) * 100
            period_high = np.max(highs[i+1:i+hold_days+1])
            period_low = np.min(lows[i+1:i+hold_days+1])

            all_signals.append({
                'symbol': sym, 'date': df.iloc[i]['date'],
                'ret5': ret5, 'ret10': ret10, 'rsi6': rsi6, 'vr': vr,
                'cd': cd, 'dist_ma20': dist_ma20, 'volatility': volatility,
                'amplitude': amplitude, 'vol_shrink': vol_shrink,
                'kdj_j': kdj_j, 'kdj_golden': int(kdj_golden),
                'macd_hist': macd_hist, 'near_boll_lower': int(near_boll),
                'hammer': hammer, 'pct': float(pcts[i]),
                'hold_ret': hold_ret,
                'max_gain': (period_high / buy - 1) * 100,
                'max_loss': (period_low / buy - 1) * 100,
                'win': int(hold_ret > 0),
            })
    print(f"\n  共收集 {len(all_signals)} 个信号 (min_vol={min_vol}, ma60={require_ma60})")
    return pd.DataFrame(all_signals)


def calc_score(row, params):
    score = 50
    ret5, rsi6, vr = row['ret5'], row['rsi6'], row['vr']
    cd, dist_ma20, ret10 = row['cd'], row['dist_ma20'], row['ret10']
    kdj_j, kdj_golden = row['kdj_j'], row['kdj_golden']
    macd_hist = row['macd_hist']
    near_boll = row['near_boll_lower']
    hammer = row['hammer']
    vol_shrink = row['vol_shrink']

    if ret5 <= -15: score += params['p_ret5_extreme']
    elif ret5 <= -12: score += params['p_ret5_strong']
    elif ret5 <= -10: score += params['p_ret5_mild']
    elif ret5 <= -8: score += params['p_ret5_weak']
    elif ret5 > 5: score -= params['p_ret5_ob']

    if rsi6 < 20: score += params['p_rsi_extreme']
    elif rsi6 < 25: score += params['p_rsi_strong']
    elif rsi6 < 30: score += params['p_rsi_mild']
    elif rsi6 < 35: score += params['p_rsi_weak']
    elif rsi6 > 75: score -= params['p_rsi_ob']
    elif rsi6 > 70: score -= params['p_rsi_high']

    if vol_shrink < 0.5: score += params['p_vs_extreme']
    elif vol_shrink < 0.7: score += params['p_vs_strong']
    elif vol_shrink < 0.9: score += params['p_vs_mild']
    elif vr > 2.0 and ret5 < 0: score -= params['p_vr_panic']

    if cd >= 5: score += params['p_cd_5']
    elif cd >= 4: score += params['p_cd_4']
    elif cd >= 3: score += params['p_cd_3']

    if -15 < dist_ma20 < -5: score += params['p_dist']
    elif dist_ma20 < -15: score -= params['p_dist_break']

    if ret10 <= -20: score -= params['p_ret10_over']
    elif ret10 <= -15: score += params['p_ret10_mild']

    if kdj_j < 0: score += params['p_kdj_extreme']
    elif kdj_j < 20: score += params['p_kdj_oversold']
    if kdj_golden and kdj_j < 50: score += params['p_kdj_golden']
    if macd_hist > 0: score += params['p_macd_pos']
    if near_boll: score += params['p_boll']
    if hammer: score += params['p_hammer']

    return max(0, min(100, score))


def evaluate(signals_df, params, min_score, conds=None):
    df = signals_df.copy()
    df['score'] = df.apply(lambda r: calc_score(r, params), axis=1)
    f = df[df['score'] >= min_score]
    if conds:
        for c in conds:
            f = c(f)
    if len(f) == 0:
        return {'win_rate': 0, 'count': 0, 'avg_ret': 0}
    return {'win_rate': f['win'].mean()*100, 'count': len(f), 'avg_ret': f['hold_ret'].mean()}


def main():
    print("=" * 70)
    print("  扩展搜索 - 寻找信号>=30且胜率>=70%的配置")
    print("=" * 70)

    market_df = load_market(600)

    # 尝试不同过滤严格度
    configs = [
        (0.025, True, "波动率>2.5%, MA60上方"),
        (0.02, True, "波动率>2%, MA60上方"),
        (0.02, False, "波动率>2%, 无MA60限制"),
        (0.015, True, "波动率>1.5%, MA60上方"),
    ]

    params = {
        'p_ret5_extreme': 18, 'p_ret5_strong': 14, 'p_ret5_mild': 9, 'p_ret5_weak': 4, 'p_ret5_ob': 8,
        'p_rsi_extreme': 18, 'p_rsi_strong': 18, 'p_rsi_mild': 9, 'p_rsi_weak': 4,
        'p_rsi_ob': 12, 'p_rsi_high': 8,
        'p_vs_extreme': 12, 'p_vs_strong': 8, 'p_vs_mild': 4, 'p_vr_panic': 6,
        'p_cd_5': 8, 'p_cd_4': 5, 'p_cd_3': 2,
        'p_dist': 6, 'p_dist_break': 6,
        'p_ret10_over': 10, 'p_ret10_mild': 4,
        'p_kdj_extreme': 10, 'p_kdj_oversold': 6, 'p_kdj_golden': 5,
        'p_macd_pos': 4, 'p_boll': 6, 'p_hammer': 8,
    }

    cond_options = [
        ([], '无条件'),
        ([lambda df: df[df['kdj_j'] < 20]], 'KDJ_J<20'),
        ([lambda df: df[df['kdj_j'] < 25]], 'KDJ_J<25'),
        ([lambda df: df[df['rsi6'] < 30]], 'RSI6<30'),
        ([lambda df: df[df['rsi6'] < 35]], 'RSI6<35'),
        ([lambda df: df[df['ret5'] <= -10]], '5日跌>=10%'),
        ([lambda df: df[df['ret5'] <= -8]], '5日跌>=8%'),
        ([lambda df: df[df['cd'] >= 3]], '连跌>=3天'),
        ([lambda df: df[df['cd'] >= 4]], '连跌>=4天'),
        ([lambda df: df[df['vol_shrink'] < 0.8]], '缩量<0.8'),
        ([lambda df: df[df['near_boll_lower'] == 1]], '触布林下轨'),
        ([lambda df: df[df['hammer'] == 1]], '锤子线'),
        ([lambda df: df[(df['kdj_j'] < 20) & (df['rsi6'] < 30)]], 'KDJ<20&RSI<30'),
        ([lambda df: df[(df['kdj_j'] < 25) & (df['ret5'] <= -10)]], 'KDJ<25&跌>=10%'),
        ([lambda df: df[(df['rsi6'] < 30) & (df['vol_shrink'] < 0.85)]], 'RSI<30&缩量'),
        ([lambda df: df[(df['cd'] >= 3) & (df['rsi6'] < 35)]], '连跌>=3&RSI<35'),
        ([lambda df: df[(df['near_boll_lower'] == 1) & (df['rsi6'] < 35)]], '布林下轨&RSI<35'),
        ([lambda df: df[(df['kdj_j'] < 25) & (df['vol_shrink'] < 0.9)]], 'KDJ<25&缩量'),
    ]

    all_results = []

    for min_vol, req_ma60, desc in configs:
        print(f"\n--- {desc} ---")
        signals_df = collect_all_signals(SYMBOLS, market_df, min_vol=min_vol, require_ma60=req_ma60)
        if signals_df.empty:
            continue

        for ms in range(65, 85, 1):
            for conds, cname in cond_options:
                res = evaluate(signals_df, params, ms, conds)
                if res['win_rate'] >= 70 and res['count'] >= 15:
                    all_results.append({
                        'config': desc, 'min_score': ms, 'condition': cname,
                        'win_rate': res['win_rate'], 'count': res['count'],
                        'avg_ret': res['avg_ret'],
                    })

    if all_results:
        # 按胜率降序，同胜率按信号数降序
        all_results.sort(key=lambda x: (-x['win_rate'], -x['count']))
        print(f"\n{'='*90}")
        print(f"  胜率>=70%的配置 (共{len(all_results)}个，展示前15)")
        print(f"{'='*90}")
        print(f"{'胜率':>8} {'信号数':>8} {'收益':>8} {'阈值':>6} {'条件':<25} {'过滤配置'}")
        print("-" * 90)
        for r in all_results[:15]:
            print(f"{r['win_rate']:>7.1f}% {r['count']:>8} {r['avg_ret']:>+7.2f}% "
                  f"{r['min_score']:>6} {r['condition']:<25} {r['config']}")

        # 找信号数最多的
        best_by_count = max(all_results, key=lambda x: x['count'])
        print(f"\n信号数最多的配置:")
        print(f"  胜率={best_by_count['win_rate']:.1f}%, 信号={best_by_count['count']}, "
              f"收益={best_by_count['avg_ret']:+.2f}%")
        print(f"  阈值={best_by_count['min_score']}, 条件={best_by_count['condition']}")

        # 找胜率最高的
        best_by_wr = max(all_results, key=lambda x: x['win_rate'])
        print(f"\n胜率最高的配置:")
        print(f"  胜率={best_by_wr['win_rate']:.1f}%, 信号={best_by_wr['count']}, "
              f"收益={best_by_wr['avg_ret']:+.2f}%")
        print(f"  阈值={best_by_wr['min_score']}, 条件={best_by_wr['condition']}")

        # 综合最优：胜率>=70%且信号数最多
        print(f"\n{'='*90}")
        print(f"  综合推荐配置（胜率>=70%且信号数最多）")
        print(f"{'='*90}")
        print(f"  胜率: {best_by_count['win_rate']:.1f}%")
        print(f"  信号数: {best_by_count['count']}")
        print(f"  平均收益: {best_by_count['avg_ret']:+.2f}%")
        print(f"  评分阈值: {best_by_count['min_score']}")
        print(f"  硬条件: {best_by_count['condition']}")
        print(f"  过滤: {best_by_count['config']}")
    else:
        print("\n未找到胜率>=70%且信号>=15的配置")


if __name__ == '__main__':
    main()
