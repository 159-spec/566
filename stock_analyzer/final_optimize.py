"""
最终优化 - 在保持胜率>=70%的前提下最大化信号数量
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


def collect_all_signals(symbols, market_df, days=500, hold_days=5):
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
            if closes[i] <= ma60: continue
            if volatility < 0.025: continue

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
    print(f"\n  共收集 {len(all_signals)} 个信号")
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
    print("  最终优化 - 最大化信号数(胜率>=70%)")
    print("=" * 70)

    market_df = load_market(600)
    signals_df = collect_all_signals(SYMBOLS, market_df)
    print(f"\n总信号: {len(signals_df)}, 整体胜率: {signals_df['win'].mean()*100:.1f}%")

    # 已验证的最佳参数（来自v2）
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

    # 搜索：不同阈值 + 不同硬条件组合，找胜率>=70%且信号最多的
    print("\n搜索最优配置（胜率>=70%，最大化信号数）...")

    cond_options = [
        [],  # 无条件
        [lambda df: df[df['kdj_j'] < 20]],
        [lambda df: df[df['kdj_j'] < 25]],
        [lambda df: df[df['rsi6'] < 30]],
        [lambda df: df[df['ret5'] <= -10]],
        [lambda df: df[df['cd'] >= 3]],
        [lambda df: df[df['vol_shrink'] < 0.8]],
        [lambda df: df[df['near_boll_lower'] == 1]],
        [lambda df: df[df['hammer'] == 1]],
        [lambda df: df[(df['kdj_j'] < 20) & (df['rsi6'] < 30)]],
        [lambda df: df[(df['kdj_j'] < 25) & (df['ret5'] <= -10)]],
        [lambda df: df[(df['rsi6'] < 30) & (df['vol_shrink'] < 0.8)]],
        [lambda df: df[(df['kdj_j'] < 20) & (df['vol_shrink'] < 0.85)]],
        [lambda df: df[(df['cd'] >= 3) & (df['rsi6'] < 35)]],
    ]
    cond_names = [
        '无条件', 'KDJ_J<20', 'KDJ_J<25', 'RSI6<30', 'ret5<=-10', 'cd>=3',
        '缩量<0.8', '触及布林下轨', '锤子线',
        'KDJ<20 & RSI<30', 'KDJ<25 & ret5<=-10',
        'RSI<30 & 缩量<0.8', 'KDJ<20 & 缩量<0.85', 'cd>=3 & RSI<35',
    ]

    best = {'win_rate': 0, 'count': 0}
    best_config = None

    for min_score in range(70, 88, 1):
        for ci, conds in enumerate(cond_options):
            res = evaluate(signals_df, params, min_score, conds)
            if res['win_rate'] >= 70 and res['count'] > best['count']:
                best = res
                best_config = (min_score, ci, cond_names[ci])
                print(f"  ✓ 阈值={min_score}, 条件={cond_names[ci]}: "
                      f"胜率={res['win_rate']:.1f}%, 信号={res['count']}, "
                      f"收益={res['avg_ret']:+.2f}%")

    if best_config:
        ms, ci, cname = best_config
        print(f"\n{'='*70}")
        print(f"  最优配置")
        print(f"{'='*70}")
        print(f"  评分阈值: {ms}")
        print(f"  硬条件: {cname}")
        print(f"  胜率: {best['win_rate']:.1f}%")
        print(f"  信号数: {best['count']}")
        print(f"  平均收益: {best['avg_ret']:+.2f}%")
        print(f"  ✓ 胜率 >= 70% 目标达成!")

        # 生成最终图表
        df = signals_df.copy()
        df['score'] = df.apply(lambda r: calc_score(r, params), axis=1)
        f = df[df['score'] >= ms]
        for c in cond_options[ci]:
            f = c(f)
        f = f.sort_values('date').reset_index(drop=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'最终优化策略 (胜率={best["win_rate"]:.1f}%, 信号={best["count"]}, 条件={cname})',
                     fontsize=13, fontweight='bold')

        # 累计净值
        ax = axes[0, 0]
        f['cum'] = (1 + f['hold_ret']/100).cumprod()
        ax.plot(range(len(f)), f['cum'], 'b-', linewidth=2, label='策略累计净值')
        base = (1 + signals_df['hold_ret']/100).cumprod()
        ax.plot(range(len(signals_df)), base, 'r--', alpha=0.6, label='全部信号基准')
        ax.axhline(1, color='gray', linestyle=':')
        ax.set_title('累计净值曲线'); ax.set_xlabel('信号序号'); ax.set_ylabel('净值')
        ax.legend(); ax.grid(alpha=0.3)

        # 收益分布
        ax = axes[0, 1]
        ax.hist(f['hold_ret'], bins=20, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(0, color='red', linestyle='--', label='0%')
        ax.axvline(f['hold_ret'].mean(), color='green', linestyle='--',
                   label=f'均值={f["hold_ret"].mean():.2f}%')
        ax.set_title('收益分布'); ax.set_xlabel('5日收益(%)'); ax.set_ylabel('频次')
        ax.legend(); ax.grid(alpha=0.3)

        # 月度胜率
        ax = axes[1, 0]
        f['month'] = pd.to_datetime(f['date']).dt.to_period('M')
        m = f.groupby('month').agg(wr=('win','mean'), cnt=('win','count')).reset_index()
        m['wr'] = m['wr'] * 100
        colors = ['green' if w>=70 else 'orange' if w>=50 else 'red' for w in m['wr']]
        ax.bar(range(len(m)), m['wr'], color=colors)
        ax.axhline(70, color='darkgreen', linestyle='--', label='70%')
        ax.axhline(50, color='gray', linestyle=':', label='50%')
        ax.set_xticks(range(len(m)))
        ax.set_xticklabels([str(x) for x in m['month']], rotation=45, fontsize=7)
        ax.set_title('月度胜率'); ax.set_ylabel('胜率(%)'); ax.set_ylim(0,100)
        ax.legend(); ax.grid(alpha=0.3)

        # 个股胜率分布
        ax = axes[1, 1]
        sym_stats = f.groupby('symbol').agg(wr=('win','mean'), cnt=('win','count'),
                                              ret=('hold_ret','mean')).reset_index()
        sym_stats = sym_stats.sort_values('wr', ascending=True)
        colors = ['green' if w>=70 else 'orange' if w>=50 else 'red' for w in sym_stats['wr']]
        ax.barh(range(len(sym_stats)), sym_stats['wr'], color=colors)
        ax.axvline(70, color='darkgreen', linestyle='--')
        ax.set_yticks(range(len(sym_stats)))
        ax.set_yticklabels(sym_stats['symbol'], fontsize=8)
        ax.set_title('各股胜率'); ax.set_xlabel('胜率(%)')
        for i, (w, c) in enumerate(zip(sym_stats['wr'], sym_stats['cnt'])):
            ax.text(w+1, i, f'{w:.0f}%({int(c)})', va='center', fontsize=7)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        p = os.path.join(OUTPUT_DIR, 'final_optimization_result.png')
        plt.savefig(p, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"\n最终图表已保存: {p}")

        # 保存信号
        csv = os.path.join(OUTPUT_DIR, 'final_optimized_signals.csv')
        f.drop(columns=['cum','month'], errors='ignore').to_csv(csv, index=False, encoding='utf-8-sig')
        print(f"信号数据已保存: {csv}")
    else:
        print("\n未找到胜率>=70%的配置")


if __name__ == '__main__':
    main()
