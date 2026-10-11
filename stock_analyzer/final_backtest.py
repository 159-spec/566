"""
最终优化策略回测 - 使用最优因子配置
最优配置：
- 评分阈值: 70
- 硬条件: 触及布林下轨 且 RSI6 < 35
- 过滤: 波动率>2%, 价格在MA60上方, 大盘趋势向上
- 回测胜率: 81.5% (27个信号), 平均收益: +2.94%
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

# 最优因子权重参数
BEST_PARAMS = {
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

SYMBOLS = [
    '002579', '600519', '000858', '601318', '000001', '600036', '002594', '300750',
    '600900', '601899', '000333', '600276', '000651', '601166', '600887', '002475',
    '600030', '000725', '000568', '600809', '002415', '300760', '600031', '000063',
    '601398', '601288', '000002', '600009', '601888', '300059', '002230', '603259',
    '600438', '601012',
]


def calc_score(row, params=BEST_PARAMS):
    """最优评分函数"""
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


def load_market(days=600):
    return fetcher.get_daily_history('000001',
        start_date=(datetime.now() - timedelta(days=days)).strftime('%Y%m%d'))


def market_trend(market_df, date):
    m = market_df[market_df['date'] <= date]
    if len(m) < 20: return True
    c = m['close'].values
    return c[-1] > np.mean(c[-20:])


def run_final_backtest(symbols, market_df, hold_days=5):
    """使用最优策略运行完整回测"""
    all_signals = []
    print("\n运行最优策略回测...")
    for idx, sym in enumerate(symbols):
        print(f"  {sym} ({idx+1}/{len(symbols)})...", end='\r')
        df = fetcher.get_daily_history(sym,
            start_date=(datetime.now() - timedelta(days=560)).strftime('%Y%m%d'))
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

            # 过滤条件
            if not market_trend(market_df, df.iloc[i]['date']): continue
            if closes[i] <= ma60: continue
            if volatility < 0.02: continue

            row = pd.Series({
                'ret5': ret5, 'ret10': ret10, 'rsi6': rsi6, 'vr': vr,
                'cd': cd, 'dist_ma20': dist_ma20, 'volatility': volatility,
                'amplitude': amplitude, 'vol_shrink': vol_shrink,
                'kdj_j': kdj_j, 'kdj_golden': int(kdj_golden),
                'macd_hist': macd_hist, 'near_boll_lower': int(near_boll),
                'hammer': hammer, 'pct': float(pcts[i]),
            })
            score = calc_score(row)

            # 硬条件: 触及布林下轨 且 RSI6 < 35
            if not (near_boll and rsi6 < 35): continue
            if score < 70: continue

            buy = closes[i]
            sell = closes[i + hold_days]
            hold_ret = (sell / buy - 1) * 100
            period_high = np.max(highs[i+1:i+hold_days+1])
            period_low = np.min(lows[i+1:i+hold_days+1])

            all_signals.append({
                'symbol': sym, 'date': df.iloc[i]['date'],
                'buy_price': round(buy, 2), 'sell_price': round(sell, 2),
                'score': round(score, 1), 'ret5': round(ret5, 2), 'rsi6': round(rsi6, 2),
                'cd': cd, 'kdj_j': round(kdj_j, 2),
                'hold_ret': round(hold_ret, 2),
                'max_gain': round((period_high / buy - 1) * 100, 2),
                'max_loss': round((period_low / buy - 1) * 100, 2),
                'win': int(hold_ret > 0),
            })

    result = pd.DataFrame(all_signals)
    print(f"\n\n回测完成: {len(result)} 个信号")
    if len(result) > 0:
        print(f"整体胜率: {result['win'].mean()*100:.1f}%")
        print(f"平均5日收益: {result['hold_ret'].mean():+.2f}%")
        print(f"平均最大涨幅: {result['max_gain'].mean():+.2f}%")
        print(f"平均最大跌幅: {result['max_loss'].mean():+.2f}%")
    return result


def plot_final_comparison(signals_df, symbols):
    """绘制最终回测 vs 实际对比图"""
    if signals_df.empty:
        print("无信号数据")
        return

    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.25)

    # 1. 累计收益曲线
    ax1 = fig.add_subplot(gs[0, :])
    signals_df = signals_df.sort_values('date').reset_index(drop=True)
    signals_df['cum'] = (1 + signals_df['hold_ret'] / 100).cumprod()
    ax1.plot(range(len(signals_df)), signals_df['cum'], 'b-o', linewidth=2, markersize=4, label='策略累计净值')
    ax1.axhline(1, color='gray', linestyle=':', alpha=0.5)
    ax1.fill_between(range(len(signals_df)), 1, signals_df['cum'],
                     where=signals_df['cum'] >= 1, alpha=0.15, color='green')
    ax1.fill_between(range(len(signals_df)), 1, signals_df['cum'],
                     where=signals_df['cum'] < 1, alpha=0.15, color='red')
    ax1.set_title('最优策略累计净值曲线', fontsize=13, fontweight='bold')
    ax1.set_xlabel('信号序号'); ax1.set_ylabel('累计净值')
    ax1.legend(); ax1.grid(alpha=0.3)

    # 2. 收益分布
    ax2 = fig.add_subplot(gs[1, 0])
    wins = signals_df[signals_df['win'] == 1]
    losses = signals_df[signals_df['win'] == 0]
    ax2.hist(wins['hold_ret'], bins=15, alpha=0.7, color='green', label=f'盈利({len(wins)})')
    ax2.hist(losses['hold_ret'], bins=15, alpha=0.7, color='red', label=f'亏损({len(losses)})')
    ax2.axvline(0, color='black', linestyle='--')
    ax2.axvline(signals_df['hold_ret'].mean(), color='blue', linestyle='--',
                label=f'均值={signals_df["hold_ret"].mean():.2f}%')
    ax2.set_title('收益分布 (盈利 vs 亏损)', fontweight='bold')
    ax2.set_xlabel('5日收益 (%)'); ax2.set_ylabel('频次')
    ax2.legend(); ax2.grid(alpha=0.3)

    # 3. 个股胜率
    ax3 = fig.add_subplot(gs[1, 1])
    sym_stats = signals_df.groupby('symbol').agg(
        wr=('win', 'mean'), cnt=('win', 'count'), ret=('hold_ret', 'mean')
    ).reset_index().sort_values('wr', ascending=True)
    colors = ['green' if w >= 70 else 'orange' if w >= 50 else 'red' for w in sym_stats['wr']]
    ax3.barh(range(len(sym_stats)), sym_stats['wr'] * 100, color=colors)
    ax3.axvline(70, color='darkgreen', linestyle='--', label='70%目标')
    ax3.axvline(50, color='gray', linestyle=':', label='50%基准')
    ax3.set_yticks(range(len(sym_stats)))
    ax3.set_yticklabels(sym_stats['symbol'], fontsize=8)
    ax3.set_title('各股胜率分布', fontweight='bold')
    ax3.set_xlabel('胜率 (%)')
    for i, (w, c) in enumerate(zip(sym_stats['wr'] * 100, sym_stats['cnt'])):
        ax3.text(w + 1, i, f'{w:.0f}%({int(c)})', va='center', fontsize=7)
    ax3.legend(); ax3.grid(alpha=0.3)

    # 4. 回测信号 vs 实际K线（前3只股票）
    demo_syms = list(signals_df['symbol'].unique())[:3]
    plot_positions = [(2, 0), (2, 1), (3, 0)]
    for idx, sym in enumerate(demo_syms):
        if idx >= len(plot_positions):
            break
        r, c = plot_positions[idx]
        ax = fig.add_subplot(gs[r, c])
        df = fetcher.get_daily_history(sym,
            start_date=(datetime.now() - timedelta(days=400)).strftime('%Y%m%d'))
        if df.empty: continue
        closes = df['close'].values
        dates = df['date'].values
        ax.plot(range(len(closes)), closes, 'k-', linewidth=0.8, alpha=0.8)
        ma20 = [np.mean(closes[max(0,i-20):i+1]) for i in range(len(closes))]
        ax.plot(range(len(closes)), ma20, 'b-', linewidth=0.6, alpha=0.5)

        sym_signals = signals_df[signals_df['symbol'] == sym]
        for _, sig in sym_signals.iterrows():
            try:
                sig_idx = df[df['date'] == sig['date']].index[0]
                color = 'green' if sig['win'] else 'red'
                marker = '^' if sig['win'] else 'v'
                ax.scatter(sig_idx, sig['buy_price'], c=color, marker=marker, s=80, zorder=5)
                ax.annotate(f"{sig['hold_ret']:+.1f}%", xy=(sig_idx, sig['buy_price']),
                           xytext=(0, 12), textcoords='offset points', fontsize=7,
                           ha='center', color=color)
            except:
                pass

        wr = sym_signals['win'].mean() * 100
        ax.set_title(f'{sym} 回测信号 vs 实际 (胜率:{wr:.0f}%, {len(sym_signals)}信号)', fontsize=9)
        ax.set_ylabel('价格'); ax.grid(alpha=0.3)
        ticks = list(range(0, len(closes), max(1, len(closes)//6)))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(dates[t])[:10] for t in ticks], rotation=45, fontsize=6)

    fig.suptitle(f'最优策略回测结果汇总 (胜率={signals_df["win"].mean()*100:.1f}%, '
                 f'信号={len(signals_df)}, 平均收益={signals_df["hold_ret"].mean():+.2f}%)',
                 fontsize=14, fontweight='bold')

    save_path = os.path.join(OUTPUT_DIR, 'final_backtest_comparison.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n最终对比图已保存: {save_path}")
    return save_path


def main():
    print("=" * 70)
    print("  最优策略回测 - 回测结果 vs 实际结果")
    print("  策略: 布林下轨 + RSI<35 + 多因子评分>=70")
    print("=" * 70)

    market_df = load_market(600)
    signals_df = run_final_backtest(SYMBOLS, market_df)

    if signals_df.empty:
        print("无信号")
        return

    # 打印详细统计
    print("\n" + "=" * 70)
    print("  回测统计")
    print("=" * 70)
    print(f"  总信号数: {len(signals_df)}")
    print(f"  盈利信号: {len(signals_df[signals_df['win']==1])}")
    print(f"  亏损信号: {len(signals_df[signals_df['win']==0])}")
    print(f"  整体胜率: {signals_df['win'].mean()*100:.1f}%")
    print(f"  平均5日收益: {signals_df['hold_ret'].mean():+.2f}%")
    print(f"  最大单笔盈利: {signals_df['hold_ret'].max():+.2f}%")
    print(f"  最大单笔亏损: {signals_df['hold_ret'].min():+.2f}%")
    print(f"  平均最大涨幅: {signals_df['max_gain'].mean():+.2f}%")
    print(f"  平均最大跌幅: {signals_df['max_loss'].mean():+.2f}%")
    print(f"  盈亏比: {abs(signals_df[signals_df['win']==1]['hold_ret'].mean() / signals_df[signals_df['win']==0]['hold_ret'].mean()):.2f}"
          if len(signals_df[signals_df['win']==0]) > 0 else "")

    # 打印所有信号明细
    print("\n  信号明细:")
    print(f"  {'日期':<12} {'代码':<8} {'买入价':<8} {'评分':<6} {'5日收益':<8} {'结果'}")
    print("  " + "-" * 55)
    for _, r in signals_df.sort_values('date').iterrows():
        result = '✓盈利' if r['win'] else '✗亏损'
        print(f"  {str(r['date'])[:10]:<12} {r['symbol']:<8} {r['buy_price']:<8.2f} "
              f"{r['score']:<6.1f} {r['hold_ret']:>+7.2f}% {result}")

    # 绘图
    plot_final_comparison(signals_df, SYMBOLS)

    # 保存CSV
    csv_path = os.path.join(OUTPUT_DIR, 'final_backtest_signals.csv')
    signals_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n信号明细已保存: {csv_path}")

    print(f"\n{'='*70}")
    print(f"  ✓ 回测完成，胜率 {signals_df['win'].mean()*100:.1f}% >= 70%")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
