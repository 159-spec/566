"""
因子优化脚本 - 目标：将整体胜率提升至70%以上
策略：
1. 收集所有信号及其因子值
2. 分析各因子与胜率的相关性
3. 迭代优化评分函数权重/阈值
4. 绘制回测 vs 实际对比图
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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fetcher = DataFetcher()
factor_engine = FactorEngine()

OUTPUT_DIR = r'c:\Users\cks\Documents\Trae-CN\default'

# 扩大股票池
SYMBOLS = [
    '002579', '600519', '000858', '601318', '000001', '600036', '002594', '300750',
    '600900', '601899', '000333', '600276', '000651', '601166', '600887', '002475',
    '600030', '000725', '000568', '600809', '002415', '300760', '600031', '000063',
    '601398', '601288', '600036', '000002', '601318', '600009', '601888', '600276',
    '300059', '002415', '002230', '300760', '603259', '600438', '601012', '002594',
]
SYMBOLS = list(dict.fromkeys(SYMBOLS))  # 去重


def load_market(days=600):
    print("加载大盘数据...")
    return fetcher.get_daily_history('000001',
        start_date=(datetime.now() - timedelta(days=days)).strftime('%Y%m%d'))


def market_trend(market_df, date):
    m = market_df[market_df['date'] <= date]
    if len(m) < 20:
        return True
    c = m['close'].values
    return c[-1] > np.mean(c[-20:])


def collect_all_signals(symbols, market_df, days=500, hold_days=5):
    """收集所有信号及其因子值，用于优化分析"""
    all_signals = []
    for idx, sym in enumerate(symbols):
        print(f"  收集 {sym} ({idx+1}/{len(symbols)})...", end='\r')
        df = fetcher.get_daily_history(sym,
            start_date=(datetime.now() - timedelta(days=days+60)).strftime('%Y%m%d'))
        if df.empty or len(df) < 65:
            continue
        closes = df['close'].values
        volumes = df['volume'].values
        highs = df['high'].values
        lows = df['low'].values
        pcts = df['pct_change'].values

        for i in range(65, len(df) - hold_days):
            window = df.iloc[:i+1].reset_index(drop=True)
            tech = factor_engine.calc_technical_factors(window)

            # 计算各因子
            ret5 = (closes[i] / closes[i-5] - 1) * 100
            ret10 = (closes[i] / closes[i-10] - 1) * 100 if i >= 10 else 0
            ret20 = (closes[i] / closes[i-20] - 1) * 100 if i >= 20 else 0
            rsi6 = calc_rsi(closes[:i+1], 6)
            rsi14 = float(tech.get('rsi14', 50))
            vr = float(tech.get('volume_ratio', 1))
            vr20 = float(tech.get('volume_ratio_20', 1))
            ma5 = float(tech.get('ma5', closes[i]))
            ma10 = float(tech.get('ma10', closes[i]))
            ma20 = float(tech.get('ma20', closes[i]))
            ma60 = float(tech.get('ma60', closes[i])) if not np.isnan(tech.get('ma60', np.nan)) else closes[i]
            pct = float(pcts[i])
            above_ma20 = closes[i] > ma20
            above_ma60 = closes[i] > ma60
            ma_bullish = ma5 > ma10 > ma20
            ma_bearish = ma5 < ma10 < ma20

            # 连续下跌天数
            cd = 0
            for j in range(i, max(i-10, 0), -1):
                if closes[j] < closes[j-1]:
                    cd += 1
                else:
                    break

            # 距MA20距离
            dist_ma20 = (closes[i] / ma20 - 1) * 100 if ma20 > 0 else 0

            # 波动率
            if i >= 20:
                daily_ret = np.diff(closes[i-20:i]) / closes[i-20:i-1]
                volatility = np.std(daily_ret)
            else:
                volatility = 0

            # 振幅
            amplitude = (highs[i] - lows[i]) / closes[i-1] * 100 if closes[i-1] > 0 else 0

            # 大盘过滤
            if not market_trend(market_df, df.iloc[i]['date']):
                continue
            # 中期趋势过滤
            if not above_ma60:
                continue
            # 波动率过滤
            if volatility < 0.025:
                continue

            # 计算持有收益
            buy = closes[i]
            sell = closes[i + hold_days]
            hold_ret = (sell / buy - 1) * 100
            period_high = np.max(highs[i+1:i+hold_days+1])
            period_low = np.min(lows[i+1:i+hold_days+1])
            max_gain = (period_high / buy - 1) * 100
            max_loss = (period_low / buy - 1) * 100

            all_signals.append({
                'symbol': sym,
                'date': df.iloc[i]['date'],
                'ret5': ret5,
                'ret10': ret10,
                'ret20': ret20,
                'rsi6': rsi6,
                'rsi14': rsi14,
                'vr': vr,
                'vr20': vr20,
                'ma_bullish': int(ma_bullish),
                'ma_bearish': int(ma_bearish),
                'above_ma20': int(above_ma20),
                'dist_ma20': dist_ma20,
                'cd': cd,
                'volatility': volatility,
                'amplitude': amplitude,
                'pct': pct,
                'hold_ret': hold_ret,
                'max_gain': max_gain,
                'max_loss': max_loss,
                'win': int(hold_ret > 0),
            })

    print(f"\n  共收集 {len(all_signals)} 个信号")
    return pd.DataFrame(all_signals)


def calc_score_v5(row, params):
    """优化版评分函数 v5"""
    score = 50
    ret5 = row['ret5']
    rsi6 = row['rsi6']
    vr = row['vr']
    cd = row['cd']
    dist_ma20 = row['dist_ma20']
    ret10 = row['ret10']
    amplitude = row['amplitude']
    volatility = row['volatility']

    # 超跌幅度得分
    if ret5 <= -15:
        score += params.get('p_ret5_extreme', 18)
    elif ret5 <= -12:
        score += params.get('p_ret5_strong', 14)
    elif ret5 <= -10:
        score += params.get('p_ret5_mild', 9)
    elif ret5 <= -8:
        score += params.get('p_ret5_weak', 4)
    elif ret5 > 5:
        score -= params.get('p_ret5_overbought', 8)

    # RSI得分
    if rsi6 < 20:
        score += params.get('p_rsi_extreme', 18)
    elif rsi6 < 25:
        score += params.get('p_rsi_strong', 14)
    elif rsi6 < 30:
        score += params.get('p_rsi_mild', 9)
    elif rsi6 < 35:
        score += params.get('p_rsi_weak', 4)
    elif rsi6 > 75:
        score -= params.get('p_rsi_overbought', 12)
    elif rsi6 > 70:
        score -= params.get('p_rsi_high', 8)

    # 缩量得分（地量见底）
    if vr < 0.6:
        score += params.get('p_vr_extreme_shrink', 10)
    elif vr < 0.8:
        score += params.get('p_vr_shrink', 6)
    elif vr > 2.0 and ret5 < 0:
        score -= params.get('p_vr_panic', 6)

    # 连续下跌得分
    if cd >= 5:
        score += params.get('p_cd_5', 8)
    elif cd >= 4:
        score += params.get('p_cd_4', 5)
    elif cd >= 3:
        score += params.get('p_cd_3', 2)

    # 距MA20距离（超跌越深越好，但不能破位太深）
    if -15 < dist_ma20 < -5:
        score += params.get('p_dist_ma20', 6)
    elif dist_ma20 < -15:
        score -= params.get('p_dist_ma20_break', 6)

    # 10日跌幅（中期超跌）
    if ret10 <= -20:
        score -= params.get('p_ret10_over', 10)
    elif ret10 <= -15:
        score += params.get('p_ret10_mild', 4)

    # 振幅（足够活跃）
    if amplitude > 5:
        score += params.get('p_amplitude', 3)

    return max(0, min(100, score))


def evaluate_params(signals_df, params, min_score):
    """评估一组参数的胜率"""
    signals_df = signals_df.copy()
    signals_df['score'] = signals_df.apply(lambda r: calc_score_v5(r, params), axis=1)
    filtered = signals_df[signals_df['score'] >= min_score]
    if len(filtered) == 0:
        return {'win_rate': 0, 'count': 0, 'avg_ret': 0}
    win_rate = filtered['win'].mean() * 100
    avg_ret = filtered['hold_ret'].mean()
    return {'win_rate': win_rate, 'count': len(filtered), 'avg_ret': avg_ret}


def optimize_factors(signals_df):
    """迭代优化因子参数"""
    print("\n" + "=" * 70)
    print("  开始因子优化（网格搜索）")
    print("=" * 70)

    # 基础参数
    base_params = {
        'p_ret5_extreme': 18, 'p_ret5_strong': 14, 'p_ret5_mild': 9, 'p_ret5_weak': 4,
        'p_rsi_extreme': 18, 'p_rsi_strong': 14, 'p_rsi_mild': 9, 'p_rsi_weak': 4,
        'p_vr_extreme_shrink': 10, 'p_vr_shrink': 6, 'p_cd_5': 8, 'p_cd_4': 5, 'p_cd_3': 2,
        'p_dist_ma20': 6,
    }

    best_result = {'win_rate': 0, 'count': 0}
    best_params = base_params.copy()
    best_min_score = 60

    # 网格搜索各参数
    param_ranges = {
        'p_ret5_strong': [10, 14, 18, 22],
        'p_rsi_strong': [10, 14, 18, 22],
        'p_rsi_mild': [5, 9, 13, 17],
        'p_vr_shrink': [3, 6, 9, 12],
        'p_cd_4': [2, 5, 8, 11],
        'p_dist_ma20': [3, 6, 9, 12],
    }

    # 先搜索评分阈值
    print("  搜索最佳评分阈值...")
    for ms in [55, 58, 60, 62, 65, 68, 70, 72, 75]:
        res = evaluate_params(signals_df, base_params, ms)
        if res['count'] >= 30 and res['win_rate'] > best_result['win_rate']:
            best_result = res
            best_min_score = ms
            print(f"    阈值={ms}: 胜率={res['win_rate']:.1f}%, 信号={res['count']} ✓")

    print(f"  最佳阈值: {best_min_score}, 胜率: {best_result['win_rate']:.1f}%")

    # 逐个优化参数
    print("\n  逐个优化因子权重...")
    for param_name, values in param_ranges.items():
        for val in values:
            test_params = best_params.copy()
            test_params[param_name] = val
            res = evaluate_params(signals_df, test_params, best_min_score)
            if res['count'] >= 30 and res['win_rate'] > best_result['win_rate']:
                best_result = res
                best_params = test_params
                print(f"    {param_name}={val}: 胜率={res['win_rate']:.1f}%, 信号={res['count']} ✓")

    # 第二轮微调阈值
    print("\n  第二轮阈值微调...")
    for ms in range(best_min_score - 3, best_min_score + 6):
        res = evaluate_params(signals_df, best_params, ms)
        if res['count'] >= 30 and res['win_rate'] > best_result['win_rate']:
            best_result = res
            best_min_score = ms
            print(f"    阈值={ms}: 胜率={res['win_rate']:.1f}%, 信号={res['count']} ✓")

    print(f"\n  优化完成:")
    print(f"    最佳胜率: {best_result['win_rate']:.1f}%")
    print(f"    信号数量: {best_result['count']}")
    print(f"    平均收益: {best_result['avg_ret']:+.2f}%")
    print(f"    评分阈值: {best_min_score}")

    return best_params, best_min_score, best_result


def plot_results(signals_df, best_params, best_min_score):
    """绘制回测 vs 实际对比图"""
    signals_df = signals_df.copy()
    signals_df['score'] = signals_df.apply(lambda r: calc_score_v5(r, best_params), axis=1)
    filtered = signals_df[signals_df['score'] >= best_min_score].copy()

    # 按日期排序
    filtered = filtered.sort_values('date').reset_index(drop=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f'因子优化回测结果 (胜率={filtered["win"].mean()*100:.1f}%, 信号数={len(filtered)})',
                 fontsize=14, fontweight='bold')

    # 1. 累计收益曲线 - 策略 vs 持有
    ax = axes[0, 0]
    if len(filtered) > 0:
        # 策略累计收益（按信号顺序）
        filtered['cum_ret'] = (1 + filtered['hold_ret'] / 100).cumprod()
        ax.plot(range(len(filtered)), filtered['cum_ret'], 'b-', label='策略累计收益', linewidth=2)
        # 基准：全市场平均持有收益
        baseline = (1 + signals_df['hold_ret'] / 100).cumprod()
        ax.plot(range(len(signals_df)), baseline, 'r--', label='全部信号基准', alpha=0.6)
        ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
        ax.set_title('策略累计收益 vs 基准')
        ax.set_xlabel('信号序号')
        ax.set_ylabel('累计净值')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 2. 胜率分布（按评分区间）
    ax = axes[0, 1]
    score_bins = [(best_min_score, 65), (65, 70), (70, 75), (75, 100)]
    bin_labels = [f'{a}-{b}' for a, b in score_bins]
    win_rates = []
    counts = []
    for a, b in score_bins:
        sub = filtered[(filtered['score'] >= a) & (filtered['score'] < b)]
        win_rates.append(sub['win'].mean() * 100 if len(sub) > 0 else 0)
        counts.append(len(sub))
    x = range(len(bin_labels))
    bars = ax.bar(x, win_rates, color=['#ff9999', '#ffcc99', '#99ff99', '#66b3ff'])
    ax.axhline(y=70, color='red', linestyle='--', label='70%目标线')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_title('各评分区间胜率')
    ax.set_xlabel('评分区间')
    ax.set_ylabel('胜率 (%)')
    ax.set_ylim(0, 100)
    ax.legend()
    for i, (bar, cnt) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{win_rates[i]:.0f}%\n(n={cnt})', ha='center', va='bottom', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. 收益分布直方图
    ax = axes[1, 0]
    if len(filtered) > 0:
        ax.hist(filtered['hold_ret'], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(x=0, color='red', linestyle='--', label='0%')
        ax.axvline(x=filtered['hold_ret'].mean(), color='green', linestyle='--',
                   label=f'均值={filtered["hold_ret"].mean():.2f}%')
        ax.set_title('策略收益分布')
        ax.set_xlabel('持有5日收益 (%)')
        ax.set_ylabel('频次')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 4. 因子相关性热力图
    ax = axes[1, 1]
    factor_cols = ['ret5', 'rsi6', 'vr', 'cd', 'dist_ma20', 'ret10', 'amplitude', 'volatility', 'win']
    corr = filtered[factor_cols].corr()
    im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1)
    ax.set_xticks(range(len(factor_cols)))
    ax.set_yticks(range(len(factor_cols)))
    ax.set_xticklabels(factor_cols, rotation=45, ha='right')
    ax.set_yticklabels(factor_cols)
    ax.set_title('因子与胜负相关性')
    for i in range(len(factor_cols)):
        for j in range(len(factor_cols)):
            ax.text(j, i, f'{corr.values[i,j]:.2f}', ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=ax)

    # 5. 最大涨幅 vs 最大跌幅散点
    ax = axes[2, 0]
    if len(filtered) > 0:
        wins = filtered[filtered['win'] == 1]
        losses = filtered[filtered['win'] == 0]
        ax.scatter(wins['max_loss'], wins['max_gain'], c='green', alpha=0.6, label='盈利', s=30)
        ax.scatter(losses['max_loss'], losses['max_gain'], c='red', alpha=0.6, label='亏损', s=30)
        ax.set_xlabel('期间最大跌幅 (%)')
        ax.set_ylabel('期间最大涨幅 (%)')
        ax.set_title('盈利/亏损交易的波动特征')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 6. 月度胜率热力图
    ax = axes[2, 1]
    if len(filtered) > 0:
        filtered['month'] = pd.to_datetime(filtered['date']).dt.to_period('M')
        monthly = filtered.groupby('month').agg(
            win_rate=('win', 'mean'),
            count=('win', 'count')
        ).reset_index()
        monthly['win_rate'] = monthly['win_rate'] * 100
        ax.bar(range(len(monthly)), monthly['win_rate'],
               color=['green' if w >= 70 else 'orange' if w >= 50 else 'red' for w in monthly['win_rate']])
        ax.axhline(y=70, color='darkgreen', linestyle='--', label='70%目标')
        ax.axhline(y=50, color='gray', linestyle=':', label='50%基准')
        ax.set_xticks(range(len(monthly)))
        ax.set_xticklabels([str(m) for m in monthly['month']], rotation=45, ha='right', fontsize=7)
        ax.set_title('月度胜率趋势')
        ax.set_ylabel('胜率 (%)')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'backtest_optimization_result.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {save_path}")
    return save_path


def plot_backtest_vs_actual(symbols, best_params, best_min_score, hold_days=5):
    """绘制回测预测 vs 实际结果对比图（以个股为例）"""
    print("\n绘制回测预测 vs 实际结果对比图...")

    # 选择3只有代表性的股票
    demo_symbols = symbols[:3]
    fig, axes = plt.subplots(len(demo_symbols), 1, figsize=(16, 4 * len(demo_symbols)))
    if len(demo_symbols) == 1:
        axes = [axes]

    for idx, sym in enumerate(demo_symbols):
        ax = axes[idx]
        df = fetcher.get_daily_history(sym,
            start_date=(datetime.now() - timedelta(days=500)).strftime('%Y%m%d'))
        if df.empty or len(df) < 100:
            continue

        closes = df['close'].values
        dates = df['date'].values

        # 找信号点
        signal_points = []
        for i in range(65, len(df) - hold_days):
            window = df.iloc[:i+1].reset_index(drop=True)
            tech = factor_engine.calc_technical_factors(window)
            ret5 = (closes[i] / closes[i-5] - 1) * 100
            rsi6 = calc_rsi(closes[:i+1], 6)
            vr = float(tech.get('volume_ratio', 1))
            cd = 0
            for j in range(i, max(i-10, 0), -1):
                if closes[j] < closes[j-1]:
                    cd += 1
                else:
                    break
            ma20 = float(tech.get('ma20', closes[i]))
            dist_ma20 = (closes[i] / ma20 - 1) * 100 if ma20 > 0 else 0
            ret10 = (closes[i] / closes[i-10] - 1) * 100 if i >= 10 else 0
            amplitude = (df['high'].values[i] - df['low'].values[i]) / closes[i-1] * 100 if closes[i-1] > 0 else 0
            volatility = np.std(np.diff(closes[i-20:i]) / closes[i-20:i-1]) if i >= 20 else 0
            ma60 = float(tech.get('ma60', closes[i])) if not np.isnan(tech.get('ma60', np.nan)) else closes[i]
            if closes[i] <= ma60 or volatility < 0.025:
                continue

            row = pd.Series({
                'ret5': ret5, 'rsi6': rsi6, 'vr': vr, 'cd': cd,
                'dist_ma20': dist_ma20, 'ret10': ret10, 'amplitude': amplitude,
                'volatility': volatility, 'pct': df['pct_change'].values[i],
            })
            score = calc_score_v5(row, best_params)
            if score >= best_min_score:
                actual_ret = (closes[i+hold_days] / closes[i] - 1) * 100
                signal_points.append({
                    'idx': i,
                    'date': dates[i],
                    'price': closes[i],
                    'score': score,
                    'actual_ret': actual_ret,
                    'win': actual_ret > 0,
                })

        # 绘制K线收盘价
        ax.plot(range(len(closes)), closes, 'k-', linewidth=1, label='收盘价', alpha=0.8)

        # 绘制MA20
        ma20_line = [np.mean(closes[max(0, i-20):i+1]) for i in range(len(closes))]
        ax.plot(range(len(closes)), ma20_line, 'b-', linewidth=0.8, label='MA20', alpha=0.6)

        # 标记信号点
        if signal_points:
            buy_points = [s for s in signal_points if s['win']]
            sell_points = [s for s in signal_points if not s['win']]
            if buy_points:
                ax.scatter([s['idx'] for s in buy_points], [s['price'] for s in buy_points],
                          c='green', marker='^', s=100, label=f'买入信号(盈利, {len(buy_points)}个)', zorder=5)
            if sell_points:
                ax.scatter([s['idx'] for s in sell_points], [s['price'] for s in sell_points],
                          c='red', marker='v', s=100, label=f'买入信号(亏损, {len(sell_points)}个)', zorder=5)

            # 标注几个信号的实际收益
            for sp in signal_points[::max(1, len(signal_points)//8)]:
                ax.annotate(f'{sp["actual_ret"]:+.1f}%',
                           xy=(sp['idx'], sp['price']),
                           xytext=(0, 15), textcoords='offset points',
                           fontsize=7, ha='center',
                           color='green' if sp['win'] else 'red')

        total_signals = len(signal_points)
        win_signals = len([s for s in signal_points if s['win']])
        wr = win_signals / total_signals * 100 if total_signals > 0 else 0
        ax.set_title(f'{sym} - 回测信号 vs 实际结果 (信号数:{total_signals}, 胜率:{wr:.1f}%)')
        ax.set_ylabel('价格')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        # 设置x轴日期
        tick_positions = list(range(0, len(closes), max(1, len(closes)//10)))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(dates[p])[:10] for p in tick_positions], rotation=45, fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'backtest_vs_actual.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"对比图已保存: {save_path}")
    return save_path


def main():
    print("=" * 70)
    print("  A股因子优化系统 - 目标胜率 >= 70%")
    print("=" * 70)

    # 1. 加载数据
    market_df = load_market(600)

    # 2. 收集所有信号
    print("\n收集历史信号数据...")
    signals_df = collect_all_signals(SYMBOLS, market_df, days=500, hold_days=5)

    if signals_df.empty:
        print("未收集到信号数据")
        return

    print(f"\n总信号数: {len(signals_df)}")
    print(f"整体胜率: {signals_df['win'].mean()*100:.1f}%")
    print(f"平均收益: {signals_df['hold_ret'].mean():+.2f}%")

    # 3. 优化因子
    best_params, best_min_score, best_result = optimize_factors(signals_df)

    # 4. 打印最终参数
    print("\n" + "=" * 70)
    print("  最佳因子参数")
    print("=" * 70)
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    print(f"    评分阈值: {best_min_score}")
    print(f"    最终胜率: {best_result['win_rate']:.1f}%")
    print(f"    信号数量: {best_result['count']}")
    print(f"    平均收益: {best_result['avg_ret']:+.2f}%")

    # 5. 检查是否达标
    if best_result['win_rate'] >= 70:
        print(f"\n  ✓ 目标达成！胜率 {best_result['win_rate']:.1f}% >= 70%")
    else:
        print(f"\n  ✗ 未达标，胜率 {best_result['win_rate']:.1f}% < 70%，继续优化...")

    # 6. 绘制结果图
    plot_results(signals_df, best_params, best_min_score)

    # 7. 绘制回测 vs 实际对比图
    plot_backtest_vs_actual(SYMBOLS, best_params, best_min_score)

    # 8. 保存优化结果
    result_df = signals_df.copy()
    result_df['score'] = result_df.apply(lambda r: calc_score_v5(r, best_params), axis=1)
    filtered = result_df[result_df['score'] >= best_min_score]
    csv_path = os.path.join(OUTPUT_DIR, 'optimized_signals.csv')
    filtered.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n优化后信号已保存: {csv_path}")

    return best_params, best_min_score, best_result


if __name__ == '__main__':
    main()
