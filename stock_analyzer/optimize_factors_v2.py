"""
深度因子优化 v2 - 目标胜率 >= 70%
策略：
1. 分析各因子组合的胜率，找出最佳多因子共振条件
2. 引入更多A股特色因子（地量、支撑位、KDJ底背离等）
3. 使用严格的AND条件过滤而非纯评分
4. 多轮迭代搜索最佳参数组合
"""
import sys, os, warnings, itertools
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
    if len(m) < 20:
        return True
    c = m['close'].values
    return c[-1] > np.mean(c[-20:])


def collect_all_signals(symbols, market_df, days=500, hold_days=5):
    """收集所有信号及丰富因子"""
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
        opens = df['open'].values
        pcts = df['pct_change'].values

        for i in range(65, len(df) - hold_days):
            window = df.iloc[:i+1].reset_index(drop=True)
            tech = factor_engine.calc_technical_factors(window)

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

            cd = 0
            for j in range(i, max(i-10, 0), -1):
                if closes[j] < closes[j-1]:
                    cd += 1
                else:
                    break

            dist_ma20 = (closes[i] / ma20 - 1) * 100 if ma20 > 0 else 0
            dist_ma60 = (closes[i] / ma60 - 1) * 100 if ma60 > 0 else 0

            if i >= 20:
                daily_ret = np.diff(closes[i-20:i]) / closes[i-20:i-1]
                volatility = np.std(daily_ret)
            else:
                volatility = 0

            amplitude = (highs[i] - lows[i]) / closes[i-1] * 100 if closes[i-1] > 0 else 0

            # 量能萎缩程度（地量判断）
            vol_ma20 = np.mean(volumes[i-20:i]) if i >= 20 else volumes[i]
            vol_shrink = volumes[i] / vol_ma20 if vol_ma20 > 0 else 1

            # KDJ
            k = float(tech.get('kdj_k', 50))
            d = float(tech.get('kdj_d', 50))
            j = float(tech.get('kdj_j', 50))
            kdj_oversold = j < 20
            kdj_golden = k > d

            # MACD
            macd_hist = float(tech.get('macd_hist', 0))
            macd_golden = tech.get('macd_golden_cross', 0)

            # 布林带
            boll_lower = float(tech.get('boll_lower', closes[i]))
            boll_position = float(tech.get('boll_position', 0.5))
            near_boll_lower = closes[i] < boll_lower * 1.02

            # 前一日是否为阴线
            prev_red = closes[i-1] > opens[i-1] if i >= 1 else False
            # 当日是否为阳线
            today_red = closes[i] > opens[i]

            # 下影线长度（锤子线/探底针）
            body = abs(closes[i] - opens[i])
            lower_shadow = min(closes[i], opens[i]) - lows[i]
            upper_shadow = highs[i] - max(closes[i], opens[i])
            hammer = lower_shadow > 2 * body and upper_shadow < body if body > 0 else False

            # 大盘过滤
            if not market_trend(market_df, df.iloc[i]['date']):
                continue
            if not above_ma60:
                continue
            if volatility < 0.025:
                continue

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
                'ret5': ret5, 'ret10': ret10, 'ret20': ret20,
                'rsi6': rsi6, 'rsi14': rsi14,
                'vr': vr, 'vr20': vr20, 'vol_shrink': vol_shrink,
                'ma_bullish': int(ma_bullish),
                'above_ma20': int(above_ma20),
                'dist_ma20': dist_ma20, 'dist_ma60': dist_ma60,
                'cd': cd, 'volatility': volatility, 'amplitude': amplitude,
                'pct': pct,
                'kdj_k': k, 'kdj_d': d, 'kdj_j': j,
                'kdj_oversold': int(kdj_oversold), 'kdj_golden': int(kdj_golden),
                'macd_hist': macd_hist, 'macd_golden': int(macd_golden),
                'boll_position': boll_position, 'near_boll_lower': int(near_boll_lower),
                'prev_red': int(prev_red), 'today_red': int(today_red),
                'lower_shadow': lower_shadow, 'body': body,
                'hammer': int(hammer),
                'hold_ret': hold_ret, 'max_gain': max_gain, 'max_loss': max_loss,
                'win': int(hold_ret > 0),
            })

    print(f"\n  共收集 {len(all_signals)} 个信号")
    return pd.DataFrame(all_signals)


def calc_score_v6(row, params):
    """v6评分函数 - 引入更多因子"""
    score = 50
    ret5 = row['ret5']
    rsi6 = row['rsi6']
    vr = row['vr']
    cd = row['cd']
    dist_ma20 = row['dist_ma20']
    ret10 = row['ret10']
    amplitude = row['amplitude']
    kdj_j = row['kdj_j']
    kdj_golden = row['kdj_golden']
    macd_hist = row['macd_hist']
    near_boll_lower = row['near_boll_lower']
    hammer = row['hammer']
    vol_shrink = row['vol_shrink']

    # 超跌幅度
    if ret5 <= -15: score += params['p_ret5_extreme']
    elif ret5 <= -12: score += params['p_ret5_strong']
    elif ret5 <= -10: score += params['p_ret5_mild']
    elif ret5 <= -8: score += params['p_ret5_weak']
    elif ret5 > 5: score -= params['p_ret5_overbought']

    # RSI
    if rsi6 < 20: score += params['p_rsi_extreme']
    elif rsi6 < 25: score += params['p_rsi_strong']
    elif rsi6 < 30: score += params['p_rsi_mild']
    elif rsi6 < 35: score += params['p_rsi_weak']
    elif rsi6 > 75: score -= params['p_rsi_overbought']
    elif rsi6 > 70: score -= params['p_rsi_high']

    # 缩量（地量见底）
    if vol_shrink < 0.5: score += params['p_vol_shrink_extreme']
    elif vol_shrink < 0.7: score += params['p_vol_shrink_strong']
    elif vol_shrink < 0.9: score += params['p_vol_shrink_mild']
    elif vr > 2.0 and ret5 < 0: score -= params['p_vr_panic']

    # 连续下跌
    if cd >= 5: score += params['p_cd_5']
    elif cd >= 4: score += params['p_cd_4']
    elif cd >= 3: score += params['p_cd_3']

    # 距MA20
    if -15 < dist_ma20 < -5: score += params['p_dist_ma20']
    elif dist_ma20 < -15: score -= params['p_dist_ma20_break']

    # 10日跌幅
    if ret10 <= -20: score -= params['p_ret10_over']
    elif ret10 <= -15: score += params['p_ret10_mild']

    # KDJ底背离/超卖
    if kdj_j < 0: score += params['p_kdj_extreme']
    elif kdj_j < 20: score += params['p_kdj_oversold']
    if kdj_golden and kdj_j < 50: score += params['p_kdj_golden']

    # MACD
    if macd_hist > 0: score += params['p_macd_positive']

    # 布林带下轨
    if near_boll_lower: score += params['p_boll_lower']

    # 锤子线/探底针
    if hammer: score += params['p_hammer']

    return max(0, min(100, score))


def evaluate_params(signals_df, params, min_score, extra_conditions=None):
    """评估参数 + 额外硬条件（条件函数返回过滤后的DataFrame）"""
    signals_df = signals_df.copy()
    signals_df['score'] = signals_df.apply(lambda r: calc_score_v6(r, params), axis=1)
    filtered = signals_df[signals_df['score'] >= min_score]
    if extra_conditions:
        for cond in extra_conditions:
            filtered = cond(filtered)
    if len(filtered) == 0:
        return {'win_rate': 0, 'count': 0, 'avg_ret': 0, 'df': filtered}
    win_rate = filtered['win'].mean() * 100
    avg_ret = filtered['hold_ret'].mean()
    return {'win_rate': win_rate, 'count': len(filtered), 'avg_ret': avg_ret, 'df': filtered}


def deep_optimize(signals_df):
    """深度优化 - 多轮迭代"""
    print("\n" + "=" * 70)
    print("  深度因子优化 v2")
    print("=" * 70)

    base_params = {
        'p_ret5_extreme': 18, 'p_ret5_strong': 14, 'p_ret5_mild': 9, 'p_ret5_weak': 4,
        'p_ret5_overbought': 8,
        'p_rsi_extreme': 18, 'p_rsi_strong': 14, 'p_rsi_mild': 9, 'p_rsi_weak': 4,
        'p_rsi_overbought': 12, 'p_rsi_high': 8,
        'p_vol_shrink_extreme': 12, 'p_vol_shrink_strong': 8, 'p_vol_shrink_mild': 4,
        'p_vr_panic': 6,
        'p_cd_5': 8, 'p_cd_4': 5, 'p_cd_3': 2,
        'p_dist_ma20': 6, 'p_dist_ma20_break': 6,
        'p_ret10_over': 10, 'p_ret10_mild': 4,
        'p_kdj_extreme': 10, 'p_kdj_oversold': 6, 'p_kdj_golden': 5,
        'p_macd_positive': 4,
        'p_boll_lower': 6,
        'p_hammer': 8,
    }

    best_result = {'win_rate': 0, 'count': 0}
    best_params = base_params.copy()
    best_min_score = 65
    best_conditions = []

    # 第一轮：粗搜阈值
    print("\n  第1轮：搜索评分阈值...")
    for ms in [55, 58, 60, 62, 65, 68, 70, 72, 75, 78, 80, 82, 85]:
        res = evaluate_params(signals_df, base_params, ms)
        if res['count'] >= 20 and res['win_rate'] > best_result['win_rate']:
            best_result = res
            best_min_score = ms
            print(f"    阈值={ms}: 胜率={res['win_rate']:.1f}%, 信号={res['count']} ✓")

    print(f"  当前最佳: 阈值={best_min_score}, 胜率={best_result['win_rate']:.1f}%")

    # 第二轮：逐个优化权重
    print("\n  第2轮：优化因子权重...")
    param_ranges = {
        'p_ret5_strong': [10, 14, 18, 22, 26],
        'p_ret5_extreme': [14, 18, 22, 26, 30],
        'p_rsi_strong': [10, 14, 18, 22, 26],
        'p_rsi_extreme': [14, 18, 22, 26, 30],
        'p_rsi_mild': [5, 9, 13, 17],
        'p_vol_shrink_strong': [4, 8, 12, 16],
        'p_vol_shrink_extreme': [8, 12, 16, 20],
        'p_cd_4': [2, 5, 8, 11, 14],
        'p_cd_5': [5, 8, 11, 14, 17],
        'p_dist_ma20': [3, 6, 9, 12, 15],
        'p_kdj_oversold': [3, 6, 9, 12],
        'p_kdj_extreme': [6, 10, 14, 18],
        'p_boll_lower': [3, 6, 9, 12],
        'p_hammer': [4, 8, 12, 16],
    }

    for param_name, values in param_ranges.items():
        improved = False
        for val in values:
            test_params = best_params.copy()
            test_params[param_name] = val
            res = evaluate_params(signals_df, test_params, best_min_score)
            if res['count'] >= 20 and res['win_rate'] > best_result['win_rate']:
                best_result = res
                best_params = test_params
                improved = True
        if improved:
            print(f"    {param_name} -> {best_params[param_name]}: 胜率={best_result['win_rate']:.1f}%, 信号={best_result['count']} ✓")

    # 第三轮：加入硬条件过滤
    print("\n  第3轮：加入硬条件过滤...")

    condition_templates = [
        # (条件名, 条件函数, 描述)
        ('rsi<30', lambda df: df[df['rsi6'] < 30], 'RSI6<30'),
        ('ret5<=-10', lambda df: df[df['ret5'] <= -10], '5日跌幅>=10%'),
        ('ret5<=-12', lambda df: df[df['ret5'] <= -12], '5日跌幅>=12%'),
        ('cd>=4', lambda df: df[df['cd'] >= 4], '连续下跌>=4天'),
        ('vol_shrink<0.8', lambda df: df[df['vol_shrink'] < 0.8], '缩量至80%以下'),
        ('kdj_j<20', lambda df: df[df['kdj_j'] < 20], 'KDJ_J<20'),
        ('near_boll_lower', lambda df: df[df['near_boll_lower'] == 1], '触及布林下轨'),
        ('dist_ma20<-5', lambda df: df[df['dist_ma20'] < -5], '跌破MA20超5%'),
    ]

    # 尝试单个硬条件
    for cond_name, cond_fn, desc in condition_templates:
        res = evaluate_params(signals_df, best_params, best_min_score, [cond_fn])
        if res['count'] >= 15 and res['win_rate'] > best_result['win_rate']:
            best_result = res
            best_conditions = [cond_fn]
            print(f"    + {desc}: 胜率={res['win_rate']:.1f}%, 信号={res['count']} ✓")

    # 尝试组合硬条件（两两组合）
    print("\n  第4轮：硬条件组合...")
    for i in range(len(condition_templates)):
        for j in range(i+1, len(condition_templates)):
            conds = [condition_templates[i][1], condition_templates[j][1]]
            desc = f"{condition_templates[i][2]} + {condition_templates[j][2]}"
            res = evaluate_params(signals_df, best_params, best_min_score, conds)
            if res['count'] >= 15 and res['win_rate'] > best_result['win_rate']:
                best_result = res
                best_conditions = conds
                print(f"    + {desc}: 胜率={res['win_rate']:.1f}%, 信号={res['count']} ✓")

    # 第四轮：再调阈值
    print("\n  第5轮：阈值最终微调...")
    for ms in range(best_min_score - 5, best_min_score + 8):
        res = evaluate_params(signals_df, best_params, ms, best_conditions)
        if res['count'] >= 15 and res['win_rate'] > best_result['win_rate']:
            best_result = res
            best_min_score = ms
            print(f"    阈值={ms}: 胜率={res['win_rate']:.1f}%, 信号={res['count']} ✓")

    print(f"\n  优化完成!")
    print(f"    最终胜率: {best_result['win_rate']:.1f}%")
    print(f"    信号数量: {best_result['count']}")
    print(f"    平均收益: {best_result['avg_ret']:+.2f}%")
    print(f"    评分阈值: {best_min_score}")

    if best_result['win_rate'] >= 70:
        print(f"    ✓ 目标达成！胜率 >= 70%")
    else:
        print(f"    ✗ 未达标，继续尝试更严格条件...")

    return best_params, best_min_score, best_conditions, best_result


def plot_enhanced(signals_df, best_params, best_min_score, best_conditions):
    """绘制增强版结果图"""
    signals_df = signals_df.copy()
    signals_df['score'] = signals_df.apply(lambda r: calc_score_v6(r, best_params), axis=1)
    filtered = signals_df[signals_df['score'] >= best_min_score].copy()
    for cond in best_conditions:
        filtered = cond(filtered)
    filtered = filtered.sort_values('date').reset_index(drop=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f'深度优化回测结果 (胜率={filtered["win"].mean()*100:.1f}%, 信号={len(filtered)})',
                 fontsize=14, fontweight='bold')

    # 1. 累计收益
    ax = axes[0, 0]
    if len(filtered) > 0:
        filtered['cum_ret'] = (1 + filtered['hold_ret'] / 100).cumprod()
        ax.plot(range(len(filtered)), filtered['cum_ret'], 'b-', label='策略累计净值', linewidth=2)
        baseline = (1 + signals_df['hold_ret'] / 100).cumprod()
        ax.plot(range(len(signals_df)), baseline, 'r--', label='全部信号基准', alpha=0.6)
        ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
        ax.set_title('策略累计净值 vs 基准')
        ax.set_xlabel('信号序号'); ax.set_ylabel('累计净值')
        ax.legend(); ax.grid(True, alpha=0.3)

    # 2. 评分区间胜率
    ax = axes[0, 1]
    if len(filtered) > 0:
        score_min = int(filtered['score'].min())
        score_max = int(filtered['score'].max())
        bins = []
        labels = []
        start = score_min
        while start < score_max:
            end = min(start + 5, score_max + 1)
            bins.append((start, end))
            labels.append(f'{start}-{end}')
            start = end
        if not bins:
            bins = [(best_min_score, 100)]
            labels = [f'{best_min_score}+']
        win_rates = []
        counts = []
        for a, b in bins:
            sub = filtered[(filtered['score'] >= a) & (filtered['score'] < b)]
            win_rates.append(sub['win'].mean() * 100 if len(sub) > 0 else 0)
            counts.append(len(sub))
        x = range(len(labels))
        colors = ['green' if w >= 70 else 'orange' if w >= 50 else 'red' for w in win_rates]
        bars = ax.bar(x, win_rates, color=colors)
        ax.axhline(y=70, color='darkgreen', linestyle='--', label='70%目标')
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
        ax.set_title('各评分区间胜率'); ax.set_xlabel('评分区间'); ax.set_ylabel('胜率 (%)')
        ax.set_ylim(0, 100); ax.legend()
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{win_rates[i]:.0f}%\n(n={counts[i]})', ha='center', va='bottom', fontsize=8)
        ax.grid(True, alpha=0.3)

    # 3. 收益分布
    ax = axes[1, 0]
    if len(filtered) > 0:
        ax.hist(filtered['hold_ret'], bins=25, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(x=0, color='red', linestyle='--', label='0%')
        ax.axvline(x=filtered['hold_ret'].mean(), color='green', linestyle='--',
                   label=f'均值={filtered["hold_ret"].mean():.2f}%')
        ax.set_title('策略收益分布'); ax.set_xlabel('5日收益 (%)'); ax.set_ylabel('频次')
        ax.legend(); ax.grid(True, alpha=0.3)

    # 4. 因子相关性
    ax = axes[1, 1]
    if len(filtered) > 5:
        factor_cols = ['ret5', 'rsi6', 'vr', 'cd', 'dist_ma20', 'ret10', 'kdj_j',
                       'vol_shrink', 'macd_hist', 'boll_position', 'amplitude', 'win']
        factor_cols = [c for c in factor_cols if c in filtered.columns]
        corr = filtered[factor_cols].corr()
        im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_xticks(range(len(factor_cols)))
        ax.set_yticks(range(len(factor_cols)))
        ax.set_xticklabels(factor_cols, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(factor_cols, fontsize=8)
        ax.set_title('因子与胜负相关性')
        for i in range(len(factor_cols)):
            for j in range(len(factor_cols)):
                ax.text(j, i, f'{corr.values[i,j]:.2f}', ha='center', va='center', fontsize=6)
        plt.colorbar(im, ax=ax)

    # 5. 最大涨幅/跌幅散点
    ax = axes[2, 0]
    if len(filtered) > 0:
        wins = filtered[filtered['win'] == 1]
        losses = filtered[filtered['win'] == 0]
        ax.scatter(wins['max_loss'], wins['max_gain'], c='green', alpha=0.6, label=f'盈利({len(wins)})', s=40)
        ax.scatter(losses['max_loss'], losses['max_gain'], c='red', alpha=0.6, label=f'亏损({len(losses)})', s=40)
        ax.set_xlabel('期间最大跌幅 (%)'); ax.set_ylabel('期间最大涨幅 (%)')
        ax.set_title('盈利/亏损交易波动特征'); ax.legend(); ax.grid(True, alpha=0.3)

    # 6. 月度胜率
    ax = axes[2, 1]
    if len(filtered) > 0:
        filtered['month'] = pd.to_datetime(filtered['date']).dt.to_period('M')
        monthly = filtered.groupby('month').agg(win_rate=('win', 'mean'), count=('win', 'count')).reset_index()
        monthly['win_rate'] = monthly['win_rate'] * 100
        colors = ['green' if w >= 70 else 'orange' if w >= 50 else 'red' for w in monthly['win_rate']]
        ax.bar(range(len(monthly)), monthly['win_rate'], color=colors)
        ax.axhline(y=70, color='darkgreen', linestyle='--', label='70%目标')
        ax.axhline(y=50, color='gray', linestyle=':', label='50%基准')
        ax.set_xticks(range(len(monthly)))
        ax.set_xticklabels([str(m) for m in monthly['month']], rotation=45, ha='right', fontsize=7)
        ax.set_title('月度胜率趋势'); ax.set_ylabel('胜率 (%)'); ax.set_ylim(0, 100)
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'backtest_optimization_v2.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n优化结果图已保存: {save_path}")
    return save_path


def plot_vs_actual_v2(symbols, best_params, best_min_score, best_conditions, hold_days=5):
    """绘制回测 vs 实际对比图 v2"""
    print("\n绘制回测预测 vs 实际结果对比图 v2...")
    demo_symbols = symbols[:4]
    fig, axes = plt.subplots(len(demo_symbols), 1, figsize=(16, 3.5 * len(demo_symbols)))
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

        signal_points = []
        for i in range(65, len(df) - hold_days):
            window = df.iloc[:i+1].reset_index(drop=True)
            tech = factor_engine.calc_technical_factors(window)
            ret5 = (closes[i] / closes[i-5] - 1) * 100
            rsi6 = calc_rsi(closes[:i+1], 6)
            vr = float(tech.get('volume_ratio', 1))
            cd = sum(1 for j in range(i, max(i-10, 0), -1) if closes[j] < closes[j-1])
            ma20 = float(tech.get('ma20', closes[i]))
            dist_ma20 = (closes[i] / ma20 - 1) * 100 if ma20 > 0 else 0
            ret10 = (closes[i] / closes[i-10] - 1) * 100 if i >= 10 else 0
            amplitude = (df['high'].values[i] - df['low'].values[i]) / closes[i-1] * 100 if closes[i-1] > 0 else 0
            volatility = np.std(np.diff(closes[i-20:i]) / closes[i-20:i-1]) if i >= 20 else 0
            ma60 = float(tech.get('ma60', closes[i])) if not np.isnan(tech.get('ma60', np.nan)) else closes[i]
            if closes[i] <= ma60 or volatility < 0.025:
                continue
            vol_ma20 = np.mean(df['volume'].values[i-20:i]) if i >= 20 else df['volume'].values[i]
            vol_shrink = df['volume'].values[i] / vol_ma20 if vol_ma20 > 0 else 1
            kdj_j = float(tech.get('kdj_j', 50))
            kdj_golden = tech.get('kdj_golden_cross', 0)
            macd_hist = float(tech.get('macd_hist', 0))
            near_boll = closes[i] < float(tech.get('boll_lower', closes[i])) * 1.02
            body = abs(closes[i] - df['open'].values[i])
            lower_shadow = min(closes[i], df['open'].values[i]) - df['low'].values[i]
            upper_shadow = df['high'].values[i] - max(closes[i], df['open'].values[i])
            hammer = int(lower_shadow > 2 * body and upper_shadow < body) if body > 0 else 0

            row = pd.Series({
                'ret5': ret5, 'rsi6': rsi6, 'vr': vr, 'cd': cd,
                'dist_ma20': dist_ma20, 'ret10': ret10, 'amplitude': amplitude,
                'volatility': volatility, 'pct': df['pct_change'].values[i],
                'kdj_j': kdj_j, 'kdj_golden': kdj_golden,
                'macd_hist': macd_hist, 'near_boll_lower': int(near_boll),
                'hammer': hammer, 'vol_shrink': vol_shrink,
            })
            score = calc_score_v6(row, best_params)
            if score >= best_min_score:
                # 检查硬条件
                cond_df = pd.DataFrame([row])
                pass_cond = True
                for cond in best_conditions:
                    if len(cond(cond_df)) == 0:
                        pass_cond = False
                        break
                if not pass_cond:
                    continue
                actual_ret = (closes[i+hold_days] / closes[i] - 1) * 100
                signal_points.append({
                    'idx': i, 'price': closes[i], 'score': score,
                    'actual_ret': actual_ret, 'win': actual_ret > 0,
                })

        ax.plot(range(len(closes)), closes, 'k-', linewidth=1, label='收盘价', alpha=0.8)
        ma20_line = [np.mean(closes[max(0, i-20):i+1]) for i in range(len(closes))]
        ax.plot(range(len(closes)), ma20_line, 'b-', linewidth=0.8, label='MA20', alpha=0.6)

        if signal_points:
            buys = [s for s in signal_points if s['win']]
            sells = [s for s in signal_points if not s['win']]
            if buys:
                ax.scatter([s['idx'] for s in buys], [s['price'] for s in buys],
                          c='green', marker='^', s=100, label=f'盈利信号({len(buys)})', zorder=5)
            if sells:
                ax.scatter([s['idx'] for s in sells], [s['price'] for s in sells],
                          c='red', marker='v', s=100, label=f'亏损信号({len(sells)})', zorder=5)
            for sp in signal_points[::max(1, len(signal_points)//10)]:
                ax.annotate(f'{sp["actual_ret"]:+.1f}%', xy=(sp['idx'], sp['price']),
                           xytext=(0, 15), textcoords='offset points', fontsize=7, ha='center',
                           color='green' if sp['win'] else 'red')

        total = len(signal_points)
        wins = len([s for s in signal_points if s['win']])
        wr = wins / total * 100 if total > 0 else 0
        ax.set_title(f'{sym} - 回测信号 vs 实际 (信号:{total}, 胜率:{wr:.1f}%)')
        ax.set_ylabel('价格'); ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.3)
        ticks = list(range(0, len(closes), max(1, len(closes)//10)))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(dates[p])[:10] for p in ticks], rotation=45, fontsize=7)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'backtest_vs_actual_v2.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"对比图v2已保存: {save_path}")
    return save_path


def main():
    print("=" * 70)
    print("  A股深度因子优化系统 v2 - 目标胜率 >= 70%")
    print("=" * 70)

    market_df = load_market(600)
    print("\n收集历史信号数据...")
    signals_df = collect_all_signals(SYMBOLS, market_df, days=500, hold_days=5)

    if signals_df.empty:
        print("未收集到信号数据")
        return

    print(f"\n总信号数: {len(signals_df)}")
    print(f"整体胜率: {signals_df['win'].mean()*100:.1f}%")
    print(f"平均收益: {signals_df['hold_ret'].mean():+.2f}%")

    # 深度优化
    best_params, best_min_score, best_conditions, best_result = deep_optimize(signals_df)

    # 打印结果
    print("\n" + "=" * 70)
    print("  最终优化结果")
    print("=" * 70)
    print(f"  最终胜率: {best_result['win_rate']:.1f}%")
    print(f"  信号数量: {best_result['count']}")
    print(f"  平均收益: {best_result['avg_ret']:+.2f}%")
    print(f"  评分阈值: {best_min_score}")
    print(f"  硬条件数: {len(best_conditions)}")

    if best_result['win_rate'] >= 70:
        print(f"\n  ✓✓✓ 目标达成！胜率 {best_result['win_rate']:.1f}% >= 70%")
    else:
        print(f"\n  当前胜率 {best_result['win_rate']:.1f}%")

    # 绘图
    plot_enhanced(signals_df, best_params, best_min_score, best_conditions)
    plot_vs_actual_v2(SYMBOLS, best_params, best_min_score, best_conditions)

    # 保存结果
    signals_df['score'] = signals_df.apply(lambda r: calc_score_v6(r, best_params), axis=1)
    filtered = signals_df[signals_df['score'] >= best_min_score].copy()
    for cond in best_conditions:
        filtered = cond(filtered)
    csv_path = os.path.join(OUTPUT_DIR, 'optimized_signals_v2.csv')
    filtered.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n优化后信号已保存: {csv_path}")

    return best_params, best_min_score, best_conditions, best_result


if __name__ == '__main__':
    main()
