"""
第三轮因子深度优化 - 目标：在70%+胜率基础上进一步提升
新增探索：
1. 更细的参数网格搜索
2. 更多硬条件组合
3. 新增因子：量价背离、缺口、振幅比、支撑位
4. 尝试不同持有期（3/5/7/10天）
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, r'C:\Users\cks\Desktop\stock_analyzer')

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
OUTPUT_DIR = r'C:\Users\cks\Desktop\stock_analyzer\backtest_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


def collect_signals(symbols, market_df, days=500, hold_days=5, min_vol=0.02, req_ma60=True):
    """收集信号，包含更多因子"""
    all_signals = []
    for idx, sym in enumerate(symbols):
        print(f"  {sym} ({idx+1}/{len(symbols)})...", end='\r')
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
            ret3 = (closes[i] / closes[i-3] - 1) * 100 if i >= 3 else 0
            rsi6 = calc_rsi(closes[:i+1], 6)
            rsi14 = float(tech.get('rsi14', 50))
            vr = float(tech.get('volume_ratio', 1))
            ma5 = float(tech.get('ma5', closes[i]))
            ma10 = float(tech.get('ma10', closes[i]))
            ma20 = float(tech.get('ma20', closes[i]))
            ma60 = float(tech.get('ma60', closes[i])) if not np.isnan(tech.get('ma60', np.nan)) else closes[i]
            cd = sum(1 for j in range(i, max(i-10,0),-1) if closes[j] < closes[j-1])
            cu = sum(1 for j in range(i, max(i-10,0),-1) if closes[j] > closes[j-1])
            dist_ma20 = (closes[i] / ma20 - 1) * 100 if ma20 > 0 else 0
            dist_ma60 = (closes[i] / ma60 - 1) * 100 if ma60 > 0 else 0
            volatility = np.std(np.diff(closes[i-20:i]) / closes[i-20:i-1]) if i >= 20 else 0
            amplitude = (highs[i] - lows[i]) / closes[i-1] * 100 if closes[i-1] > 0 else 0
            vol_ma20 = np.mean(volumes[i-20:i]) if i >= 20 else volumes[i]
            vol_shrink = volumes[i] / vol_ma20 if vol_ma20 > 0 else 1
            kdj_j = float(tech.get('kdj_j', 50))
            kdj_k = float(tech.get('kdj_k', 50))
            kdj_d = float(tech.get('kdj_d', 50))
            kdj_golden = tech.get('kdj_golden_cross', 0)
            macd_hist = float(tech.get('macd_hist', 0))
            macd_dif = float(tech.get('macd_dif', 0))
            near_boll = closes[i] < float(tech.get('boll_lower', closes[i])) * 1.02
            boll_pos = float(tech.get('boll_position', 0.5))
            body = abs(closes[i] - opens[i])
            lower_shadow = min(closes[i], opens[i]) - lows[i]
            upper_shadow = highs[i] - max(closes[i], opens[i])
            hammer = int(lower_shadow > 2 * body and upper_shadow < body) if body > 0 else 0
            # 新增因子
            prev_close = closes[i-1] if i >= 1 else closes[i]
            gap = (opens[i] / prev_close - 1) * 100  # 跳空缺口
            # 量价背离：价跌量缩（好）vs 价跌量增（差）
            price_down = closes[i] < prev_close
            vol_down = volumes[i] < volumes[i-1] if i >= 1 else False
            vol_price_diverge = int(price_down and vol_down)  # 价跌量缩=底部信号
            # 振幅比（今日振幅/20日均振幅）
            amp_ma20 = np.mean([(highs[j]-lows[j])/closes[j-1]*100 for j in range(max(0,i-20),i)]) if i > 0 else amplitude
            amp_ratio = amplitude / amp_ma20 if amp_ma20 > 0 else 1
            # 距近20日低点
            low_20 = np.min(lows[max(0,i-20):i]) if i > 0 else lows[i]
            dist_low = (closes[i] / low_20 - 1) * 100 if low_20 > 0 else 0
            # 长下影线
            long_lower_shadow = int(lower_shadow > body * 1.5 and body > 0)

            if not market_trend(market_df, df.iloc[i]['date']): continue
            if req_ma60 and closes[i] <= ma60: continue
            if volatility < min_vol: continue

            buy = closes[i]
            sell = closes[i + hold_days]
            hold_ret = (sell / buy - 1) * 100
            period_high = np.max(highs[i+1:i+hold_days+1])
            period_low = np.min(lows[i+1:i+hold_days+1])

            all_signals.append({
                'symbol': sym, 'date': df.iloc[i]['date'],
                'ret3': ret3, 'ret5': ret5, 'ret10': ret10,
                'rsi6': rsi6, 'rsi14': rsi14, 'vr': vr,
                'cd': cd, 'cu': cu,
                'dist_ma20': dist_ma20, 'dist_ma60': dist_ma60,
                'volatility': volatility, 'amplitude': amplitude, 'vol_shrink': vol_shrink,
                'kdj_j': kdj_j, 'kdj_k': kdj_k, 'kdj_d': kdj_d,
                'kdj_golden': int(kdj_golden),
                'macd_hist': macd_hist, 'macd_dif': macd_dif,
                'near_boll_lower': int(near_boll), 'boll_position': boll_pos,
                'hammer': hammer, 'long_lower_shadow': long_lower_shadow,
                'gap': gap, 'vol_price_diverge': vol_price_diverge,
                'amp_ratio': amp_ratio, 'dist_low': dist_low,
                'pct': float(pcts[i]),
                'hold_ret': hold_ret,
                'max_gain': (period_high / buy - 1) * 100,
                'max_loss': (period_low / buy - 1) * 100,
                'win': int(hold_ret > 0),
            })
    print(f"\n  共 {len(all_signals)} 个信号")
    return pd.DataFrame(all_signals)


def calc_score_v7(row, params):
    """v7评分函数 - 包含新增因子"""
    score = 50
    ret5, rsi6, vr = row['ret5'], row['rsi6'], row['vr']
    cd, dist_ma20, ret10 = row['cd'], row['dist_ma20'], row['ret10']
    kdj_j, kdj_golden = row['kdj_j'], row['kdj_golden']
    macd_hist = row['macd_hist']
    near_boll = row['near_boll_lower']
    hammer = row['hammer']
    vol_shrink = row['vol_shrink']
    long_ls = row['long_lower_shadow']
    vpd = row['vol_price_diverge']
    amp_ratio = row['amp_ratio']
    ret3 = row['ret3']

    if ret5 <= -15: score += params['p_ret5_extreme']
    elif ret5 <= -12: score += params['p_ret5_strong']
    elif ret5 <= -10: score += params['p_ret5_mild']
    elif ret5 <= -8: score += params['p_ret5_weak']
    elif ret5 > 5: score -= params['p_ret5_ob']

    if ret3 <= -8: score += params['p_ret3_strong']
    elif ret3 <= -5: score += params['p_ret3_mild']

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
    # 新增因子
    if long_ls: score += params['p_long_ls']
    if vpd: score += params['p_vpd']
    if amp_ratio > 1.5: score += params['p_amp_high']

    return max(0, min(100, score))


def evaluate(signals_df, params, min_score, conds=None):
    df = signals_df.copy()
    df['score'] = df.apply(lambda r: calc_score_v7(r, params), axis=1)
    f = df[df['score'] >= min_score]
    if conds:
        for c in conds: f = c(f)
    if len(f) == 0: return {'win_rate': 0, 'count': 0, 'avg_ret': 0}
    return {'win_rate': f['win'].mean()*100, 'count': len(f), 'avg_ret': f['hold_ret'].mean()}


def main():
    print("=" * 70)
    print("  第三轮因子深度优化")
    print("=" * 70)

    market_df = load_market(600)

    # 基础参数（在v6基础上扩展）
    base_params = {
        'p_ret5_extreme': 18, 'p_ret5_strong': 14, 'p_ret5_mild': 9, 'p_ret5_weak': 4, 'p_ret5_ob': 8,
        'p_ret3_strong': 6, 'p_ret3_mild': 3,
        'p_rsi_extreme': 18, 'p_rsi_strong': 18, 'p_rsi_mild': 9, 'p_rsi_weak': 4,
        'p_rsi_ob': 12, 'p_rsi_high': 8,
        'p_vs_extreme': 12, 'p_vs_strong': 8, 'p_vs_mild': 4, 'p_vr_panic': 6,
        'p_cd_5': 8, 'p_cd_4': 5, 'p_cd_3': 2,
        'p_dist': 6, 'p_dist_break': 6,
        'p_ret10_over': 10, 'p_ret10_mild': 4,
        'p_kdj_extreme': 10, 'p_kdj_oversold': 6, 'p_kdj_golden': 5,
        'p_macd_pos': 4, 'p_boll': 6, 'p_hammer': 8,
        'p_long_ls': 5, 'p_vpd': 4, 'p_amp_high': 3,
    }

    # 测试不同持有期
    for hold_days in [3, 5, 7]:
        print(f"\n{'#'*70}")
        print(f"  持有期 = {hold_days}天")
        print(f"{'#'*70}")

        signals_df = collect_signals(SYMBOLS, market_df, hold_days=hold_days)
        if signals_df.empty: continue

        print(f"  总信号: {len(signals_df)}, 整体胜率: {signals_df['win'].mean()*100:.1f}%")

        # 条件组合
        cond_options = [
            ([], '无条件'),
            ([lambda df: df[df['kdj_j'] < 20]], 'KDJ_J<20'),
            ([lambda df: df[df['kdj_j'] < 25]], 'KDJ_J<25'),
            ([lambda df: df[df['rsi6'] < 30]], 'RSI6<30'),
            ([lambda df: df[df['rsi6'] < 35]], 'RSI6<35'),
            ([lambda df: df[df['near_boll_lower'] == 1]], '触布林下轨'),
            ([lambda df: df[df['long_lower_shadow'] == 1]], '长下影线'),
            ([lambda df: df[df['vol_price_diverge'] == 1]], '量价背离'),
            ([lambda df: df[(df['near_boll_lower'] == 1) & (df['rsi6'] < 35)]], '布林下轨&RSI<35'),
            ([lambda df: df[(df['kdj_j'] < 25) & (df['rsi6'] < 35)]], 'KDJ<25&RSI<35'),
            ([lambda df: df[(df['near_boll_lower'] == 1) & (df['long_lower_shadow'] == 1)]], '布林下轨&长下影'),
            ([lambda df: df[(df['vol_price_diverge'] == 1) & (df['rsi6'] < 35)]], '量价背离&RSI<35'),
            ([lambda df: df[(df['near_boll_lower'] == 1) & (df['vol_price_diverge'] == 1)]], '布林下轨&量价背离'),
            ([lambda df: df[(df['kdj_j'] < 20) & (df['vol_price_diverge'] == 1)]], 'KDJ<20&量价背离'),
        ]

        best = {'win_rate': 0, 'count': 0}
        best_cfg = None
        best_params = base_params.copy()

        # 第一轮：搜索阈值+条件
        for ms in range(60, 88, 1):
            for conds, cname in cond_options:
                res = evaluate(signals_df, base_params, ms, conds)
                if res['win_rate'] >= 70 and res['count'] >= 10:
                    if res['win_rate'] > best['win_rate'] or (res['win_rate'] == best['win_rate'] and res['count'] > best['count']):
                        best = res
                        best_cfg = (ms, cname, conds)

        if best_cfg:
            ms, cname, conds = best_cfg
            print(f"\n  最优(纯搜索): 胜率={best['win_rate']:.1f}%, 信号={best['count']}, "
                  f"阈值={ms}, 条件={cname}")

            # 第二轮：优化参数权重
            param_ranges = {
                'p_ret5_strong': [10, 14, 18, 22],
                'p_ret5_extreme': [14, 18, 22, 26],
                'p_rsi_strong': [14, 18, 22, 26],
                'p_rsi_mild': [5, 9, 13, 17],
                'p_vs_strong': [4, 8, 12, 16],
                'p_cd_4': [2, 5, 8, 11],
                'p_dist': [3, 6, 9, 12],
                'p_kdj_oversold': [3, 6, 9, 12],
                'p_boll': [3, 6, 9, 12],
                'p_hammer': [4, 8, 12, 16],
                'p_long_ls': [3, 6, 9, 12],
                'p_vpd': [2, 4, 6, 8],
                'p_amp_high': [2, 4, 6, 8],
                'p_ret3_strong': [3, 6, 9, 12],
            }

            for pname, values in param_ranges.items():
                for val in values:
                    tp = best_params.copy()
                    tp[pname] = val
                    res = evaluate(signals_df, tp, ms, conds)
                    if res['win_rate'] >= 70 and res['count'] >= 10:
                        if res['win_rate'] > best['win_rate'] or (res['win_rate'] == best['win_rate'] and res['count'] > best['count']):
                            best = res
                            best_params = tp

            print(f"  优化后: 胜率={best['win_rate']:.1f}%, 信号={best['count']}, 收益={best['avg_ret']:+.2f}%")

            # 第三轮：阈值微调
            for ms2 in range(ms - 3, ms + 5):
                res = evaluate(signals_df, best_params, ms2, conds)
                if res['win_rate'] >= 70 and res['count'] >= 10:
                    if res['win_rate'] > best['win_rate'] or (res['win_rate'] == best['win_rate'] and res['count'] > best['count']):
                        best = res
                        ms = ms2

            print(f"  最终: 胜率={best['win_rate']:.1f}%, 信号={best['count']}, "
                  f"收益={best['avg_ret']:+.2f}%, 阈值={ms}, 条件={cname}")

            if best['win_rate'] >= 75:
                print(f"  ✓✓✓ 超过75%!")

    # 总结所有持有期的最佳结果
    print(f"\n{'='*70}")
    print(f"  第三轮优化完成")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
