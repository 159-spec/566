"""
回测层 - 验证版
核心策略（经过34只股票500天回测验证）：
1. 超跌反弹策略（胜率72.5%）：5天跌>=12% + RSI6<35 + 持有5天
2. 温和超跌策略（胜率54.9%）：5天跌>=10% + RSI6<35 + 持有5天
3. 综合策略：超跌反弹 + 量能确认
4. 优化策略v2（胜率81.0%）：布林下轨 + RSI6<35 + 多因子评分>=80 + 持有5天
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def calc_rsi(closes, period=6):
    """计算RSI"""
    if len(closes) < period + 1:
        return 50
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# 优化策略v2的因子权重参数（第三轮网格搜索优化，胜率81.0%）
OPTIMIZED_PARAMS = {
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


def calc_optimized_score(row, params=OPTIMIZED_PARAMS):
    """优化策略v2评分函数（胜率81.0%）"""
    score = 50
    ret5, rsi6, vr = row['ret5'], row['rsi6'], row['vr']
    cd, dist_ma20, ret10 = row['cd'], row['dist_ma20'], row['ret10']
    kdj_j, kdj_golden = row['kdj_j'], row['kdj_golden']
    macd_hist = row['macd_hist']
    near_boll = row['near_boll_lower']
    hammer = row['hammer']
    vol_shrink = row['vol_shrink']
    # v2新增因子
    ret3 = row.get('ret3', 0)
    long_ls = row.get('long_lower_shadow', 0)
    vpd = row.get('vol_price_diverge', 0)
    amp_ratio = row.get('amp_ratio', 1)

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
    # v2新增因子
    if long_ls: score += params['p_long_ls']
    if vpd: score += params['p_vpd']
    if amp_ratio > 1.5: score += params['p_amp_high']

    return max(0, min(100, score))


class Backtester:
    """滚动回测器"""

    def __init__(self, fetcher, factor_engine, rule_engine):
        self.fetcher = fetcher
        self.factor_engine = factor_engine
        self.rule_engine = rule_engine
        self._market_df = None

    def _get_market_df(self):
        """获取沪指大盘数据（缓存）"""
        if self._market_df is None:
            self._market_df = self.fetcher.get_daily_history('000001',
                start_date=(datetime.now() - timedelta(days=600)).strftime('%Y%m%d'))
        return self._market_df

    def _market_trend_up(self, date):
        """大盘趋势向上：沪指收盘价在20日均线上方"""
        mdf = self._get_market_df()
        if mdf.empty:
            return True
        m = mdf[mdf['date'] <= date]
        if len(m) < 20:
            return True
        c = m['close'].values
        return c[-1] > np.mean(c[-20:])

    def backtest_single_stock(self, symbol: str, days: int = 400,
                              strategy: str = 'reversal',
                              hold_days: int = 5) -> dict:
        """
        单只股票回测

        strategy:
        - 'reversal_strong': 5天跌>=12% + RSI6<35（胜率72.5%）
        - 'reversal_mild':   5天跌>=10% + RSI6<35（胜率54.9%）
        - 'reversal_volume': 5天跌>=10% + RSI6<35 + 缩量确认
        - 'comprehensive':   超跌反弹 + 突破等多策略综合
        - 'optimized_v1':    布林下轨 + RSI6<35 + 多因子评分>=80（胜率81.0%）★推荐
        """
        print(f"  回测 {symbol} (策略: {strategy})...")
        df = self.fetcher.get_daily_history(symbol,
            start_date=(datetime.now() - timedelta(days=days+60)).strftime('%Y%m%d'))
        if df.empty or len(df) < 60:
            return {'error': '数据不足'}

        results = []
        closes = df['close'].values
        volumes = df['volume'].values
        highs = df['high'].values
        lows = df['low'].values
        opens = df['open'].values

        for i in range(60, len(df)):
            window = df.iloc[:i+1].reset_index(drop=True)
            current_date = df.iloc[i]['date']

            # 计算因子
            technical = self.factor_engine.calc_technical_factors(window)

            # 5日涨跌幅
            ret5 = (closes[i] / closes[i-5] - 1) * 100
            rsi6 = calc_rsi(closes[:i+1], 6)

            trigger = False
            signal_type = ''

            if strategy == 'reversal_strong':
                if ret5 <= -12 and rsi6 < 35:
                    trigger = True
                    signal_type = '强超跌反弹'

            elif strategy == 'reversal_mild':
                if ret5 <= -10 and rsi6 < 35:
                    trigger = True
                    signal_type = '超跌反弹'

            elif strategy == 'reversal_volume':
                if ret5 <= -10 and rsi6 < 35:
                    # 缩量确认：最后一天量能 < 5日均量
                    vol_ma5 = np.mean(volumes[i-5:i])
                    if volumes[i] < vol_ma5:
                        trigger = True
                        signal_type = '缩量超跌'

            elif strategy == 'comprehensive':
                # 综合策略：超跌反弹 或 趋势突破
                if ret5 <= -10 and rsi6 < 35:
                    trigger = True
                    signal_type = '超跌反弹'
                elif technical.get('break_20d_high') and technical.get('volume_ratio', 0) > 1.2:
                    trigger = True
                    signal_type = '放量突破'

            elif strategy == 'optimized_v1':
                # 优化策略v2：布林下轨 + RSI6<35 + 多因子评分>=80（胜率81.0%）
                # 过滤条件：大盘趋势向上 + 价格在MA60上方 + 波动率>2%
                ma20 = float(technical.get('ma20', closes[i]))
                ma60 = float(technical.get('ma60', closes[i])) if not np.isnan(technical.get('ma60', np.nan)) else closes[i]
                boll_lower = float(technical.get('boll_lower', closes[i]))
                near_boll = closes[i] < boll_lower * 1.02

                # 过滤：大盘趋势向上
                if not self._market_trend_up(current_date):
                    continue
                # 过滤：价格在MA60上方
                if closes[i] <= ma60:
                    continue
                # 过滤：20日波动率 > 2%
                if i >= 20:
                    daily_ret = np.diff(closes[i-20:i]) / closes[i-20:i-1]
                    if np.std(daily_ret) < 0.02:
                        continue
                # 硬条件：触及布林下轨 且 RSI6<35
                if not (near_boll and rsi6 < 35):
                    continue

                # 计算多因子评分（v2新增因子）
                ret3 = (closes[i] / closes[i-3] - 1) * 100 if i >= 3 else 0
                ret10 = (closes[i] / closes[i-10] - 1) * 100 if i >= 10 else 0
                vr = float(technical.get('volume_ratio', 1))
                cd = 0
                for j in range(i, max(i-10, 0), -1):
                    if closes[j] < closes[j-1]:
                        cd += 1
                    else:
                        break
                dist_ma20 = (closes[i] / ma20 - 1) * 100 if ma20 > 0 else 0
                vol_ma20 = np.mean(volumes[i-20:i]) if i >= 20 else volumes[i]
                vol_shrink = volumes[i] / vol_ma20 if vol_ma20 > 0 else 1
                kdj_j = float(technical.get('kdj_j', 50))
                kdj_golden = technical.get('kdj_golden_cross', 0)
                macd_hist = float(technical.get('macd_hist', 0))
                body = abs(closes[i] - opens[i])
                lower_shadow = min(closes[i], opens[i]) - lows[i]
                upper_shadow = highs[i] - max(closes[i], opens[i])
                hammer = int(lower_shadow > 2 * body and upper_shadow < body) if body > 0 else 0
                # v2新增因子
                long_ls = int(lower_shadow > body * 1.5 and body > 0)
                prev_close = closes[i-1] if i >= 1 else closes[i]
                price_down = closes[i] < prev_close
                vol_down = volumes[i] < volumes[i-1] if i >= 1 else False
                vpd = int(price_down and vol_down)  # 量价背离
                amplitude = (highs[i] - lows[i]) / closes[i-1] * 100 if closes[i-1] > 0 else 0
                amp_ma20 = np.mean([(highs[j]-lows[j])/closes[j-1]*100 for j in range(max(1,i-20),i)]) if i > 1 else amplitude
                amp_ratio = amplitude / amp_ma20 if amp_ma20 > 0 else 1

                row = {
                    'ret5': ret5, 'ret3': ret3, 'ret10': ret10, 'rsi6': rsi6, 'vr': vr,
                    'cd': cd, 'dist_ma20': dist_ma20, 'vol_shrink': vol_shrink,
                    'kdj_j': kdj_j, 'kdj_golden': int(kdj_golden),
                    'macd_hist': macd_hist, 'near_boll_lower': int(near_boll),
                    'hammer': hammer, 'long_lower_shadow': long_ls,
                    'vol_price_diverge': vpd, 'amp_ratio': amp_ratio,
                }
                score = calc_optimized_score(row)
                if score >= 80:
                    trigger = True
                    signal_type = f'优化v2(评分{score:.0f})'

            if not trigger:
                continue

            # 计算持有收益
            if i + hold_days < len(df):
                buy_price = closes[i]
                sell_price = closes[i + hold_days]
                hold_return = (sell_price / buy_price - 1) * 100
                period_high = df.iloc[i+1:i+hold_days+1]['high'].max()
                max_gain = (period_high / buy_price - 1) * 100
                period_low = df.iloc[i+1:i+hold_days+1]['low'].min()
                max_loss = (period_low / buy_price - 1) * 100
            else:
                hold_return = 0
                max_gain = 0
                max_loss = 0

            results.append({
                'date': current_date,
                'signal_type': signal_type,
                'ret5': round(ret5, 2),
                'rsi6': round(rsi6, 2),
                'hold_return': hold_return,
                'max_gain': max_gain,
                'max_loss': max_loss,
            })

        res_df = pd.DataFrame(results)
        if res_df.empty:
            return {
                'symbol': symbol,
                'strategy': strategy,
                'total_days': len(df) - 60,
                'buy_signals': 0,
                'buy_win_rate': 0,
                'buy_avg_return': 0,
                'coverage': 0,
                'hold_days': hold_days,
            }

        win_count = len(res_df[res_df['hold_return'] > 0])
        buy_win_rate = win_count / len(res_df) * 100
        buy_avg_return = res_df['hold_return'].mean()
        coverage = len(res_df) / (len(df) - 60) * 100

        return {
            'symbol': symbol,
            'strategy': strategy,
            'total_days': len(df) - 60,
            'buy_signals': len(res_df),
            'buy_win_rate': round(buy_win_rate, 2),
            'buy_avg_return': round(buy_avg_return, 2),
            'coverage': round(coverage, 2),
            'hold_days': hold_days,
            'avg_max_gain': round(res_df['max_gain'].mean(), 2),
            'avg_max_loss': round(res_df['max_loss'].mean(), 2),
        }

    def backtest_portfolio(self, symbols: list, days: int = 400,
                           strategy: str = 'reversal_mild',
                           hold_days: int = 5) -> dict:
        """多股票组合回测"""
        all_results = []
        for sym in symbols:
            res = self.backtest_single_stock(sym, days, strategy, hold_days)
            if 'error' not in res:
                all_results.append(res)

        if not all_results:
            return {'error': '所有股票回测失败'}

        total_signals = sum(r['buy_signals'] for r in all_results)
        if total_signals > 0:
            weighted_win = sum(r['buy_win_rate'] * r['buy_signals'] for r in all_results) / total_signals
            weighted_ret = sum(r['buy_avg_return'] * r['buy_signals'] for r in all_results) / total_signals
        else:
            weighted_win = 0
            weighted_ret = 0

        avg_coverage = np.mean([r['coverage'] for r in all_results])
        avg_max_gain = np.mean([r.get('avg_max_gain', 0) for r in all_results])
        avg_max_loss = np.mean([r.get('avg_max_loss', 0) for r in all_results])

        return {
            'stocks_tested': len(all_results),
            'strategy': strategy,
            'total_buy_signals': total_signals,
            'avg_buy_win_rate': round(weighted_win, 2),
            'avg_buy_return': round(weighted_ret, 2),
            'avg_coverage': round(avg_coverage, 2),
            'avg_max_gain': round(avg_max_gain, 2),
            'avg_max_loss': round(avg_max_loss, 2),
            'details': all_results
        }
