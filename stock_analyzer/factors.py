"""
因子工程层
包含：技术面、基本面、资金面、A股特色、情绪面、反人性因子
"""
import pandas as pd
import numpy as np
import re


class FactorEngine:
    """多因子计算引擎"""

    def __init__(self):
        # 情绪词典（A股常用词）
        self.positive_words = [
            '利好', '上涨', '涨停', '突破', '反弹', '买入', '看好', '牛', '涨',
            '业绩增长', '超预期', '增持', '回购', '分红', '重组', '并购', '中标',
            '订单', '签约', '产能', '扩产', '新品', '合作', '入选', '获批'
        ]
        self.negative_words = [
            '利空', '下跌', '跌停', '破位', '跳水', '卖出', '看空', '熊', '跌',
            '业绩下滑', '不及预期', '减持', '质押', '平仓', '退市', 'ST', '违规',
            '处罚', '诉讼', '亏损', '爆雷', '造假', '立案', '调查', '解禁',
            '减持', '套现', '闪崩', '熔断'
        ]

    # ============ 技术面因子 ============
    def calc_technical_factors(self, df: pd.DataFrame) -> dict:
        """计算技术面因子"""
        factors = {}
        if df.empty or len(df) < 30:
            return factors

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        turnover_rate = df['turnover_rate'].values

        # 1. 均线系统
        factors['ma5'] = self._ma(close, 5)
        factors['ma10'] = self._ma(close, 10)
        factors['ma20'] = self._ma(close, 20)
        factors['ma60'] = self._ma(close, 60) if len(close) >= 60 else np.nan

        # 均线多头排列
        factors['ma_bullish'] = 1 if (factors['ma5'] > factors['ma10'] > factors['ma20']) else 0
        # 均线空头排列
        factors['ma_bearish'] = 1 if (factors['ma5'] < factors['ma10'] < factors['ma20']) else 0

        # 价格相对均线位置
        factors['price_vs_ma5'] = (close[-1] / factors['ma5'] - 1) * 100 if factors['ma5'] else 0
        factors['price_vs_ma20'] = (close[-1] / factors['ma20'] - 1) * 100 if factors['ma20'] else 0

        # 2. MACD
        dif, dea, macd = self._macd(close)
        factors['macd_dif'] = dif
        factors['macd_dea'] = dea
        factors['macd_hist'] = macd
        factors['macd_golden_cross'] = 1 if (dif > dea and self._prev_dif <= self._prev_dea) else 0
        factors['macd_dead_cross'] = 1 if (dif < dea and self._prev_dif >= self._prev_dea) else 0

        # 3. RSI
        rsi6 = self._rsi(close, 6)
        rsi14 = self._rsi(close, 14)
        factors['rsi6'] = rsi6
        factors['rsi14'] = rsi14
        factors['rsi_oversold'] = 1 if rsi6 < 30 else 0  # 超卖
        factors['rsi_overbought'] = 1 if rsi6 > 70 else 0  # 超买

        # 4. KDJ
        k, d, j = self._kdj(high, low, close)
        factors['kdj_k'] = k
        factors['kdj_d'] = d
        factors['kdj_j'] = j
        factors['kdj_golden_cross'] = 1 if (k > d and self._prev_k <= self._prev_d) else 0

        # 5. 布林带
        upper, middle, lower = self._boll(close)
        factors['boll_upper'] = upper
        factors['boll_middle'] = middle
        factors['boll_lower'] = lower
        factors['boll_position'] = (close[-1] - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
        factors['boll_break_upper'] = 1 if close[-1] > upper else 0
        factors['boll_break_lower'] = 1 if close[-1] < lower else 0

        # 6. 成交量
        vol_ma5 = np.mean(volume[-5:])
        vol_ma20 = np.mean(volume[-20:])
        factors['volume_ratio'] = volume[-1] / vol_ma5 if vol_ma5 > 0 else 1
        factors['volume_ratio_20'] = volume[-1] / vol_ma20 if vol_ma20 > 0 else 1
        factors['volume_increase'] = 1 if volume[-1] > vol_ma5 * 1.5 else 0
        factors['volume_shrink'] = 1 if volume[-1] < vol_ma5 * 0.5 else 0

        # 7. 量价配合
        factors['price_up_vol_up'] = 1 if (close[-1] > close[-2] and volume[-1] > volume[-2]) else 0
        factors['price_down_vol_up'] = 1 if (close[-1] < close[-2] and volume[-1] > volume[-2]) else 0

        # 8. 动量因子
        factors['momentum_5d'] = (close[-1] / close[-6] - 1) * 100 if len(close) >= 6 else 0
        factors['momentum_10d'] = (close[-1] / close[-11] - 1) * 100 if len(close) >= 11 else 0
        factors['momentum_20d'] = (close[-1] / close[-21] - 1) * 100 if len(close) >= 21 else 0

        # 9. 波动率
        returns = np.diff(close) / close[:-1]
        factors['volatility_20d'] = np.std(returns[-20:]) * np.sqrt(252) * 100 if len(returns) >= 20 else 0

        # 10. 振幅
        factors['amplitude'] = (high[-1] - low[-1]) / close[-2] * 100 if close[-2] > 0 else 0

        # 11. 换手率
        factors['turnover_rate'] = turnover_rate[-1] if not np.isnan(turnover_rate[-1]) else 0
        factors['turnover_rate_avg5'] = np.mean(turnover_rate[-5:]) if len(turnover_rate) >= 5 else 0

        # 12. 高低点突破
        factors['break_20d_high'] = 1 if close[-1] > np.max(close[-21:-1]) else 0
        factors['break_20d_low'] = 1 if close[-1] < np.min(close[-21:-1]) else 0

        # 13. K线形态
        factors['doji'] = 1 if abs(close[-1] - df['open'].values[-1]) / (high[-1] - low[-1] + 1e-8) < 0.1 else 0
        factors['hammer'] = self._is_hammer(df)
        factors['engulfing_bullish'] = self._is_bullish_engulfing(df)

        return factors

    # ============ 基本面因子 ============
    def calc_fundamental_factors(self, info: dict) -> dict:
        """计算基本面因子"""
        factors = {}
        factors['roe'] = info.get('roe', np.nan)
        factors['gross_margin'] = info.get('gross_margin', np.nan)
        factors['net_margin'] = info.get('net_margin', np.nan)
        factors['revenue_yoy'] = info.get('revenue_yoy', np.nan)
        factors['profit_yoy'] = info.get('profit_yoy', np.nan)

        # 基本面评分（0-100）
        score = 50
        if not np.isnan(factors.get('roe', np.nan)):
            score += min(max((factors['roe'] - 10) * 2, -20), 20)
        if not np.isnan(factors.get('profit_yoy', np.nan)):
            score += min(max(factors['profit_yoy'] * 0.5, -20), 20)
        if not np.isnan(factors.get('gross_margin', np.nan)):
            score += min(max((factors['gross_margin'] - 30) * 0.5, -10), 10)
        factors['fundamental_score'] = max(0, min(100, score))
        return factors

    # ============ 资金面因子 ============
    def calc_capital_factors(self, capital_df: pd.DataFrame, symbol: str = None) -> dict:
        """计算资金面因子"""
        factors = {}
        if capital_df.empty or len(capital_df) < 5:
            return factors

        main = capital_df['main_net_inflow'].values
        factors['main_inflow_today'] = main[-1]
        factors['main_inflow_3d'] = np.sum(main[-3:])
        factors['main_inflow_5d'] = np.sum(main[-5:])
        factors['main_inflow_10d'] = np.sum(main[-10:]) if len(main) >= 10 else np.sum(main)

        # 主力资金连续流入天数
        consecutive = 0
        for v in main[::-1]:
            if v > 0:
                consecutive += 1
            else:
                break
        factors['main_inflow_consecutive_days'] = consecutive

        # 主力资金占比
        if abs(main[-1]) > 0:
            factors['main_inflow_ratio'] = main[-1] / (abs(main[-1]) + 1e-8)
        else:
            factors['main_inflow_ratio'] = 0

        return factors

    # ============ A股特色因子 ============
    def calc_ashare_special_factors(self, symbol: str, df: pd.DataFrame,
                                     limit_up_df: pd.DataFrame,
                                     dragon_tiger_df: pd.DataFrame,
                                     sector_flow_df: pd.DataFrame) -> dict:
        """计算A股特色因子"""
        factors = {}
        if df.empty:
            return factors

        close = df['close'].values
        pct = df['pct_change'].values

        # 1. 涨停/跌停统计
        factors['is_limit_up'] = 1 if pct[-1] >= 9.8 else 0
        factors['is_limit_down'] = 1 if pct[-1] <= -9.8 else 0

        # 近N日涨停次数
        limit_up_count = np.sum(pct[-20:] >= 9.8) if len(pct) >= 20 else np.sum(pct >= 9.8)
        factors['limit_up_count_20d'] = limit_up_count

        # 连板数
        consecutive_limit = 0
        for v in pct[::-1]:
            if v >= 9.8:
                consecutive_limit += 1
            else:
                break
        factors['consecutive_limit_up'] = consecutive_limit

        # 2. 龙虎榜因子
        if not dragon_tiger_df.empty:
            try:
                # 东方财富龙虎榜字段：SECURITY_CODE, BILLBOARD_NET_AMT等
                code_col = None
                for c in dragon_tiger_df.columns:
                    if 'code' in str(c).lower() or '代码' in str(c):
                        code_col = c
                        break
                if code_col:
                    stock_lhb = dragon_tiger_df[dragon_tiger_df[code_col].astype(str) == symbol]
                else:
                    stock_lhb = pd.DataFrame()

                if not stock_lhb.empty:
                    factors['on_dragon_tiger'] = 1
                    # 净买入
                    net_col = None
                    for c in stock_lhb.columns:
                        if 'net' in str(c).lower() and 'amt' in str(c).lower():
                            net_col = c
                            break
                    if net_col is None:
                        for c in stock_lhb.columns:
                            if '净买' in str(c):
                                net_col = c
                                break
                    if net_col:
                        factors['lhb_net_buy'] = float(stock_lhb[net_col].iloc[0])
                    else:
                        factors['lhb_net_buy'] = 0
                else:
                    factors['on_dragon_tiger'] = 0
                    factors['lhb_net_buy'] = 0
            except:
                factors['on_dragon_tiger'] = 0
                factors['lhb_net_buy'] = 0
        else:
            factors['on_dragon_tiger'] = 0
            factors['lhb_net_buy'] = 0

        # 3. 涨停板封板强度（如果当天涨停）
        if factors['is_limit_up'] and len(df) >= 2:
            open_price = df['open'].values[-1]
            limit_price = round(close[-2] * 1.1, 2)
            factors['limit_strength'] = 1 if open_price >= limit_price * 0.99 else 0
        else:
            factors['limit_strength'] = 0

        return factors

    # ============ 情绪面因子 ============
    def calc_sentiment_factors(self, news_df: pd.DataFrame, guba_df: pd.DataFrame) -> dict:
        """计算舆情情绪因子"""
        factors = {}

        # 新闻情绪
        news_score = 0
        news_count = 0
        if not news_df.empty:
            for _, row in news_df.iterrows():
                text = str(row.get('title', '')) + str(row.get('content', ''))
                score = self._analyze_text_sentiment(text)
                news_score += score
                news_count += 1
        factors['news_sentiment'] = news_score / news_count if news_count > 0 else 0
        factors['news_count'] = news_count

        # 股吧情绪
        guba_score = 0
        guba_count = 0
        if not guba_df.empty:
            for _, row in guba_df.iterrows():
                text = str(row.values)
                score = self._analyze_text_sentiment(text)
                guba_score += score
                guba_count += 1
        factors['guba_sentiment'] = guba_score / guba_count if guba_count > 0 else 0
        factors['guba_count'] = guba_count

        # 综合情绪
        total = news_count + guba_count
        factors['overall_sentiment'] = (news_score + guba_score) / total if total > 0 else 0

        return factors

    # ============ 反人性因子（A股特色） ============
    def calc_contrarian_factors(self, df: pd.DataFrame, technical: dict,
                                market_overview: dict, sentiment: dict) -> dict:
        """
        A股反人性因子：
        1. 极度恐慌时给买入信号
        2. 极度狂热时给卖出信号
        3. 超跌反弹
        4. 涨停后回调低吸
        """
        factors = {}
        if df.empty or len(df) < 10:
            return factors

        close = df['close'].values
        pct = df['pct_change'].values

        # 1. RSI超卖反弹信号（反人性：别人恐慌我贪婪）
        if technical.get('rsi6', 50) < 25:
            factors['extreme_fear'] = 1
            factors['contrarian_buy'] = 1  # 逆向买入信号
        else:
            factors['extreme_fear'] = 0
            factors['contrarian_buy'] = 0

        # 2. RSI超买风险信号（反人性：别人贪婪我恐惧）
        if technical.get('rsi6', 50) > 80:
            factors['extreme_greed'] = 1
            factors['contrarian_sell'] = 1  # 逆向卖出信号
        else:
            factors['extreme_greed'] = 0
            factors['contrarian_sell'] = 0

        # 3. 连续大跌后反弹概率
        recent_5d_pct = np.sum(pct[-5:]) if len(pct) >= 5 else 0
        if recent_5d_pct < -15:  # 5日跌超15%，极度超卖
            factors['oversold_rebound'] = 1
        else:
            factors['oversold_rebound'] = 0

        # 4. 涨停后次日低开高走概率
        if len(pct) >= 2 and pct[-2] >= 9.8:
            # 昨日涨停
            if close[-1] < close[-2] and pct[-1] > -3:
                factors['limit_up_pullback'] = 1  # 涨停后回调
            else:
                factors['limit_up_pullback'] = 0
        else:
            factors['limit_up_pullback'] = 0

        # 5. 市场情绪极端值（大盘跌停家数过多）
        limit_down_count = market_overview.get('limit_down_count', 0)
        limit_up_count = market_overview.get('limit_up_count', 0)
        if limit_down_count > 50:
            factors['market_panic'] = 1
        else:
            factors['market_panic'] = 0
        if limit_up_count > 100:
            factors['market_overheat'] = 1
        else:
            factors['market_overheat'] = 0

        # 6. 舆情极度负面时的逆向信号
        overall_sentiment = sentiment.get('overall_sentiment', 0)
        if overall_sentiment < -0.3:
            factors['sentiment_contrarian_buy'] = 1
        else:
            factors['sentiment_contrarian_buy'] = 0
        if overall_sentiment > 0.5:
            factors['sentiment_contrarian_sell'] = 1
        else:
            factors['sentiment_contrarian_sell'] = 0

        return factors

    # ============ 综合评分 ============
    def calc_composite_score(self, technical: dict, fundamental: dict,
                             capital: dict, ashare: dict, sentiment: dict,
                             contrarian: dict) -> dict:
        """计算综合评分和买卖信号"""
        score = 50  # 基准分
        signals = {}

        # 技术面权重 30%
        tech_score = 50
        if technical.get('ma_bullish'):
            tech_score += 10
        if technical.get('macd_golden_cross'):
            tech_score += 10
        if technical.get('kdj_golden_cross'):
            tech_score += 5
        if technical.get('price_up_vol_up'):
            tech_score += 5
        if technical.get('break_20d_high'):
            tech_score += 5
        if technical.get('rsi_oversold'):
            tech_score += 8  # 超卖反弹
        if technical.get('rsi_overbought'):
            tech_score -= 8
        if technical.get('macd_dead_cross'):
            tech_score -= 8
        if technical.get('ma_bearish'):
            tech_score -= 10
        tech_score = max(0, min(100, tech_score))
        score += (tech_score - 50) * 0.3

        # 基本面权重 15%
        fund_score = fundamental.get('fundamental_score', 50)
        score += (fund_score - 50) * 0.15

        # 资金面权重 25%
        cap_score = 50
        if capital.get('main_inflow_today', 0) > 0:
            cap_score += 10
        if capital.get('main_inflow_consecutive_days', 0) >= 3:
            cap_score += 10
        if capital.get('main_inflow_5d', 0) > 0:
            cap_score += 5
        if capital.get('main_inflow_today', 0) < 0:
            cap_score -= 10
        cap_score = max(0, min(100, cap_score))
        score += (cap_score - 50) * 0.25

        # A股特色权重 15%
        ashare_score = 50
        if ashare.get('on_dragon_tiger') and ashare.get('lhb_net_buy', 0) > 0:
            ashare_score += 15
        if ashare.get('limit_up_count_20d', 0) >= 3:
            ashare_score += 5  # 股性活跃
        if ashare.get('is_limit_down'):
            ashare_score -= 20
        ashare_score = max(0, min(100, ashare_score))
        score += (ashare_score - 50) * 0.15

        # 情绪面权重 5%
        sent_score = 50 + sentiment.get('overall_sentiment', 0) * 50
        sent_score = max(0, min(100, sent_score))
        score += (sent_score - 50) * 0.05

        # 反人性因子权重 10%（关键）
        contrarian_score = 50
        if contrarian.get('contrarian_buy'):
            contrarian_score += 15
        if contrarian.get('oversold_rebound'):
            contrarian_score += 15
        if contrarian.get('limit_up_pullback'):
            contrarian_score += 8
        if contrarian.get('sentiment_contrarian_buy'):
            contrarian_score += 10
        if contrarian.get('contrarian_sell'):
            contrarian_score -= 15
        if contrarian.get('market_overheat'):
            contrarian_score -= 8
        if contrarian.get('sentiment_contrarian_sell'):
            contrarian_score -= 5
        contrarian_score = max(0, min(100, contrarian_score))
        score += (contrarian_score - 50) * 0.10

        score = max(0, min(100, score))

        return {
            'composite_score': round(score, 2),
            'technical_score': tech_score,
            'fundamental_score': fund_score,
            'capital_score': cap_score,
            'ashare_score': ashare_score,
            'sentiment_score': sent_score,
            'contrarian_score': contrarian_score,
            'recommendation': self._get_recommendation(score),
            'risk_level': self._get_risk_level(score, technical, contrarian),
            'stop_loss': self._calc_stop_loss(technical, df=None),
        }

    # ============ 辅助方法 ============
    def _analyze_text_sentiment(self, text: str) -> float:
        """简单文本情绪分析（-1 到 1）"""
        if not text:
            return 0
        pos_count = sum(1 for w in self.positive_words if w in text)
        neg_count = sum(1 for w in self.negative_words if w in text)
        total = pos_count + neg_count
        if total == 0:
            return 0
        return (pos_count - neg_count) / total

    @staticmethod
    def _ma(data, n):
        return np.mean(data[-n:]) if len(data) >= n else np.nan

    @staticmethod
    def _ema(data, n):
        alpha = 2 / (n + 1)
        ema = data[0]
        for i in range(1, len(data)):
            ema = alpha * data[i] + (1 - alpha) * ema
        return ema

    def _macd(self, close):
        ema12 = self._ema(close, 12)
        ema26 = self._ema(close, 26)
        dif = ema12 - ema26
        # 计算DEA
        dif_list = []
        ema12 = close[0]
        ema26 = close[0]
        for c in close:
            ema12 = 2 / 13 * c + 11 / 13 * ema12
            ema26 = 2 / 27 * c + 25 / 27 * ema26
            dif_list.append(ema12 - ema26)
        dea = self._ema(np.array(dif_list), 9)
        macd = 2 * (dif - dea)
        # 保存前一日值用于判断金叉死叉
        if len(dif_list) >= 2:
            self._prev_dif = dif_list[-2]
            self._prev_dea = self._ema(np.array(dif_list[:-1]), 9)
        else:
            self._prev_dif = dif
            self._prev_dea = dea
        return dif, dea, macd

    @staticmethod
    def _rsi(close, n=14):
        if len(close) < n + 1:
            return 50
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-n:])
        avg_loss = np.mean(losses[-n:])
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _kdj(self, high, low, close, n=9):
        if len(close) < n:
            return 50, 50, 50
        lowest = np.min(low[-n:])
        highest = np.max(high[-n:])
        rsv = (close[-1] - lowest) / (highest - lowest + 1e-8) * 100
        k = 2 / 3 * 50 + 1 / 3 * rsv
        d = 2 / 3 * 50 + 1 / 3 * k
        j = 3 * k - 2 * d
        self._prev_k = k
        self._prev_d = d
        return k, d, j

    @staticmethod
    def _boll(close, n=20, k=2):
        if len(close) < n:
            return close[-1], close[-1], close[-1]
        middle = np.mean(close[-n:])
        std = np.std(close[-n:])
        upper = middle + k * std
        lower = middle - k * std
        return upper, middle, lower

    @staticmethod
    def _is_hammer(df):
        if len(df) < 1:
            return 0
        o = df['open'].values[-1]
        c = df['close'].values[-1]
        h = df['high'].values[-1]
        l = df['low'].values[-1]
        body = abs(c - o)
        upper_shadow = h - max(c, o)
        lower_shadow = min(c, o) - l
        if lower_shadow > 2 * body and upper_shadow < body * 0.5:
            return 1
        return 0

    @staticmethod
    def _is_bullish_engulfing(df):
        if len(df) < 2:
            return 0
        o1, c1 = df['open'].values[-2], df['close'].values[-2]
        o2, c2 = df['open'].values[-1], df['close'].values[-1]
        if c1 < o1 and c2 > o2 and c2 > o1 and o2 < c1:
            return 1
        return 0

    @staticmethod
    def _get_recommendation(score):
        if score >= 75:
            return '强烈推荐买入'
        elif score >= 65:
            return '推荐买入'
        elif score >= 55:
            return '谨慎买入'
        elif score >= 45:
            return '观望'
        elif score >= 35:
            return '谨慎卖出'
        else:
            return '建议卖出'

    @staticmethod
    def _get_risk_level(score, technical, contrarian):
        risk = '中'
        if contrarian.get('extreme_greed') or technical.get('rsi_overbought'):
            risk = '高'
        if score < 40:
            risk = '高'
        if contrarian.get('extreme_fear') or technical.get('rsi_oversold'):
            risk = '低（逆向）'
        return risk

    @staticmethod
    def _calc_stop_loss(technical, df):
        # 简单止损：基于布林带下轨或ATR
        boll_lower = technical.get('boll_lower')
        if boll_lower and not np.isnan(boll_lower):
            return round(boll_lower * 0.98, 2)
        return None
