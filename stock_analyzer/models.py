"""
模型层
多模型集成：XGBoost + LightGBM + 随机森林 + 逻辑回归 + 规则引擎
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


class ModelEnsemble:
    """多模型集成预测器"""

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.is_trained = False
        self._init_models()

    def _init_models(self):
        """初始化所有模型"""
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='logloss', n_jobs=-1
        )
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, verbose=-1
        )
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        )
        self.models['logistic'] = LogisticRegression(
            max_iter=1000, random_state=42
        )
        self.models['gbdt'] = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )

    def create_features_from_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        从历史数据构建训练特征和标签
        标签：次日是否上涨（1=涨，0=跌）
        """
        if df.empty or len(df) < 30:
            return pd.DataFrame()

        data = df.copy().reset_index(drop=True)
        features = []
        labels = []

        for i in range(20, len(data) - 1):
            window = data.iloc[i-20:i+1]
            feat = self._extract_window_features(window)
            # 标签：次日收盘 > 当日收盘
            label = 1 if data.iloc[i+1]['close'] > data.iloc[i]['close'] else 0
            features.append(feat)
            labels.append(label)

        feature_df = pd.DataFrame(features)
        feature_df['label'] = labels
        self.feature_cols = [c for c in feature_df.columns if c != 'label']
        return feature_df

    def _extract_window_features(self, window: pd.DataFrame) -> dict:
        """从一个时间窗口提取特征"""
        feat = {}
        close = window['close'].values
        high = window['high'].values
        low = window['low'].values
        volume = window['volume'].values
        pct = window['pct_change'].values

        # 收益率特征
        feat['ret_1d'] = pct[-1]
        feat['ret_3d'] = (close[-1] / close[-4] - 1) * 100 if len(close) >= 4 else 0
        feat['ret_5d'] = (close[-1] / close[-6] - 1) * 100 if len(close) >= 6 else 0
        feat['ret_10d'] = (close[-1] / close[-11] - 1) * 100 if len(close) >= 11 else 0
        feat['ret_20d'] = (close[-1] / close[0] - 1) * 100

        # 均线偏离
        feat['ma5'] = np.mean(close[-5:])
        feat['ma10'] = np.mean(close[-10:])
        feat['ma20'] = np.mean(close)
        feat['price_vs_ma5'] = (close[-1] / feat['ma5'] - 1) * 100
        feat['price_vs_ma20'] = (close[-1] / feat['ma20'] - 1) * 100

        # 成交量
        feat['vol_ratio'] = volume[-1] / np.mean(volume[-5:]) if np.mean(volume[-5:]) > 0 else 1
        feat['vol_ma5'] = np.mean(volume[-5:])
        feat['vol_change'] = (volume[-1] / volume[-2] - 1) if volume[-2] > 0 else 0

        # 波动率
        returns = np.diff(close) / close[:-1]
        feat['volatility_5d'] = np.std(returns[-5:]) * 100 if len(returns) >= 5 else 0
        feat['volatility_20d'] = np.std(returns) * 100

        # 高低点
        feat['high_20d'] = np.max(high)
        feat['low_20d'] = np.min(low)
        feat['price_position'] = (close[-1] - feat['low_20d']) / (feat['high_20d'] - feat['low_20d'] + 1e-8)

        # RSI
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
        feat['rsi'] = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-8)))

        # MACD
        ema12 = self._ema(close, 12)
        ema26 = self._ema(close, 26)
        feat['macd'] = ema12 - ema26

        # 涨跌统计
        feat['up_days_5d'] = np.sum(pct[-5:] > 0)
        feat['limit_up_count'] = np.sum(pct >= 9.8)
        feat['max_pct'] = np.max(pct)
        feat['min_pct'] = np.min(pct)

        # 振幅
        feat['amplitude'] = (high[-1] - low[-1]) / close[-2] * 100 if len(close) >= 2 else 0

        return feat

    @staticmethod
    def _ema(data, n):
        alpha = 2 / (n + 1)
        ema = data[0]
        for i in range(1, len(data)):
            ema = alpha * data[i] + (1 - alpha) * ema
        return ema

    def train(self, feature_df: pd.DataFrame) -> dict:
        """训练所有模型并返回各模型准确率"""
        if feature_df.empty or len(feature_df) < 50:
            return {'error': '样本不足，至少需要50条数据'}

        X = feature_df[self.feature_cols].fillna(0).values
        y = feature_df['label'].values

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=3)
        results = {}

        for name, model in self.models.items():
            accuracies = []
            for train_idx, test_idx in tscv.split(X_scaled):
                try:
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    accuracies.append(acc)
                except:
                    accuracies.append(0.5)
            results[name] = {
                'accuracy': round(np.mean(accuracies) * 100, 2),
                'accuracies': [round(a * 100, 2) for a in accuracies]
            }

        # 用全量数据重新训练
        for name, model in self.models.items():
            try:
                model.fit(X_scaled, y)
            except:
                pass

        self.is_trained = True
        return results

    def predict(self, latest_features: dict) -> dict:
        """预测单只股票"""
        if not self.is_trained:
            return {'error': '模型未训练'}

        # 构建特征向量
        feat_vec = []
        for col in self.feature_cols:
            feat_vec.append(latest_features.get(col, 0))

        X = np.array([feat_vec]).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        predictions = {}
        probabilities = []

        for name, model in self.models.items():
            try:
                prob = model.predict_proba(X_scaled)[0][1]  # 上涨概率
                pred = 1 if prob > 0.5 else 0
                predictions[name] = {
                    'prediction': '上涨' if pred == 1 else '下跌',
                    'probability': round(prob * 100, 2)
                }
                probabilities.append(prob)
            except:
                predictions[name] = {'prediction': '未知', 'probability': 50}
                probabilities.append(0.5)

        # 集成投票（加权平均）
        avg_prob = np.mean(probabilities) * 100
        final_pred = '上涨' if avg_prob > 50 else '下跌'

        # 一致性评分（各模型是否一致）
        up_count = sum(1 for p in probabilities if p > 0.5)
        consistency = up_count / len(probabilities) * 100

        return {
            'ensemble': {
                'prediction': final_pred,
                'up_probability': round(avg_prob, 2),
                'consistency': round(consistency, 2),
                'model_votes': f'{up_count}/{len(probabilities)} 模型看多'
            },
            'individual': predictions
        }


class RuleEngine:
    """A股规则引擎（基于经验的硬规则）"""

    def __init__(self):
        pass

    def evaluate(self, technical: dict, capital: dict, ashare: dict,
                 sentiment: dict, contrarian: dict) -> dict:
        """
        规则引擎评估
        返回：买入/卖出信号和置信度
        """
        buy_signals = []
        sell_signals = []
        buy_confidence = 0
        sell_confidence = 0

        # ============ 买入规则 ============

        # 规则1：MACD金叉 + 量增价升
        if technical.get('macd_golden_cross') and technical.get('price_up_vol_up'):
            buy_signals.append('MACD金叉+量增价升')
            buy_confidence += 15

        # 规则2：主力资金连续3日流入
        if capital.get('main_inflow_consecutive_days', 0) >= 3:
            buy_signals.append('主力资金连续3日流入')
            buy_confidence += 15

        # 规则3：龙虎榜净买入
        if ashare.get('on_dragon_tiger') and ashare.get('lhb_net_buy', 0) > 0:
            buy_signals.append('龙虎榜机构净买入')
            buy_confidence += 20

        # 规则4：超跌反弹（RSI<25）
        if contrarian.get('contrarian_buy'):
            buy_signals.append('RSI超卖-逆向买入')
            buy_confidence += 15

        # 规则5：涨停后回调
        if contrarian.get('limit_up_pullback'):
            buy_signals.append('涨停后回调低吸')
            buy_confidence += 10

        # 规则6：均线多头排列 + 突破20日新高
        if technical.get('ma_bullish') and technical.get('break_20d_high'):
            buy_signals.append('均线多头+突破新高')
            buy_confidence += 15

        # 规则7：KDJ金叉 + RSI未超买
        if technical.get('kdj_golden_cross') and not technical.get('rsi_overbought'):
            buy_signals.append('KDJ金叉')
            buy_confidence += 8

        # 规则8：5日跌超15%超卖反弹
        if contrarian.get('oversold_rebound'):
            buy_signals.append('超跌反弹')
            buy_confidence += 15

        # 规则9：舆情极度负面（逆向）
        if contrarian.get('sentiment_contrarian_buy'):
            buy_signals.append('舆情极度悲观-逆向买入')
            buy_confidence += 10

        # ============ 卖出规则 ============

        # 规则1：MACD死叉
        if technical.get('macd_dead_cross'):
            sell_signals.append('MACD死叉')
            sell_confidence += 15

        # 规则2：RSI超买
        if technical.get('rsi_overbought'):
            sell_signals.append('RSI超买')
            sell_confidence += 15

        # 规则3：主力资金大幅流出
        if capital.get('main_inflow_today', 0) < -10000000:  # 流出超1000万
            sell_signals.append('主力资金大幅流出')
            sell_confidence += 15

        # 规则4：跌停
        if ashare.get('is_limit_down'):
            sell_signals.append('跌停')
            sell_confidence += 30

        # 规则5：均线空头排列
        if technical.get('ma_bearish'):
            sell_signals.append('均线空头排列')
            sell_confidence += 15

        # 规则6：舆情极度狂热（逆向）
        if contrarian.get('sentiment_contrarian_sell'):
            sell_signals.append('舆情极度狂热-逆向卖出')
            sell_confidence += 10

        # 规则7：市场过热
        if contrarian.get('market_overheat'):
            sell_signals.append('市场过热')
            sell_confidence += 8

        buy_confidence = min(buy_confidence, 100)
        sell_confidence = min(sell_confidence, 100)

        if buy_confidence > sell_confidence:
            action = '买入'
            confidence = buy_confidence
        elif sell_confidence > buy_confidence:
            action = '卖出'
            confidence = sell_confidence
        else:
            action = '观望'
            confidence = 50

        return {
            'action': action,
            'confidence': round(confidence, 2),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'buy_confidence': buy_confidence,
            'sell_confidence': sell_confidence
        }
