"""
A股智能选股系统 - 主程序
作者：专属工具（仅本地运行）

功能：
1. 全市场多因子扫描
2. 技术面 + 基本面 + 资金面 + A股特色 + 情绪面 + 反人性因子
3. 多模型集成预测（XGBoost + LightGBM + RF + LR + GBDT）
4. 规则引擎（基于A股经验）
5. 真实历史回测
"""
import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime

from data_fetcher import DataFetcher
from factors import FactorEngine
from models import ModelEnsemble, RuleEngine
from backtest import Backtester


class StockAnalyzer:
    """A股智能选股系统主类"""

    def __init__(self):
        self.fetcher = DataFetcher()
        self.factor_engine = FactorEngine()
        self.model_ensemble = ModelEnsemble()
        self.rule_engine = RuleEngine()
        self.backtester = Backtester(self.fetcher, self.factor_engine, self.rule_engine)
        self.market_overview = {}
        self.limit_up_data = {}
        self.dragon_tiger = pd.DataFrame()
        self.sector_flow = pd.DataFrame()

    def init_market_data(self):
        """初始化大盘环境数据"""
        print("[1/4] 正在获取大盘概况...")
        self.market_overview = self.fetcher.get_market_overview()

        print("[2/4] 正在获取涨跌停数据...")
        self.limit_up_data = self.fetcher.get_limit_up_down()

        print("[3/4] 正在获取龙虎榜数据...")
        self.dragon_tiger = self.fetcher.get_dragon_tiger()

        print("[4/4] 正在获取板块资金流向...")
        self.sector_flow = self.fetcher.get_sector_fund_flow()

        print("\n=== 大盘概况 ===")
        print(f"上涨家数: {self.market_overview.get('up_count', 'N/A')}")
        print(f"下跌家数: {self.market_overview.get('down_count', 'N/A')}")
        print(f"涨停家数: {self.market_overview.get('limit_up_count', 'N/A')}")
        print(f"跌停家数: {self.market_overview.get('limit_down_count', 'N/A')}")
        if 'sh_change_5d' in self.market_overview:
            print(f"沪指5日涨跌: {self.market_overview['sh_change_5d']:.2f}%")
        print("=" * 50)

    def analyze_stock(self, symbol: str, with_news: bool = True) -> dict:
        """分析单只股票"""
        result = {'symbol': symbol, 'name': '', 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        # 1. 获取历史行情
        df = self.fetcher.get_daily_history(symbol)
        if df.empty:
            result['error'] = '无法获取行情数据'
            return result

        # 股票名称
        stocks = self.fetcher.get_all_stocks()
        name_row = stocks[stocks['代码'] == symbol]
        result['name'] = name_row['名称'].values[0] if not name_row.empty else symbol

        # 2. 技术面因子
        technical = self.factor_engine.calc_technical_factors(df)

        # 3. 超跌反弹信号检测（经过回测验证胜率72.5%的策略）
        reversal_signal = self._detect_reversal_signal(df)
        result['reversal_signal'] = reversal_signal

        # 4. 基本面
        fundamental_info = self.fetcher.get_fundamental(symbol)
        fundamental = self.factor_engine.calc_fundamental_factors(fundamental_info)

        # 5. 资金面
        capital_df = self.fetcher.get_capital_flow(symbol)
        capital = self.factor_engine.calc_capital_factors(capital_df, symbol)

        # 6. A股特色
        limit_up_df = self.limit_up_data.get('limit_up', pd.DataFrame())
        ashare = self.factor_engine.calc_ashare_special_factors(
            symbol, df, limit_up_df, self.dragon_tiger, self.sector_flow
        )

        # 7. 情绪面（舆情）
        sentiment = {'overall_sentiment': 0, 'news_sentiment': 0, 'guba_sentiment': 0}
        if with_news:
            try:
                news_df = self.fetcher.get_news_sentiment(symbol)
                sentiment = self.factor_engine.calc_sentiment_factors(news_df, pd.DataFrame())
            except:
                pass

        # 8. 反人性因子
        contrarian = self.factor_engine.calc_contrarian_factors(
            df, technical, self.market_overview, sentiment
        )

        # 9. 综合评分
        composite = self.factor_engine.calc_composite_score(
            technical, fundamental, capital, ashare, sentiment, contrarian
        )
        result['composite'] = composite

        # 10. 规则引擎
        rule_result = self.rule_engine.evaluate(technical, capital, ashare, sentiment, contrarian)
        result['rule_engine'] = rule_result

        # 11. ML模型预测
        ml_result = {'ensemble': {'up_probability': None}}
        try:
            feature_df = self.model_ensemble.create_features_from_history(df)
            if not feature_df.empty and len(feature_df) >= 50:
                train_res = self.model_ensemble.train(feature_df)
                latest_feat = self.model_ensemble._extract_window_features(df.tail(21))
                ml_result = self.model_ensemble.predict(latest_feat)
        except:
            pass
        result['ml'] = ml_result

        # 12. 综合决策（超跌反弹信号优先）
        result['final_decision'] = self._make_final_decision(
            composite, rule_result, ml_result, reversal_signal
        )

        return result

    def _detect_reversal_signal(self, df: pd.DataFrame) -> dict:
        """
        检测超跌反弹信号
        策略：5天跌>=10% + RSI6<35
        回测胜率：54.9%（温和）/ 72.5%（强超跌>=12%）
        """
        if len(df) < 10:
            return {'triggered': False}

        from backtest import calc_rsi

        closes = df['close'].values
        ret5 = (closes[-1] / closes[-6] - 1) * 100 if len(closes) >= 6 else 0
        rsi6 = calc_rsi(closes, 6)

        signal = {
            'ret5': round(ret5, 2),
            'rsi6': round(rsi6, 2),
            'triggered': False,
            'strength': 'none',
            'hold_days': 5,
            'historic_win_rate': 0,
        }

        if ret5 <= -12 and rsi6 < 35:
            signal['triggered'] = True
            signal['strength'] = 'strong'
            signal['historic_win_rate'] = 72.5
        elif ret5 <= -10 and rsi6 < 35:
            signal['triggered'] = True
            signal['strength'] = 'mild'
            signal['historic_win_rate'] = 54.9

        return signal

    def _make_final_decision(self, composite: dict, rule: dict, ml: dict,
                             reversal: dict = None) -> dict:
        """综合所有维度做出最终决策"""
        score = composite.get('composite_score', 50)
        rule_action = rule.get('action', '观望')
        rule_conf = rule.get('confidence', 50)
        ml_prob = ml.get('ensemble', {}).get('up_probability')

        # 综合得分
        final_score = score * 0.4 + rule_conf * (1 if rule_action == '买入' else -1 if rule_action == '卖出' else 0) * 0.3
        if ml_prob is not None:
            final_score += (ml_prob - 50) * 0.3

        final_score = max(0, min(100, final_score + 50))

        # 超跌反弹信号加分（经过回测验证）
        reversal_bonus = 0
        if reversal and reversal.get('triggered'):
            if reversal.get('strength') == 'strong':
                reversal_bonus = 20  # 强超跌反弹，胜率72.5%
            else:
                reversal_bonus = 10  # 温和超跌反弹，胜率54.9%
            final_score = min(100, final_score + reversal_bonus)

        if final_score >= 75:
            recommendation = '强烈推荐买入'
        elif final_score >= 65:
            recommendation = '推荐买入'
        elif final_score >= 55:
            recommendation = '谨慎买入'
        elif final_score >= 45:
            recommendation = '观望'
        elif final_score >= 35:
            recommendation = '谨慎卖出'
        else:
            recommendation = '建议卖出'

        return {
            'final_score': round(final_score, 2),
            'recommendation': recommendation,
            'ml_up_probability': ml_prob,
            'rule_action': rule_action,
            'rule_confidence': rule_conf,
            'reversal_signal': reversal,
        }

    def scan_market(self, top_n: int = 20, score_threshold: float = 65) -> list:
        """扫描全市场，返回评分最高的股票"""
        print("\n正在扫描全市场股票...")
        stocks = self.fetcher.get_all_stocks()
        if stocks.empty:
            print("无法获取股票列表")
            return []

        # 过滤：价格在2-100元，换手率>1%，成交量>0
        stocks = stocks[(stocks['最新价'] >= 2) & (stocks['最新价'] <= 100)]
        stocks = stocks[stocks['换手率'] > 1]
        stocks = stocks[stocks['成交量'] > 0]
        # 按成交额降序，先看活跃股
        stocks = stocks.sort_values('成交额', ascending=False).head(200)

        print(f"候选股票数: {len(stocks)}")

        results = []
        for idx, (_, row) in enumerate(stocks.iterrows()):
            symbol = row['代码']
            name = row['名称']
            try:
                print(f"  分析 [{idx+1}/{len(stocks)}] {name}({symbol})...", end='\r')
                res = self.analyze_stock(symbol, with_news=False)
                if 'error' not in res:
                    final_score = res['final_decision']['final_score']
                    results.append({
                        'symbol': symbol,
                        'name': name,
                        'final_score': final_score,
                        'composite_score': res['composite']['composite_score'],
                        'recommendation': res['final_decision']['recommendation'],
                        'ml_up_prob': res['final_decision'].get('ml_up_probability'),
                        'price': row['最新价'],
                        'pct_change': row['涨跌幅'],
                        'turnover_rate': row['换手率'],
                    })
            except Exception as e:
                continue
            time.sleep(0.1)  # 避免请求过快

        print("\n扫描完成，正在排序...")
        results = sorted(results, key=lambda x: x['final_score'], reverse=True)
        return results[:top_n]

    def print_stock_report(self, result: dict):
        """打印单只股票分析报告"""
        if 'error' in result:
            print(f"分析失败: {result['error']}")
            return

        print("\n" + "=" * 60)
        print(f"  股票: {result['name']} ({result['symbol']})")
        print(f"  时间: {result['time']}")
        print("=" * 60)

        # 超跌反弹信号（重点展示）
        rev = result.get('reversal_signal', {})
        if rev.get('triggered'):
            strength_desc = '强超跌反弹' if rev.get('strength') == 'strong' else '超跌反弹'
            print(f"\n【超跌反弹信号】{strength_desc} ✓")
            print(f"  5日涨跌幅: {rev.get('ret5')}%")
            print(f"  RSI6: {rev.get('rsi6')}")
            print(f"  历史回测胜率: {rev.get('historic_win_rate')}%")
            print(f"  建议持有: {rev.get('hold_days')}天")

        comp = result.get('composite', {})
        print(f"\n【综合评分】{comp.get('composite_score', 'N/A')} / 100")
        print(f"  - 技术面: {comp.get('technical_score', 'N/A')}")
        print(f"  - 基本面: {comp.get('fundamental_score', 'N/A')}")
        print(f"  - 资金面: {comp.get('capital_score', 'N/A')}")
        print(f"  - A股特色: {comp.get('ashare_score', 'N/A')}")
        print(f"  - 情绪面: {comp.get('sentiment_score', 'N/A')}")
        print(f"  - 反人性: {comp.get('contrarian_score', 'N/A')}")
        print(f"  - 操作建议: {comp.get('recommendation', 'N/A')}")
        print(f"  - 风险等级: {comp.get('risk_level', 'N/A')}")
        if comp.get('stop_loss'):
            print(f"  - 建议止损: {comp['stop_loss']}")

        rule = result.get('rule_engine', {})
        print(f"\n【规则引擎】动作: {rule.get('action', 'N/A')} (置信度: {rule.get('confidence', 'N/A')}%)")
        if rule.get('buy_signals'):
            print(f"  买入信号: {', '.join(rule['buy_signals'])}")
        if rule.get('sell_signals'):
            print(f"  卖出信号: {', '.join(rule['sell_signals'])}")

        ml = result.get('ml', {})
        ens = ml.get('ensemble', {})
        if ens.get('up_probability') is not None:
            print(f"\n【ML模型】上涨概率: {ens.get('up_probability')}%")
            print(f"  模型一致性: {ens.get('consistency')}% ({ens.get('model_votes', 'N/A')})")

        final = result.get('final_decision', {})
        print(f"\n{'='*60}")
        print(f"  【最终决策】{final.get('recommendation', 'N/A')}")
        print(f"  综合得分: {final.get('final_score', 'N/A')} / 100")
        print(f"{'='*60}")

    def print_scan_results(self, results: list):
        """打印扫描结果"""
        if not results:
            print("未找到符合条件的股票")
            return

        print("\n" + "=" * 80)
        print(f"{'排名':<4} {'代码':<8} {'名称':<8} {'综合分':<8} {'建议':<10} {'现价':<8} {'涨跌幅':<8} {'换手率':<8}")
        print("-" * 80)
        for i, r in enumerate(results, 1):
            print(f"{i:<4} {r['symbol']:<8} {r['name']:<8} {r['final_score']:<8.1f} "
                  f"{r['recommendation']:<10} {r['price']:<8.2f} {r['pct_change']:<8.2f}% "
                  f"{r['turnover_rate']:<8.2f}%")
        print("=" * 80)

    def run_backtest(self, symbols: list = None, days: int = 400,
                     strategy: str = 'reversal_mild', hold_days: int = 5):
        """
        运行回测
        strategy可选:
        - reversal_strong: 5天跌>=12%+RSI<35 (胜率72.5%)
        - reversal_mild:   5天跌>=10%+RSI<35 (胜率54.9%)
        - reversal_volume: 5天跌>=10%+RSI<35+缩量
        - comprehensive:   超跌反弹+放量突破
        """
        if symbols is None:
            symbols = ['600519', '000858', '601318', '000001', '600036',
                       '002594', '300750', '600900', '601899', '000333']

        print(f"\n开始回测（{len(symbols)}只股票，{days}天，策略={strategy}，持有{hold_days}天）...")
        result = self.backtester.backtest_portfolio(symbols, days, strategy, hold_days)

        if 'error' in result:
            print(f"回测失败: {result['error']}")
            return

        print("\n" + "=" * 60)
        print("  回测结果汇总")
        print("=" * 60)
        print(f"  测试股票数: {result['stocks_tested']}")
        print(f"  策略: {result['strategy']}")
        print(f"  总买入信号数: {result['total_buy_signals']}")
        print(f"  平均买入胜率: {result['avg_buy_win_rate']}%")
        print(f"  平均{hold_days}日收益: {result['avg_buy_return']}%")
        print(f"  平均覆盖率: {result['avg_coverage']}%")
        print(f"  平均期间最大涨幅: {result['avg_max_gain']}%")
        print(f"  平均期间最大跌幅: {result['avg_max_loss']}%")
        print("=" * 60)

        print("\n各股票详情:")
        for d in result.get('details', []):
            print(f"  {d['symbol']}: 信号数={d['buy_signals']}, 胜率={d['buy_win_rate']}%, "
                  f"平均收益={d['buy_avg_return']}%, 覆盖率={d['coverage']}%")

        return result


def main():
    print("=" * 60)
    print("  A股智能选股系统 v1.0")
    print("  多因子 + 多模型 + 反人性 + 舆情分析")
    print("  声明：本工具仅供学习参考，不构成投资建议")
    print("=" * 60)

    analyzer = StockAnalyzer()

    while True:
        print("\n请选择功能:")
        print("  1. 分析单只股票")
        print("  2. 全市场扫描选股")
        print("  3. 历史回测（验证策略胜率）")
        print("  4. 退出")

        choice = input("\n请输入选项 (1-4): ").strip()

        if choice == '1':
            symbol = input("请输入股票代码（如600519）: ").strip()
            if symbol:
                analyzer.init_market_data()
                result = analyzer.analyze_stock(symbol)
                analyzer.print_stock_report(result)

        elif choice == '2':
            analyzer.init_market_data()
            top_n = input("请输入返回数量（默认20）: ").strip()
            top_n = int(top_n) if top_n.isdigit() else 20
            results = analyzer.scan_market(top_n=top_n)
            analyzer.print_scan_results(results)

        elif choice == '3':
            print("\n选择回测策略:")
            print("  1. 强超跌反弹（5天跌>=12%+RSI<35，历史胜率72.5%）")
            print("  2. 温和超跌反弹（5天跌>=10%+RSI<35，历史胜率54.9%）")
            print("  3. 缩量超跌反弹")
            print("  4. 综合策略（超跌反弹+放量突破）")
            print("  5. 优化策略v1（布林下轨+RSI<35+多因子评分，回测胜率81.5%）★推荐")
            strat_choice = input("请选择策略 (1-5，默认5): ").strip()
            strategy_map = {'1': 'reversal_strong', '2': 'reversal_mild',
                            '3': 'reversal_volume', '4': 'comprehensive',
                            '5': 'optimized_v1'}
            strategy = strategy_map.get(strat_choice, 'optimized_v1')

            days = input("回测天数（默认400）: ").strip()
            days = int(days) if days.isdigit() else 400
            hold = input("持有天数（默认5）: ").strip()
            hold = int(hold) if hold.isdigit() else 5
            analyzer.run_backtest(days=days, strategy=strategy, hold_days=hold)

        elif choice == '4':
            print("退出系统。")
            break

        else:
            print("无效选项，请重新输入")


if __name__ == '__main__':
    main()
