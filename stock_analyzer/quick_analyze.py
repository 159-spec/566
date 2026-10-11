"""单股快速分析脚本 - 使用胜率最高的强超跌反弹策略"""
import sys
sys.path.insert(0, r'c:\Users\cks\Documents\trae_projects\stock_analyzer')

import warnings
warnings.filterwarnings('ignore')

from data_fetcher import DataFetcher
from factors import FactorEngine
from models import ModelEnsemble, RuleEngine
from backtest import Backtester, calc_rsi
from datetime import datetime
import pandas as pd
import numpy as np

def quick_analyze(symbol):
    fetcher = DataFetcher()
    factor_engine = FactorEngine()
    
    print(f"\n{'='*60}")
    print(f"  分析股票: {symbol}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # 获取历史数据
    df = fetcher.get_daily_history(symbol)
    if df.empty:
        print("无法获取行情数据")
        return
    
    # 股票名称
    stocks = fetcher.get_all_stocks()
    name_row = stocks[stocks['代码'] == symbol]
    name = name_row['名称'].values[0] if not name_row.empty else symbol
    print(f"  股票名称: {name}")
    
    latest = df.iloc[-1]
    print(f"  最新日期: {latest['date'].strftime('%Y-%m-%d')}")
    print(f"  最新收盘价: {latest['close']:.2f}")
    print(f"  当日涨跌幅: {latest['pct_change']:.2f}%")
    
    # 计算技术因子
    technical = factor_engine.calc_technical_factors(df)
    
    # 超跌反弹信号检测（强策略：胜率72.5%）
    closes = df['close'].values
    ret5 = (closes[-1] / closes[-6] - 1) * 100 if len(closes) >= 6 else 0
    rsi6 = calc_rsi(closes, 6)
    rsi14 = calc_rsi(closes, 14)
    
    print(f"\n--- 超跌反弹信号检测（胜率最高策略）---")
    print(f"  5日涨跌幅: {ret5:+.2f}%")
    print(f"  RSI6: {rsi6:.1f}")
    print(f"  RSI14: {rsi14:.1f}")
    
    triggered_strong = ret5 <= -12 and rsi6 < 35
    triggered_mild = ret5 <= -10 and rsi6 < 35
    
    if triggered_strong:
        print(f"\n  ★★★ 强超跌反弹信号触发！★★★")
        print(f"  历史回测胜率: 72.5%")
        print(f"  历史平均收益: +4.90%")
        print(f"  建议操作: 买入，持有5天")
    elif triggered_mild:
        print(f"\n  ★★ 温和超跌反弹信号触发 ★★")
        print(f"  历史回测胜率: 54.9%")
        print(f"  历史平均收益: +2.29%")
        print(f"  建议操作: 可考虑买入，持有5天")
    else:
        print(f"\n  当前未触发超跌反弹信号")
        if ret5 > 0:
            print(f"  (近期上涨，无超跌机会)")
        elif rsi6 >= 35:
            print(f"  (RSI6={rsi6:.1f}，未到超卖区<35)")
    
    # 关键技术指标
    print(f"\n--- 关键技术指标 ---")
    print(f"  MA5:  {technical.get('ma5', 0):.2f}")
    print(f"  MA10: {technical.get('ma10', 0):.2f}")
    print(f"  MA20: {technical.get('ma20', 0):.2f}")
    print(f"  MA60: {technical.get('ma60', 0):.2f}")
    print(f"  量比: {technical.get('volume_ratio', 0):.2f}")
    print(f"  MACD金叉: {'是' if technical.get('macd_golden_cross') else '否'}")
    print(f"  均线多头: {'是' if technical.get('ma_bullish') else '否'}")
    print(f"  突破20日新高: {'是' if technical.get('break_20d_high') else '否'}")
    
    print(f"\n{'='*60}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        quick_analyze(sys.argv[1])
    else:
        print("用法: python quick_analyze.py <股票代码>")
        print("例如: python quick_analyze.py 600519")
