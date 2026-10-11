"""优化策略回测验证"""
import sys
sys.path.insert(0, r'c:\Users\cks\Documents\trae_projects\stock_analyzer')

from main import StockAnalyzer

analyzer = StockAnalyzer()
# 10只代表性股票
symbols = ['600519', '000858', '601318', '000001', '600036',
           '002594', '300750', '600900', '601899', '000333']
result = analyzer.run_backtest(symbols=symbols, days=300, threshold=60, hold_days=3)
