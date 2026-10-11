"""
交互式量化回测比对程序
功能：
1. 输入股票代码
2. 对该股票运行多种策略回测并比对
3. 保存回测数据到本地CSV（数据不变）
4. 绘制回测信号 vs 实际K线图
5. 回测结果持久化保存

使用方法：
    python interactive_backtest.py
    然后输入股票代码（如 600519）
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 确保能导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import DataFetcher
from factors import FactorEngine
from backtest import Backtester, calc_rsi, calc_optimized_score, OPTIMIZED_PARAMS

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 数据保存目录（回测数据持久化）
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_data')
os.makedirs(DATA_DIR, exist_ok=True)

# 可用策略列表
STRATEGIES = [
    ('optimized_v1', '优化策略v1(布林下轨+RSI<35+多因子)', '历史胜率81.5%'),
    ('reversal_strong', '强超跌反弹(5日跌>=12%+RSI<35)', '历史胜率72.5%'),
    ('reversal_mild', '温和超跌反弹(5日跌>=10%+RSI<35)', '历史胜率54.9%'),
    ('reversal_volume', '缩量超跌反弹', '缩量确认'),
    ('comprehensive', '综合策略(超跌+突破)', '多策略综合'),
]


def get_stock_name(fetcher, symbol):
    """获取股票名称"""
    try:
        stocks = fetcher.get_all_stocks()
        row = stocks[stocks['代码'] == symbol]
        if not row.empty:
            return row['名称'].values[0]
    except:
        pass
    return symbol


def run_comparison_backtest(symbol, days=400, hold_days=5):
    """对单只股票运行所有策略并比对"""
    fetcher = DataFetcher()
    factor_engine = FactorEngine()
    rule_engine = type('RuleEngine', (), {'evaluate': lambda *a, **k: {}})()
    bt = Backtester(fetcher, factor_engine, rule_engine)

    name = get_stock_name(fetcher, symbol)
    print(f"\n{'='*70}")
    print(f"  股票: {name} ({symbol})")
    print(f"  回测天数: {days}, 持有天数: {hold_days}")
    print(f"{'='*70}")

    all_results = []
    for strat_key, strat_name, strat_desc in STRATEGIES:
        print(f"\n  运行策略: {strat_name} ({strat_desc})...")
        res = bt.backtest_single_stock(symbol, days=days, strategy=strat_key, hold_days=hold_days)
        if 'error' in res:
            print(f"    失败: {res['error']}")
            continue
        res['strategy_name'] = strat_name
        res['strategy_desc'] = strat_desc
        all_results.append(res)
        print(f"    信号数: {res['buy_signals']}, "
              f"胜率: {res['buy_win_rate']}%, "
              f"平均收益: {res['buy_avg_return']}%, "
              f"覆盖率: {res['coverage']}%")

    # 汇总比对
    print(f"\n{'='*70}")
    print(f"  策略比对汇总")
    print(f"{'='*70}")
    print(f"{'策略':<35} {'信号数':>6} {'胜率':>8} {'平均收益':>10} {'覆盖率':>8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['strategy_name']:<35} {r['buy_signals']:>6} "
              f"{r['buy_win_rate']:>7.1f}% {r['buy_avg_return']:>+9.2f}% "
              f"{r['coverage']:>7.1f}%")

    # 保存回测数据到CSV（持久化）
    if all_results:
        summary_df = pd.DataFrame(all_results)
        csv_path = os.path.join(DATA_DIR, f"{symbol}_backtest_summary.csv")
        summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n  回测汇总已保存: {csv_path}")

    return all_results, name


def plot_stock_backtest(symbol, name, days=400, hold_days=5):
    """绘制单只股票的回测信号 vs 实际K线图"""
    fetcher = DataFetcher()
    factor_engine = FactorEngine()

    df = fetcher.get_daily_history(symbol,
        start_date=(datetime.now() - timedelta(days=days+60)).strftime('%Y%m%d'))
    if df.empty or len(df) < 60:
        print(f"  {symbol} 数据不足，无法绘图")
        return

    closes = df['close'].values
    dates = df['date'].values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    volumes = df['volume'].values

    # 收集优化策略v1的信号点
    signals = []
    for i in range(60, len(df)):
        window = df.iloc[:i+1].reset_index(drop=True)
        technical = factor_engine.calc_technical_factors(window)
        ret5 = (closes[i] / closes[i-5] - 1) * 100
        rsi6 = calc_rsi(closes[:i+1], 6)
        ma20 = float(technical.get('ma20', closes[i]))
        ma60 = float(technical.get('ma60', closes[i])) if not np.isnan(technical.get('ma60', np.nan)) else closes[i]
        boll_lower = float(technical.get('boll_lower', closes[i]))
        near_boll = closes[i] < boll_lower * 1.02

        if closes[i] <= ma60:
            continue
        if i >= 20:
            daily_ret = np.diff(closes[i-20:i]) / closes[i-20:i-1]
            if np.std(daily_ret) < 0.02:
                continue
        if not (near_boll and rsi6 < 35):
            continue

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

        row = {
            'ret5': ret5, 'ret10': ret10, 'rsi6': rsi6, 'vr': vr,
            'cd': cd, 'dist_ma20': dist_ma20, 'vol_shrink': vol_shrink,
            'kdj_j': kdj_j, 'kdj_golden': int(kdj_golden),
            'macd_hist': macd_hist, 'near_boll_lower': int(near_boll),
            'hammer': hammer,
        }
        score = calc_optimized_score(row)
        if score >= 70:
            buy_price = closes[i]
            sell_price = closes[i + hold_days] if i + hold_days < len(df) else closes[-1]
            hold_ret = (sell_price / buy_price - 1) * 100
            signals.append({
                'idx': i, 'date': dates[i], 'buy_price': buy_price,
                'sell_price': sell_price, 'hold_ret': hold_ret,
                'score': score, 'win': hold_ret > 0,
            })

    # 保存信号明细
    if signals:
        sig_df = pd.DataFrame(signals)
        sig_path = os.path.join(DATA_DIR, f"{symbol}_signals.csv")
        sig_df.to_csv(sig_path, index=False, encoding='utf-8-sig')
        print(f"  信号明细已保存: {sig_path}")

    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])
    fig.suptitle(f'{name}({symbol}) 优化策略v1回测信号 vs 实际走势', fontsize=14, fontweight='bold')

    # K线收盘价 + 均线 + 布林带
    ax1.plot(range(len(closes)), closes, 'k-', linewidth=1, label='收盘价', alpha=0.9)
    ma20_line = [np.mean(closes[max(0, i-20):i+1]) for i in range(len(closes))]
    ax1.plot(range(len(closes)), ma20_line, 'b-', linewidth=0.8, label='MA20', alpha=0.6)

    # 布林带
    upper_list, lower_list = [], []
    for i in range(len(closes)):
        if i >= 20:
            mid = np.mean(closes[i-20:i])
            std = np.std(closes[i-20:i])
            upper_list.append(mid + 2 * std)
            lower_list.append(mid - 2 * std)
        else:
            upper_list.append(closes[i])
            lower_list.append(closes[i])
    ax1.plot(range(len(closes)), upper_list, 'gray', linewidth=0.5, alpha=0.4)
    ax1.plot(range(len(closes)), lower_list, 'gray', linewidth=0.5, alpha=0.4)
    ax1.fill_between(range(len(closes)), lower_list, upper_list, alpha=0.05, color='blue')

    # 标记信号点
    if signals:
        buys = [s for s in signals if s['win']]
        sells = [s for s in signals if not s['win']]
        if buys:
            ax1.scatter([s['idx'] for s in buys], [s['buy_price'] for s in buys],
                       c='green', marker='^', s=100, label=f'盈利信号({len(buys)})', zorder=5)
        if sells:
            ax1.scatter([s['idx'] for s in sells], [s['buy_price'] for s in sells],
                       c='red', marker='v', s=100, label=f'亏损信号({len(sells)})', zorder=5)
        for s in signals:
            ax1.annotate(f"{s['hold_ret']:+.1f}%", xy=(s['idx'], s['buy_price']),
                        xytext=(0, 14), textcoords='offset points', fontsize=7,
                        ha='center', color='green' if s['win'] else 'red')

    ax1.set_ylabel('价格')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ticks = list(range(0, len(closes), max(1, len(closes)//12)))
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([str(dates[t])[:10] for t in ticks], rotation=45, fontsize=7)

    # 成交量
    colors = ['red' if closes[i] >= opens[i] else 'green' for i in range(len(closes))]
    ax2.bar(range(len(closes)), volumes, color=colors, alpha=0.6, width=1)
    ax2.set_ylabel('成交量')
    ax2.set_xlabel('日期')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([str(dates[t])[:10] for t in ticks], rotation=45, fontsize=7)

    plt.tight_layout()
    img_path = os.path.join(DATA_DIR, f"{symbol}_backtest_chart.png")
    plt.savefig(img_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  回测图表已保存: {img_path}")

    # 统计
    if signals:
        wins = len([s for s in signals if s['win']])
        wr = wins / len(signals) * 100
        avg_ret = np.mean([s['hold_ret'] for s in signals])
        print(f"\n  优化策略v1信号统计:")
        print(f"    总信号数: {len(signals)}")
        print(f"    盈利: {wins}, 亏损: {len(signals)-wins}")
        print(f"    胜率: {wr:.1f}%")
        print(f"    平均5日收益: {avg_ret:+.2f}%")

    return img_path


def main():
    print("=" * 70)
    print("  A股量化回测比对程序")
    print("  输入股票代码，自动运行多策略回测比对并保存数据")
    print("  数据保存目录: backtest_data/")
    print("=" * 70)

    while True:
        print("\n请选择操作:")
        print("  1. 输入股票代码进行回测比对")
        print("  2. 批量回测多只股票")
        print("  3. 查看已保存的回测数据")
        print("  4. 退出")

        choice = input("\n请输入选项 (1-4): ").strip()

        if choice == '1':
            symbol = input("请输入股票代码（如 600519）: ").strip()
            if not symbol:
                print("股票代码不能为空")
                continue
            days_input = input("回测天数（默认400）: ").strip()
            days = int(days_input) if days_input.isdigit() else 400
            hold_input = input("持有天数（默认5）: ").strip()
            hold = int(hold_input) if hold_input.isdigit() else 5

            # 运行多策略比对
            results, name = run_comparison_backtest(symbol, days, hold)

            # 绘制回测 vs 实际图
            print(f"\n  正在绘制 {name}({symbol}) 回测图表...")
            plot_stock_backtest(symbol, name, days, hold)

            print(f"\n  ✓ {name}({symbol}) 回测比对完成，数据已保存到 backtest_data/")

        elif choice == '2':
            symbols_input = input("请输入股票代码，用逗号分隔（如 600519,000858,300750）: ").strip()
            if not symbols_input:
                print("股票代码不能为空")
                continue
            symbols = [s.strip() for s in symbols_input.split(',') if s.strip()]
            days_input = input("回测天数（默认400）: ").strip()
            days = int(days_input) if days_input.isdigit() else 400
            hold_input = input("持有天数（默认5）: ").strip()
            hold = int(hold_input) if hold_input.isdigit() else 5

            print(f"\n批量回测 {len(symbols)} 只股票...")
            all_summary = []
            for sym in symbols:
                results, name = run_comparison_backtest(sym, days, hold)
                for r in results:
                    all_summary.append(r)

            if all_summary:
                batch_df = pd.DataFrame(all_summary)
                batch_path = os.path.join(DATA_DIR, f"batch_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                batch_df.to_csv(batch_path, index=False, encoding='utf-8-sig')
                print(f"\n批量回测汇总已保存: {batch_path}")

        elif choice == '3':
            print(f"\n已保存的回测数据目录: {DATA_DIR}")
            files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.csv', '.png'))]
            if not files:
                print("  暂无保存的数据")
            else:
                for f in sorted(files):
                    fpath = os.path.join(DATA_DIR, f)
                    size = os.path.getsize(fpath)
                    print(f"  {f}  ({size/1024:.1f} KB)")

        elif choice == '4':
            print("退出程序。")
            break

        else:
            print("无效选项，请重新输入")


if __name__ == '__main__':
    main()
