"""
数据获取层 - 直接调用东方财富公开API
避免 akshare 依赖问题，更稳定
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import warnings
warnings.filterwarnings('ignore')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://quote.eastmoney.com/',
}


class DataFetcher:
    """A股数据获取器（东方财富API）"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._all_stocks = None

    def _get(self, url, params=None, timeout=15):
        """统一请求方法"""
        try:
            r = self.session.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return None

    def get_all_stocks(self) -> pd.DataFrame:
        """获取A股全部股票列表（新浪财经API）"""
        if self._all_stocks is not None:
            return self._all_stocks
        try:
            all_items = []
            page = 1
            while True:
                url = "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
                params = {
                    'page': page, 'num': 80, 'sort': 'symbol', 'asc': 1,
                    'node': 'hs_a', 'symbol': '', '_s_r_a': 'auto',
                }
                r = self.session.get(url, params=params, timeout=15)
                r.raise_for_status()
                items = r.json()
                if not items:
                    break
                all_items.extend(items)
                if len(items) < 80:
                    break
                page += 1
                time.sleep(0.2)

            if not all_items:
                return pd.DataFrame()

            df = pd.DataFrame(all_items)
            # 转换数值列
            for col in ['trade', 'changepercent', 'pricechange', 'volume', 'amount',
                        'turnoverratio', 'per', 'pb', 'mktcap', 'nmc', 'high', 'low',
                        'open', 'settlement']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            # 重命名列
            df = df.rename(columns={
                'code': '代码', 'name': '名称', 'trade': '最新价',
                'changepercent': '涨跌幅', 'pricechange': '涨跌额',
                'volume': '成交量', 'amount': '成交额',
                'turnoverratio': '换手率', 'per': '市盈率',
                'pb': '市净率', 'mktcap': '总市值', 'nmc': '流通市值',
                'high': '最高', 'low': '最低', 'open': '今开',
                'settlement': '昨收',
            })
            # 过滤ST、退市、北交所
            df = df[~df['名称'].str.contains('ST|退|N ', na=False, regex=True)]
            df = df[df['代码'].str.startswith(('0', '3', '6'))]
            df = df[df['最新价'] > 0]
            self._all_stocks = df
            return df
        except Exception as e:
            print(f"获取股票列表失败: {e}")
            return pd.DataFrame()

    def get_daily_history(self, symbol: str, period: str = "daily",
                          start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取个股日线历史数据（新浪财经为主，腾讯备用）"""
        # 市场前缀
        prefix = 'sh' if symbol.startswith('6') else 'sz'
        full_code = f"{prefix}{symbol}"

        datalen = 500
        if start_date:
            try:
                sd = datetime.strptime(start_date, '%Y%m%d')
                datalen = max(60, (datetime.now() - sd).days)
            except:
                pass

        # 优先使用新浪财经
        df = self._get_history_sina(full_code, datalen)
        if df.empty:
            # 备用：腾讯财经
            df = self._get_history_tencent(full_code, datalen)

        if df.empty:
            return df

        # 按日期过滤
        if start_date:
            try:
                sd = pd.to_datetime(start_date)
                df = df[df['date'] >= sd]
            except:
                pass
        if end_date:
            try:
                ed = pd.to_datetime(end_date)
                df = df[df['date'] <= ed]
            except:
                pass

        return df.reset_index(drop=True)

    def _get_history_sina(self, full_code: str, datalen: int) -> pd.DataFrame:
        """新浪财经日线数据"""
        try:
            url = "https://quotes.sina.cn/cn/api/jsonp.php/var=/CN_MarketDataService.getKLineData"
            params = {'symbol': full_code, 'scale': 240, 'ma': 'no', 'datalen': datalen}
            r = self.session.get(url, params=params, timeout=15)
            r.raise_for_status()
            text = r.text
            start = text.index('(') + 1
            end = text.rindex(')')
            import json
            data = json.loads(text[start:end])
            if not data:
                return pd.DataFrame()

            rows = []
            for item in data:
                rows.append({
                    'date': item['day'],
                    'open': float(item['open']),
                    'close': float(item['close']),
                    'high': float(item['high']),
                    'low': float(item['low']),
                    'volume': float(item['volume']),
                    'turnover': 0, 'amplitude': 0, 'pct_change': 0,
                    'change': 0, 'turnover_rate': 0,
                })
            df = pd.DataFrame(rows)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

            if len(df) > 1:
                df['change'] = df['close'].diff()
                df['pct_change'] = df['close'].pct_change() * 100
                df['amplitude'] = (df['high'] - df['low']) / df['close'].shift(1) * 100
                df = df.fillna(0)
            return df
        except:
            return pd.DataFrame()

    def _get_history_tencent(self, full_code: str, datalen: int) -> pd.DataFrame:
        """腾讯财经日线数据（备用）"""
        try:
            url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
            params = {'_var': 'kline_dayqfq', 'param': f'{full_code},day,,,{datalen},qfq'}
            r = self.session.get(url, params=params, timeout=15)
            r.raise_for_status()
            text = r.text
            idx = text.index('=') + 1
            import json
            data = json.loads(text[idx:])
            if not data or 'data' not in data:
                return pd.DataFrame()
            kdata = data['data'].get(full_code, {})
            day = kdata.get('qfqday') or kdata.get('day')
            if not day:
                return pd.DataFrame()

            rows = []
            for item in day:
                rows.append({
                    'date': item[0],
                    'open': float(item[1]),
                    'close': float(item[2]),
                    'high': float(item[3]),
                    'low': float(item[4]),
                    'volume': float(item[5]),
                    'turnover': 0, 'amplitude': 0, 'pct_change': 0,
                    'change': 0, 'turnover_rate': 0,
                })
            df = pd.DataFrame(rows)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

            if len(df) > 1:
                df['change'] = df['close'].diff()
                df['pct_change'] = df['close'].pct_change() * 100
                df['amplitude'] = (df['high'] - df['low']) / df['close'].shift(1) * 100
                df = df.fillna(0)
            return df
        except:
            return pd.DataFrame()

    def get_fundamental(self, symbol: str) -> dict:
        """获取个股基本面数据（主要财务指标）"""
        info = {}
        try:
            # 个股基本信息
            market = 1 if symbol.startswith('6') else 0
            url = "https://push2.eastmoney.com/api/qt/stock/get"
            params = {
                'secid': f"{market}.{symbol}",
                'fields': 'f55,f57,f58,f84,f85,f116,f117,f162,f167,f173',
            }
            data = self._get(url, params)
            if data and 'data' in data and data['data']:
                d = data['data']
                info['roe'] = d.get('f173')
                info['pe'] = d.get('f162')
                info['pb'] = d.get('f167')
                info['total_market_cap'] = d.get('f116')
                info['float_market_cap'] = d.get('f117')
        except:
            pass
        return info

    def get_capital_flow(self, symbol: str) -> pd.DataFrame:
        """获取个股资金流向（近多日）"""
        try:
            market = 1 if symbol.startswith('6') else 0
            url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
            params = {
                'lmt': 60,
                'klt': 101,
                'fields1': 'f1,f2,f3,f7',
                'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65',
                'ut': 'b2884a393a59ad64002292a3e90d46a5',
                'secid': f"{market}.{symbol}",
            }
            data = self._get(url, params)
            if not data or 'data' not in data or not data['data']:
                return pd.DataFrame()
            klines = data['data'].get('klines', [])
            if not klines:
                return pd.DataFrame()
            rows = []
            for line in klines:
                parts = line.split(',')
                rows.append({
                    'date': parts[0],
                    'main_net_inflow': float(parts[1]),
                    'small_net_inflow': float(parts[2]),
                    'medium_net_inflow': float(parts[3]),
                    'large_net_inflow': float(parts[4]),
                    'super_large_net_inflow': float(parts[5]),
                })
            df = pd.DataFrame(rows)
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date').reset_index(drop=True)
        except:
            return pd.DataFrame()

    def get_north_flow(self) -> pd.DataFrame:
        """获取北向资金历史"""
        try:
            url = "https://push2his.eastmoney.com/api/qt/kamt.kline/get"
            params = {
                'fields1': 'f1,f2,f3,f5',
                'fields2': 'f51,f52,f53,f54,f55,f56',
                'klt': 101, 'lmt': 60,
            }
            data = self._get(url, params)
            if not data or 'data' not in data:
                return pd.DataFrame()
            # 北向资金 = 沪股通 + 深股通
            hk = data['data'].get('hk2sh', [])
            sz = data['data'].get('hk2sz', [])
            rows = []
            for i in range(min(len(hk), len(sz))):
                parts_h = hk[i].split(',')
                parts_s = sz[i].split(',')
                rows.append({
                    'date': parts_h[0],
                    'north_net_inflow': float(parts_h[1]) + float(parts_s[1]),
                })
            df = pd.DataFrame(rows)
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date').reset_index(drop=True)
        except:
            return pd.DataFrame()

    def get_limit_up_down(self, date: str = None) -> dict:
        """获取涨跌停数据"""
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        result = {'limit_up': pd.DataFrame(), 'limit_down': pd.DataFrame(), 'strong': pd.DataFrame()}
        try:
            # 涨停池
            url = "https://push2ex.eastmoney.com/getTopicZTPool"
            params = {'ut': '7eea3edcaed734bea9cbfc24409ed989', 'dpt': 'wz.ztzt',
                      'Pageindex': 0, 'pagesize': 500, 'sort': 'fbt:asc',
                      'date': date, '_': int(time.time() * 1000)}
            data = self._get(url, params)
            if data and 'data' in data and data['data']:
                result['limit_up'] = pd.DataFrame(data['data'].get('pool', []))
        except:
            pass
        try:
            # 跌停池
            url = "https://push2ex.eastmoney.com/getTopicDTPool"
            params = {'ut': '7eea3edcaed734bea9cbfc24409ed989', 'dpt': 'wz.dtzt',
                      'Pageindex': 0, 'pagesize': 500, 'sort': 'fund:asc',
                      'date': date, '_': int(time.time() * 1000)}
            data = self._get(url, params)
            if data and 'data' in data and data['data']:
                result['limit_down'] = pd.DataFrame(data['data'].get('pool', []))
        except:
            pass
        return result

    def get_dragon_tiger(self, date: str = None) -> pd.DataFrame:
        """获取龙虎榜数据"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        try:
            url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
            params = {
                'sortColumns': 'SECURITY_CODE,TRADE_DATE',
                'sortTypes': '-1,-1',
                'pageSize': 500, 'pageNumber': 1,
                'reportName': 'RPT_DAILYBILLBOARD_DETAILSNEW',
                'columns': 'ALL',
                'source': 'WEB', 'client': 'WEB',
                'filter': f'(TRADE_DATE=\'{date}\')',
            }
            data = self._get(url, params)
            if data and 'result' in data and data['result']:
                return pd.DataFrame(data['result'].get('data', []))
        except:
            pass
        return pd.DataFrame()

    def get_sector_fund_flow(self) -> pd.DataFrame:
        """获取板块资金流向"""
        try:
            url = "https://push2.eastmoney.com/api/qt/clist/get"
            params = {
                'pn': 1, 'pz': 100, 'po': 1, 'np': 1,
                'fltt': 2, 'invt': 2, 'fid': 'f62',
                'fs': 'm:90+t:2+f:!50',
                'fields': 'f1,f2,f3,f4,f12,f14,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87,f204,f205,f124',
            }
            data = self._get(url, params)
            if data and 'data' in data and data['data']:
                return pd.DataFrame(data['data'].get('diff', []))
        except:
            pass
        return pd.DataFrame()

    def get_news_sentiment(self, symbol: str = None, limit: int = 50) -> pd.DataFrame:
        """获取新闻数据"""
        try:
            if symbol:
                url = "https://search-api-web.eastmoney.com/search/jsonp"
                params = {
                    'cb': 'jQuery',
                    'param': json.dumps({
                        "uid": "", "keyword": symbol,
                        "type": ["cmsArticleWebOld"],
                        "client": "web", "clientType": "web",
                        "clientVersion": "curr", "param": {
                            "cmsArticleWebOld": {
                                "searchScope": "default", "sort": "default",
                                "pageIndex": 1, "pageSize": limit,
                                "preTag": "", "postTag": ""
                            }
                        }
                    })
                }
                r = self.session.get(url, params=params, timeout=10)
                text = r.text
                # 解析JSONP
                start = text.index('(') + 1
                end = text.rindex(')')
                data = json.loads(text[start:end])
                articles = data.get('result', {}).get('cmsArticleWebOld', [])
                rows = []
                for a in articles:
                    rows.append({
                        'title': a.get('title', ''),
                        'content': a.get('content', ''),
                        'time': a.get('date', ''),
                    })
                return pd.DataFrame(rows)
        except:
            pass
        return pd.DataFrame()

    def get_market_overview(self) -> dict:
        """获取大盘概况"""
        overview = {}
        try:
            # 上证指数
            url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
            params = {
                'secid': '1.000001',
                'fields1': 'f1,f2,f3,f4,f5,f6',
                'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
                'klt': 101, 'fqt': 0,
                'beg': (datetime.now() - timedelta(days=60)).strftime('%Y%m%d'),
                'end': datetime.now().strftime('%Y%m%d'),
            }
            data = self._get(url, params)
            if data and 'data' in data and data['data']:
                klines = data['data'].get('klines', [])
                if len(klines) >= 20:
                    closes = [float(k.split(',')[2]) for k in klines]
                    overview['sh_change_5d'] = (closes[-1] / closes[-6] - 1) * 100
                    overview['sh_change_20d'] = (closes[-1] / closes[-21] - 1) * 100
        except:
            pass

        try:
            # 全市场涨跌家数
            df = self.get_all_stocks()
            if not df.empty:
                overview['up_count'] = len(df[df['涨跌幅'] > 0])
                overview['down_count'] = len(df[df['涨跌幅'] < 0])
                overview['flat_count'] = len(df[df['涨跌幅'] == 0])
                overview['limit_up_count'] = len(df[df['涨跌幅'] >= 9.9])
                overview['limit_down_count'] = len(df[df['涨跌幅'] <= -9.9])
        except:
            pass
        return overview

    @staticmethod
    def _safe_float(val) -> float:
        try:
            if val is None or val == '' or val == '--':
                return np.nan
            return float(val)
        except:
            return np.nan
