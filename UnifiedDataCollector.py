"""
UnifiedDataCollector - 统一数据采集模块
用于gamma squeeze信号捕捉系统的数据感知层
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import json
import logging
from abc import ABC, abstractmethod
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

proxy_url = "http://127.0.0.1:7890"
# 数据类型枚举
class DataType(Enum):
    OPTION = "option"
    SPOT = "spot"
    ORDERBOOK = "orderbook"
    GREEK = "greek"
    DERIVED = "derived"

# 数据结构定义
@dataclass
class DataPoint:
    """统一数据点结构"""
    timestamp: datetime
    source: str
    symbol: str
    data_type: DataType
    data: Dict[str, Any]
    
# 数据源基类
class DataSource(ABC):
    """数据源抽象基类"""
    def __init__(self, name: str, symbols: List[str]):
        self.name = name
        self.symbols = symbols
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    @abstractmethod
    async def fetch(self) -> List[DataPoint]:
        """获取数据的抽象方法"""
        pass

# Deribit期权数据源
class DeribitSource(DataSource):
    """Deribit期权数据采集"""
    def __init__(self, symbols: List[str] = ["BTC", "ETH"]):
        super().__init__("deribit", symbols)
        self.base_url = "https://www.deribit.com/api/v2/public"
        
    async def fetch(self) -> List[DataPoint]:
        """获取期权数据"""
        data_points = []
        
        for symbol in self.symbols:
            try:
                # 获取活跃期权合约
                instruments = await self._get_instruments(symbol)
                
                # 获取期权数据
                for instrument in instruments[:60]:  # 限制数量避免过多请求
                    option_data = await self._get_option_data(instrument['instrument_name'])
                    if option_data:
                        data_points.append(DataPoint(
                            timestamp=datetime.now(timezone.utc),
                            source=self.name,
                            symbol=symbol,
                            data_type=DataType.OPTION,
                            data={
                                'instrument': instrument['instrument_name'],
                                'strike': instrument.get('strike'),
                                'expiry': instrument.get('expiration_timestamp'),
                                'type': instrument.get('option_type'),
                                'open_interest': option_data.get('open_interest'),
                                'volume': option_data.get('stats', {}).get('volume'),
                                'iv': option_data.get('mark_iv'),
                                'bid': option_data.get('best_bid_price'),
                                'ask': option_data.get('best_ask_price'),
                                'mark': option_data.get('mark_price')
                            }
                        ))
            except Exception as e:
                logger.error(f"Error fetching Deribit data for {symbol}: {e}")
                
        return data_points
            
    async def _get_instruments(self, currency: str) -> List[Dict]:
        """获取期权合约列表 - 优化版"""
        url = f"{self.base_url}/get_instruments"
        params = {
            'currency': currency,
            'kind': 'option',
            'expired': 'false'
        }
        
        async with self.session.get(url, proxy=proxy_url, params=params) as resp:
            data = await resp.json()
            instruments = data.get('result', [])
            
        if not instruments:
            return []
        
        spot_price = await self._get_spot_price(currency)
        if not spot_price:
            logger.warning(f"无法获取{currency}现货价格，使用默认排序")
            return instruments[:60]  # 统一为60个
        
        # 过滤和排序期权
        valid_instruments = []
        for inst in instruments:
            # 解析到期时间
            expiry_ms = inst.get('expiration_timestamp', 0)
            days_to_expiry = (expiry_ms - datetime.now().timestamp() * 1000) / (1000 * 86400)
            
            # 只选择90天内到期的期权
            if days_to_expiry <= 0 or days_to_expiry > 90:
                continue
                
            # 计算行权价相对现货的偏离度
            strike = inst.get('strike', 0)
            if strike <= 0:
                continue
                
            moneyness = abs(strike - spot_price) / spot_price
            
            # 只选择偏离度在30%以内的期权
            if moneyness > 0.3:
                continue
                
            inst['moneyness'] = moneyness
            inst['days_to_expiry'] = days_to_expiry
            valid_instruments.append(inst)
        
        # 按照多个维度排序
        # 1. 先按到期日分组（近月优先）
        # 2. 每个到期日内，按行权价距离排序
        valid_instruments.sort(key=lambda x: (
            int(x['days_to_expiry'] / 7),  # 按周分组
            x['moneyness']  # 按价格距离排序
        ))
        
        # 确保包含不同行权价的期权
        selected = []
        strikes_above = set()
        strikes_below = set()
        strikes_at = set()
        
        for inst in valid_instruments:
            strike = inst['strike']
            
            # 分类期权
            if strike > spot_price * 1.01:  # 高于现价1%
                if strike not in strikes_above:
                    strikes_above.add(strike)
                    selected.append(inst)
            elif strike < spot_price * 0.99:  # 低于现价1%
                if strike not in strikes_below:
                    strikes_below.add(strike)
                    selected.append(inst)
            else:  # ATM附近
                if strike not in strikes_at:
                    strikes_at.add(strike)
                    selected.append(inst)
            
            # 限制总数但确保平衡
            if len(selected) >= 60:  # 增加到60个
                break
        
        # 确保有足够的高低行权价期权
        logger.info(f"\n{currency}期权选择: ATM={len(strikes_at)}, "
                    f"Above={len(strikes_above)}, Below={len(strikes_below)}")
        
        return selected

    async def _get_spot_price(self, currency: str) -> Optional[float]:
        """获取现货价格"""
        url = f"{self.base_url}/get_index_price"
        params = {'index_name': f'{currency.lower()}_usd'}
        
        try:
            async with self.session.get(url, proxy=proxy_url, params=params) as resp:
                data = await resp.json()
                return data.get('result', {}).get('index_price')
        except Exception as e:
            logger.error(f"获取{currency}现货价格失败: {e}")
            return None
    

    async def _get_option_data(self, instrument: str) -> Optional[Dict]:
        """获取单个期权数据"""
        url = f"{self.base_url}/ticker"
        params = {'instrument_name': instrument}
        
        async with self.session.get(url, proxy=proxy_url, params=params) as resp:
            data = await resp.json()
            return data.get('result')

# Binance现货数据源
class BinanceSource(DataSource):
    """Binance现货数据采集"""
    def __init__(self, symbols: List[str] = ["BTCUSDT", "ETHUSDT"]):
        super().__init__("binance", symbols)
        self.base_url = "https://api.binance.com/api/v3"
        
    async def fetch(self) -> List[DataPoint]:
        """获取现货数据"""
        data_points = []
        
        for symbol in self.symbols:
            try:
                # 获取价格数据
                ticker = await self._get_ticker(symbol)
                if ticker:
                    data_points.append(DataPoint(
                        timestamp=datetime.now(timezone.utc),
                        source=self.name,
                        symbol=symbol,
                        data_type=DataType.SPOT,
                        data={
                            'price': float(ticker['lastPrice']),
                            'bid': float(ticker['bidPrice']),
                            'ask': float(ticker['askPrice']),
                            'volume': float(ticker['volume']),
                            'quote_volume': float(ticker['quoteVolume']),
                            'high_24h': float(ticker['highPrice']),
                            'low_24h': float(ticker['lowPrice']),
                            'price_change_24h': float(ticker['priceChangePercent'])
                        }
                    ))
                    
                # 获取订单簿数据
                orderbook = await self._get_orderbook(symbol)
                if orderbook:
                    data_points.append(DataPoint(
                        timestamp=datetime.now(timezone.utc),
                        source=self.name,
                        symbol=symbol,
                        data_type=DataType.ORDERBOOK,
                        data={
                            'bids': orderbook['bids'][:10],  # Top 10
                            'asks': orderbook['asks'][:10],
                            'bid_volume': sum(float(b[1]) for b in orderbook['bids'][:10]),
                            'ask_volume': sum(float(a[1]) for a in orderbook['asks'][:10])
                        }
                    ))
            except Exception as e:
                logger.error(f"Error fetching Binance data for {symbol}: {e}")
                
        return data_points
        
    async def _get_ticker(self, symbol: str) -> Optional[Dict]:
        """获取ticker数据"""
        url = f"{self.base_url}/ticker/24hr"
        params = {'symbol': symbol}
        
        async with self.session.get(url, proxy=proxy_url, params=params) as resp:
            return await resp.json()
            
    async def _get_orderbook(self, symbol: str) -> Optional[Dict]:
        """获取订单簿数据"""
        url = f"{self.base_url}/depth"
        params = {'symbol': symbol, 'limit': 20}
        
        async with self.session.get(url, proxy=proxy_url, params=params) as resp:
            return await resp.json()

# Greeks计算器
class GreeksCalculator:
    """期权Greeks计算"""
    @staticmethod
    def calculate_bs_greeks(S: float, K: float, T: float, r: float, 
                           sigma: float, option_type: str = 'call') -> Dict[str, float]:
        """Black-Scholes Greeks计算"""
        from scipy.stats import norm
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        else:
            delta = -norm.cdf(-d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
            
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega
        }

# 衍生指标计算器
class DerivedMetricsCalculator:
    """衍生指标计算"""
    def __init__(self):
        self.price_history = defaultdict(list)
        self.volume_history = defaultdict(list)
        self.window_size = 20
        
    def update_history(self, symbol: str, price: float, volume: float):
        """更新历史数据"""
        self.price_history[symbol].append(price)
        self.volume_history[symbol].append(volume)
        
        # 保持窗口大小
        if len(self.price_history[symbol]) > self.window_size:
            self.price_history[symbol].pop(0)
        if len(self.volume_history[symbol]) > self.window_size:
            self.volume_history[symbol].pop(0)
            
    def calculate_price_acceleration(self, symbol: str) -> float:
        """计算价格加速度"""
        prices = self.price_history[symbol]
        if len(prices) < 3:
            return 0.0
            
        # 计算一阶差分（速度）
        velocity = np.diff(prices)
        # 计算二阶差分（加速度）
        acceleration = np.diff(velocity)
        
        return float(np.mean(acceleration)) if len(acceleration) > 0 else 0.0
        
    def calculate_volume_anomaly(self, symbol: str) -> float:
        """计算成交量异常度"""
        volumes = self.volume_history[symbol]
        if len(volumes) < 5:
            return 0.0
            
        mean_vol = np.mean(volumes[:-1])
        std_vol = np.std(volumes[:-1])
        
        if std_vol == 0:
            return 0.0
            
        # Z-score
        z_score = (volumes[-1] - mean_vol) / std_vol
        return float(z_score)

# 统一数据采集器
class UnifiedDataCollector:
    """统一数据采集器主类"""
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.sources: Dict[str, DataSource] = {}
        self.calculators = {
            'greeks': GreeksCalculator(),
            'derived': DerivedMetricsCalculator()
        }
        self.data_buffer: List[pd.DataFrame] = []
        self.tasks: List[asyncio.Task] = []
        self._running = False
        
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'deribit': {
                'enabled': True,
                'symbols': ['BTC', 'ETH'],
                'interval': 30  # 秒
            },
            'binance': {
                'enabled': True,
                'symbols': ['BTCUSDT', 'ETHUSDT'],
                'interval': 1  # 秒
            },
            'buffer_size': 1000,
            'export_interval': 60  # 秒
        }
        
    async def initialize(self):
        """初始化数据源"""
        if self.config['deribit']['enabled']:
            self.sources['deribit'] = DeribitSource(
                symbols=self.config['deribit']['symbols']
            )
            
        if self.config['binance']['enabled']:
            self.sources['binance'] = BinanceSource(
                symbols=self.config['binance']['symbols']
            )
            
    async def start(self):
        """启动数据采集"""
        logger.info("Starting UnifiedDataCollector...")
        self._running = True
        
        # 初始化session
        for source in self.sources.values():
            await source.__aenter__()
            
        # 启动采集任务
        for name, source in self.sources.items():
            interval = self.config[name]['interval']
            task = asyncio.create_task(
                self._collect_loop(source, interval)
            )
            self.tasks.append(task)
            
        # 启动数据处理任务
        process_task = asyncio.create_task(self._process_loop())
        self.tasks.append(process_task)
        
    async def stop(self):
        """停止数据采集"""
        logger.info("Stopping UnifiedDataCollector...")
        self._running = False
        
        # 取消所有任务
        for task in self.tasks:
            task.cancel()
            
        # 等待任务完成
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # 关闭session
        for source in self.sources.values():
            await source.__aexit__(None, None, None)
            
    async def _collect_loop(self, source: DataSource, interval: float):
        """数据采集循环"""
        while self._running:
            try:
                # 采集数据
                data_points = await source.fetch()
                
                # 处理数据点
                for dp in data_points:
                    await self._process_data_point(dp)
                    
                # 等待下次采集
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in collect loop for {source.name}: {e}")
                await asyncio.sleep(interval)
                
    async def _process_data_point(self, data_point: DataPoint):
        """处理单个数据点"""
        # 更新衍生指标历史
        if data_point.data_type == DataType.SPOT:
            self.calculators['derived'].update_history(
                data_point.symbol,
                data_point.data['price'],
                data_point.data['volume']
            )
            
        # 添加到缓冲区
        df_row = {
            'timestamp': data_point.timestamp,
            'source': data_point.source,
            'symbol': data_point.symbol,
            'data_type': data_point.data_type.value,
            **data_point.data
        }
        
        self.data_buffer.append(pd.DataFrame([df_row]))
        
        # 检查缓冲区大小
        if len(self.data_buffer) > self.config['buffer_size']:
            self.data_buffer.pop(0)
            
    async def _process_loop(self):
        """数据处理循环"""
        while self._running:
            try:
                # 计算衍生指标
                await self._calculate_derived_metrics()
                
                # 定期导出数据
                await asyncio.sleep(self.config['export_interval'])
                
            except Exception as e:
                logger.error(f"Error in process loop: {e}")
                await asyncio.sleep(10)
                
    async def _calculate_derived_metrics(self):
        """计算衍生指标"""
        # 获取最新数据
        if not self.data_buffer:
            return
            
        latest_data = pd.concat(self.data_buffer[-100:], ignore_index=True)
        spot_data = latest_data[latest_data['data_type'] == 'spot']
        # 计算每个symbol的衍生指标
        for symbol in spot_data['symbol'].unique():
            symbol_data = latest_data[latest_data['symbol'] == symbol]
            
            # 价格加速度
            acceleration = self.calculators['derived'].calculate_price_acceleration(symbol)
            
            # 成交量异常度
            volume_anomaly = self.calculators['derived'].calculate_volume_anomaly(symbol)
            
            # 创建衍生指标数据点
            derived_dp = DataPoint(
                timestamp=datetime.now(timezone.utc),
                source='calculated',
                symbol=symbol,
                data_type=DataType.DERIVED,
                data={
                    'price_acceleration': acceleration,
                    'volume_anomaly': volume_anomaly
                }
            )
            
            await self._process_data_point(derived_dp)
            
    def get_latest_data(self, window_seconds: int = 60) -> pd.DataFrame:
        """获取最新数据"""
        if not self.data_buffer:
            return pd.DataFrame()
            
        # 合并所有缓冲数据
        df = pd.concat(self.data_buffer, ignore_index=True)
        
        # 筛选时间窗口
        cutoff_time = datetime.now(timezone.utc) - pd.Timedelta(seconds=window_seconds)
        df = df[df['timestamp'] > cutoff_time]
        
        return df
        
    def export_data(self, filepath: str):
        """导出数据到文件"""
        df = self.get_latest_data(window_seconds=3600)  # 导出最近1小时
        df.to_csv(filepath, index=False)
        logger.info(f"Data exported to {filepath}")

# 使用示例
async def main():
    # 配置
    config = {
        'deribit': {
            'enabled': True,
            'symbols': ['BTC', 'ETH'],
            'interval': 30
        },
        'binance': {
            'enabled': True, 
            'symbols': ['BTCUSDT', 'ETHUSDT'],
            'interval': 1
        },
        'buffer_size': 1000,
        'export_interval': 300  # 5分钟
    }
    
    # 创建采集器
    collector = UnifiedDataCollector(config)
    
    # 初始化
    await collector.initialize()
    
    # 启动采集
    await collector.start()
    
    try:
        # 运行一段时间
        await asyncio.sleep(60)
        
        # 获取最新数据
        latest_data = collector.get_latest_data(window_seconds=30)
        print(f"Collected {len(latest_data)} data points")
        print(latest_data.head())
        
        # 导出数据
        collector.export_data('gamma_squeeze_data.csv')
        
    finally:
        # 停止采集
        await collector.stop()

if __name__ == "__main__":
    asyncio.run(main())