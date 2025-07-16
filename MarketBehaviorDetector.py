"""
MarketBehaviorDetector - 市场行为检测模块
用于gamma squeeze信号捕捉系统的模式识别层
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque
from scipy import stats
from sklearn.preprocessing import StandardScaler
import copy
logger = logging.getLogger(__name__)

@dataclass
class SweepOrder:
    """扫单数据结构"""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    volume: float
    price: float
    frequency: float  # 订单频率
    anomaly_score: float  # 异常分数 0-1

@dataclass
class Divergence:
    """背离信号数据结构"""
    symbol: str
    divergence_type: str  # 'price_volume', 'momentum', 'breadth'
    strength: float  # 0-1
    duration: int  # 持续周期数
    details: Dict[str, Any]

@dataclass
class CrossMarketSignal:
    """跨市场信号数据结构"""
    lead_market: str
    lag_market: str
    correlation: float
    lag_time: float  # 秒
    signal_strength: float
    propagation_path: List[str]

class MarketBehaviorDetector:
    """市场行为检测器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.order_flow_analyzer = OrderFlowAnalyzer(self.config['order_flow'])
        self.divergence_detector = DivergenceDetector(self.config['divergence'])
        self.cross_market_analyzer = CrossMarketAnalyzer(self.config['cross_market'])
        self.anomaly_detector = AnomalyDetector()
        
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'order_flow': {
                'sweep_threshold': 3.0,  # 扫单识别阈值（标准差）
                'frequency_window': 60,  # 频率计算窗口（秒）
            },
            'divergence': {
                'lookback_period': 20,  # 回看周期
                'significance_level': 0.001,  # 显著性水平
                'min_duration': 3  # 最小持续周期
            },
            'cross_market': {
                'correlation_threshold': 0.7,  # 相关性阈值
                'max_lag': 300,  # 最大延迟（秒）
                'min_observations': 100  # 最小观测数
            },
            'market_regime': {
                'anomaly_threshold': 0.7,
                'volatility_percentile': 0.3,
                'volume_percentile': 0.7
            },
            'learning_params': {
                'enable_ml': True,
                'update_frequency': 3600  # 模型更新频率（秒）
            }
        }
    
    def detect(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """主检测方法"""
        result = {
            'timestamp': datetime.utcnow(),
            'sweep_orders': [],
            'divergences': [],
            'cross_market_signals': [],
            'anomaly_scores': {},
            'market_regime': {},
            'raw_metrics': {}
        }
        
        # 按数据类型分组
        spot_data = market_data[market_data['data_type'] == 'spot']
        orderbook_data = market_data[market_data['data_type'] == 'orderbook']
        
        # 1. 检测异常扫单
        sweeps = self.order_flow_analyzer.detect_sweeps(orderbook_data, spot_data)
        result['sweep_orders'] = sweeps
        
        # 2. 检测价格量背离
        divergences = self.divergence_detector.detect_divergences(spot_data)
        result['divergences'] = divergences
        
        # 3. 检测跨市场信号
        cross_signals = self.cross_market_analyzer.analyze_propagation(spot_data)
        result['cross_market_signals'] = cross_signals
        
        # 4. 综合异常评分
        anomaly_scores = self.anomaly_detector.calculate_scores(
            market_data, sweeps, divergences, cross_signals
        )
        result['anomaly_scores'] = anomaly_scores
        
        # 5. 市场状态识别
        regime = self._identify_market_regime(market_data, anomaly_scores)
        result['market_regime'] = regime
        
        # 6. 保存原始指标供学习
        result['raw_metrics'] = self._extract_raw_metrics(
            market_data, sweeps, divergences, cross_signals
        )
        
        return result
    
    def _identify_market_regime(self, data: pd.DataFrame, 
                               anomaly_scores: Dict) -> Dict[str, Any]:
        """识别市场状态"""
        regime = {
            'state': 'normal',  # normal, squeeze, breakout, consolidation
            'confidence': 0.0,
            'characteristics': {},
            'transition_probability': {}
        }
        
        # 提取市场特征
        if not data.empty:
            # 波动率特征
            spot_data = data[data['data_type'] == 'spot']
            if not spot_data.empty:
                returns = spot_data.groupby('symbol')['price'].pct_change()
                volatility = returns.std()
                
                volume_mean = spot_data.groupby('symbol')['volume'].mean()
                volume_std = spot_data.groupby('symbol')['volume'].std()
                
                # 使用配置中的阈值
                config = self.config.get('market_regime', {})
                anomaly_threshold = config.get('anomaly_threshold', 0.7)
                volatility_percentile = config.get('volatility_percentile', 0.3)
                volume_percentile = config.get('volume_percentile', 0.7)
                
                high_anomaly = any(score > anomaly_threshold for score in anomaly_scores.values())
                low_volatility = volatility.mean() < np.quantile(volatility, volatility_percentile)
                high_volume = volume_mean.mean() > np.quantile(volume_mean, volume_percentile)
                
                if high_anomaly and low_volatility:
                    regime['state'] = 'squeeze'
                    regime['confidence'] = 0.8
                elif high_anomaly and high_volume:
                    regime['state'] = 'breakout'
                    regime['confidence'] = 0.7
                elif low_volatility and not high_anomaly:
                    regime['state'] = 'consolidation'
                    regime['confidence'] = 0.6
                    
                regime['characteristics'] = {
                    'volatility_percentile': float(volatility.mean()),
                    'volume_percentile': float(volume_mean.mean()),
                    'anomaly_level': float(np.mean(list(anomaly_scores.values())))
                }
                
        return regime
    
    def _extract_raw_metrics(self, data: pd.DataFrame, sweeps: List[SweepOrder],
                           divergences: List[Divergence], 
                           cross_signals: List[CrossMarketSignal]) -> Dict:
        """提取原始指标"""
        metrics = {
            'sweep_intensity': len(sweeps) / max(len(data), 1) if data is not None else 0,
            'divergence_count': len(divergences),
            'cross_market_correlation': np.mean([s.correlation for s in cross_signals]) if cross_signals else 0,
            'feature_matrix': None  # 预留给ML特征
        }
        
        # 构建特征矩阵（为学习模块准备）
        if self.config['learning_params']['enable_ml']:
            features = []
            
            # 订单流特征
            if sweeps:
                features.extend([
                    np.mean([s.anomaly_score for s in sweeps]),
                    np.std([s.volume for s in sweeps]),
                    len([s for s in sweeps if s.side == 'buy']) / max(len(sweeps), 1)
                ])
            else:
                features.extend([0, 0, 0.5])
                
            # 背离特征
            if divergences:
                features.extend([
                    np.mean([d.strength for d in divergences]),
                    max([d.duration for d in divergences]),
                    len(set(d.divergence_type for d in divergences))
                ])
            else:
                features.extend([0, 0, 0])
                
            metrics['feature_matrix'] = np.array(features)
            
        return metrics

    def update_parameters(self, updates: Dict[str, Any]) -> bool:
        """动态更新参数"""
        try:
            # 深度更新配置
            def update_nested(target, source):
                for key, value in source.items():
                    if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                        update_nested(target[key], value)
                    else:
                        target[key] = value
            
            # 更新主配置
            update_nested(self.config, updates)
            
            # 同步更新子模块配置
            if 'order_flow' in updates:
                update_nested(self.order_flow_analyzer.config, updates['order_flow'])
            
            if 'divergence' in updates:
                update_nested(self.divergence_detector.config, updates['divergence'])
                
            if 'cross_market' in updates:
                update_nested(self.cross_market_analyzer.config, updates['cross_market'])
            
            logger.info(f"MarketBehaviorDetector parameters updated: {updates}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update parameters: {e}")
            return False

    def get_current_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return copy.deepcopy(self.config)

    def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标供学习模块使用"""
        metrics = {}
        
        # 扫单检测准确性（基于历史）
        if hasattr(self, 'sweep_detection_history'):
            metrics['sweep_detection_rate'] = len(self.sweep_detection_history) / max(self.total_checks, 1)
        
        # 背离检测统计
        if hasattr(self, 'divergence_history'):
            metrics['divergence_detection_rate'] = len(self.divergence_history) / max(self.total_checks, 1)
        
        # 跨市场信号质量
        if hasattr(self, 'cross_market_history'):
            valid_signals = sum(1 for s in self.cross_market_history if s.correlation > 0.8)
            metrics['high_quality_cross_market_rate'] = valid_signals / max(len(self.cross_market_history), 1)
        
        return metrics
    
    def get_ml_ready_features(self) -> Dict[str, np.ndarray]:
        """获取ML就绪的特征供AdaptiveLearner使用"""
        if not self.config['learning_params']['enable_ml']:
            return {}
        
        # 返回最近计算的特征矩阵
        features = {}
        for symbol in self.anomaly_detector.anomaly_history:
            history = self.anomaly_detector.anomaly_history[symbol]
            if history:
                features[symbol] = np.array([
                    h['components']['sweep'] for h in history[-100:]
                ])
        
        return features

class OrderFlowAnalyzer:
    """订单流分析器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.volume_history = defaultdict(lambda: deque(maxlen=1000))
        self.frequency_tracker = defaultdict(list)
        self.orderbook_fingerprint = {}

    def detect_sweeps(self, orderbook_data: pd.DataFrame, 
                     spot_data: pd.DataFrame) -> List[SweepOrder]:
        """检测扫单行为"""
        sweeps = []
        
        if orderbook_data.empty or spot_data.empty:
            return sweeps
            
        # 按symbol分组分析
        for symbol in orderbook_data['symbol'].unique():
            symbol_ob = orderbook_data[orderbook_data['symbol'] == symbol]
            symbol_spot = spot_data[spot_data['symbol'] == symbol]
            
            if symbol_spot.empty:
                continue
                
            # 分析每个时间点的订单簿
            for _, ob in symbol_ob.iterrows():
                # 提取买卖盘数据
                bids = ob.get('bids', [])
                asks = ob.get('asks', [])
                
                if not bids or not asks:
                    continue
                    
                # 检测大额扫单
                sweep = self._analyze_orderbook_sweep(
                    bids, asks, 
                    symbol_spot.iloc[-1]['price'],
                    symbol, 
                    ob['timestamp']
                )
                
                if sweep:
                    sweeps.append(sweep)
                    
        return sweeps
    

    def _analyze_orderbook_sweep(self, bids: List, asks: List, spot_price: float,
                                symbol: str, timestamp: datetime) -> Optional[SweepOrder]:
        """分析订单簿扫单"""
        # 1. 计算当前深度
        bid_depth = sum(float(b[1]) for b in bids[:5])
        ask_depth = sum(float(a[1]) for a in asks[:5])
        
        # 2. 更新历史
        self.volume_history[symbol].append({
            'timestamp': timestamp,
            'bid_depth': bid_depth,
            'ask_depth': ask_depth
        })
        
        # 3. 需要足够的历史数据
        if len(self.volume_history[symbol]) < 20:
            return None
        
        # 4. 计算历史统计（排除当前值）
        min_points = self.config.get('min_history_points', 30)
        if len(self.volume_history[symbol]) < min_points:
            return None
            
        history = list(self.volume_history[symbol])[:-1]
        bid_depths = [h['bid_depth'] for h in history]
        ask_depths = [h['ask_depth'] for h in history]
        
        # 5. 基础统计
        bid_mean = np.mean(bid_depths)
        bid_std = np.std(bid_depths)
        ask_mean = np.mean(ask_depths)
        ask_std = np.std(ask_depths)
        
        if bid_std == 0 or ask_std == 0:
            return None
        
        # 6. 计算Z-score
        bid_zscore = (bid_depth - bid_mean) / bid_std
        ask_zscore = (ask_depth - ask_mean) / ask_std
        
        # 7. 使用配置的阈值判断
        threshold = self.get_adaptive_threshold(symbol)
        
        # 8. 检测扫单
        sweep = None
        
        if bid_zscore > threshold:
            # 买方扫单
            frequency = self._calculate_order_frequency(symbol, 'buy', timestamp)
            sweep = SweepOrder(
                timestamp=timestamp,
                symbol=symbol,
                side='buy',
                volume=bid_depth,
                price=spot_price,
                frequency=frequency,
                anomaly_score=min(bid_zscore / (threshold * 2), 1.0)  # 归一化到0-1
            )
        elif ask_zscore > threshold:
            # 卖方扫单
            frequency = self._calculate_order_frequency(symbol, 'sell', timestamp)
            sweep = SweepOrder(
                timestamp=timestamp,
                symbol=symbol,
                side='sell',
                volume=ask_depth,
                price=spot_price,
                frequency=frequency,
                anomaly_score=min(ask_zscore / (threshold * 2), 1.0)
            )
        
        return sweep

    def get_adaptive_threshold(self, symbol: str) -> float:
        """获取自适应阈值（用于高级场景）"""
        if len(self.volume_history[symbol]) < 100:
            return self.config['sweep_threshold']
        
        # 基于最近100个样本的分布
        recent = list(self.volume_history[symbol])[-100:]
        all_depths = []
        for h in recent:
            all_depths.extend([h['bid_depth'], h['ask_depth']])
        
        # 使用95分位数作为动态阈值参考
        if all_depths:
            mean = np.mean(all_depths)
            std = np.std(all_depths)
            if std > 0:
                p95 = np.percentile(all_depths, 95)
                # 计算95分位数对应的z-score
                adaptive_z = (p95 - mean) / std
                # 在原阈值基础上调整
                return max(self.config['sweep_threshold'], adaptive_z * 0.8)
        
        return self.config['sweep_threshold']
    
    def _calculate_order_frequency(self, symbol: str, side: str, 
                                  timestamp: datetime) -> float:
        """计算订单频率"""
        key = f"{symbol}_{side}"
        self.frequency_tracker[key].append(timestamp)
        
        # 清理过期记录
        cutoff = timestamp - timedelta(seconds=self.config['frequency_window'])
        self.frequency_tracker[key] = [
            t for t in self.frequency_tracker[key] if t > cutoff
        ]
        
        # 计算频率（每分钟）
        count = len(self.frequency_tracker[key])
        return count * 60 / self.config['frequency_window']


class DivergenceDetector:
    """背离检测器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.price_volume_history = defaultdict(lambda: deque(maxlen=100))
        # 背离状态追踪：记录每种背离类型的连续检测次数
        self.divergence_duration_tracker = defaultdict(int)
        # RSI计算缓存
        self.rsi_cache = defaultdict(lambda: deque(maxlen=100))
        
    def detect_divergences(self, spot_data: pd.DataFrame) -> List[Divergence]:
        """检测价格量背离"""
        divergences = []
        
        if spot_data.empty:
            return divergences
            
        # 按symbol分组
        for symbol in spot_data['symbol'].unique():
            symbol_data = spot_data[spot_data['symbol'] == symbol].sort_values('timestamp')
            
            if len(symbol_data) < self.config['lookback_period']:
                continue
                
            # 更新历史
            for _, row in symbol_data.iterrows():
                self.price_volume_history[symbol].append({
                    'timestamp': row['timestamp'],
                    'price': row['price'],
                    'volume': row['volume']
                })
                
            # 1. 检测价格-成交量背离
            pv_div = self._detect_price_volume_divergence(symbol)
            if pv_div:
                divergences.append(pv_div)
            else:
                # 重置价量背离计数
                self._reset_divergence_duration(symbol, ['pv_bullish', 'pv_bearish'])
                
            # 2. 检测动量背离
            momentum_div = self._detect_momentum_divergence(symbol)
            if momentum_div:
                divergences.append(momentum_div)
            else:
                # 重置动量背离计数
                self._reset_divergence_duration(symbol, ['momentum_bullish', 'momentum_bearish'])
                
        return divergences
    
    def _detect_price_volume_divergence(self, symbol: str) -> Optional[Divergence]:
        """检测价格-成交量背离"""
        history = list(self.price_volume_history[symbol])
        if len(history) < self.config['lookback_period']:
            return None
            
        prices = [h['price'] for h in history]
        volumes = [h['volume'] for h in history]
        
        price_trend = self._calculate_trend(prices[-self.config['lookback_period']:])
        volume_trend = self._calculate_trend(volumes[-self.config['lookback_period']:])
        
        # 检测背离条件
        divergence_type = None
        
        if abs(price_trend['slope']) > self.config['significance_level']:
            if price_trend['slope'] > 0 and volume_trend['slope'] < 0:
                divergence_type = 'pv_bullish'  # 价涨量缩，可能见顶
            elif price_trend['slope'] < 0 and volume_trend['slope'] > 0:
                divergence_type = 'pv_bearish'  # 价跌量增，可能见底
        
        if divergence_type:
            # 更新持续时间
            duration_key = f"{symbol}_{divergence_type}"
            self.divergence_duration_tracker[duration_key] += 1
            duration = self.divergence_duration_tracker[duration_key]
            
            # 达到最小持续时间才生成信号
            if duration >= self.config.get('min_duration', 3):
                strength = abs(volume_trend['slope']) / (abs(price_trend['slope']) + 1e-6)
                return Divergence(
                    symbol=symbol,
                    divergence_type='price_volume',
                    strength=min(strength, 1.0),
                    duration=duration,
                    details={
                        'type': divergence_type.split('_')[1],  # bullish/bearish
                        'price_trend': price_trend['slope'],
                        'volume_trend': volume_trend['slope'],
                        'correlation': price_trend['correlation']
                    }
                )
                
        return None
    
    def _detect_momentum_divergence(self, symbol: str) -> Optional[Divergence]:
        """检测动量背离（RSI背离）"""
        history = list(self.price_volume_history[symbol])
        if len(history) < self.config['lookback_period'] * 2:
            return None
            
        prices = [h['price'] for h in history]
        
        # 计算RSI
        rsi_values = self._calculate_rsi(prices, 14)
        if len(rsi_values) < self.config['lookback_period']:
            return None
            
        # 寻找价格和RSI的高低点
        lookback = self.config['lookback_period']
        recent_prices = prices[-lookback:]
        recent_rsi = rsi_values[-lookback:]
        
        # 找出局部高低点
        price_peaks, price_troughs = self._find_peaks_and_troughs(recent_prices)
        rsi_peaks, rsi_troughs = self._find_peaks_and_troughs(recent_rsi)
        
        divergence_type = None
        strength = 0.0
        
        # 看涨背离：价格新低但RSI未创新低
        if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
            # 比较最近两个低点
            if (price_troughs[-1]['value'] < price_troughs[-2]['value'] and 
                rsi_troughs[-1]['value'] > rsi_troughs[-2]['value']):
                divergence_type = 'momentum_bullish'
                strength = 0.7 + 0.3 * (rsi_troughs[-1]['value'] - rsi_troughs[-2]['value']) / 50
                
        # 看跌背离：价格新高但RSI未创新高
        elif len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            if (price_peaks[-1]['value'] > price_peaks[-2]['value'] and 
                rsi_peaks[-1]['value'] < rsi_peaks[-2]['value']):
                divergence_type = 'momentum_bearish'
                strength = 0.7 + 0.3 * (rsi_peaks[-2]['value'] - rsi_peaks[-1]['value']) / 50
                
        if divergence_type:
            # 更新持续时间
            duration_key = f"{symbol}_{divergence_type}"
            self.divergence_duration_tracker[duration_key] += 1
            duration = self.divergence_duration_tracker[duration_key]
            
            # 达到最小持续时间才生成信号
            if duration >= self.config.get('min_duration', 3):
                return Divergence(
                    symbol=symbol,
                    divergence_type='momentum',
                    strength=min(strength, 1.0),
                    duration=duration,
                    details={
                        'type': divergence_type.split('_')[1],  # bullish/bearish
                        'latest_rsi': recent_rsi[-1],
                        'price_change': (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                    }
                )
                
        return None
    
    def _find_peaks_and_troughs(self, data: List[float], min_distance: int = 3) -> Tuple[List[Dict], List[Dict]]:
        """找出局部高点和低点"""
        peaks = []
        troughs = []
        
        for i in range(min_distance, len(data) - min_distance):
            # 检查是否为局部高点
            if all(data[i] >= data[j] for j in range(i - min_distance, i + min_distance + 1) if j != i):
                peaks.append({'index': i, 'value': data[i]})
            # 检查是否为局部低点
            elif all(data[i] <= data[j] for j in range(i - min_distance, i + min_distance + 1) if j != i):
                troughs.append({'index': i, 'value': data[i]})
                
        return peaks, troughs
    
    def _reset_divergence_duration(self, symbol: str, divergence_types: List[str]):
        """重置指定背离类型的持续时间"""
        for div_type in divergence_types:
            key = f"{symbol}_{div_type}"
            if key in self.divergence_duration_tracker:
                self.divergence_duration_tracker[key] = 0
    
    def _calculate_trend(self, data: List[float]) -> Dict[str, float]:
        """计算趋势"""
        x = np.arange(len(data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'correlation': r_value,
            'p_value': p_value
        }
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """计算RSI"""
        if len(prices) < period + 1:
            return []
            
        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 100
        rsi = [100 - 100 / (1 + rs)]
        
        for delta in deltas[period:]:
            if delta > 0:
                up = (up * (period - 1) + delta) / period
                down = down * (period - 1) / period
            else:
                up = up * (period - 1) / period
                down = (down * (period - 1) - delta) / period
                
            rs = up / down if down != 0 else 100
            rsi.append(100 - 100 / (1 + rs))
            
        return rsi


class CrossMarketAnalyzer:
    """跨市场分析器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.price_matrix = defaultdict(lambda: deque(maxlen=1000))
        self.correlation_cache = {}
        
    def analyze_propagation(self, spot_data: pd.DataFrame) -> List[CrossMarketSignal]:
        """分析跨市场传导"""
        signals = []
        
        if spot_data.empty:
            return signals
            
        # 更新价格矩阵
        for _, row in spot_data.iterrows():
            self.price_matrix[row['symbol']].append({
                'timestamp': row['timestamp'],
                'price': row['price']
            })
            
        # 获取所有symbol对
        symbols = list(self.price_matrix.keys())
        
        # 计算两两之间的关系
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                signal = self._analyze_pair(symbol1, symbol2)
                if signal:
                    signals.append(signal)
                    
        return signals
    
    def _analyze_pair(self, symbol1: str, symbol2: str) -> Optional[CrossMarketSignal]:
        """分析symbol对"""
        data1 = list(self.price_matrix[symbol1])
        data2 = list(self.price_matrix[symbol2])
        
        if len(data1) < self.config['min_observations'] or \
           len(data2) < self.config['min_observations']:
            return None
            
        # 对齐时间序列
        aligned_data = self._align_timeseries(data1, data2)
        if len(aligned_data) < self.config['min_observations']:
            return None
            
        # 计算领先滞后关系
        lead_lag = self._calculate_lead_lag(aligned_data)
        
        if lead_lag['correlation'] > self.config['correlation_threshold']:
            # 确定领先市场
            if lead_lag['optimal_lag'] > 0:
                lead_market = symbol1
                lag_market = symbol2
            else:
                lead_market = symbol2
                lag_market = symbol1
                
            return CrossMarketSignal(
                lead_market=lead_market,
                lag_market=lag_market,
                correlation=lead_lag['correlation'],
                lag_time=abs(lead_lag['optimal_lag']),
                signal_strength=lead_lag['signal_strength'],
                propagation_path=[lead_market, lag_market]
            )
            
        return None
    
    def _align_timeseries(self, data1: List[Dict], data2: List[Dict]) -> pd.DataFrame:
        """对齐时间序列"""
        df1 = pd.DataFrame(data1).set_index('timestamp')
        df2 = pd.DataFrame(data2).set_index('timestamp')
        
        # 合并并前向填充
        merged = pd.merge(
            df1, df2, 
            left_index=True, right_index=True, 
            how='outer', 
            suffixes=('_1', '_2')
        ).ffill().dropna()
        
        return merged
    
    def _calculate_lead_lag(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算领先滞后关系"""
        prices1 = data['price_1'].values
        prices2 = data['price_2'].values
        
        # 计算不同滞后期的相关性
        correlations = []
        max_lag = min(self.config['max_lag'], len(prices1) // 4)
        
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                corr = np.corrcoef(prices1[:lag], prices2[-lag:])[0, 1]
            elif lag > 0:
                corr = np.corrcoef(prices1[lag:], prices2[:-lag])[0, 1]
            else:
                corr = np.corrcoef(prices1, prices2)[0, 1]
                
            correlations.append((lag, corr))
            
        # 找到最优滞后期
        optimal_lag, max_corr = max(correlations, key=lambda x: x[1])
        
        # 计算信号强度
        signal_strength = self._calculate_signal_strength(
            prices1, prices2, optimal_lag, max_corr
        )
        
        return {
            'optimal_lag': optimal_lag,
            'correlation': max_corr,
            'signal_strength': signal_strength
        }
    
    def _calculate_signal_strength(self, prices1: np.ndarray, prices2: np.ndarray,
                                lag: int, correlation: float) -> float:
        """计算信号强度"""
        strength = abs(correlation)
        
        if len(prices1) > 100:
            window = 50
            rolling_corrs = []
            
            # 修正循环范围，确保切片有效
            start_idx = window + abs(lag)
            end_idx = len(prices1) - abs(lag)
            
            for i in range(start_idx, end_idx):
                try:
                    if lag < 0:
                        # prices1领先prices2
                        slice1 = prices1[i-window:i]
                        slice2 = prices2[i-window-lag:i-lag]
                    elif lag > 0:
                        # prices2领先prices1
                        slice1 = prices1[i-window:i]
                        slice2 = prices2[i-window+lag:i+lag]
                    else:
                        slice1 = prices1[i-window:i]
                        slice2 = prices2[i-window:i]
                    
                    # 确保切片长度相同且非空
                    if len(slice1) == len(slice2) == window:
                        if np.std(slice1) == 0 or np.std(slice2) == 0:
                            continue  
                        if np.any(np.isnan(slice1)) or np.any(np.isnan(slice2)):
                            continue  
                        
                        corr = np.corrcoef(slice1, slice2)[0, 1]
                        rolling_corrs.append(corr)
                except:
                    continue
                    
            if rolling_corrs:
                stability = 1 - np.std(rolling_corrs)
                strength *= max(0, stability)
                
        return min(strength, 1.0)


class AnomalyDetector:
    """综合异常检测器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_history = defaultdict(list)
        
    def calculate_scores(self, market_data: pd.DataFrame, 
                        sweeps: List[SweepOrder],
                        divergences: List[Divergence],
                        cross_signals: List[CrossMarketSignal]) -> Dict[str, float]:
        """计算综合异常分数"""
        scores = {}
        
        # 获取所有symbols
        symbols = set()
        if not market_data.empty:
            symbols.update(market_data['symbol'].unique())
            
        for symbol in symbols:
            # 计算各维度分数
            sweep_score = self._calculate_sweep_score(symbol, sweeps)
            divergence_score = self._calculate_divergence_score(symbol, divergences)
            cross_market_score = self._calculate_cross_market_score(symbol, cross_signals)
            
            # 综合评分（可以使用更复杂的融合方法）
            composite_score = (
                sweep_score * 0.4 +
                divergence_score * 0.3 +
                cross_market_score * 0.3
            )
            
            scores[symbol] = min(composite_score, 1.0)
            
            # 更新历史
            self.anomaly_history[symbol].append({
                'timestamp': datetime.utcnow(),
                'score': scores[symbol],
                'components': {
                    'sweep': sweep_score,
                    'divergence': divergence_score,
                    'cross_market': cross_market_score
                }
            })
            
        return scores
    
    def _calculate_sweep_score(self, symbol: str, sweeps: List[SweepOrder]) -> float:
        """计算扫单异常分数"""
        symbol_sweeps = [s for s in sweeps if s.symbol == symbol]
        if not symbol_sweeps:
            return 0.0
            
        # 考虑频率和强度
        avg_anomaly = np.mean([s.anomaly_score for s in symbol_sweeps])
        frequency_factor = min(len(symbol_sweeps) / 10, 1.0)  # 10次以上满分
        
        return avg_anomaly * 0.7 + frequency_factor * 0.3
    
    def _calculate_divergence_score(self, symbol: str, 
                                   divergences: List[Divergence]) -> float:
        """计算背离异常分数"""
        symbol_divs = [d for d in divergences if d.symbol == symbol]
        if not symbol_divs:
            return 0.0
            
        # 考虑强度和持续时间
        avg_strength = np.mean([d.strength for d in symbol_divs])
        max_duration = max([d.duration for d in symbol_divs])
        duration_factor = min(max_duration / 10, 1.0)  # 10周期以上满分
        
        return avg_strength * 0.6 + duration_factor * 0.4
    
    def _calculate_cross_market_score(self, symbol: str,
                                     cross_signals: List[CrossMarketSignal]) -> float:
        """计算跨市场异常分数"""
        # 找到涉及该symbol的信号
        related_signals = [
            s for s in cross_signals 
            if s.lead_market == symbol or s.lag_market == symbol
        ]
        
        if not related_signals:
            return 0.0
            
        # 领先市场加分更多
        lead_count = sum(1 for s in related_signals if s.lead_market == symbol)
        lag_count = sum(1 for s in related_signals if s.lag_market == symbol)
        
        lead_score = min(lead_count / 3, 1.0) * 0.7
        lag_score = min(lag_count / 3, 1.0) * 0.3
        
        return lead_score + lag_score