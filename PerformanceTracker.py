"""
PerformanceTracker - 增强版信号表现跟踪模块
支持连续决策监控和多维度绩效评价
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
import logging
import json
import os
from collections import defaultdict, deque
import asyncio
from SignalEvaluator import TradingSignal

logger = logging.getLogger(__name__)

@dataclass
class DecisionSnapshot:
    """决策快照 - 记录每个决策时刻的完整信息"""
    timestamp: datetime
    decision_type: str  # 'signal_generated', 'no_signal', 'signal_suppressed'
    assets_analyzed: List[str]
    
    # 市场快照
    market_snapshot: Dict[str, 'MarketSnapshot']
    
    # 决策因素
    gamma_metrics: Dict[str, Any]
    behavior_metrics: Dict[str, Any]
    scores: Dict[str, float]
    
    # 决策结果
    signal_generated: Optional['SignalPerformance'] = None
    suppression_reason: Optional[str] = None
    
    # 反事实分析数据
    counterfactual_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketSnapshot:
    """市场快照 - 记录决策时刻的市场状态"""
    asset: str
    timestamp: datetime
    
    # 价格数据
    price: float
    bid: float
    ask: float
    spread: float
    
    # 微观结构
    orderbook_depth: Dict[str, float]  # {'bid_depth_5': xx, 'ask_depth_5': xx}
    orderbook_imbalance: float
    recent_trades: List[Dict[str, Any]]
    
    # 期权结构
    gamma_distribution: Dict[str, float]
    iv_surface: Dict[str, float]
    put_call_skew: float
    nearest_gamma_wall: Dict[str, Any]
    
    # 技术指标
    trend_strength: float
    support_levels: List[float]
    resistance_levels: List[float]
    momentum_indicators: Dict[str, float]
    
    # 市场状态
    volatility_regime: str
    liquidity_score: float
    market_regime: str

@dataclass
class PricePathMetrics:
    """价格路径指标"""
    max_favorable_move: float
    max_adverse_move: float
    time_to_max_favorable: float  # 小时
    time_to_max_adverse: float
    path_volatility: float
    momentum_score: float
    reversal_count: int
    avg_drawdown: float
    sharpe_ratio: float

@dataclass
class SignalPerformance:
    """增强的信号表现记录"""
    signal_id: str
    signal_timestamp: datetime
    asset: str
    signal_type: str
    direction: str
    initial_price: float
    strength: float
    confidence: float
    expected_move: str
    time_horizon: str
    
    # 市场快照（信号发出时）
    market_snapshot: MarketSnapshot
    
    # 多时间尺度价格记录
    price_5m: Optional[float] = None
    price_15m: Optional[float] = None
    price_30m: Optional[float] = None
    price_1h: Optional[float] = None
    price_2h: Optional[float] = None
    price_4h: Optional[float] = None
    price_8h: Optional[float] = None
    price_1d: Optional[float] = None
    
    # 多时间尺度收益
    return_5m: Optional[float] = None
    return_15m: Optional[float] = None
    return_30m: Optional[float] = None
    return_1h: Optional[float] = None
    return_2h: Optional[float] = None
    return_4h: Optional[float] = None
    return_8h: Optional[float] = None
    return_1d: Optional[float] = None
    
    # 细粒度评分
    direction_score: Optional[float] = None  # -1到1，考虑偏离程度
    timing_score: Optional[float] = None     # 0到1，信号时机
    persistence_score: Optional[float] = None # 0到1，有效持续时间
    robustness_score: Optional[float] = None # 0到1，不同市场状态下的稳定性
    
    # 价格路径指标
    path_metrics: Optional[PricePathMetrics] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluation_complete: bool = False
    evaluation_timestamp: Optional[datetime] = None

class PerformanceTracker:
    """增强版性能跟踪器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # 数据存储路径
        self.signal_db_path = self.config['signal_db_path']
        self.decision_db_path = self.config['decision_db_path']
        
        # 活跃信号跟踪（保持兼容性）
        self.active_signals: Dict[str, SignalPerformance] = {}
        
        # 决策历史
        self.decision_history: deque = deque(maxlen=10000)
        self.market_snapshots: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # 价格获取器
        self.price_fetcher = None
        self.market_data_fetcher = None
        
        # 统计缓存
        self.performance_cache = {}
        self.last_cache_update = datetime.utcnow()
        
        # 初始化
        self._ensure_db_exists()
        self._load_active_signals()
        
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'signal_db_path': 'signal_performance_enhanced.csv',
            'decision_db_path': 'decision_history.csv',
            'check_intervals': [5/60, 15/60, 30/60, 1, 2, 4, 8, 24],  # 小时
            'decision_interval': 60,  # 决策记录间隔（秒）
            'update_interval': 300,   # 价格更新间隔
            'report_interval': 1800,  # 报告间隔
            'expected_move_ranges': {
                "1-2%": (1, 2),
                "2-5%": (2, 5),
                "5-10%": (5, 10),
                "10%+": (10, 20)
            },
            'time_horizon_hours': {
                "0-1h": 1,
                "1-2h": 2,
                "2-4h": 3,
                "4-8h": 6,
                "8-24h": 16
            },
            'market_snapshot_features': {
                'orderbook_levels': 5,
                'trade_history_size': 100,
                'technical_indicators': ['rsi', 'macd', 'bb']
            }
        }
    
    def _ensure_db_exists(self):
        """确保数据库文件存在"""
        # 信号数据库
        if not os.path.exists(self.signal_db_path):
            columns = [
                'signal_id', 'signal_timestamp', 'asset', 'signal_type',
                'direction', 'initial_price', 'strength', 'confidence',
                'expected_move', 'time_horizon'
            ]
            # 添加价格列
            for interval in ['5m', '15m', '30m', '1h', '2h', '4h', '8h', '1d']:
                columns.extend([f'price_{interval}', f'return_{interval}'])
            # 添加评分列
            columns.extend(['direction_score', 'timing_score', 'persistence_score', 'robustness_score'])
            # 添加路径指标列
            columns.extend([
                'max_favorable_move', 'max_adverse_move', 'path_volatility',
                'momentum_score', 'sharpe_ratio'
            ])
            columns.extend(['metadata', 'evaluation_complete', 'evaluation_timestamp'])
            
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.signal_db_path, index=False)
        
        # 决策数据库
        if not os.path.exists(self.decision_db_path):
            columns = [
                'timestamp', 'decision_type', 'assets_analyzed', 'signal_generated',
                'suppression_reason', 'gamma_metrics', 'behavior_metrics', 'scores',
                'market_snapshot', 'counterfactual_data'
            ]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.decision_db_path, index=False)
    
    def _load_active_signals(self):
        """加载未完成评估的信号"""
        try:
            df = pd.read_csv(self.signal_db_path)
            active_df = df[df['evaluation_complete'] == False]
            
            for _, row in active_df.iterrows():
                # 重建SignalPerformance对象（简化版，不包含完整市场快照）
                perf = SignalPerformance(
                    signal_id=row['signal_id'],
                    signal_timestamp=pd.to_datetime(row['signal_timestamp']),
                    asset=row['asset'],
                    signal_type=row['signal_type'],
                    direction=row['direction'],
                    initial_price=row['initial_price'],
                    strength=row['strength'],
                    confidence=row['confidence'],
                    expected_move=row['expected_move'],
                    time_horizon=row['time_horizon'],
                    market_snapshot=None,  # 暂时为None
                    metadata=json.loads(row['metadata']) if pd.notna(row['metadata']) else {}
                )
                
                # 恢复已有的价格数据
                for interval in ['5m', '15m', '30m', '1h', '2h', '4h', '8h', '1d']:
                    price_key = f'price_{interval}'
                    return_key = f'return_{interval}'
                    if pd.notna(row.get(price_key)):
                        setattr(perf, price_key, row[price_key])
                        setattr(perf, return_key, row.get(return_key))
                
                self.active_signals[perf.signal_id] = perf
                
        except Exception as e:
            logger.error(f"Error loading active signals: {e}")
    
    async def record_decision(self, assets_analyzed: List[str], 
                            gamma_analysis: Dict[str, Any],
                            market_behavior: Dict[str, Any],
                            scores: Dict[str, Dict[str, float]],
                            signals_generated: List[TradingSignal],
                            suppressed_signals: Dict[str, str]):
        """记录决策时刻"""
        timestamp = datetime.utcnow()
        
        # 获取市场快照
        market_snapshots = {}
        for asset in assets_analyzed:
            snapshot = await self._capture_market_snapshot(asset, gamma_analysis, market_behavior)
            if snapshot:
                market_snapshots[asset] = snapshot
                self.market_snapshots[asset].append(snapshot)
        
        # 构建决策快照
        decision = DecisionSnapshot(
            timestamp=timestamp,
            decision_type='signal_generated' if signals_generated else 'no_signal',
            assets_analyzed=assets_analyzed,
            market_snapshot=market_snapshots,
            gamma_metrics=self._extract_gamma_metrics(gamma_analysis),
            behavior_metrics=self._extract_behavior_metrics(market_behavior),
            scores=scores,
            signal_generated=None,
            suppression_reason=None
        )
        
        # 处理生成的信号
        if signals_generated:
            for signal in signals_generated:
                perf = self._create_signal_performance(signal, market_snapshots.get(signal.asset))
                decision.signal_generated = perf
                self.active_signals[perf.signal_id] = perf
                self._save_signal(perf)
        
        # 记录被抑制的信号
        if suppressed_signals:
            decision.suppression_reason = json.dumps(suppressed_signals)
        
        # 添加反事实分析数据
        decision.counterfactual_data = await self._generate_counterfactual_data(
            assets_analyzed, market_snapshots, scores
        )
        
        # 保存决策
        self.decision_history.append(decision)
        self._save_decision(decision)
    
    def track_signal(self, signal: TradingSignal, initial_price: float):
        """跟踪新信号（保持向后兼容）"""
        market_snapshot = self._get_latest_market_snapshot(signal.asset)
        perf = self._create_signal_performance(signal, market_snapshot, initial_price)
        self.active_signals[perf.signal_id] = perf
        self._save_signal(perf)
        logger.info(f"Started tracking signal {perf.signal_id}")
    
    def _create_signal_performance(self, signal: TradingSignal, 
                                 market_snapshot: Optional[MarketSnapshot],
                                 initial_price: Optional[float] = None) -> SignalPerformance:
        """创建信号性能记录"""
        signal_id = f"{signal.asset}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        if initial_price is None and market_snapshot:
            initial_price = market_snapshot.price
        
        return SignalPerformance(
            signal_id=signal_id,
            signal_timestamp=signal.timestamp,
            asset=signal.asset,
            signal_type=signal.signal_type,
            direction=signal.direction,
            initial_price=initial_price,
            strength=signal.strength,
            confidence=signal.confidence,
            expected_move=signal.expected_move,
            time_horizon=signal.time_horizon,
            market_snapshot=market_snapshot,
            metadata=signal.metadata
        )
    
    async def update_prices(self):
        """更新所有活跃信号的价格"""
        if not self.price_fetcher:
            logger.error("Price fetcher not set")
            return
        
        current_time = datetime.utcnow()
        
        for signal_id, performance in list(self.active_signals.items()):
            try:
                current_price = await self.price_fetcher(performance.asset)
                if current_price is None:
                    continue
                
                elapsed_hours = (current_time - performance.signal_timestamp).total_seconds() / 3600
                
                # 更新多时间尺度价格
                for interval_str, hours in [
                    ('5m', 5/60), ('15m', 15/60), ('30m', 0.5),
                    ('1h', 1), ('2h', 2), ('4h', 4), ('8h', 8), ('1d', 24)
                ]:
                    price_key = f'price_{interval_str}'
                    return_key = f'return_{interval_str}'
                    
                    if elapsed_hours >= hours and getattr(performance, price_key) is None:
                        setattr(performance, price_key, current_price)
                        returns = ((current_price - performance.initial_price) / 
                                 performance.initial_price * 100)
                        setattr(performance, return_key, returns)
                        
                        logger.info(f"Updated {interval_str} price for {signal_id}: "
                                  f"{current_price} ({returns:+.2f}%)")
                
                # 更新价格路径指标
                self._update_price_path_metrics(performance, current_price, elapsed_hours)
                
                # 检查是否完成评估
                if elapsed_hours >= 24:  # 24小时后完成评估
                    self._evaluate_performance(performance)
                
            except Exception as e:
                logger.error(f"Error updating prices for {signal_id}: {e}")
    
    def _update_price_path_metrics(self, performance: SignalPerformance, 
                                  current_price: float, elapsed_hours: float):
        """更新价格路径指标"""
        if performance.path_metrics is None:
            performance.path_metrics = PricePathMetrics(
                max_favorable_move=0,
                max_adverse_move=0,
                time_to_max_favorable=0,
                time_to_max_adverse=0,
                path_volatility=0,
                momentum_score=0,
                reversal_count=0,
                avg_drawdown=0,
                sharpe_ratio=0
            )
        
        # 计算当前收益
        current_return = (current_price - performance.initial_price) / performance.initial_price * 100
        
        # 更新最大有利/不利波动
        if performance.direction == 'BULLISH':
            if current_return > performance.path_metrics.max_favorable_move:
                performance.path_metrics.max_favorable_move = current_return
                performance.path_metrics.time_to_max_favorable = elapsed_hours
            if current_return < performance.path_metrics.max_adverse_move:
                performance.path_metrics.max_adverse_move = current_return
                performance.path_metrics.time_to_max_adverse = elapsed_hours
        else:  # BEARISH
            if current_return < performance.path_metrics.max_favorable_move:
                performance.path_metrics.max_favorable_move = current_return
                performance.path_metrics.time_to_max_favorable = elapsed_hours
            if current_return > performance.path_metrics.max_adverse_move:
                performance.path_metrics.max_adverse_move = current_return
                performance.path_metrics.time_to_max_adverse = elapsed_hours
    
    def _evaluate_performance(self, performance: SignalPerformance):
        """评估信号表现"""
        # 1. 方向评分（-1到1）
        performance.direction_score = self._calculate_direction_score(performance)
        
        # 2. 时机评分（0到1）
        performance.timing_score = self._calculate_timing_score(performance)
        
        # 3. 持续性评分（0到1）
        performance.persistence_score = self._calculate_persistence_score(performance)
        
        # 4. 稳健性评分（0到1）
        performance.robustness_score = self._calculate_robustness_score(performance)
        
        # 标记评估完成
        performance.evaluation_complete = True
        performance.evaluation_timestamp = datetime.utcnow()
        
        # 保存结果
        self._save_signal(performance)
        
        # 从活跃信号中移除
        del self.active_signals[performance.signal_id]
        
        # 输出总结
        self._print_performance_summary(performance)
    
    def _calculate_direction_score(self, perf: SignalPerformance) -> float:
        """计算方向评分"""
        # 获取所有时间点的收益
        returns = []
        for interval in ['5m', '15m', '30m', '1h', '2h', '4h', '8h', '1d']:
            ret = getattr(perf, f'return_{interval}')
            if ret is not None:
                returns.append(ret)
        
        if not returns:
            return 0.0
        
        # 计算方向正确性和幅度
        if perf.direction == 'BULLISH':
            correct_returns = [r for r in returns if r > 0]
            avg_return = np.mean(returns)
        else:  # BEARISH
            correct_returns = [r for r in returns if r < 0]
            avg_return = -np.mean(returns)
        
        # 方向正确率
        direction_accuracy = len(correct_returns) / len(returns)
        
        # 考虑幅度的评分
        magnitude_factor = min(abs(avg_return) / 5, 1)  # 5%为满分
        
        # 综合评分
        score = direction_accuracy * 0.7 + magnitude_factor * 0.3
        
        # 如果方向完全错误，给负分
        if direction_accuracy == 0:
            score = -magnitude_factor
        
        return np.clip(score, -1, 1)
    
    def _calculate_timing_score(self, perf: SignalPerformance) -> float:
        """计算时机评分"""
        if perf.path_metrics is None:
            return 0.0
        
        # 评估信号是否在趋势开始时发出
        time_to_peak = perf.path_metrics.time_to_max_favorable
        total_time = 24  # 总观察时间
        
        # 越早达到峰值，时机越好
        early_peak_score = 1 - (time_to_peak / total_time) if time_to_peak > 0 else 0
        
        # 评估不利波动的控制
        adverse_control = 1 - min(abs(perf.path_metrics.max_adverse_move) / 5, 1)
        
        # 综合评分
        return early_peak_score * 0.6 + adverse_control * 0.4
    
    def _calculate_persistence_score(self, perf: SignalPerformance) -> float:
        """计算持续性评分"""
        # 统计信号在各时间段的有效性
        valid_periods = 0
        total_periods = 0
        
        expected_range = self.config['expected_move_ranges'].get(perf.expected_move, (1, 5))
        
        for interval in ['5m', '15m', '30m', '1h', '2h', '4h', '8h', '1d']:
            ret = getattr(perf, f'return_{interval}')
            if ret is not None:
                total_periods += 1
                # 检查是否在预期范围内且方向正确
                if perf.direction == 'BULLISH' and ret > expected_range[0]:
                    valid_periods += 1
                elif perf.direction == 'BEARISH' and ret < -expected_range[0]:
                    valid_periods += 1
        
        return valid_periods / total_periods if total_periods > 0 else 0
    
    def _calculate_robustness_score(self, perf: SignalPerformance) -> float:
        """计算稳健性评分"""
        if perf.path_metrics is None:
            return 0.0
        
        # 评估价格路径的稳定性
        volatility_score = 1 - min(perf.path_metrics.path_volatility / 10, 1)
        
        # 评估回撤控制
        drawdown_score = 1 - min(abs(perf.path_metrics.avg_drawdown) / 5, 1)
        
        # 评估风险调整收益
        sharpe_score = min(max(perf.path_metrics.sharpe_ratio, 0) / 2, 1)
        
        return np.mean([volatility_score, drawdown_score, sharpe_score])
    
    async def _capture_market_snapshot(self, asset: str, 
                                     gamma_analysis: Dict[str, Any],
                                     market_behavior: Dict[str, Any]) -> Optional[MarketSnapshot]:
        """捕获市场快照"""
        try:
            # 获取当前市场数据
            if self.market_data_fetcher:
                market_data = await self.market_data_fetcher(asset)
            else:
                market_data = self._extract_market_data_from_analysis(asset, gamma_analysis, market_behavior)
            
            if not market_data:
                return None
            
            # 构建市场快照
            snapshot = MarketSnapshot(
                asset=asset,
                timestamp=datetime.utcnow(),
                price=market_data.get('price', 0),
                bid=market_data.get('bid', 0),
                ask=market_data.get('ask', 0),
                spread=market_data.get('ask', 0) - market_data.get('bid', 0),
                orderbook_depth=market_data.get('orderbook_depth', {}),
                orderbook_imbalance=market_data.get('orderbook_imbalance', 0),
                recent_trades=market_data.get('recent_trades', []),
                gamma_distribution=self._extract_gamma_distribution(asset, gamma_analysis),
                iv_surface=market_data.get('iv_surface', {}),
                put_call_skew=market_data.get('put_call_skew', 0),
                nearest_gamma_wall=self._extract_nearest_gamma_wall(asset, gamma_analysis),
                trend_strength=market_data.get('trend_strength', 0),
                support_levels=market_data.get('support_levels', []),
                resistance_levels=market_data.get('resistance_levels', []),
                momentum_indicators=market_data.get('momentum_indicators', {}),
                volatility_regime=market_data.get('volatility_regime', 'normal'),
                liquidity_score=market_data.get('liquidity_score', 0.5),
                market_regime=market_behavior.get('market_regime', {}).get('state', 'normal')
            )
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error capturing market snapshot for {asset}: {e}")
            return None
    
    def _extract_market_data_from_analysis(self, asset: str, 
                                         gamma_analysis: Dict[str, Any],
                                         market_behavior: Dict[str, Any]) -> Dict[str, Any]:
        """从分析结果中提取市场数据"""
        # 这是一个后备方法，当没有专门的市场数据获取器时使用
        market_data = {
            'price': 0,
            'bid': 0,
            'ask': 0,
            'orderbook_depth': {},
            'orderbook_imbalance': 0,
            'recent_trades': [],
            'iv_surface': {},
            'put_call_skew': 0,
            'trend_strength': 0,
            'support_levels': [],
            'resistance_levels': [],
            'momentum_indicators': {},
            'volatility_regime': 'normal',
            'liquidity_score': 0.5
        }
        
        # 从gamma_analysis提取
        if gamma_analysis and 'raw_data' in gamma_analysis:
            raw_data = gamma_analysis['raw_data'].get(asset, {})
            if 'spot_snapshot' in raw_data and raw_data['spot_snapshot']:
                latest_spot = raw_data['spot_snapshot'][-1]
                market_data['price'] = latest_spot.get('price', 0)
                market_data['bid'] = latest_spot.get('bid', 0)
                market_data['ask'] = latest_spot.get('ask', 0)
        
        return market_data
    
    def _extract_gamma_distribution(self, asset: str, gamma_analysis: Dict[str, Any]) -> Dict[str, float]:
        """提取gamma分布"""
        gamma_dist = {}
        
        if gamma_analysis and 'gamma_distribution' in gamma_analysis:
            asset_gamma = gamma_analysis['gamma_distribution'].get(asset, {})
            if 'profile' in asset_gamma:
                for item in asset_gamma['profile']:
                    strike = item.get('strike')
                    if strike:
                        gamma_dist[str(strike)] = item.get('gamma_exposure', 0)
        
        return gamma_dist
    
    def _extract_nearest_gamma_wall(self, asset: str, gamma_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """提取最近的gamma墙"""
        if not gamma_analysis or 'gamma_walls' not in gamma_analysis:
            return {}
        
        walls = gamma_analysis['gamma_walls']
        # 简化处理：返回第一个墙的信息
        if walls and hasattr(walls[0], 'strike'):
            wall = walls[0]
            return {
                'strike': wall.strike,
                'gamma_exposure': wall.gamma_exposure,
                'distance_pct': wall.distance_pct,
                'position': wall.position
            }
        
        return {}
    
    def _extract_gamma_metrics(self, gamma_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """提取gamma指标"""
        metrics = {}
        
        if gamma_analysis:
            # 提取总体gamma指标
            for asset, dist in gamma_analysis.get('gamma_distribution', {}).items():
                metrics[f'{asset}_total_gamma'] = dist.get('total_exposure', 0)
                metrics[f'{asset}_net_gamma'] = dist.get('net_exposure', 0)
                metrics[f'{asset}_concentration'] = dist.get('concentration', 0)
            
            # 提取gamma墙数量
            metrics['total_gamma_walls'] = len(gamma_analysis.get('gamma_walls', []))
        
        return metrics
    
    def _extract_behavior_metrics(self, market_behavior: Dict[str, Any]) -> Dict[str, Any]:
        """提取市场行为指标"""
        metrics = {}
        
        if market_behavior:
            # 扫单统计
            metrics['sweep_count'] = len(market_behavior.get('sweep_orders', []))
            metrics['divergence_count'] = len(market_behavior.get('divergences', []))
            metrics['cross_market_signals'] = len(market_behavior.get('cross_market_signals', []))
            
            # 市场状态
            regime = market_behavior.get('market_regime', {})
            metrics['market_state'] = regime.get('state', 'normal')
            metrics['regime_confidence'] = regime.get('confidence', 0)
        
        return metrics
    
    async def _generate_counterfactual_data(self, assets: List[str],
                                          snapshots: Dict[str, MarketSnapshot],
                                          scores: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """生成反事实分析数据"""
        counterfactual = {}
        
        # 对每个资产评估"如果发出信号"的潜在结果
        for asset in assets:
            if asset not in snapshots or asset not in scores:
                continue
            
            snapshot = snapshots[asset]
            asset_scores = scores[asset]
            
            # 评估潜在信号强度
            potential_strength = np.mean(list(asset_scores.values()))
            
            # 基于市场状态评估潜在收益
            if snapshot.market_regime == 'squeeze':
                potential_move = potential_strength * 0.03  # 3%基准
            elif snapshot.market_regime == 'breakout':
                potential_move = potential_strength * 0.05  # 5%基准
            else:
                potential_move = potential_strength * 0.02  # 2%基准
            
            counterfactual[asset] = {
                'potential_signal_strength': potential_strength,
                'expected_move_percent': potential_move * 100,
                'market_favorability': self._assess_market_favorability(snapshot),
                'missed_opportunity_score': self._calculate_missed_opportunity_score(
                    potential_strength, snapshot
                )
            }
        
        return counterfactual
    
    def _assess_market_favorability(self, snapshot: MarketSnapshot) -> float:
        """评估市场有利程度"""
        score = 0.5  # 基准分
        
        # 流动性好加分
        score += snapshot.liquidity_score * 0.2
        
        # 趋势强度加分
        score += abs(snapshot.trend_strength) * 0.2
        
        # 低波动率环境加分（适合突破）
        if snapshot.volatility_regime == 'low':
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_missed_opportunity_score(self, potential_strength: float,
                                          snapshot: MarketSnapshot) -> float:
        """计算错失机会分数"""
        # 基于信号强度和市场有利程度
        favorability = self._assess_market_favorability(snapshot)
        
        # 只有当信号强度和市场条件都好时，才算真正的错失机会
        if potential_strength > 70 and favorability > 0.7:
            return (potential_strength / 100) * favorability
        
        return 0.0
    
    def _get_latest_market_snapshot(self, asset: str) -> Optional[MarketSnapshot]:
        """获取最新的市场快照"""
        if asset in self.market_snapshots and self.market_snapshots[asset]:
            return self.market_snapshots[asset][-1]
        return None
    
    def _save_signal(self, performance: SignalPerformance):
        """保存信号到CSV"""
        df = pd.read_csv(self.signal_db_path)
        
        # 转换为字典
        data = {
            'signal_id': performance.signal_id,
            'signal_timestamp': performance.signal_timestamp,
            'asset': performance.asset,
            'signal_type': performance.signal_type,
            'direction': performance.direction,
            'initial_price': performance.initial_price,
            'strength': performance.strength,
            'confidence': performance.confidence,
            'expected_move': performance.expected_move,
            'time_horizon': performance.time_horizon
        }
        
        # 添加价格和收益数据
        for interval in ['5m', '15m', '30m', '1h', '2h', '4h', '8h', '1d']:
            data[f'price_{interval}'] = getattr(performance, f'price_{interval}')
            data[f'return_{interval}'] = getattr(performance, f'return_{interval}')
        
        # 添加评分数据
        data['direction_score'] = performance.direction_score
        data['timing_score'] = performance.timing_score
        data['persistence_score'] = performance.persistence_score
        data['robustness_score'] = performance.robustness_score
        
        # 添加路径指标
        if performance.path_metrics:
            data['max_favorable_move'] = performance.path_metrics.max_favorable_move
            data['max_adverse_move'] = performance.path_metrics.max_adverse_move
            data['path_volatility'] = performance.path_metrics.path_volatility
            data['momentum_score'] = performance.path_metrics.momentum_score
            data['sharpe_ratio'] = performance.path_metrics.sharpe_ratio
        
        data['metadata'] = json.dumps(performance.metadata)
        data['evaluation_complete'] = performance.evaluation_complete
        data['evaluation_timestamp'] = performance.evaluation_timestamp
        
        # 更新或添加记录
        mask = df['signal_id'] == performance.signal_id
        if mask.any():
            for key, value in data.items():
                df.loc[mask, key] = value
        else:
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
        
        df.to_csv(self.signal_db_path, index=False)
    
    def _save_decision(self, decision: DecisionSnapshot):
        """保存决策记录"""
        try:
            df = pd.read_csv(self.decision_db_path)
            
            # 转换为可序列化格式
            data = {
                'timestamp': decision.timestamp,
                'decision_type': decision.decision_type,
                'assets_analyzed': json.dumps(decision.assets_analyzed),
                'signal_generated': decision.signal_generated.signal_id if decision.signal_generated else None,
                'suppression_reason': decision.suppression_reason,
                'gamma_metrics': json.dumps(decision.gamma_metrics),
                'behavior_metrics': json.dumps(decision.behavior_metrics),
                'scores': json.dumps(decision.scores),
                'market_snapshot': json.dumps({
                    asset: {
                        'price': snapshot.price,
                        'spread': snapshot.spread,
                        'liquidity_score': snapshot.liquidity_score,
                        'market_regime': snapshot.market_regime
                    } for asset, snapshot in decision.market_snapshot.items()
                }),
                'counterfactual_data': json.dumps(decision.counterfactual_data)
            }
            
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
            df.to_csv(self.decision_db_path, index=False)
            
        except Exception as e:
            logger.error(f"Error saving decision: {e}")
    
    def _print_performance_summary(self, performance: SignalPerformance):
        """打印表现总结"""
        returns = []
        for interval in ['5m', '15m', '30m', '1h', '2h', '4h', '8h', '1d']:
            ret = getattr(performance, f'return_{interval}')
            if ret is not None:
                returns.append(f"{interval}: {ret:+.2f}%")
        
        summary = f"""
=== Enhanced Signal Performance Summary ===
Signal ID: {performance.signal_id}
Asset: {performance.asset}
Direction: {performance.direction}

Performance Scores:
- Direction Score: {performance.direction_score:.2f} (-1 to 1)
- Timing Score: {performance.timing_score:.2f} (0 to 1)
- Persistence Score: {performance.persistence_score:.2f} (0 to 1)
- Robustness Score: {performance.robustness_score:.2f} (0 to 1)

Returns Timeline:
{' | '.join(returns)}

Path Metrics:
- Max Favorable: {performance.path_metrics.max_favorable_move:.2f}%
- Max Adverse: {performance.path_metrics.max_adverse_move:.2f}%
- Sharpe Ratio: {performance.path_metrics.sharpe_ratio:.2f}
==========================================
"""
        print(summary)
    
    def get_performance_stats(self, lookback_days: int = 7) -> Dict[str, Any]:
        """获取性能统计（增强版）"""
        df = pd.read_csv(self.signal_db_path)
        
        # 筛选时间范围
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        df['signal_timestamp'] = pd.to_datetime(df['signal_timestamp'])
        df = df[df['signal_timestamp'] > cutoff]
        
        # 只统计已完成的信号
        df = df[df['evaluation_complete'] == True]
        
        if df.empty:
            return {}
        
        # 基础统计
        stats = {
            'total_signals': len(df),
            'avg_direction_score': df['direction_score'].mean(),
            'avg_timing_score': df['timing_score'].mean(),
            'avg_persistence_score': df['persistence_score'].mean(),
            'avg_robustness_score': df['robustness_score'].mean(),
            'composite_score': df[['direction_score', 'timing_score', 
                                  'persistence_score', 'robustness_score']].mean().mean(),
            'by_signal_type': {},
            'by_asset': {},
            'by_timeframe': {},
            'failure_patterns': self._identify_failure_patterns(df),
            'success_patterns': self._identify_success_patterns(df),
            'missed_opportunities': self._analyze_missed_opportunities()
        }
        
        # 按信号类型统计
        for signal_type in df['signal_type'].unique():
            type_df = df[df['signal_type'] == signal_type]
            stats['by_signal_type'][signal_type] = {
                'count': len(type_df),
                'avg_direction_score': type_df['direction_score'].mean(),
                'avg_composite_score': type_df[['direction_score', 'timing_score', 
                                              'persistence_score', 'robustness_score']].mean().mean()
            }
        
        # 按资产统计
        for asset in df['asset'].unique():
            asset_df = df[df['asset'] == asset]
            stats['by_asset'][asset] = {
                'count': len(asset_df),
                'avg_direction_score': asset_df['direction_score'].mean(),
                'best_timeframe': self._find_best_timeframe(asset_df)
            }
        
        # 按时间框架统计表现
        for timeframe in ['5m', '15m', '30m', '1h', '2h', '4h', '8h', '1d']:
            returns = df[f'return_{timeframe}'].dropna()
            if len(returns) > 0:
                stats['by_timeframe'][timeframe] = {
                    'avg_return': returns.mean(),
                    'hit_rate': len(returns[returns > 0]) / len(returns),
                    'sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0
                }
        
        return stats
    
    def _find_best_timeframe(self, df: pd.DataFrame) -> str:
        """找出表现最好的时间框架"""
        best_timeframe = None
        best_score = -float('inf')
        
        for timeframe in ['5m', '15m', '30m', '1h', '2h', '4h', '8h', '1d']:
            returns = df[f'return_{timeframe}'].dropna()
            if len(returns) > 0:
                # 综合考虑平均收益和稳定性
                avg_return = returns.mean()
                sharpe = avg_return / returns.std() if returns.std() > 0 else 0
                score = avg_return + sharpe * 10  # 权重调整
                
                if score > best_score:
                    best_score = score
                    best_timeframe = timeframe
        
        return best_timeframe or 'N/A'
    
    def _identify_failure_patterns(self, df: pd.DataFrame) -> Dict[str, int]:
        """识别失败模式（增强版）"""
        patterns = defaultdict(int)
        
        # 方向完全错误的信号
        wrong_direction = df[df['direction_score'] < -0.5]
        patterns['wrong_direction'] = len(wrong_direction)
        
        # 时机太晚的信号
        late_timing = df[df['timing_score'] < 0.3]
        patterns['late_timing'] = len(late_timing)
        
        # 缺乏持续性的信号
        no_persistence = df[df['persistence_score'] < 0.3]
        patterns['no_persistence'] = len(no_persistence)
        
        # 不稳健的信号
        not_robust = df[df['robustness_score'] < 0.3]
        patterns['not_robust'] = len(not_robust)
        
        # 分析元数据中的失败原因
        for _, signal in df.iterrows():
            metadata = json.loads(signal['metadata']) if pd.notna(signal['metadata']) else {}
            
            # 低流动性
            if metadata.get('market_metrics', {}).get('sweep_count', 0) < 2:
                patterns['low_liquidity'] += 1
            
            # 市场状态不利
            if metadata.get('market_metrics', {}).get('anomaly_score', 0) < 0.3:
                patterns['unfavorable_market'] += 1
        
        return dict(patterns)
    
    def _identify_success_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """识别成功模式"""
        patterns = {}
        
        # 找出表现最好的信号
        high_performers = df[df['direction_score'] > 0.7]
        
        if len(high_performers) > 0:
            # 分析成功信号的共同特征
            patterns['high_performer_characteristics'] = {
                'avg_strength': high_performers['strength'].mean(),
                'avg_confidence': high_performers['confidence'].mean(),
                'common_signal_type': high_performers['signal_type'].mode()[0] if len(high_performers['signal_type'].mode()) > 0 else 'N/A',
                'common_timeframe': self._find_best_timeframe(high_performers)
            }
            
            # 分析成功的市场条件
            successful_metadata = []
            for _, signal in high_performers.iterrows():
                metadata = json.loads(signal['metadata']) if pd.notna(signal['metadata']) else {}
                successful_metadata.append(metadata)
            
            if successful_metadata:
                patterns['favorable_conditions'] = {
                    'avg_sweep_count': np.mean([m.get('market_metrics', {}).get('sweep_count', 0) 
                                               for m in successful_metadata]),
                    'avg_anomaly_score': np.mean([m.get('market_metrics', {}).get('anomaly_score', 0) 
                                                 for m in successful_metadata])
                }
        
        return patterns
    
    def _analyze_missed_opportunities(self) -> Dict[str, Any]:
        """分析错失的机会"""
        try:
            # 读取决策历史
            decision_df = pd.read_csv(self.decision_db_path)
            
            # 筛选没有生成信号的决策
            no_signal_decisions = decision_df[decision_df['decision_type'] == 'no_signal']
            
            missed_opportunities = []
            
            for _, decision in no_signal_decisions.iterrows():
                counterfactual = json.loads(decision['counterfactual_data']) if pd.notna(decision['counterfactual_data']) else {}
                
                # 找出高分但被抑制的机会
                for asset, data in counterfactual.items():
                    if data.get('missed_opportunity_score', 0) > 0.7:
                        missed_opportunities.append({
                            'timestamp': decision['timestamp'],
                            'asset': asset,
                            'potential_strength': data['potential_signal_strength'],
                            'expected_move': data['expected_move_percent'],
                            'opportunity_score': data['missed_opportunity_score']
                        })
            
            # 汇总分析
            if missed_opportunities:
                return {
                    'total_missed': len(missed_opportunities),
                    'avg_opportunity_score': np.mean([m['opportunity_score'] for m in missed_opportunities]),
                    'top_missed_opportunities': sorted(missed_opportunities, 
                                                     key=lambda x: x['opportunity_score'], 
                                                     reverse=True)[:5]
                }
            
        except Exception as e:
            logger.error(f"Error analyzing missed opportunities: {e}")
        
        return {'total_missed': 0}
    
    def set_price_fetcher(self, fetcher: callable):
        """设置价格获取器"""
        self.price_fetcher = fetcher
    
    def set_market_data_fetcher(self, fetcher: callable):
        """设置市场数据获取器"""
        self.market_data_fetcher = fetcher
    
    async def periodic_update(self):
        """定期更新任务"""
        while True:
            try:
                await self.update_prices()
                await asyncio.sleep(self.config['update_interval'])
            except Exception as e:
                logger.error(f"Error in periodic update: {e}")
                await asyncio.sleep(60)