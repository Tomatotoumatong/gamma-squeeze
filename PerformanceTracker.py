"""
PerformanceTracker - 信号表现跟踪模块
用于gamma squeeze信号捕捉系统的学习进化层
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import json
import os
from collections import defaultdict
import asyncio
from SignalEvaluator import TradingSignal

logger = logging.getLogger(__name__)

@dataclass
class SignalPerformance:
    """信号表现记录"""
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
    
    # 实际表现
    price_2h: Optional[float] = None
    price_4h: Optional[float] = None
    price_6h: Optional[float] = None
    price_12h: Optional[float] = None
    
    # 计算指标
    actual_move_2h: Optional[float] = None
    actual_move_4h: Optional[float] = None
    actual_move_6h: Optional[float] = None
    actual_move_12h: Optional[float] = None
    
    direction_hit: Optional[bool] = None
    magnitude_accuracy: Optional[float] = None
    timing_accuracy: Optional[float] = None
    
    # 元数据
    metadata: Dict[str, Any] = None
    evaluation_complete: bool = False
    evaluation_timestamp: Optional[datetime] = None

class PerformanceTracker:
    """信号表现跟踪器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.db_path = self.config['db_path']
        self.active_signals: Dict[str, SignalPerformance] = {}
        self.price_fetcher = None  # 价格获取器，需要注入
        self._ensure_db_exists()
        self._load_active_signals()
        
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'db_path': 'signal_performance.csv',
            'check_intervals': [2, 4, 6, 12],  # 小时
            'expected_move_ranges': {
                "1-2%": (1, 2),
                "2-5%": (2, 5),
                "5-10%": (5, 10),
                "10%+": (10, 20)
            },
            'time_horizon_hours': {
                "0-2h": 2,
                "2-4h": 4,
                "4-8h": 6,
                "8-24h": 12
            }
        }
        
    def _ensure_db_exists(self):
        """确保数据库文件存在"""
        if not os.path.exists(self.db_path):
            # 创建空的CSV文件
            df = pd.DataFrame(columns=[
                'signal_id', 'signal_timestamp', 'asset', 'signal_type',
                'direction', 'initial_price', 'strength', 'confidence',
                'expected_move', 'time_horizon', 'price_2h', 'price_4h',
                'price_6h', 'price_12h', 'actual_move_2h', 'actual_move_4h',
                'actual_move_6h', 'actual_move_12h', 'direction_hit',
                'magnitude_accuracy', 'timing_accuracy', 'metadata',
                'evaluation_complete', 'evaluation_timestamp'
            ])
            df.to_csv(self.db_path, index=False)
            
    def _load_active_signals(self):
        """加载未完成评估的信号"""
        try:
            df = pd.read_csv(self.db_path)
            
            # 筛选未完成的信号
            active_df = df[df['evaluation_complete'] == False]
            
            for _, row in active_df.iterrows():
                # 重建SignalPerformance对象
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
                    metadata=json.loads(row['metadata']) if pd.notna(row['metadata']) else {}
                )
                
                # 恢复已有的价格数据
                for interval in self.config['check_intervals']:
                    price_key = f'price_{interval}h'
                    if pd.notna(row[price_key]):
                        setattr(perf, price_key, row[price_key])
                        
                self.active_signals[perf.signal_id] = perf
                
        except Exception as e:
            logger.error(f"Error loading active signals: {e}")
            
    def track_signal(self, signal: 'TradingSignal', initial_price: float):
        """开始跟踪新信号"""
        # 生成唯一ID
        signal_id = f"{signal.asset}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # 创建表现记录
        performance = SignalPerformance(
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
            metadata=signal.metadata
        )
        
        # 添加到活跃信号
        self.active_signals[signal_id] = performance
        
        # 保存到数据库
        self._save_signal(performance)
        
        logger.info(f"Started tracking signal {signal_id}")
        
    async def update_prices(self):
        """更新所有活跃信号的价格"""
        if not self.price_fetcher:
            logger.error("Price fetcher not set")
            return
            
        current_time = datetime.utcnow()
        
        for signal_id, performance in self.active_signals.items():
            try:
                # 获取当前价格
                current_price = await self.price_fetcher(performance.asset)
                
                if current_price is None:
                    continue
                    
                # 计算经过的时间
                elapsed_hours = (current_time - performance.signal_timestamp).total_seconds() / 3600
                
                # 更新对应时间点的价格
                for interval in self.config['check_intervals']:
                    price_key = f'price_{interval}h'
                    move_key = f'actual_move_{interval}h'
                    
                    # 如果已经过了这个时间点且还没记录
                    if elapsed_hours >= interval and getattr(performance, price_key) is None:
                        setattr(performance, price_key, current_price)
                        
                        # 计算实际涨跌幅
                        actual_move = ((current_price - performance.initial_price) / 
                                     performance.initial_price * 100)
                        setattr(performance, move_key, actual_move)
                        
                        logger.info(f"Updated {interval}h price for {signal_id}: {current_price} ({actual_move:.2f}%)")
                        
                # 检查是否所有时间点都已记录
                if all(getattr(performance, f'price_{i}h') is not None 
                      for i in self.config['check_intervals']):
                    # 计算最终指标
                    self._evaluate_performance(performance)
                    
            except Exception as e:
                logger.error(f"Error updating prices for {signal_id}: {e}")
                
    def _evaluate_performance(self, performance: SignalPerformance):
        """评估信号表现"""
        # 1. 方向准确性
        direction_correct = False
        if performance.direction == 'BULLISH':
            direction_correct = any(
                getattr(performance, f'actual_move_{i}h', 0) > 0 
                for i in self.config['check_intervals']
            )
        else:  # BEARISH
            direction_correct = any(
                getattr(performance, f'actual_move_{i}h', 0) < 0 
                for i in self.config['check_intervals']
            )
        performance.direction_hit = direction_correct
        
        # 2. 幅度准确性
        expected_range = self.config['expected_move_ranges'].get(
            performance.expected_move, (0, 100)
        )
        
        # 找到最接近预期的实际波动
        actual_moves = [
            abs(getattr(performance, f'actual_move_{i}h', 0))
            for i in self.config['check_intervals']
        ]
        max_move = max(actual_moves)
        
        # 计算与预期范围的匹配度
        if max_move < expected_range[0]:
            magnitude_accuracy = max_move / expected_range[0]
        elif max_move > expected_range[1]:
            magnitude_accuracy = expected_range[1] / max_move
        else:
            magnitude_accuracy = 1.0
            
        performance.magnitude_accuracy = magnitude_accuracy
        
        # 3. 时间准确性
        expected_hours = self.config['time_horizon_hours'].get(
            performance.time_horizon, 6
        )
        
        # 找到达到显著波动的时间
        significant_move_time = None
        for i in self.config['check_intervals']:
            if abs(getattr(performance, f'actual_move_{i}h', 0)) >= expected_range[0]:
                significant_move_time = i
                break
                
        if significant_move_time:
            timing_accuracy = min(expected_hours, significant_move_time) / max(expected_hours, significant_move_time)
        else:
            timing_accuracy = 0.0
            
        performance.timing_accuracy = timing_accuracy
        
        # 标记评估完成
        performance.evaluation_complete = True
        performance.evaluation_timestamp = datetime.utcnow()
        
        # 保存结果
        self._save_signal(performance)
        
        # 从活跃信号中移除
        del self.active_signals[performance.signal_id]
        
        # 输出总结
        self._print_performance_summary(performance)
        
    def _save_signal(self, performance: SignalPerformance):
        """保存信号到CSV"""
        # 读取现有数据
        df = pd.read_csv(self.db_path)
        
        # 转换为字典
        data = asdict(performance)
        data['metadata'] = json.dumps(data['metadata']) if data['metadata'] else ''
        
        # 检查是否已存在
        mask = df['signal_id'] == performance.signal_id
        if mask.any():
            # 更新现有记录
            for key, value in data.items():
                df.loc[mask, key] = value
        else:
            # 添加新记录
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
            
        # 保存
        df.to_csv(self.db_path, index=False)
        
    def _print_performance_summary(self, performance: SignalPerformance):
        """打印表现总结"""
        summary = f"""
=== Signal Performance Summary ===
Signal ID: {performance.signal_id}
Asset: {performance.asset}
Direction: {performance.direction} (Hit: {'✓' if performance.direction_hit else '✗'})
Expected Move: {performance.expected_move} | Actual Max: {max(abs(getattr(performance, f'actual_move_{i}h', 0)) for i in self.config['check_intervals']):.2f}%
Magnitude Accuracy: {performance.magnitude_accuracy:.2%}
Timing Accuracy: {performance.timing_accuracy:.2%}

Price Trajectory:
Initial: ${performance.initial_price:.2f}
2h:  ${performance.price_2h:.2f} ({performance.actual_move_2h:+.2f}%)
4h:  ${performance.price_4h:.2f} ({performance.actual_move_4h:+.2f}%)
6h:  ${performance.price_6h:.2f} ({performance.actual_move_6h:+.2f}%)
12h: ${performance.price_12h:.2f} ({performance.actual_move_12h:+.2f}%)
================================
"""
        print(summary)
        
    def get_performance_stats(self, lookback_days: int = 7) -> Dict[str, Any]:
        """获取性能统计"""
        df = pd.read_csv(self.db_path)
        
        # 筛选时间范围
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        df['signal_timestamp'] = pd.to_datetime(df['signal_timestamp'])
        df = df[df['signal_timestamp'] > cutoff]
        
        # 只统计已完成的信号
        df = df[df['evaluation_complete'] == True]
        
        if df.empty:
            return {}
            
        stats = {
            'total_signals': len(df),
            'direction_accuracy': df['direction_hit'].mean(),
            'avg_magnitude_accuracy': df['magnitude_accuracy'].mean(),
            'avg_timing_accuracy': df['timing_accuracy'].mean(),
            'by_signal_type': {},
            'by_asset': {},
            'failure_patterns': self._identify_failure_patterns(df)
        }
        
        # 按信号类型统计
        for signal_type in df['signal_type'].unique():
            type_df = df[df['signal_type'] == signal_type]
            stats['by_signal_type'][signal_type] = {
                'count': len(type_df),
                'direction_accuracy': type_df['direction_hit'].mean(),
                'magnitude_accuracy': type_df['magnitude_accuracy'].mean()
            }
            
        # 按资产统计
        for asset in df['asset'].unique():
            asset_df = df[df['asset'] == asset]
            stats['by_asset'][asset] = {
                'count': len(asset_df),
                'direction_accuracy': asset_df['direction_hit'].mean(),
                'magnitude_accuracy': asset_df['magnitude_accuracy'].mean()
            }
            
        return stats
        
    def _identify_failure_patterns(self, df: pd.DataFrame) -> Dict[str, int]:
        """识别失败模式"""
        patterns = defaultdict(int)
        
        # 失败的信号
        failed = df[df['direction_hit'] == False]
        
        for _, signal in failed.iterrows():
            # 分析失败原因
            metadata = json.loads(signal['metadata']) if pd.notna(signal['metadata']) else {}
            
            # 低流动性
            if metadata.get('market_metrics', {}).get('sweep_count', 0) < 2:
                patterns['low_liquidity'] += 1
                
            # 高置信度但失败
            if signal['confidence'] > 0.7:
                patterns['overconfidence'] += 1
                
            # 时间窗口错误
            if signal['timing_accuracy'] < 0.3:
                patterns['timing_error'] += 1
                
            # 幅度预测错误
            if signal['magnitude_accuracy'] < 0.3:
                patterns['magnitude_error'] += 1
                
        return dict(patterns)
        
    def set_price_fetcher(self, fetcher: callable):
        """设置价格获取器"""
        self.price_fetcher = fetcher
        
    async def periodic_update(self):
        """定期更新任务"""
        while True:
            try:
                await self.update_prices()
                await asyncio.sleep(300)  # 5分钟更新一次
            except Exception as e:
                logger.error(f"Error in periodic update: {e}")
                await asyncio.sleep(60)