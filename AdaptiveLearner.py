"""
AdaptiveLearner - 自适应学习模块
用于gamma squeeze信号捕捉系统的学习进化层
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json
import os
from collections import defaultdict
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """优化结果"""
    timestamp: datetime
    parameter_type: str  # 'threshold', 'weight', 'rule'
    parameter_name: str
    old_value: Any
    new_value: Any
    reason: str
    expected_improvement: float
    confidence: float

@dataclass
class FailurePattern:
    """失败模式"""
    pattern_id: str
    pattern_type: str
    description: str
    frequency: int
    conditions: Dict[str, Any]
    suggested_actions: List[str]

class AdaptiveLearner:
    """自适应学习器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.optimization_history = []
        self.failure_patterns = {}
        self.parameter_performance = defaultdict(list)
        self._load_optimization_history()
        
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'optimization_db': 'optimization_history.json',
            'failure_patterns_db': 'failure_patterns.json',
            'learning_rate': 0.1,  # 参数调整幅度
            'min_samples': 10,  # 最小样本数
            'confidence_threshold': 0.7,  # 优化置信度阈值
            'parameter_bounds': {
                'thresholds': {
                    'gamma_pressure.critical': (70, 90),
                    'gamma_pressure.high': (50, 70),
                    'gamma_pressure.medium': (30, 50),
                    'signal_generation.min_strength': (40, 60),
                    'signal_generation.min_confidence': (0.4, 0.7),
                    # 新增市场行为检测参数
                    'market_behavior.order_flow.sweep_threshold': (1.5, 4.0),
                    'market_behavior.order_flow.volume_multiplier': (1.5, 4.0),
                    'market_behavior.order_flow.frequency_window': (30, 120),
                    'market_behavior.divergence.min_duration': (2, 10),
                    'market_behavior.divergence.lookback_period': (10, 50)
                },
                'weights': {
                    'gamma_pressure.wall_proximity_weight': (0.1, 0.5),
                    'gamma_pressure.hedge_flow_weight': (0.1, 0.5),
                    'market_momentum.sweep_weight': (0.2, 0.6),
                    'market_momentum.divergence_weight': (0.1, 0.5)
                },
                # 新增市场状态阈值
                'regime_thresholds': {
                    'market_regime.anomaly_threshold': (0.5, 0.85),
                    'market_regime.volatility_percentile': (0.2, 0.4),
                    'market_regime.volume_percentile': (0.6, 0.8)
                }
            },
            'failure_thresholds': {
                'low_liquidity': 0.2,  # 20%失败率触发
                'overconfidence': 0.3,
                'timing_error': 0.25,
                'magnitude_error': 0.25
            }
        }
        
    def _load_optimization_history(self):
        """加载优化历史"""
        if os.path.exists(self.config['optimization_db']):
            with open(self.config['optimization_db'], 'r') as f:
                data = json.load(f)
                self.optimization_history = [
                    OptimizationResult(**item) for item in data
                ]
                
        if os.path.exists(self.config['failure_patterns_db']):
            with open(self.config['failure_patterns_db'], 'r') as f:
                patterns = json.load(f)
                self.failure_patterns = {
                    k: FailurePattern(**v) for k, v in patterns.items()
                }
                
    def learn_from_performance(self, performance_stats: Dict[str, Any], 
                             current_config: Dict[str, Any]) -> Dict[str, Any]:
        """从性能数据学习并优化配置"""
        optimizations = {}
        
        # 检查样本量
        if performance_stats.get('total_signals', 0) < self.config['min_samples']:
            logger.info(f"Insufficient samples for learning: {performance_stats.get('total_signals', 0)}")
            return optimizations
            
        # 1. 优化阈值
        threshold_opts = self._optimize_thresholds(performance_stats, current_config)
        optimizations.update(threshold_opts)
        
        # 2. 优化权重
        weight_opts = self._optimize_weights(performance_stats, current_config)
        optimizations.update(weight_opts)
        
        # 3. 识别并处理失败模式
        pattern_opts = self._handle_failure_patterns(performance_stats, current_config)
        optimizations.update(pattern_opts)
        
        # 记录优化历史
        self._save_optimizations(optimizations)
        
        return optimizations
        
    def _optimize_thresholds(self, stats: Dict, config: Dict) -> Dict[str, Any]:
        """优化阈值参数"""
        optimizations = {}
        
        # 分析方向准确率
        direction_accuracy = stats.get('direction_accuracy', 0.5)
        
        # 如果准确率太低，提高门槛
        if direction_accuracy < 0.45:
            # 提高最小信号强度
            param = 'signal_generation.min_strength'
            current = self._get_nested_value(config, param)
            bounds = self.config['parameter_bounds']['thresholds'][param]
            
            new_value = min(
                current * (1 + self.config['learning_rate']),
                bounds[1]
            )
            
            if new_value != current:
                optimizations[param] = new_value
                self._record_optimization(
                    OptimizationResult(
                        timestamp=datetime.utcnow(),
                        parameter_type='threshold',
                        parameter_name=param,
                        old_value=current,
                        new_value=new_value,
                        reason=f"Low direction accuracy: {direction_accuracy:.2%}",
                        expected_improvement=0.05,
                        confidence=0.8
                    )
                )
                
        # 如果准确率很高但信号太少，降低门槛
        elif direction_accuracy > 0.65 and stats['total_signals'] < 20:
            param = 'signal_generation.min_strength'
            current = self._get_nested_value(config, param)
            bounds = self.config['parameter_bounds']['thresholds'][param]
            
            new_value = max(
                current * (1 - self.config['learning_rate']),
                bounds[0]
            )
            
            if new_value != current:
                optimizations[param] = new_value
                self._record_optimization(
                    OptimizationResult(
                        timestamp=datetime.utcnow(),
                        parameter_type='threshold',
                        parameter_name=param,
                        old_value=current,
                        new_value=new_value,
                        reason=f"High accuracy but low signal count",
                        expected_improvement=0.1,
                        confidence=0.7
                    )
                )
                
        # 根据信号类型调整相应阈值
        for signal_type, type_stats in stats.get('by_signal_type', {}).items():
            if type_stats['count'] < 5:
                continue
                
            if signal_type == 'GAMMA_SQUEEZE' and type_stats['direction_accuracy'] < 0.4:
                # 调整gamma压力阈值
                param = 'gamma_pressure.high'
                current = self._get_nested_value(config, param)
                bounds = self.config['parameter_bounds']['thresholds'][param]
                
                new_value = min(
                    current * (1 + self.config['learning_rate'] * 0.5),
                    bounds[1]
                )
                
                if new_value != current:
                    optimizations[param] = new_value
                    self._record_optimization(
                        OptimizationResult(
                            timestamp=datetime.utcnow(),
                            parameter_type='threshold',
                            parameter_name=param,
                            old_value=current,
                            new_value=new_value,
                            reason=f"Poor GAMMA_SQUEEZE performance",
                            expected_improvement=0.05,
                            confidence=0.6
                        )
                    )
                    
        return optimizations
        
    def _optimize_weights(self, stats: Dict, config: Dict) -> Dict[str, Any]:
        """优化权重参数"""
        optimizations = {}
        
        # 分析各维度贡献
        by_type = stats.get('by_signal_type', {})
        
        # 计算各类型信号的相对表现
        type_performance = {}
        for signal_type, type_stats in by_type.items():
            if type_stats['count'] >= 3:
                # 综合评分 = 方向准确率 * 幅度准确率
                score = type_stats['direction_accuracy'] * type_stats['magnitude_accuracy']
                type_performance[signal_type] = score
                
        if not type_performance:
            return optimizations
            
        # 找出表现最好和最差的类型
        best_type = max(type_performance.items(), key=lambda x: x[1])
        worst_type = min(type_performance.items(), key=lambda x: x[1])
        
        # 如果差异显著，调整权重
        if best_type[1] - worst_type[1] > 0.2:
            # 提升表现好的维度权重
            if best_type[0] == 'MOMENTUM_BREAKOUT':
                param = 'market_momentum.sweep_weight'
                current = self._get_nested_value(config, param)
                bounds = self.config['parameter_bounds']['weights'][param]
                
                new_value = min(
                    current * (1 + self.config['learning_rate']),
                    bounds[1]
                )
                
                if new_value != current:
                    optimizations[param] = new_value
                    self._record_optimization(
                        OptimizationResult(
                            timestamp=datetime.utcnow(),
                            parameter_type='weight',
                            parameter_name=param,
                            old_value=current,
                            new_value=new_value,
                            reason=f"MOMENTUM signals performing well: {best_type[1]:.2f}",
                            expected_improvement=0.05,
                            confidence=0.75
                        )
                    )
                    
            # 降低表现差的维度权重
            if worst_type[0] == 'GAMMA_SQUEEZE' and worst_type[1] < 0.3:
                param = 'gamma_pressure.hedge_flow_weight'
                current = self._get_nested_value(config, param)
                bounds = self.config['parameter_bounds']['weights'][param]
                
                new_value = max(
                    current * (1 - self.config['learning_rate']),
                    bounds[0]
                )
                
                if new_value != current:
                    optimizations[param] = new_value
                    self._record_optimization(
                        OptimizationResult(
                            timestamp=datetime.utcnow(),
                            parameter_type='weight',
                            parameter_name=param,
                            old_value=current,
                            new_value=new_value,
                            reason=f"GAMMA signals underperforming: {worst_type[1]:.2f}",
                            expected_improvement=0.03,
                            confidence=0.6
                        )
                    )
                    
        return optimizations
        
    def _handle_failure_patterns(self, stats: Dict, config: Dict) -> Dict[str, Any]:
        """处理失败模式"""
        optimizations = {}
        failure_patterns = stats.get('failure_patterns', {})
        
        total_failures = sum(failure_patterns.values())
        if total_failures == 0:
            return optimizations
            
        # 分析每种失败模式
        for pattern_type, count in failure_patterns.items():
            failure_rate = count / stats.get('total_signals', 1)
            
            # 检查是否超过阈值
            threshold = self.config['failure_thresholds'].get(pattern_type, 0.2)
            
            if failure_rate > threshold:
                # 创建或更新失败模式记录
                pattern_id = f"{pattern_type}_{datetime.utcnow().strftime('%Y%m%d')}"
                
                if pattern_type == 'low_liquidity':
                    # 流动性误判 - 需要更严格的成交量过滤
                    pattern = FailurePattern(
                        pattern_id=pattern_id,
                        pattern_type=pattern_type,
                        description="Signals generated in low liquidity conditions",
                        frequency=count,
                        conditions={'sweep_count': '<2'},
                        suggested_actions=[
                            "Increase minimum sweep count requirement",
                            "Add volume anomaly threshold",
                            "Check time-of-day liquidity patterns"
                        ]
                    )
                    self.failure_patterns[pattern_id] = pattern
                    
                    # 自动调整：降低低流动性环境下的信号置信度
                    param = 'signal_generation.risk_assessment.low_liquidity_threshold'
                    current = self._get_nested_value(config, param, 0.3)
                    new_value = min(current * 1.2, 0.5)
                    optimizations[param] = new_value
                    
                elif pattern_type == 'overconfidence':
                    # 过度自信 - 高置信度但失败
                    pattern = FailurePattern(
                        pattern_id=pattern_id,
                        pattern_type=pattern_type,
                        description="High confidence signals that failed",
                        frequency=count,
                        conditions={'confidence': '>0.7', 'direction_hit': False},
                        suggested_actions=[
                            "Review confidence calculation logic",
                            "Add more conservative factors",
                            "Implement confidence decay over time"
                        ]
                    )
                    self.failure_patterns[pattern_id] = pattern
                    
                    # 自动调整：提高最小置信度要求
                    param = 'signal_generation.min_confidence'
                    current = self._get_nested_value(config, param)
                    bounds = self.config['parameter_bounds']['thresholds'][param]
                    new_value = min(current * 1.1, bounds[1])
                    optimizations[param] = new_value
                    
                elif pattern_type == 'timing_error':
                    # 时间预测错误
                    pattern = FailurePattern(
                        pattern_id=pattern_id,
                        pattern_type=pattern_type,
                        description="Poor timing predictions",
                        frequency=count,
                        conditions={'timing_accuracy': '<0.3'},
                        suggested_actions=[
                            "Adjust time horizon estimation",
                            "Consider market regime in timing",
                            "Add volatility-based adjustments"
                        ]
                    )
                    self.failure_patterns[pattern_id] = pattern
                    
                # 记录模式识别
                logger.warning(f"Failure pattern detected: {pattern_type} ({failure_rate:.1%} of signals)")
                
        # 保存失败模式
        self._save_failure_patterns()
        
        return optimizations
        
    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """获取优化建议"""
        suggestions = []
        
        # 基于历史优化效果
        recent_opts = self.optimization_history[-20:]
        
        # 分析哪些优化有效
        for opt in recent_opts:
            if opt.confidence > 0.7:
                suggestions.append({
                    'type': 'successful_optimization',
                    'parameter': opt.parameter_name,
                    'change': f"{opt.old_value} → {opt.new_value}",
                    'reason': opt.reason
                })
                
        # 基于失败模式
        for pattern in self.failure_patterns.values():
            if pattern.frequency > 5:
                suggestions.append({
                    'type': 'failure_pattern',
                    'pattern': pattern.pattern_type,
                    'description': pattern.description,
                    'actions': pattern.suggested_actions
                })
                
        return suggestions
        
    def _get_nested_value(self, config: Dict, path: str, default: Any = None) -> Any:
        """获取嵌套配置值"""
        keys = path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
        
    def _record_optimization(self, result: OptimizationResult):
        """记录优化结果"""
        self.optimization_history.append(result)
        
        # 保持历史长度
        if len(self.optimization_history) > 1000:
            self.optimization_history = self.optimization_history[-1000:]
            
    def _save_optimizations(self, optimizations: Dict[str, Any]):
        """保存优化历史"""
        # 转换为可序列化格式
        history_data = []
        for opt in self.optimization_history:
            item = {
                'timestamp': opt.timestamp.isoformat(),
                'parameter_type': opt.parameter_type,
                'parameter_name': opt.parameter_name,
                'old_value': opt.old_value,
                'new_value': opt.new_value,
                'reason': opt.reason,
                'expected_improvement': opt.expected_improvement,
                'confidence': opt.confidence
            }
            history_data.append(item)
            
        with open(self.config['optimization_db'], 'w') as f:
            json.dump(history_data, f, indent=2)
            
    def _save_failure_patterns(self):
        """保存失败模式"""
        patterns_data = {}
        
        for key, pattern in self.failure_patterns.items():
            patterns_data[key] = {
                'pattern_id': pattern.pattern_id,
                'pattern_type': pattern.pattern_type,
                'description': pattern.description,
                'frequency': pattern.frequency,
                'conditions': pattern.conditions,
                'suggested_actions': pattern.suggested_actions
            }
            
        with open(self.config['failure_patterns_db'], 'w') as f:
            json.dump(patterns_data, f, indent=2)
            
    def validate_optimization(self, param: str, old_value: Any, 
                            new_value: Any, validation_stats: Dict) -> float:
        """验证优化效果"""
        # 记录参数变化前后的性能
        self.parameter_performance[param].append({
            'timestamp': datetime.utcnow(),
            'value': new_value,
            'performance': validation_stats.get('direction_accuracy', 0.5)
        })
        
        # 计算改进幅度
        history = self.parameter_performance[param]
        if len(history) >= 2:
            old_perf = np.mean([h['performance'] for h in history[:-1]])
            new_perf = history[-1]['performance']
            improvement = (new_perf - old_perf) / old_perf if old_perf > 0 else 0
            return improvement
            
        return 0.0
        
    def get_learning_report(self) -> Dict[str, Any]:
        """生成学习报告"""
        report = {
            'total_optimizations': len(self.optimization_history),
            'recent_optimizations': [],
            'active_failure_patterns': [],
            'parameter_trends': {},
            'recommendations': []
        }
        
        # 最近的优化
        for opt in self.optimization_history[-10:]:
            report['recent_optimizations'].append({
                'parameter': opt.parameter_name,
                'change': f"{opt.old_value} → {opt.new_value}",
                'reason': opt.reason,
                'timestamp': opt.timestamp.isoformat()
            })
            
        # 活跃的失败模式
        for pattern in self.failure_patterns.values():
            if pattern.frequency > 3:
                report['active_failure_patterns'].append({
                    'type': pattern.pattern_type,
                    'frequency': pattern.frequency,
                    'actions': pattern.suggested_actions
                })
                
        # 参数趋势
        for param, history in self.parameter_performance.items():
            if len(history) > 3:
                values = [h['value'] for h in history]
                performances = [h['performance'] for h in history]
                
                # 计算趋势
                if len(values) > 1:
                    trend = np.polyfit(range(len(values)), performances, 1)[0]
                    report['parameter_trends'][param] = {
                        'trend': 'improving' if trend > 0 else 'declining',
                        'current_value': values[-1],
                        'avg_performance': np.mean(performances)
                    }
                    
        # 生成建议
        if report['parameter_trends']:
            # 找出表现最好的参数设置
            best_param = max(
                report['parameter_trends'].items(),
                key=lambda x: x[1]['avg_performance']
            )
            report['recommendations'].append(
                f"Parameter '{best_param[0]}' shows best performance at value {best_param[1]['current_value']}"
            )
            
        return report