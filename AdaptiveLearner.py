"""
AdaptiveLearner - 增强版自适应学习模块
专注于优化其他模块参数，而非重复实现功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import os
from collections import defaultdict, deque
from scipy import stats
import copy

logger = logging.getLogger(__name__)

@dataclass
class ParameterGradient:
    """参数梯度信息"""
    parameter: str
    gradient: float
    confidence: float
    sensitivity: float
    recent_impact: List[float] = field(default_factory=list)

@dataclass
class LearningDecision:
    """学习决策记录"""
    timestamp: datetime
    market_regime: str
    parameter_adjustments: Dict[str, float]
    expected_improvement: float
    exploration_factor: float
    decision_basis: str

class GradientAwareLearner:
    """梯度感知学习器 - 学习参数调整方向"""
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.gradient_history = defaultdict(lambda: deque(maxlen=100))
        self.parameter_performance = defaultdict(list)
        
    def estimate_parameter_gradients(self, 
                                   recent_decisions: List[Dict],
                                   performance_data: Dict) -> Dict[str, ParameterGradient]:
        """估计参数梯度"""
        gradients = {}
        
        # 关注的参数列表
        adjustable_params = [
            'market_behavior.order_flow.sweep_threshold',
            'market_behavior.divergence.min_duration',
            'market_behavior.divergence.lookback_period',
            'gamma_analysis.wall_percentile',
            'gamma_analysis.hedge_flow_threshold',
            'signal_generation.min_strength',
            'signal_generation.min_confidence',
        ]
        
        for param in adjustable_params:
            gradient_info = self._analyze_parameter_performance(
                param, recent_decisions, performance_data
            )
            if gradient_info:
                gradients[param] = gradient_info
                
        return gradients
    
    def _analyze_parameter_performance(self, 
                                     param: str,
                                     decisions: List[Dict],
                                     performance: Dict) -> Optional[ParameterGradient]:
        """分析单个参数的性能影响"""
        param_values = []
        performances = []
        
        for decision in decisions:
            # 从config_snapshot中提取参数值
            config = decision.get('config_snapshot', {})
            value = self._get_nested_value(config, param)
            
            if value is not None:
                param_values.append(value)
                # 使用决策质量作为性能指标
                perf = decision.get('decision_quality', 0.5)
                performances.append(perf)
                
        if len(param_values) < 5:
            return None
            
        # 计算局部梯度
        gradient = self._calculate_local_gradient(param_values, performances)
        confidence = self._calculate_gradient_confidence(param_values, performances)
        sensitivity = self._calculate_parameter_sensitivity(performances)
        
        return ParameterGradient(
            parameter=param,
            gradient=gradient,
            confidence=confidence,
            sensitivity=sensitivity,
            recent_impact=performances[-5:]
        )
    
    def _calculate_local_gradient(self, values: List[float], 
                                performances: List[float]) -> float:
        """计算局部梯度"""
        if len(values) < 2:
            return 0.0
            
        # 使用加权线性回归
        weights = np.exp(np.linspace(-1, 0, len(values)))
        values_norm = (values - np.mean(values)) / (np.std(values) + 1e-8)
        
        try:
            slope, _ = np.polyfit(values_norm, performances, 1, w=weights)
            gradient = slope / (np.std(values) + 1e-8)
        except:
            gradient = 0.0
            
        return np.clip(gradient, -1, 1)
    
    def _calculate_gradient_confidence(self, values: List[float], 
                                     performances: List[float]) -> float:
        """计算梯度置信度"""
        if len(values) < 3:
            return 0.0
            
        correlation = abs(np.corrcoef(values, performances)[0, 1])
        sample_factor = min(len(values) / 20, 1.0)
        value_spread = (max(values) - min(values)) / (np.mean(values) + 1e-8)
        spread_factor = min(value_spread / 0.2, 1.0)
        
        return min(correlation * sample_factor * spread_factor, 1.0)
    
    def _calculate_parameter_sensitivity(self, performances: List[float]) -> float:
        """计算参数敏感度"""
        if len(performances) < 2:
            return 0.0
            
        perf_std = np.std(performances)
        perf_range = max(performances) - min(performances)
        
        return min(perf_std * perf_range, 1.0)
    
    def _get_nested_value(self, config: Dict, path: str) -> Any:
        """获取嵌套配置值"""
        keys = path.split('.')
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value


class ConditionalParameterOptimizer:
    """条件参数优化器 - 基于市场状态优化参数"""
    
    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]]):
        self.parameter_bounds = parameter_bounds
        self.regime_parameters = defaultdict(dict)
        self.exploration_rate = defaultdict(lambda: 0.2)
        
    def optimize_for_regime(self, 
                          regime: str,
                          gradients: Dict[str, ParameterGradient],
                          current_config: Dict) -> Dict[str, float]:
        """为特定市场状态优化参数"""
        adjustments = {}
        
        # 选择最敏感的参数进行优化
        sensitive_params = self._select_sensitive_parameters(gradients, n=3)
        
        for param in sensitive_params:
            gradient_info = gradients[param]
            adjustment = self._calculate_adjustment(
                param, gradient_info, regime, current_config
            )
            
            if abs(adjustment) > 1e-6:
                adjustments[param] = adjustment
                
        return adjustments
    
    def _select_sensitive_parameters(self, 
                                   gradients: Dict[str, ParameterGradient],
                                   n: int = 3) -> List[str]:
        """选择最敏感的参数"""
        param_scores = {
            param: grad.sensitivity * grad.confidence
            for param, grad in gradients.items()
        }
        
        sorted_params = sorted(param_scores.items(), key=lambda x: x[1], reverse=True)
        return [param for param, _ in sorted_params[:n]]
    
    def _calculate_adjustment(self,
                            param: str,
                            gradient_info: ParameterGradient,
                            regime: str,
                            current_config: Dict) -> float:
        """计算参数调整量"""
        current_value = self._get_param_value(current_config, param)
        if current_value is None:
            return 0
            
        # 基础调整量
        base_adjustment = gradient_info.gradient * 0.1 * gradient_info.confidence
        
        # 探索因子
        exploration = self._get_exploration_adjustment(param, regime)
        
        # 总调整量
        adjustment = base_adjustment + exploration
        
        # 确保在边界内
        bounds = self.parameter_bounds.get(param, (0, 100))
        new_value = current_value + adjustment * (bounds[1] - bounds[0])
        new_value = np.clip(new_value, bounds[0], bounds[1])
        
        return new_value - current_value
    
    def _get_exploration_adjustment(self, param: str, regime: str) -> float:
        """获取探索性调整"""
        key = f"{regime}_{param}"
        exploration_rate = self.exploration_rate[key]
        
        if np.random.random() < exploration_rate:
            return np.random.normal(0, 0.05)
        return 0
    
    def _get_param_value(self, config: Dict, param_path: str) -> Optional[float]:
        """获取参数值"""
        keys = param_path.split('.')
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return float(value) if value is not None else None


class EnhancedAdaptiveLearner:
    """增强版自适应学习器 - 优化其他模块参数"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # 核心组件
        self.gradient_learner = GradientAwareLearner(
            learning_rate=self.config['learning_rate']
        )
        self.parameter_optimizer = ConditionalParameterOptimizer(
            parameter_bounds=self.config['parameter_bounds']
        )
        
        # 学习历史
        self.learning_history = []
        self.continuous_decisions = deque(maxlen=1000)  # 存储连续决策
        
        # 学习控制
        self.learning_enabled = True
        self.last_optimization = datetime.utcnow()
        self.optimization_cooldown = timedelta(minutes=30)
        
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'learning_rate': 0.1,
            'min_decisions_for_learning': 20,
            'parameter_bounds': {
                'market_behavior.order_flow.sweep_threshold': (2.0, 4.0),
                'market_behavior.divergence.min_duration': (2, 10),
                'market_behavior.divergence.lookback_period': (10, 30),
                'gamma_analysis.wall_percentile': (70, 95),
                'gamma_analysis.hedge_flow_threshold': (0.5, 0.9),
                'signal_generation.min_strength': (40, 70),
                'signal_generation.min_confidence': (0.3, 0.8),
            }
        }
    
    def record_continuous_decision(self, decision: Dict):
        """记录连续决策（每5分钟）"""
        # 计算决策质量
        decision['decision_quality'] = self._evaluate_decision_quality(decision)
        self.continuous_decisions.append(decision)
        
        logger.info(f"Recorded continuous decision: type={decision.get('decision_type')}, "
                   f"quality={decision['decision_quality']:.2f}")

    def _evaluate_decision_quality(self, decision: Dict) -> float:
        """评估决策质量 - 连续化版本"""
        quality = 0.5  # 基准分
        
        # 1. 反事实分析 - 使用连续函数
        counterfactual = decision.get('counterfactual_data', {})
        for asset, cf_data in counterfactual.items():
            potential = cf_data.get('potential_signal_strength', 0) / 100  # 归一化到0-1
            opportunity = cf_data.get('missed_opportunity_score', 0)
            
            # 使用sigmoid型函数，在中间区域更敏感
            # potential接近0.5时斜率最大，两端趋于平缓
            potential_impact = (potential - 0.5) * 2  # -1 到 1
            quality_adjustment = -0.2 * (1 / (1 + np.exp(-5 * potential_impact)))
            quality += quality_adjustment
            
            # 错失机会的连续惩罚
            # 使用平方根使小机会的惩罚较轻，大机会的惩罚递增但不过度
            quality -= 0.3 * np.sqrt(opportunity)
        
        # 2. 市场行为指标 - 组合多个信号
        behavior = decision.get('behavior_metrics', {})
        sweep_count = behavior.get('sweep_count', 0)
        anomaly_scores = behavior.get('anomaly_scores', {})
        
        if anomaly_scores:
            max_anomaly = max(anomaly_scores.values(), default=0)
            # 组合扫单数量和异常分数，使用乘积形式
            # 这样只有两者都高时才会有显著影响
            market_activity = (sweep_count / 10) * max_anomaly  # 归一化
            quality -= 0.15 * np.tanh(market_activity * 2)  # tanh限制在[-1,1]
        
        # 3. 评分一致性分析
        scores = decision.get('scores', {})
        all_scores = []
        for asset_scores in scores.values():
            if isinstance(asset_scores, dict):
                all_scores.extend(asset_scores.values())
        
        if all_scores:
            # 归一化分数到0-1
            normalized_scores = [s/100 for s in all_scores]
            max_score = max(normalized_scores)
            score_std = np.std(normalized_scores)
            
            # 高分信号的连续惩罚
            high_score_penalty = -0.15 * (max_score - 0.7) if max_score > 0.7 else 0
            
            # 分散度奖励 - 使用对数函数避免过度奖励
            dispersion_bonus = 0.1 * np.log1p(score_std * 5)
            
            quality += high_score_penalty + dispersion_bonus
        
        # 4. 添加一些正则化，避免极端值
        # 使用soft clipping，避免硬边界
        quality = 0.5 + 0.4 * np.tanh((quality - 0.5) * 3)
        
        return quality
    
    def learn_from_continuous_decisions(self) -> Dict[str, Any]:
        """从连续决策中学习"""
        if not self.learning_enabled:
            return {}
            
        # 检查是否有足够的决策数据
        if len(self.continuous_decisions) < self.config['min_decisions_for_learning']:
            logger.info(f"Insufficient continuous decisions: {len(self.continuous_decisions)}")
            return {}
            
        # 检查冷却期
        if datetime.utcnow() - self.last_optimization < self.optimization_cooldown:
            return {}
            
        try:
            # 转换为列表供分析
            recent_decisions = list(self.continuous_decisions)[-100:]
            
            # 提取当前市场状态（从最新决策）
            latest_decision = recent_decisions[-1]
            market_regime = latest_decision.get('behavior_metrics', {}).get('market_regime', 'normal')
            
            # 计算性能统计
            performance_stats = self._calculate_performance_stats(recent_decisions)
            
            # 估计参数梯度
            gradients = self.gradient_learner.estimate_parameter_gradients(
                recent_decisions, performance_stats
            )
            
            # 获取当前配置
            current_config = latest_decision.get('config_snapshot', {})
            
            # 优化参数
            adjustments = self.parameter_optimizer.optimize_for_regime(
                market_regime, gradients, current_config
            )
            
            # 创建学习决策
            learning_decision = self._create_learning_decision(
                market_regime, adjustments, gradients
            )
            
            # 记录学习历史
            self._record_learning_history(learning_decision, performance_stats)
            
            self.last_optimization = datetime.utcnow()
            
            return {
                'adjustments': adjustments,
                'regime': market_regime,
                'learning_decision': learning_decision,
                'performance_stats': performance_stats
            }
            
        except Exception as e:
            logger.error(f"Error in continuous learning: {e}", exc_info=True)
            return {}
    
    def _calculate_performance_stats(self, decisions: List[Dict]) -> Dict:
        """计算性能统计"""
        stats = {
            'avg_decision_quality': np.mean([d.get('decision_quality', 0.5) for d in decisions]),
            'missed_opportunities': sum(1 for d in decisions if d.get('missed_opportunity')),
            'signals_generated': sum(1 for d in decisions if d.get('signal_generated')),
            'correct_no_signals': sum(1 for d in decisions if d.get('avoided_bad_signal')),
        }
        
        # 计算趋势
        qualities = [d.get('decision_quality', 0.5) for d in decisions]
        if len(qualities) > 10:
            recent_avg = np.mean(qualities[-10:])
            older_avg = np.mean(qualities[-20:-10]) if len(qualities) >= 20 else np.mean(qualities[:-10])
            stats['quality_trend'] = recent_avg - older_avg
        else:
            stats['quality_trend'] = 0
            
        return stats
    
    def _create_learning_decision(self, 
                                regime: str,
                                adjustments: Dict[str, float],
                                gradients: Dict[str, ParameterGradient]) -> LearningDecision:
        """创建学习决策记录"""
        expected_improvement = 0
        for param, adjustment in adjustments.items():
            if param in gradients:
                grad_info = gradients[param]
                improvement = grad_info.gradient * adjustment * grad_info.confidence
                expected_improvement += improvement
                
        decision_basis = 'gradient' if adjustments else 'no_adjustment'
            
        return LearningDecision(
            timestamp=datetime.utcnow(),
            market_regime=regime,
            parameter_adjustments=adjustments,
            expected_improvement=expected_improvement,
            exploration_factor=0.1,
            decision_basis=decision_basis
        )
    
    def _record_learning_history(self, decision: LearningDecision, 
                               performance: Dict):
        """记录学习历史"""
        self.learning_history.append({
            'decision': decision,
            'performance_before': copy.deepcopy(performance),
            'timestamp': datetime.utcnow()
        })
        
        if len(self.learning_history) > 1000:
            self.learning_history = self.learning_history[-1000:]
    
    def get_learning_report(self) -> Dict[str, Any]:
        """生成学习报告"""
        report = {
            'total_continuous_decisions': len(self.continuous_decisions),
            'total_learning_cycles': len(self.learning_history),
            'parameter_adjustments': self._summarize_adjustments(),
            'performance_trend': self._analyze_performance_trend(),
            'learning_effectiveness': self._evaluate_effectiveness()
        }
        
        return report
    
    def _summarize_adjustments(self) -> Dict[str, Any]:
        """总结参数调整"""
        adjustments = defaultdict(list)
        
        for record in self.learning_history[-20:]:
            for param, value in record['decision'].parameter_adjustments.items():
                adjustments[param].append(value)
                
        summary = {}
        for param, values in adjustments.items():
            if values:
                summary[param] = {
                    'total_adjustments': len(values),
                    'net_change': sum(values),
                    'avg_adjustment': np.mean(values)
                }
                
        return summary
    
    def _analyze_performance_trend(self) -> Dict[str, float]:
        """分析性能趋势"""
        if len(self.continuous_decisions) < 50:
            return {'trend': 0.0, 'confidence': 0.0}
            
        decisions = list(self.continuous_decisions)
        qualities = [d.get('decision_quality', 0.5) for d in decisions]
        
        # 分成前后两半
        mid = len(qualities) // 2
        first_half = np.mean(qualities[:mid])
        second_half = np.mean(qualities[mid:])
        
        return {
            'trend': second_half - first_half,
            'confidence': min(len(qualities) / 200, 1.0),
            'current_avg': np.mean(qualities[-20:])
        }
    
    def _evaluate_effectiveness(self) -> Dict[str, float]:
        """评估学习效果"""
        if len(self.learning_history) < 2:
            return {'effectiveness': 0, 'confidence': 0}
            
        improvements = []
        for i in range(1, min(len(self.learning_history), 10)):
            before = self.learning_history[i-1]['performance_before'].get('avg_decision_quality', 0.5)
            after = self.learning_history[i]['performance_before'].get('avg_decision_quality', 0.5)
            improvements.append(after - before)
            
        return {
            'avg_improvement': np.mean(improvements) if improvements else 0,
            'positive_rate': sum(1 for i in improvements if i > 0) / len(improvements) if improvements else 0,
            'confidence': min(len(improvements) / 10, 1.0)
        }
