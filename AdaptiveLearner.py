"""
Enhanced AdaptiveLearner - 修复冷启动悖论的自适应学习模块
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
    exploration_bonus: float = 0.0  # 新增：探索奖励

@dataclass
class LearningDecision:
    """学习决策记录"""
    timestamp: datetime
    market_regime: str
    parameter_adjustments: Dict[str, float]
    expected_improvement: float
    exploration_factor: float
    decision_basis: str  # 'gradient', 'exploration', 'hybrid', 'bootstrap'

class GradientAwareLearner:
    """梯度感知学习器 - 带探索机制"""
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.gradient_history = defaultdict(lambda: deque(maxlen=100))
        self.parameter_performance = defaultdict(list)
        self.exploration_count = defaultdict(int)  # 新增：探索计数
        
    def estimate_parameter_gradients(self,
                                   recent_decisions: List[Dict],
                                   performance_data: Dict) -> Dict[str, ParameterGradient]:
        """估计参数梯度 - 增强版"""
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
            gradient_info = self._analyze_parameter_performance_enhanced(
                param, recent_decisions, performance_data
            )
            if gradient_info:
                gradients[param] = gradient_info
                
        return gradients
    
    def _analyze_parameter_performance_enhanced(self,
                                              param: str,
                                              decisions: List[Dict],
                                              performance: Dict) -> Optional[ParameterGradient]:
        """分析单个参数的性能影响 - 增强版处理参数无变化的情况"""
        param_values = []
        performances = []
        
        for decision in decisions:
            config = decision.get('config_snapshot', {})
            value = self._get_nested_value(config, param)
            
            if value is not None:
                param_values.append(value)
                perf = decision.get('decision_quality', 0.5)
                performances.append(perf)
        
        if len(param_values) < 3:  # 降低最小要求
            return self._create_exploration_gradient(param, performance)
        
        # 检查参数变化情况
        param_std = np.std(param_values)
        
        if param_std < 1e-8:  # 参数几乎没有变化
            # 使用探索梯度
            return self._create_exploration_gradient(param, performance, performances)
        
        # 正常梯度计算
        gradient = self._calculate_local_gradient(param_values, performances)
        confidence = self._calculate_gradient_confidence(param_values, performances)
        sensitivity = self._calculate_parameter_sensitivity(performances)
        
        # 添加探索奖励
        exploration_bonus = self._calculate_exploration_bonus(param, len(param_values))
        
        return ParameterGradient(
            parameter=param,
            gradient=gradient,
            confidence=confidence,
            sensitivity=sensitivity,
            recent_impact=performances[-5:],
            exploration_bonus=exploration_bonus
        )
    
    def _create_exploration_gradient(self, param: str,
                                   performance: Dict,
                                   recent_performances: List[float] = None) -> ParameterGradient:
        """创建探索性梯度"""
        # 基于性能趋势决定探索方向
        if recent_performances and len(recent_performances) > 1:
            # 如果性能在下降，倾向于更大的探索
            perf_trend = recent_performances[-1] - recent_performances[0]
            exploration_gradient = -0.1 if perf_trend < 0 else 0.1
        else:
            # 随机探索方向
            exploration_gradient = np.random.choice([-0.1, 0.1])
        
        # 基于参数类型调整探索强度
        if 'threshold' in param or 'min_' in param:
            sensitivity = 0.5  # 阈值类参数通常比较敏感
        else:
            sensitivity = 0.3
        
        return ParameterGradient(
            parameter=param,
            gradient=exploration_gradient,
            confidence=0.3,  # 低置信度表示这是探索
            sensitivity=sensitivity,
            recent_impact=recent_performances[-5:] if recent_performances else [0.5],
            exploration_bonus=0.5  # 高探索奖励
        )
    
    def _calculate_exploration_bonus(self, param: str, sample_count: int) -> float:
        """计算探索奖励"""
        # 样本越少，探索奖励越高
        base_bonus = max(0, 1 - sample_count / 50)
        
        # 如果某个参数很少被探索，给予额外奖励
        exploration_count = self.exploration_count[param]
        if exploration_count < 5:
            base_bonus += 0.2
        
        return min(base_bonus, 1.0)
    
    def _calculate_local_gradient(self, values: List[float],
                                performances: List[float]) -> float:
        """计算局部梯度 - 改进版"""
        if len(values) < 2:
            return 0.0
        
        # 处理标准差为0的情况
        values_std = np.std(values)
        if values_std < 1e-8:
            return 0.0
        
        # 使用加权线性回归
        weights = np.exp(np.linspace(-1, 0, len(values)))
        values_norm = (values - np.mean(values)) / values_std
        
        try:
            slope, _ = np.polyfit(values_norm, performances, 1, w=weights)
            gradient = slope / values_std
        except:
            gradient = 0.0
            
        return np.clip(gradient, -1, 1)
    
    def _calculate_gradient_confidence(self, values: List[float],
                                     performances: List[float]) -> float:
        """计算梯度置信度 - 改进版"""
        if len(values) < 3:
            return 0.0
        
        # 处理相关性计算中的异常
        try:
            if np.std(values) < 1e-8 or np.std(performances) < 1e-8:
                return 0.0
            
            correlation = abs(np.corrcoef(values, performances)[0, 1])
            if np.isnan(correlation):
                correlation = 0.0
        except:
            correlation = 0.0
        
        sample_factor = min(len(values) / 20, 1.0)
        value_spread = (max(values) - min(values)) / (np.mean(values) + 1e-8)
        spread_factor = min(value_spread / 0.2, 1.0)
        
        return min(correlation * sample_factor * spread_factor, 1.0)
    
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
    """条件参数优化器 - 增强探索机制"""
    
    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]]):
        self.parameter_bounds = parameter_bounds
        self.regime_parameters = defaultdict(dict)
        self.exploration_rate = defaultdict(lambda: 0.3)  # 提高初始探索率
        self.parameter_history = defaultdict(list)  # 参数历史
        
    def optimize_for_regime(self,
                          regime: str,
                          gradients: Dict[str, ParameterGradient],
                          current_config: Dict,
                          force_exploration: bool = False) -> Dict[str, float]:
        """为特定市场状态优化参数"""
        adjustments = {}
        
        # 如果没有梯度或强制探索，进行探索性调整
        if not gradients or force_exploration:
            return self._exploration_adjustments(regime, current_config)
        
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
    def _select_sensitive_parameters(self, gradients: Dict[str, ParameterGradient], n: int = 3) -> List[str]:
        if not gradients:
            return []
        
        # 计算每个参数的优先级分数
        parameter_scores = []
        
        for param, gradient_info in gradients.items():
            # 综合考虑多个因素
            # 1. 梯度绝对值（影响大小）
            gradient_magnitude = abs(gradient_info.gradient)
            
            # 2. 置信度（可靠性）
            confidence = gradient_info.confidence
            
            # 3. 敏感性（参数的影响力）
            sensitivity = gradient_info.sensitivity
            
            # 4. 探索奖励（鼓励探索不充分的参数）
            exploration_bonus = gradient_info.exploration_bonus
            
            # 计算综合分数
            # 基础分数：梯度 * 置信度 * 敏感性
            base_score = gradient_magnitude * confidence * sensitivity
            
            # 加入探索奖励，但权重较低，避免过度探索
            exploration_weight = 0.2
            final_score = base_score * (1 - exploration_weight) + exploration_bonus * exploration_weight
            
            parameter_scores.append({
                'param': param,
                'score': final_score,
                'gradient': gradient_info.gradient,
                'confidence': confidence
            })
        
        # 按分数排序
        parameter_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # 额外的筛选逻辑
        selected_params = []
        
        for item in parameter_scores:
            # 跳过置信度太低的参数（避免基于噪声调整）
            if item['confidence'] < 0.2:
                continue
                
            # 确保参数在允许的边界内
            if item['param'] not in self.parameter_bounds:
                continue
                
            selected_params.append(item['param'])
            
            if len(selected_params) >= n:
                break
        
        # 如果选中的参数太少，考虑添加一些探索性参数
        if len(selected_params) < n:
            # 找出最近调整较少的参数
            remaining_params = [p for p in gradients.keys()
                              if p not in selected_params and p in self.parameter_bounds]
            
            # 按探索奖励排序
            remaining_params.sort(
                key=lambda p: gradients[p].exploration_bonus,
                reverse=True
            )
            
            # 补充参数
            for param in remaining_params:
                selected_params.append(param)
                if len(selected_params) >= n:
                    break
        
        return selected_params
        
    def _exploration_adjustments(self, regime: str, current_config: Dict) -> Dict[str, float]:
        """生成探索性参数调整"""
        adjustments = {}
        
        # 随机选择2-3个参数进行探索
        all_params = list(self.parameter_bounds.keys())
        n_params = np.random.randint(2, min(4, len(all_params) + 1))
        selected_params = np.random.choice(all_params, n_params, replace=False)
        
        for param in selected_params:
            current_value = self._get_param_value(current_config, param)
            if current_value is None:
                continue
            
            bounds = self.parameter_bounds[param]
            
            # 使用智能探索策略
            if self._should_explore_direction(param, 'up'):
                # 向上探索
                adjustment = np.random.uniform(0.05, 0.15) * (bounds[1] - bounds[0])
            else:
                # 向下探索
                adjustment = -np.random.uniform(0.05, 0.15) * (bounds[1] - bounds[0])
            
            # 确保在边界内
            new_value = current_value + adjustment
            new_value = np.clip(new_value, bounds[0], bounds[1])
            
            adjustments[param] = new_value - current_value
            
            # 记录探索
            self._record_exploration(param, regime)
        
        return adjustments
    
    def _should_explore_direction(self, param: str, direction: str) -> bool:
        """决定探索方向"""
        # 检查参数历史
        if param in self.parameter_history and len(self.parameter_history[param]) > 2:
            recent_values = self.parameter_history[param][-3:]
            
            # 如果最近都在增加，考虑减少
            if all(recent_values[i] > recent_values[i-1] for i in range(1, len(recent_values))):
                return direction == 'down'
            # 如果最近都在减少，考虑增加
            elif all(recent_values[i] < recent_values[i-1] for i in range(1, len(recent_values))):
                return direction == 'up'
        
        # 否则随机
        return np.random.random() > 0.5
    
    def _record_exploration(self, param: str, regime: str):
        """记录探索行为"""
        key = f"{regime}_{param}"
        # 动态调整探索率
        self.exploration_rate[key] *= 0.95  # 逐渐降低探索率
        self.exploration_rate[key] = max(self.exploration_rate[key], 0.1)  # 保持最小探索
    
    def _calculate_adjustment(self,
                            param: str,
                            gradient_info: ParameterGradient,
                            regime: str,
                            current_config: Dict) -> float:
        """计算参数调整量 - 增强版"""
        current_value = self._get_param_value(current_config, param)
        if current_value is None:
            return 0
        
        # 记录参数历史
        self.parameter_history[param].append(current_value)
        if len(self.parameter_history[param]) > 20:
            self.parameter_history[param].pop(0)
        
        # 基础调整量
        base_adjustment = gradient_info.gradient * 0.1 * gradient_info.confidence
        
        # 探索因子
        exploration = self._get_exploration_adjustment(param, regime)
        
        # 考虑探索奖励
        if gradient_info.exploration_bonus > 0.3:
            exploration *= (1 + gradient_info.exploration_bonus)
        
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
            # 使用自适应探索幅度
            exploration_std = 0.05 * (1 + exploration_rate)
            return np.random.normal(0, exploration_std)
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
    """增强版自适应学习器 - 解决冷启动问题"""
    
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
        self.continuous_decisions = deque(maxlen=1000)
        
        # 学习控制
        self.learning_enabled = True
        self.last_optimization = datetime.utcnow()
        self.optimization_cooldown = timedelta(minutes=30)
        
        # 冷启动控制
        self.bootstrap_phase = True
        self.bootstrap_cycles = 0
        self.min_bootstrap_cycles = 3  # 至少进行3次强制探索
        
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'learning_rate': 0.1,
            'min_decisions_for_learning': 20,
            'bootstrap_decisions_threshold': 10,  # 冷启动阶段的决策阈值
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
        """记录连续决策"""
        # 计算决策质量
        decision['decision_quality'] = self._evaluate_decision_quality(decision)
        self.continuous_decisions.append(decision)
        
        logger.info(f"Recorded continuous decision: type={decision.get('decision_type')}, "
                   f"quality={decision['decision_quality']:.2f}")
    
    def learn_from_continuous_decisions(self) -> Dict[str, Any]:
        """从连续决策中学习 - 增强版处理冷启动"""
        if not self.learning_enabled:
            return {}
        
        # 冷启动阶段使用更低的阈值
        min_decisions = (self.config['bootstrap_decisions_threshold']
                        if self.bootstrap_phase
                        else self.config['min_decisions_for_learning'])
        
        # 检查是否有足够的决策数据
        if len(self.continuous_decisions) < min_decisions:
            logger.info(f"Insufficient continuous decisions: {len(self.continuous_decisions)} < {min_decisions}")
            return {}
        
        # 检查冷却期
        if datetime.utcnow() - self.last_optimization < self.optimization_cooldown:
            return {}
        
        try:
            # 转换为列表供分析
            recent_decisions = list(self.continuous_decisions)[-100:]
            
            # 提取当前市场状态
            latest_decision = recent_decisions[-1]
            market_regime = latest_decision.get('behavior_metrics', {}).get('market_regime', 'normal')
            
            # 计算性能统计
            performance_stats = self._calculate_performance_stats(recent_decisions)
            
            # 如果在冷启动阶段，强制探索
            if self.bootstrap_phase and self.bootstrap_cycles < self.min_bootstrap_cycles:
                logger.info(f"Bootstrap phase: forcing exploration (cycle {self.bootstrap_cycles + 1}/{self.min_bootstrap_cycles})")
                
                # 获取当前配置
                current_config = latest_decision.get('config_snapshot', {})
                
                # 生成探索性调整
                adjustments = self.parameter_optimizer._exploration_adjustments(
                    market_regime, current_config
                )
                
                # 创建学习决策
                learning_decision = LearningDecision(
                    timestamp=datetime.utcnow(),
                    market_regime=market_regime,
                    parameter_adjustments=adjustments,
                    expected_improvement=0.0,  # 探索阶段没有预期改进
                    exploration_factor=1.0,  # 完全探索
                    decision_basis='bootstrap'
                )
                
                self.bootstrap_cycles += 1
                
                # 检查是否可以退出冷启动
                if self.bootstrap_cycles >= self.min_bootstrap_cycles:
                    self._check_bootstrap_exit(performance_stats)
                
            else:
                # 正常学习流程
                # 估计参数梯度
                gradients = self.gradient_learner.estimate_parameter_gradients(
                    recent_decisions, performance_stats
                )
                
                # 获取当前配置
                current_config = latest_decision.get('config_snapshot', {})
                
                # 优化参数
                adjustments = self.parameter_optimizer.optimize_for_regime(
                    market_regime, gradients, current_config,
                    force_exploration=(len(gradients) == 0)  # 如果没有梯度，强制探索
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
                'performance_stats': performance_stats,
                'bootstrap_phase': self.bootstrap_phase
            }
            
        except Exception as e:
            logger.error(f"Error in continuous learning: {e}", exc_info=True)
            return {}
    
    def _check_bootstrap_exit(self, performance_stats: Dict):
        """检查是否可以退出冷启动阶段"""
        # 如果性能稳定且有足够的参数变化，退出冷启动
        if performance_stats.get('quality_trend', 0) > -0.1:  # 性能没有明显下降
            logger.info("Exiting bootstrap phase - performance stable")
            self.bootstrap_phase = False
    
    def _evaluate_decision_quality(self, decision: Dict) -> float:
        """评估决策质量 - 连续化版本"""
        quality = 0.5  # 基准分
        
        # 1. 反事实分析
        counterfactual = decision.get('counterfactual_data', {})
        for asset, cf_data in counterfactual.items():
            potential = cf_data.get('potential_signal_strength', 0) / 100
            opportunity = cf_data.get('missed_opportunity_score', 0)
            
            potential_impact = (potential - 0.5) * 2
            quality_adjustment = -0.2 * (1 / (1 + np.exp(-5 * potential_impact)))
            quality += quality_adjustment
            
            quality -= 0.3 * np.sqrt(opportunity)
        
        # 2. 市场行为指标
        behavior = decision.get('behavior_metrics', {})
        sweep_count = behavior.get('sweep_count', 0)
        anomaly_scores = behavior.get('anomaly_scores', {})
        
        if anomaly_scores:
            max_anomaly = max(anomaly_scores.values(), default=0)
            market_activity = (sweep_count / 10) * max_anomaly
            quality -= 0.15 * np.tanh(market_activity * 2)
        
        # 3. 评分一致性分析
        scores = decision.get('scores', {})
        all_scores = []
        for asset_scores in scores.values():
            if isinstance(asset_scores, dict):
                all_scores.extend(asset_scores.values())
        
        if all_scores:
            normalized_scores = [s/100 for s in all_scores]
            max_score = max(normalized_scores)
            score_std = np.std(normalized_scores)
            
            high_score_penalty = -0.15 * (max_score - 0.7) if max_score > 0.7 else 0
            dispersion_bonus = 0.1 * np.log1p(score_std * 5)
            
            quality += high_score_penalty + dispersion_bonus
        
        # 4. 正则化
        quality = 0.5 + 0.45 * np.tanh((quality - 0.5) * 3)
        
        return quality
    
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
        exploration_factor = 0
        
        for param, adjustment in adjustments.items():
            if param in gradients:
                grad_info = gradients[param]
                improvement = grad_info.gradient * adjustment * grad_info.confidence
                expected_improvement += improvement
                exploration_factor += grad_info.exploration_bonus
        
        # 归一化探索因子
        if adjustments:
            exploration_factor /= len(adjustments)
        
        # 决定决策基础
        if exploration_factor > 0.5:
            decision_basis = 'exploration'
        elif expected_improvement > 0.1:
            decision_basis = 'gradient'
        else:
            decision_basis = 'hybrid'
            
        return LearningDecision(
            timestamp=datetime.utcnow(),
            market_regime=regime,
            parameter_adjustments=adjustments,
            expected_improvement=expected_improvement,
            exploration_factor=exploration_factor,
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
            'bootstrap_status': {
                'active': self.bootstrap_phase,
                'cycles_completed': self.bootstrap_cycles,
                'min_required': self.min_bootstrap_cycles
            },
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
                    'avg_adjustment': np.mean(values),
                    'adjustment_variance': np.var(values)
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
            'current_avg': np.mean(qualities[-20:]),
            'volatility': np.std(qualities[-20:])
        }
    
    def _evaluate_effectiveness(self) -> Dict[str, float]:
        """评估学习效果"""
        if len(self.learning_history) < 2:
            return {'effectiveness': 0, 'confidence': 0}
            
        improvements = []
        exploration_ratios = []
        
        for i in range(1, min(len(self.learning_history), 10)):
            before = self.learning_history[i-1]['performance_before'].get('avg_decision_quality', 0.5)
            after = self.learning_history[i]['performance_before'].get('avg_decision_quality', 0.5)
            improvements.append(after - before)
            
            # 记录探索比例
            decision = self.learning_history[i]['decision']
            exploration_ratios.append(decision.exploration_factor)
            
        return {
            'avg_improvement': np.mean(improvements) if improvements else 0,
            'positive_rate': sum(1 for i in improvements if i > 0) / len(improvements) if improvements else 0,
            'confidence': min(len(improvements) / 10, 1.0),
            'avg_exploration_ratio': np.mean(exploration_ratios) if exploration_ratios else 0,
            'learning_diversity': np.std(improvements) if len(improvements) > 1 else 0
        }
