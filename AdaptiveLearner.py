"""
EnhancedAdaptiveLearner - 增强版自适应学习模块
采用梯度感知学习、市场状态识别和条件参数优化
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import copy

logger = logging.getLogger(__name__)

@dataclass
class MarketRegime:
    """市场状态定义"""
    regime_id: str
    characteristics: Dict[str, float]
    typical_parameters: Dict[str, Any]
    performance_history: List[float] = field(default_factory=list)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    stability_score: float = 0.5

@dataclass
class ParameterGradient:
    """参数梯度信息"""
    parameter: str
    gradient: float  # 正值表示应增加，负值表示应减少
    confidence: float  # 梯度估计的置信度
    sensitivity: float  # 参数敏感度
    recent_impact: List[float] = field(default_factory=list)

@dataclass
class LearningDecision:
    """学习决策记录"""
    timestamp: datetime
    market_regime: str
    parameter_adjustments: Dict[str, float]
    expected_improvement: float
    exploration_factor: float
    decision_basis: str  # 'gradient', 'exploration', 'pattern_memory'

class MarketRegimeIdentifier:
    """市场状态识别器"""
    
    def __init__(self, n_regimes: int = 5):
        self.n_regimes = n_regimes
        self.scaler = StandardScaler()
        self.kmeans = None
        self.regime_history = deque(maxlen=1000)
        self.regime_transitions = defaultdict(lambda: defaultdict(int))
        self.feature_importance = {}
        
    def identify_regime(self, market_features: Dict[str, float]) -> Tuple[str, float]:
        """识别当前市场状态"""
        # 提取关键特征
        feature_vector = self._extract_features(market_features)
        
        if self.kmeans is None:
            # 首次调用，返回默认状态
            return "regime_0", 0.5
            
        # 标准化特征
        feature_scaled = self.scaler.transform([feature_vector])[0]
        
        # 预测状态
        regime_id = self.kmeans.predict([feature_scaled])[0]
        
        # 计算置信度（到各中心的距离）
        distances = self.kmeans.transform([feature_scaled])[0]
        confidence = 1.0 / (1.0 + distances[regime_id])
        
        regime_name = f"regime_{regime_id}"
        
        # 更新历史
        self._update_history(regime_name, market_features)
        
        return regime_name, confidence
        
    def _extract_features(self, market_data: Dict[str, float]) -> List[float]:
        """提取市场特征向量"""
        features = []
        
        # 波动率相关
        features.append(market_data.get('volatility', 0))
        features.append(market_data.get('volatility_percentile', 0.5))
        
        # Gamma相关
        features.append(market_data.get('total_gamma_exposure', 0))
        features.append(market_data.get('gamma_concentration', 0))
        features.append(market_data.get('nearest_wall_distance', 10))
        
        # 市场动量
        features.append(market_data.get('sweep_intensity', 0))
        features.append(market_data.get('price_momentum', 0))
        features.append(market_data.get('volume_anomaly', 0))
        
        # 市场微观结构
        features.append(market_data.get('spread_percentile', 0.5))
        features.append(market_data.get('orderbook_imbalance', 0))
        
        if 'behavior_ml_features' in market_data:
            ml_features = market_data['behavior_ml_features']
            features.extend([
                ml_features.get('sweep_anomaly_mean', 0),
                ml_features.get('sweep_volume_std', 0), 
                ml_features.get('buy_ratio', 0.5),
                ml_features.get('divergence_strength_mean', 0),
                ml_features.get('divergence_max_duration', 0),
                ml_features.get('divergence_type_count', 0)
            ])
        
        return features
    

    def _update_history(self, regime: str, features: Dict[str, float]):
        """更新状态历史"""
        self.regime_history.append({
            'timestamp': datetime.utcnow(),
            'regime': regime,
            'features': features
        })
        
        # 更新转换矩阵
        if len(self.regime_history) > 1:
            prev_regime = self.regime_history[-2]['regime']
            self.regime_transitions[prev_regime][regime] += 1
            
    def train_on_history(self, historical_features: pd.DataFrame):
        """基于历史数据训练状态识别"""
        if len(historical_features) < 100:
            logger.warning("Insufficient data for regime training")
            return
            
        # 准备特征矩阵
        feature_matrix = []
        for _, row in historical_features.iterrows():
            features = self._extract_features(row.to_dict())
            feature_matrix.append(features)
            
        # 标准化
        feature_matrix = self.scaler.fit_transform(feature_matrix)
        
        # 聚类
        self.kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
        self.kmeans.fit(feature_matrix)
        
        # 分析特征重要性
        self._analyze_feature_importance(feature_matrix)
        
    def _analyze_feature_importance(self, feature_matrix: np.ndarray):
        """分析特征重要性"""
        # 计算每个特征对聚类的贡献
        cluster_centers = self.kmeans.cluster_centers_
        feature_vars = np.var(cluster_centers, axis=0)
        
        feature_names = [
            'volatility', 'volatility_pct', 'gamma_exposure', 'gamma_conc',
            'wall_distance', 'sweep_intensity', 'momentum', 'volume_anomaly',
            'spread_pct', 'ob_imbalance'
        ]
        
        for i, name in enumerate(feature_names):
            self.feature_importance[name] = feature_vars[i]


class GradientAwareLearner:
    """梯度感知学习器"""
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.gradient_history = defaultdict(lambda: deque(maxlen=100))
        self.parameter_performance = defaultdict(list)
        self.gradient_momentum = defaultdict(float)
        self.gradient_variance = defaultdict(float)
        
    def estimate_parameter_gradients(self, 
                                   recent_decisions: List[Dict],
                                   performance_data: Dict) -> Dict[str, ParameterGradient]:
        """估计参数梯度"""
        gradients = {}
        
        # 获取可调参数列表
        adjustable_params = self._get_adjustable_parameters()
        
        for param in adjustable_params:
            # 分析参数值与性能的关系
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
        # 收集参数值和对应的性能
        param_values = []
        performances = []
        
        for decision in decisions:
            if 'config_used' in decision and param in self._flatten_dict(decision['config_used']):
                value = self._get_nested_value(decision['config_used'], param)
                perf = decision.get('performance_score', 0.5)
                
                param_values.append(value)
                performances.append(perf)
                
        if len(param_values) < 5:
            return None
            
        # 计算局部梯度
        gradient = self._calculate_local_gradient(param_values, performances)
        
        # 计算置信度
        confidence = self._calculate_gradient_confidence(param_values, performances)
        
        # 计算敏感度
        sensitivity = self._calculate_parameter_sensitivity(performances)
        
        # 更新动量
        self._update_gradient_momentum(param, gradient)
        
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
            
        # 使用加权线性回归，最近的数据权重更高
        weights = np.exp(np.linspace(-1, 0, len(values)))
        
        # 归一化值以提高数值稳定性
        values_norm = (values - np.mean(values)) / (np.std(values) + 1e-8)
        
        # 加权最小二乘
        X = np.column_stack([values_norm, np.ones(len(values))])
        W = np.diag(weights)
        
        # 正规方程：(X'WX)^(-1)X'Wy
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ performances
        
        try:
            coeffs = np.linalg.solve(XtWX, XtWy)
            gradient = coeffs[0] / (np.std(values) + 1e-8)  # 还原到原始尺度
        except:
            gradient = 0.0
            
        return np.clip(gradient, -1, 1)
        
    def _calculate_gradient_confidence(self, values: List[float], 
                                     performances: List[float]) -> float:
        """计算梯度估计的置信度"""
        if len(values) < 3:
            return 0.0
            
        # 基于R²和样本数量
        correlation = abs(np.corrcoef(values, performances)[0, 1])
        sample_factor = min(len(values) / 20, 1.0)
        
        # 检查值的分散程度
        value_spread = (max(values) - min(values)) / (np.mean(values) + 1e-8)
        spread_factor = min(value_spread / 0.2, 1.0)  # 至少20%的变化范围
        
        confidence = correlation * sample_factor * spread_factor
        return min(confidence, 1.0)
        
    def _calculate_parameter_sensitivity(self, performances: List[float]) -> float:
        """计算参数敏感度"""
        if len(performances) < 2:
            return 0.0
            
        # 性能变化幅度
        perf_std = np.std(performances)
        perf_range = max(performances) - min(performances)
        
        # 敏感度 = 标准差 * 范围
        sensitivity = perf_std * perf_range
        
        return min(sensitivity, 1.0)
        
    def _update_gradient_momentum(self, param: str, gradient: float):
        """更新梯度动量"""
        momentum_decay = 0.9
        self.gradient_momentum[param] = (
            momentum_decay * self.gradient_momentum[param] + 
            (1 - momentum_decay) * gradient
        )
        
        # 更新梯度方差（用于自适应学习率）
        variance_decay = 0.95
        self.gradient_variance[param] = (
            variance_decay * self.gradient_variance[param] + 
            (1 - variance_decay) * gradient ** 2
        )
        
    def _get_adjustable_parameters(self) -> List[str]:
        """获取可调参数列表"""
        return [
            'gamma_pressure.thresholds.critical',
            'gamma_pressure.thresholds.high',
            'gamma_pressure.wall_proximity_weight',
            'gamma_pressure.hedge_flow_weight',
            'market_momentum.sweep_weight',
            'market_momentum.divergence_weight',
            'signal_generation.min_strength',
            'signal_generation.min_confidence',
            'market_behavior.order_flow.sweep_threshold',
            'market_behavior.divergence.min_duration'
        ]
        
    def _flatten_dict(self, d: Dict, parent_key: str = '') -> Dict:
        """扁平化嵌套字典"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
        
    def _get_nested_value(self, config: Dict, path: str) -> Any:
        """获取嵌套值"""
        keys = path.split('.')
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value


class ConditionalParameterOptimizer:
    """条件参数优化器"""
    
    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]]):
        self.parameter_bounds = parameter_bounds
        self.regime_parameters = defaultdict(dict)
        self.optimization_history = defaultdict(list)
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
            
            # 基于梯度和探索率决定调整
            adjustment = self._calculate_adjustment(
                param, gradient_info, regime, current_config
            )
            
            if adjustment != 0:
                adjustments[param] = adjustment
                
        # 更新探索率
        self._update_exploration_rates(regime, adjustments)
        
        return adjustments
        
    def _select_sensitive_parameters(self, 
                                   gradients: Dict[str, ParameterGradient],
                                   n: int = 3) -> List[str]:
        """选择最敏感的参数"""
        # 综合考虑敏感度和置信度
        param_scores = {}
        
        for param, grad_info in gradients.items():
            score = grad_info.sensitivity * grad_info.confidence
            param_scores[param] = score
            
        # 选择得分最高的n个参数
        sorted_params = sorted(param_scores.items(), key=lambda x: x[1], reverse=True)
        return [param for param, _ in sorted_params[:n]]
        
    def _calculate_adjustment(self,
                            param: str,
                            gradient_info: ParameterGradient,
                            regime: str,
                            current_config: Dict) -> float:
        """计算参数调整量"""
        # 获取当前值
        current_value = self._get_param_value(current_config, param)
        if current_value is None:
            return 0
            
        # 基础调整量 = 梯度 * 学习率 * 置信度
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
        
        # 随机探索
        if np.random.random() < exploration_rate:
            return np.random.normal(0, 0.1)
        return 0
        
    def _update_exploration_rates(self, regime: str, adjustments: Dict[str, float]):
        """更新探索率"""
        decay_factor = 0.95
        
        for param in adjustments:
            key = f"{regime}_{param}"
            self.exploration_rate[key] *= decay_factor
            self.exploration_rate[key] = max(self.exploration_rate[key], 0.05)
            
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


class PatternMemory:
    """模式记忆库"""
    
    def __init__(self, memory_size: int = 1000):
        self.memory_size = memory_size
        self.success_patterns = deque(maxlen=memory_size)
        self.failure_patterns = deque(maxlen=memory_size)
        self.pattern_index = {}
        
    def remember_pattern(self, 
                       market_state: Dict,
                       decision: Dict,
                       outcome: Dict):
        """记忆决策模式"""
        pattern = {
            'timestamp': datetime.utcnow(),
            'market_features': self._extract_pattern_features(market_state),
            'decision_features': self._extract_decision_features(decision),
            'outcome_score': outcome.get('composite_score', 0.5),
            'regime': market_state.get('regime', 'unknown')
        }
        
        # 分类存储
        if pattern['outcome_score'] > 0.7:
            self.success_patterns.append(pattern)
        elif pattern['outcome_score'] < 0.3:
            self.failure_patterns.append(pattern)
            
        # 更新索引
        self._update_pattern_index(pattern)
        
    def recall_similar_patterns(self, 
                              current_state: Dict,
                              n_patterns: int = 5) -> List[Dict]:
        """召回相似模式"""
        current_features = self._extract_pattern_features(current_state)
        
        # 计算与所有模式的相似度
        similarities = []
        
        for pattern in self.success_patterns:
            sim = self._calculate_similarity(current_features, pattern['market_features'])
            similarities.append((sim, pattern))
            
        # 返回最相似的模式
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [pattern for _, pattern in similarities[:n_patterns]]
        
    def _extract_pattern_features(self, state: Dict) -> np.ndarray:
        """提取模式特征"""
        features = []
        
        # 市场特征
        features.extend([
            state.get('volatility', 0),
            state.get('gamma_concentration', 0),
            state.get('sweep_intensity', 0),
            state.get('price_momentum', 0),
            state.get('volume_profile', 0),
        ])
        
        return np.array(features)
        
    def _extract_decision_features(self, decision: Dict) -> np.ndarray:
        """提取决策特征"""
        features = []
        
        # 决策参数
        features.extend([
            decision.get('signal_strength_threshold', 50),
            decision.get('confidence_threshold', 0.5),
            decision.get('risk_level', 0.5),
        ])
        
        return np.array(features)
        
    def _calculate_similarity(self, features1: np.ndarray, 
                            features2: np.ndarray) -> float:
        """计算特征相似度"""
        # 余弦相似度
        dot_product = np.dot(features1, features2)
        norm_product = np.linalg.norm(features1) * np.linalg.norm(features2)
        
        if norm_product == 0:
            return 0
            
        return dot_product / norm_product
        
    def _update_pattern_index(self, pattern: Dict):
        """更新模式索引"""
        regime = pattern['regime']
        if regime not in self.pattern_index:
            self.pattern_index[regime] = []
        self.pattern_index[regime].append(pattern)


class EnhancedAdaptiveLearner:
    """增强版自适应学习器 - 整合所有组件"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # 核心组件
        self.regime_identifier = MarketRegimeIdentifier(n_regimes=5)
        self.gradient_learner = GradientAwareLearner(
            learning_rate=self.config['learning_rate']
        )
        self.parameter_optimizer = ConditionalParameterOptimizer(
            parameter_bounds=self.config['parameter_bounds']
        )
        self.pattern_memory = PatternMemory(memory_size=1000)
        
        # 状态追踪
        self.current_regime = None
        self.regime_parameters = defaultdict(dict)
        self.learning_history = []
        self.performance_tracker = None
        
        # 学习控制
        self.learning_enabled = True
        self.last_optimization = datetime.utcnow()
        self.optimization_cooldown = timedelta(minutes=30)
        
        # 持久化
        self._load_state()
        
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'learning_rate': 0.1,
            'optimization_interval': 1800,  # 30分钟
            'min_decisions_for_learning': 20,
            'regime_stability_threshold': 10,  # 在同一regime停留的最小决策数
            'parameter_bounds': {
                'gamma_pressure.thresholds.critical': (70, 95),
                'gamma_pressure.thresholds.high': (50, 80),
                'gamma_pressure.wall_proximity_weight': (0.1, 0.5),
                'gamma_pressure.hedge_flow_weight': (0.1, 0.5),
                'market_momentum.sweep_weight': (0.2, 0.6),
                'market_momentum.divergence_weight': (0.1, 0.5),
                'signal_generation.min_strength': (40, 70),
                'signal_generation.min_confidence': (0.3, 0.8),
                'market_behavior.order_flow.sweep_threshold': (2.0, 4.0),
                'market_behavior.divergence.min_duration': (2, 10)
            },
            'exploration_config': {
                'initial_rate': 0.2,
                'decay_rate': 0.95,
                'min_rate': 0.05
            },
            'state_file': 'adaptive_learner_state.json'
        }
        
    def learn_from_decisions(self, 
                           decision_history: List[Dict],
                           performance_stats: Dict,
                           current_config: Dict) -> Dict[str, Any]:
        """从决策历史中学习并优化配置"""
        if not self.learning_enabled:
            return {}
            
        # 检查是否有足够的数据
        if len(decision_history) < self.config['min_decisions_for_learning']:
            logger.info(f"Insufficient decisions for learning: {len(decision_history)}")
            return {}
            
        # 检查冷却期
        if datetime.utcnow() - self.last_optimization < self.optimization_cooldown:
            return {}
            
        try:
            # 1. 提取市场特征
            market_features = self._extract_market_features(decision_history)
            
            # 2. 识别当前市场状态
            regime, confidence = self.regime_identifier.identify_regime(market_features)
            
            # 3. 检查regime稳定性
            if not self._check_regime_stability(regime, decision_history):
                logger.info(f"Regime {regime} not stable enough for optimization")
                return {}
                
            # 4. 估计参数梯度
            gradients = self.gradient_learner.estimate_parameter_gradients(
                decision_history, performance_stats
            )
            
            # 5. 优化参数
            adjustments = self.parameter_optimizer.optimize_for_regime(
                regime, gradients, current_config
            )
            
            # 6. 生成学习决策
            learning_decision = self._create_learning_decision(
                regime, adjustments, gradients
            )
            
            # 7. 记录学习历史
            self._record_learning_history(learning_decision, performance_stats)
            
            # 8. 保存状态
            self._save_state()
            
            self.last_optimization = datetime.utcnow()
            
            return {
                'adjustments': adjustments,
                'regime': regime,
                'confidence': confidence,
                'learning_decision': learning_decision
            }
            
        except Exception as e:
            logger.error(f"Error in adaptive learning: {e}", exc_info=True)
            return {}
            
    def _extract_market_features(self, decision_history: List[Dict]) -> Dict[str, float]:
        """从决策历史提取市场特征"""
        features = {}
        
        # 聚合最近的市场指标
        recent_decisions = decision_history[-20:]  # 最近20个决策
        behavior_features = []
        # Gamma指标
        gamma_metrics = [d.get('gamma_metrics', {}) for d in recent_decisions]
        features['total_gamma_exposure'] = np.mean([
            m.get('total_gamma', 0) for m in gamma_metrics if m
        ])
        features['gamma_concentration'] = np.mean([
            m.get('concentration', 0) for m in gamma_metrics if m
        ])
        
        # 市场行为指标
        behavior_metrics = [d.get('behavior_metrics', {}) for d in recent_decisions]
        features['sweep_intensity'] = np.mean([
            m.get('sweep_count', 0) for m in behavior_metrics if m
        ])
        
        # 价格动量（从性能数据推断）
        features['price_momentum'] = self._calculate_price_momentum(decision_history)
        
        # 波动率（从市场快照推断）
        features['volatility'] = self._estimate_volatility(decision_history)
        
        # 其他特征
        features['volume_anomaly'] = 0  # 可从decision数据中提取
        features['spread_percentile'] = 0.5
        features['orderbook_imbalance'] = 0
        features['nearest_wall_distance'] = 10
        features['volatility_percentile'] = 0.5
        
        for d in recent_decisions:
            if 'behavior_metrics' in d and 'feature_matrix' in d['behavior_metrics']:
                behavior_features.append(d['behavior_metrics']['feature_matrix'])
        
        if behavior_features:
            # 聚合ML特征
            features['behavior_ml_features'] = {
                'sweep_anomaly_mean': np.mean([f[0] for f in behavior_features]),
                'sweep_volume_std': np.mean([f[1] for f in behavior_features]),
                'buy_ratio': np.mean([f[2] for f in behavior_features]),
                'divergence_strength_mean': np.mean([f[3] for f in behavior_features]),
                'divergence_max_duration': np.mean([f[4] for f in behavior_features]),
                'divergence_type_count': np.mean([f[5] for f in behavior_features])
            }
        
        return features
    
    def _calculate_price_momentum(self, decisions: List[Dict]) -> float:
        """计算价格动量"""
        # 从决策记录中提取价格信息
        prices = []
        for d in decisions:
            if 'market_snapshot' in d:
                snapshot = d['market_snapshot']
                if isinstance(snapshot, dict):
                    for asset_data in snapshot.values():
                        if 'price' in asset_data:
                            prices.append(asset_data['price'])
                            
        if len(prices) < 2:
            return 0
            
        # 简单动量计算
        returns = np.diff(prices) / prices[:-1]
        return np.mean(returns) * 100  # 转换为百分比
        
    def _estimate_volatility(self, decisions: List[Dict]) -> float:
        """估算波动率"""
        prices = []
        for d in decisions:
            if 'market_snapshot' in d:
                snapshot = d['market_snapshot']
                if isinstance(snapshot, dict):
                    for asset_data in snapshot.values():
                        if 'price' in asset_data:
                            prices.append(asset_data['price'])
                            
        if len(prices) < 3:
            return 0.01  # 默认1%
            
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)
        
    def _check_regime_stability(self, regime: str, decisions: List[Dict]) -> bool:
        """检查regime是否稳定"""
        # 计算最近在同一regime的决策数
        recent_regimes = []
        for d in decisions[-20:]:
            if 'regime' in d:
                recent_regimes.append(d['regime'])
                
        if not recent_regimes:
            return False
            
        # 计算当前regime的占比
        regime_ratio = recent_regimes.count(regime) / len(recent_regimes)
        
        return regime_ratio > 0.5  # 至少50%的决策在同一regime
        
    def _create_learning_decision(self, 
                                regime: str,
                                adjustments: Dict[str, float],
                                gradients: Dict[str, ParameterGradient]) -> LearningDecision:
        """创建学习决策记录"""
        # 计算预期改进
        expected_improvement = 0
        for param, adjustment in adjustments.items():
            if param in gradients:
                # 改进 = 梯度 * 调整量 * 置信度
                grad_info = gradients[param]
                improvement = grad_info.gradient * adjustment * grad_info.confidence
                expected_improvement += improvement
                
        # 确定决策基础
        if not adjustments:
            decision_basis = 'no_adjustment'
        elif any(abs(adj) > 0.1 for adj in adjustments.values()):
            decision_basis = 'gradient'
        else:
            decision_basis = 'exploration'
            
        return LearningDecision(
            timestamp=datetime.utcnow(),
            market_regime=regime,
            parameter_adjustments=adjustments,
            expected_improvement=expected_improvement,
            exploration_factor=self.parameter_optimizer.exploration_rate.get(regime, 0.2),
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
        
        # 保持历史长度
        if len(self.learning_history) > 1000:
            self.learning_history = self.learning_history[-1000:]
            
    def suggest_manual_review(self) -> Dict[str, Any]:
        """建议需要人工审查的参数"""
        suggestions = {
            'high_uncertainty_params': [],
            'conflicting_gradients': [],
            'exploration_candidates': [],
            'regime_specific_issues': {}
        }
        
        # 分析最近的学习历史
        if len(self.learning_history) < 10:
            return suggestions
            
        recent_history = self.learning_history[-10:]
        
        # 1. 识别高不确定性参数
        param_uncertainties = defaultdict(list)
        for record in recent_history:
            decision = record['decision']
            for param, adj in decision.parameter_adjustments.items():
                param_uncertainties[param].append(abs(adj))
                
        for param, adjustments in param_uncertainties.items():
            if np.std(adjustments) > 0.1:
                suggestions['high_uncertainty_params'].append({
                    'parameter': param,
                    'variance': np.std(adjustments),
                    'recent_adjustments': adjustments[-5:]
                })
                
        # 2. 识别冲突的梯度
        param_gradients = defaultdict(list)
        for record in recent_history:
            if hasattr(record, 'gradients'):
                for param, grad_info in record['gradients'].items():
                    param_gradients[param].append(grad_info.gradient)
                    
        for param, gradients in param_gradients.items():
            if len(gradients) > 3:
                # 检查梯度方向是否频繁改变
                direction_changes = sum(
                    1 for i in range(1, len(gradients))
                    if np.sign(gradients[i]) != np.sign(gradients[i-1])
                )
                if direction_changes > len(gradients) / 2:
                    suggestions['conflicting_gradients'].append({
                        'parameter': param,
                        'direction_changes': direction_changes,
                        'gradients': gradients[-5:]
                    })
                    
        return suggestions
        
    def get_learning_report(self) -> Dict[str, Any]:
        """生成学习报告"""
        report = {
            'total_learning_decisions': len(self.learning_history),
            'current_regime': self.current_regime,
            'regime_distribution': self._calculate_regime_distribution(),
            'parameter_evolution': self._analyze_parameter_evolution(),
            'learning_effectiveness': self._evaluate_learning_effectiveness(),
            'recommendations': self.suggest_manual_review()
        }
        
        return report
        
    def _calculate_regime_distribution(self) -> Dict[str, float]:
        """计算regime分布"""
        regime_counts = defaultdict(int)
        
        for record in self.learning_history:
            regime = record['decision'].market_regime
            regime_counts[regime] += 1
            
        total = sum(regime_counts.values())
        if total == 0:
            return {}
            
        return {
            regime: count / total 
            for regime, count in regime_counts.items()
        }
        
    def _analyze_parameter_evolution(self) -> Dict[str, Any]:
        """分析参数演化"""
        evolution = {}
        
        # 追踪每个参数的变化轨迹
        param_trajectories = defaultdict(list)
        
        for record in self.learning_history:
            for param, adj in record['decision'].parameter_adjustments.items():
                param_trajectories[param].append({
                    'timestamp': record['timestamp'],
                    'adjustment': adj,
                    'regime': record['decision'].market_regime
                })
                
        # 分析趋势
        for param, trajectory in param_trajectories.items():
            if len(trajectory) > 5:
                adjustments = [t['adjustment'] for t in trajectory]
                evolution[param] = {
                    'total_adjustments': len(trajectory),
                    'net_change': sum(adjustments),
                    'volatility': np.std(adjustments),
                    'trend': 'increasing' if sum(adjustments) > 0 else 'decreasing',
                    'recent_adjustments': adjustments[-5:]
                }
                
        return evolution
        
    def _evaluate_learning_effectiveness(self) -> Dict[str, float]:
        """评估学习效果"""
        if len(self.learning_history) < 2:
            return {'effectiveness': 0, 'confidence': 0}
            
        # 比较学习前后的性能
        improvements = []
        
        for i in range(1, min(len(self.learning_history), 10)):
            before = self.learning_history[i-1]['performance_before']
            after = self.learning_history[i]['performance_before']
            
            if 'composite_score' in before and 'composite_score' in after:
                improvement = after['composite_score'] - before['composite_score']
                improvements.append(improvement)
                
        if not improvements:
            return {'effectiveness': 0, 'confidence': 0}
            
        return {
            'effectiveness': np.mean(improvements),
            'confidence': min(len(improvements) / 10, 1.0),
            'improvement_rate': sum(1 for i in improvements if i > 0) / len(improvements)
        }
        
    def _save_state(self):
        """保存学习器状态"""
        state = {
            'regime_parameters': dict(self.regime_parameters),
            'learning_history': [
                {
                    'decision': {
                        'timestamp': rec['decision'].timestamp.isoformat(),
                        'market_regime': rec['decision'].market_regime,
                        'parameter_adjustments': rec['decision'].parameter_adjustments,
                        'expected_improvement': rec['decision'].expected_improvement
                    },
                    'performance_before': rec['performance_before'],
                    'timestamp': rec['timestamp'].isoformat()
                }
                for rec in self.learning_history[-100:]  # 只保存最近100条
            ],
            'current_regime': self.current_regime,
            'last_optimization': self.last_optimization.isoformat()
        }
        
        with open(self.config['state_file'], 'w') as f:
            json.dump(state, f, indent=2)
            
    def _load_state(self):
        """加载学习器状态"""
        if not os.path.exists(self.config['state_file']):
            return
            
        try:
            with open(self.config['state_file'], 'r') as f:
                state = json.load(f)
                
            self.regime_parameters = defaultdict(dict, state.get('regime_parameters', {}))
            self.current_regime = state.get('current_regime')
            
            if 'last_optimization' in state:
                self.last_optimization = datetime.fromisoformat(state['last_optimization'])
                
            # 简化历史加载，只保留关键信息
            logger.info(f"Loaded state with {len(state.get('learning_history', []))} history records")
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            
    def set_performance_tracker(self, tracker):
        """设置性能追踪器引用"""
        self.performance_tracker = tracker