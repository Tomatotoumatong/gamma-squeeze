"""
SignalEvaluator - 信号评估模块
用于gamma squeeze信号捕捉系统的信号生成层
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import copy
logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """交易信号数据结构"""
    timestamp: datetime
    asset: str
    signal_type: str
    direction: str
    strength: float
    confidence: float
    key_levels: List[float]
    expected_move: str
    time_horizon: str
    risk_factors: List[str]
    metadata: Dict[str, Any]

class SignalEvaluator:
    """支持自主学习的信号评估器"""
    
    def __init__(self, config: Optional[Dict] = None):
        # 基础配置
        self.base_config = self._default_config()
        
        # 当前运行配置（可被动态调整）
        self.config = copy.deepcopy(self.base_config)
        if config:
            self._update_config(config)
            
        # 参数历史追踪
        self.parameter_history = defaultdict(list)
        self.performance_history = []
        
        # 学习状态
        self.learning_enabled = True
        self.last_update = datetime.utcnow()
        
        # 初始化评分器
        self._init_scorers()
        
        # 信号历史
        self.signal_history = defaultdict(list)
        self.manual_overrides = {}
        
    def _default_config(self) -> Dict:
        """默认配置 - 作为基线"""
        return {
            'gamma_pressure': {
                'wall_proximity_weight': 0.3,
                'hedge_flow_weight': 0.3,
                'concentration_weight': 0.2,
                'dealer_position_weight': 0.2,
                'thresholds': {
                    'critical': 80,
                    'high': 60,
                    'medium': 40,
                    'low': 20
                }
            },
            'market_momentum': {
                'sweep_weight': 0.4,
                'divergence_weight': 0.3,
                'cross_market_weight': 0.3,
                'lookback_periods': [5, 10, 30],
                'momentum_decay': 0.95
            },
            'technical': {
                'support_resistance_weight': 0.4,
                'trend_alignment_weight': 0.3,
                'volume_profile_weight': 0.3,
                'key_level_tolerance': 0.005
            },
            'signal_generation': {
                'min_strength': 50,
                'min_confidence': 0.5,
                'signal_cooldown': 300,
                'risk_assessment': {
                    'low_liquidity_threshold': 0.3,
                    'weekend_penalty': 0.2,
                    'news_event_boost': 0.1
                }
            },
            'adaptive_learning': {
                'enabled': True,
                'learning_rate': 0.1,
                'parameter_bounds': {
                    # 阈值边界
                    'signal_generation.min_strength': (40, 70),
                    'signal_generation.min_confidence': (0.3, 0.8),
                    'gamma_pressure.thresholds.high': (50, 80),
                    'gamma_pressure.thresholds.critical': (70, 95),
                    # 权重边界
                    'gamma_pressure.wall_proximity_weight': (0.1, 0.5),
                    'gamma_pressure.hedge_flow_weight': (0.1, 0.5),
                    'market_momentum.sweep_weight': (0.2, 0.6),
                    'market_momentum.divergence_weight': (0.1, 0.5),
                }
            },
            'manual_scoring': {
                'enabled': True,
                'weight': 0.2
            }
        }
    
    def _init_scorers(self):
        """初始化评分器"""
        from SignalEvaluator import GammaPressureScorer, MarketMomentumScorer, TechnicalConfirmationScorer
        
        self.score_calculators = {
            'gamma_pressure': GammaPressureScorer(self.config['gamma_pressure']),
            'market_momentum': MarketMomentumScorer(self.config['market_momentum']),
            'technical': TechnicalConfirmationScorer(self.config['technical'])
        }
    
    def update_parameters(self, parameter_updates: Dict[str, Any], 
                         reason: str = "Manual update") -> bool:
        """
        动态更新参数
        
        Args:
            parameter_updates: 嵌套字典形式的参数更新
            reason: 更新原因
            
        Returns:
            bool: 更新是否成功
        """
        try:
            # 验证参数边界
            if not self._validate_parameters(parameter_updates):
                logger.error("Parameter updates violate bounds")
                return False
            
            # 记录原始配置
            old_config = copy.deepcopy(self.config)
            
            # 应用更新
            self._update_config(parameter_updates)
            
            # 重新初始化评分器
            self._init_scorers()
            
            # 记录参数变化
            self._log_parameter_change(old_config, parameter_updates, reason)
            
            logger.info(f"Parameters updated successfully: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update parameters: {e}")
            return False
    
    def _validate_parameters(self, updates: Dict[str, Any]) -> bool:
        """验证参数是否在允许范围内"""
        bounds = self.config['adaptive_learning']['parameter_bounds']
        
        for param_path, new_value in self._flatten_dict(updates).items():
            if param_path in bounds:
                min_val, max_val = bounds[param_path]
                if not (min_val <= new_value <= max_val):
                    logger.warning(f"Parameter {param_path}={new_value} outside bounds [{min_val}, {max_val}]")
                    return False
        return True
    
    def _update_config(self, updates: Dict[str, Any]):
        """递归更新配置"""
        def update_nested(target, source):
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    update_nested(target[key], value)
                else:
                    target[key] = value
        
        update_nested(self.config, updates)
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """扁平化嵌套字典"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _log_parameter_change(self, old_config: Dict, updates: Dict, reason: str):
        """记录参数变化"""
        timestamp = datetime.utcnow()
        
        change_record = {
            'timestamp': timestamp,
            'reason': reason,
            'changes': {},
            'config_snapshot': copy.deepcopy(self.config)
        }
        
        # 记录具体变化
        flat_updates = self._flatten_dict(updates)
        flat_old = self._flatten_dict(old_config)
        
        for param_path, new_value in flat_updates.items():
            old_value = flat_old.get(param_path, 'N/A')
            change_record['changes'][param_path] = {
                'old': old_value,
                'new': new_value
            }
        
        self.parameter_history[timestamp] = change_record
    
    def get_parameter_effectiveness(self) -> Dict[str, float]:
        """分析参数调整的有效性"""
        if len(self.performance_history) < 2:
            return {}
        
        effectiveness = {}
        
        # 分析最近的参数变化对性能的影响
        recent_changes = list(self.parameter_history.items())[-5:]  # 最近5次变化
        
        for timestamp, change_record in recent_changes:
            # 找到变化前后的性能数据
            before_perf = [p for p in self.performance_history if p['timestamp'] < timestamp]
            after_perf = [p for p in self.performance_history if p['timestamp'] > timestamp]
            
            if before_perf and after_perf:
                before_avg = np.mean([p['accuracy'] for p in before_perf[-10:]])
                after_avg = np.mean([p['accuracy'] for p in after_perf[:10]])
                
                improvement = after_avg - before_avg
                
                for param_path in change_record['changes']:
                    effectiveness[param_path] = effectiveness.get(param_path, 0) + improvement
        
        return effectiveness
    
    def suggest_parameter_adjustments(self) -> Dict[str, Any]:
        """基于历史表现建议参数调整"""
        if not self.learning_enabled:
            return {}
        
        suggestions = {}
        effectiveness = self.get_parameter_effectiveness()
        
        # 基于表现建议调整
        for param_path, effect in effectiveness.items():
            if param_path in self.config['adaptive_learning']['parameter_bounds']:
                bounds = self.config['adaptive_learning']['parameter_bounds'][param_path]
                current_value = self._get_nested_value(param_path)
                
                if effect < -0.05:  # 负面影响，尝试反向调整
                    if current_value > bounds[0]:
                        suggestions[param_path] = max(
                            current_value * 0.9,
                            bounds[0]
                        )
                elif effect > 0.05:  # 正面影响，继续同方向
                    if current_value < bounds[1]:
                        suggestions[param_path] = min(
                            current_value * 1.1,
                            bounds[1]
                        )
        
        return suggestions
    
    def _get_nested_value(self, path: str) -> Any:
        """获取嵌套配置值"""
        keys = path.split('.')
        value = self.config
        for key in keys:
            value = value[key]
        return value
    
    def evaluate(self, gamma_analysis: Dict[str, Any], 
                market_behavior: Dict[str, Any],
                market_data: pd.DataFrame) -> List[TradingSignal]:
        """主评估方法"""
        signals = []
        
        # 获取所有交易资产
        assets = set()
        if gamma_analysis.get('gamma_distribution'):
            assets.update(gamma_analysis['gamma_distribution'].keys())
        if market_behavior.get('anomaly_scores'):
            assets.update(market_behavior['anomaly_scores'].keys())
            
        # 对每个资产进行评估
        for asset in assets:
            # 1. 计算各维度评分
            scores = self._calculate_scores(
                asset, gamma_analysis, market_behavior, market_data
            )
            
            # 2. 检查是否应该生成信号
            if self._should_generate_signal(asset, scores):
                # 3. 生成信号
                signal = self._generate_signal(
                    asset, scores, gamma_analysis, market_behavior, market_data
                )
                if signal:
                    signals.append(signal)
                    self._update_signal_history(asset, signal)
                    
        return signals
    
    def _calculate_scores(self, asset: str, gamma_analysis: Dict,
                         market_behavior: Dict, market_data: pd.DataFrame) -> Dict[str, float]:
        """计算各维度评分"""
        scores = {}
        
        # 1. Gamma压力分数
        scores['gamma_pressure'] = self.score_calculators['gamma_pressure'].calculate(
            asset, gamma_analysis
        )
        
        # 2. 市场动量分数
        scores['market_momentum'] = self.score_calculators['market_momentum'].calculate(
            asset, market_behavior, market_data
        )
        
        # 3. 技术面确认分数
        scores['technical'] = self.score_calculators['technical'].calculate(
            asset, market_data
        )
        
        # 4. 应用手动评分（如果有）
        if self.config['manual_scoring']['enabled'] and asset in self.manual_overrides:
            manual_score = self.manual_overrides[asset]
            weight = self.config['manual_scoring']['weight']
            
            # 加权平均
            for key in scores:
                scores[key] = scores[key] * (1 - weight) + manual_score * weight
                
        return scores
    
    def _should_generate_signal(self, asset: str, scores: Dict[str, float]) -> bool:
        """判断是否应该生成信号"""
        # 检查冷却期
        if not self._check_cooldown(asset):
            return False
            
        # 计算综合分数
        composite_score = np.mean(list(scores.values()))
        
        # 检查最小阈值
        if composite_score < self.config['signal_generation']['min_strength']:
            return False
            
        # 至少有一个维度达到高分
        high_threshold = self.config['gamma_pressure']['thresholds']['high']
        if not any(score >= high_threshold for score in scores.values()):
            return False
            
        return True
    
    def _check_cooldown(self, asset: str) -> bool:
        """检查冷却期"""
        if asset not in self.signal_history:
            return True
            
        if not self.signal_history[asset]:
            return True
            
        last_signal = self.signal_history[asset][-1]
        cooldown = self.config['signal_generation']['signal_cooldown']
        
        return (datetime.utcnow() - last_signal.timestamp).total_seconds() > cooldown
    
    def _generate_signal(self, asset: str, scores: Dict[str, float],
                        gamma_analysis: Dict, market_behavior: Dict,
                        market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """生成交易信号"""
        # 确定信号方向
        direction = self._determine_direction(
            asset, gamma_analysis, market_behavior, market_data
        )
        
        if not direction:
            return None
        
        # 计算信号强度和置信度
        strength = self._calculate_signal_strength(scores)
        confidence = self._calculate_confidence(scores, market_behavior)
        min_confidence = self.config['signal_generation']['min_confidence']
        
        if confidence < min_confidence:
            return None
        # 识别关键价位
        key_levels = self._identify_key_levels(
            asset, gamma_analysis, market_data
        )
        
        # 估算预期波动
        expected_move = self._estimate_expected_move(
            asset, gamma_analysis, strength
        )
        
        # 确定时间范围
        time_horizon = self._determine_time_horizon(
            asset, market_behavior, strength
        )
        
        # 评估风险因素
        risk_factors = self._assess_risk_factors(
            asset, market_behavior, market_data
        )
        
        # 构建信号
        signal = TradingSignal(
            timestamp=datetime.utcnow(),
            asset=asset,
            signal_type=self._determine_signal_type(scores),
            direction=direction,
            strength=strength,
            confidence=confidence,
            key_levels=key_levels,
            expected_move=expected_move,
            time_horizon=time_horizon,
            risk_factors=risk_factors,
            metadata={
                'scores': scores,
                'gamma_metrics': self._extract_gamma_metrics(asset, gamma_analysis),
                'market_metrics': self._extract_market_metrics(asset, market_behavior),
                'config_used': copy.deepcopy(self.config),
                'adaptive_features': {
                    'parameter_version': len(self.parameter_history),
                    'learning_enabled': self.learning_enabled
                }
            }
        )
        
        return signal
    
    def _determine_direction(self, asset: str, gamma_analysis: Dict,
                           market_behavior: Dict, market_data: pd.DataFrame) -> Optional[str]:
        """确定信号方向"""
        direction_votes = []
        
        # 基于gamma墙位置
        gamma_dist = gamma_analysis.get('gamma_distribution', {}).get(asset, {})
        if gamma_dist:
            walls = gamma_analysis.get('gamma_walls', [])
            asset_walls = [w for w in walls if w.strike]  # 简化处理
            
            if asset_walls:
                # 获取当前价格
                asset_data = market_data[market_data['symbol'] == asset]
                if not asset_data.empty:
                    current_price = asset_data.iloc[-1]['price']
                    
                    # 找到上下方最近的墙
                    upper_walls = [w for w in asset_walls if w.strike > current_price]
                    lower_walls = [w for w in asset_walls if w.strike < current_price]
                    
                    if upper_walls and lower_walls:
                        upper_distance = min(w.strike - current_price for w in upper_walls)
                        lower_distance = current_price - max(w.strike for w in lower_walls)
                        
                        # 距离下方墙更近，可能反弹
                        if lower_distance < upper_distance * 0.5:
                            direction_votes.append('BULLISH')
                        # 距离上方墙更近，可能回调
                        elif upper_distance < lower_distance * 0.5:
                            direction_votes.append('BEARISH')
                            
        # 基于对冲流方向
        hedge_flows = gamma_analysis.get('hedge_flows', [])
        buy_flows = sum(1 for f in hedge_flows if f.direction == 'buy')
        sell_flows = sum(1 for f in hedge_flows if f.direction == 'sell')
        
        if buy_flows > sell_flows * 1.5:
            direction_votes.append('BULLISH')
        elif sell_flows > buy_flows * 1.5:
            direction_votes.append('BEARISH')
            
        # 基于市场动量
        sweeps = market_behavior.get('sweep_orders', [])
        asset_sweeps = [s for s in sweeps if s.symbol == asset]
        
        buy_sweeps = sum(1 for s in asset_sweeps if s.side == 'buy')
        sell_sweeps = sum(1 for s in asset_sweeps if s.side == 'sell')
        
        if buy_sweeps > sell_sweeps:
            direction_votes.append('BULLISH')
        elif sell_sweeps > buy_sweeps:
            direction_votes.append('BEARISH')
            
        # 投票决定
        if not direction_votes:
            return None
            
        bullish_votes = sum(1 for v in direction_votes if v == 'BULLISH')
        bearish_votes = sum(1 for v in direction_votes if v == 'BEARISH')
        
        if bullish_votes > bearish_votes:
            return 'BULLISH'
        elif bearish_votes > bullish_votes:
            return 'BEARISH'
        else:
            return None
    
    def _calculate_signal_strength(self, scores: Dict[str, float]) -> float:
        """计算信号强度"""
        # 加权平均，gamma压力权重最高
        weights = {
            'gamma_pressure': 0.5,
            'market_momentum': 0.3,
            'technical': 0.2
        }
        
        weighted_sum = sum(scores.get(k, 0) * v for k, v in weights.items())
        return round(weighted_sum, 1)
    
    def _calculate_confidence(self, scores: Dict[str, float],
                            market_behavior: Dict) -> float:
        """计算置信度"""
        confidence_factors = []
        
        # 1. 各维度一致性
        score_std = np.std(list(scores.values()))
        consistency = 1 - (score_std / 50)  # 标准差越小，一致性越高
        confidence_factors.append(max(0, consistency))
        
        # 2. 市场状态清晰度
        regime = market_behavior.get('market_regime', {})
        if regime.get('confidence', 0) > 0:
            confidence_factors.append(regime['confidence'])
            
        # 3. 异常程度
        anomaly_scores = market_behavior.get('anomaly_scores', {})
        if anomaly_scores:
            avg_anomaly = np.mean(list(anomaly_scores.values()))
            confidence_factors.append(avg_anomaly)
            
        return round(np.mean(confidence_factors), 2) if confidence_factors else 0.5
    
    def _identify_key_levels(self, asset: str, gamma_analysis: Dict,
                           market_data: pd.DataFrame) -> List[float]:
        """识别关键价位"""
        key_levels = []
        
        # 1. Gamma墙位置
        walls = gamma_analysis.get('gamma_walls', [])
        for wall in walls:
            if hasattr(wall, 'strike'):
                key_levels.append(wall.strike)
                
        # 2. 近期高低点
        asset_data = market_data[market_data['symbol'] == asset]
        if not asset_data.empty and 'price' in asset_data.columns:
            prices = asset_data['price'].values
            if len(prices) > 20:
                key_levels.append(float(np.max(prices[-20:])))
                key_levels.append(float(np.min(prices[-20:])))
                
        # 去重并排序
        key_levels = sorted(list(set(key_levels)))
        
        # 保留最重要的5个
        return key_levels[:5]
    
    def _estimate_expected_move(self, asset: str, gamma_analysis: Dict,
                              strength: float) -> str:
        """估算预期波动"""
        # 基于gamma敞口和信号强度
        gamma_dist = gamma_analysis.get('gamma_distribution', {}).get(asset, {})
        
        base_move = 1.0  # 基础波动1%
        
        # 信号强度调整
        strength_multiplier = strength / 50  # 50分对应1倍
        
        # Gamma集中度调整
        if gamma_dist:
            concentration = gamma_dist.get('concentration', 0)
            gamma_multiplier = 1 + concentration
        else:
            gamma_multiplier = 1
            
        expected_pct = base_move * strength_multiplier * gamma_multiplier
        
        # 分档返回
        if expected_pct < 2:
            return "1-2%"
        elif expected_pct < 5:
            return "2-5%"
        elif expected_pct < 10:
            return "5-10%"
        else:
            return "10%+"
    
    def _determine_time_horizon(self, asset: str, market_behavior: Dict,
                              strength: float) -> str:
        """确定时间范围"""
        # 基于市场状态和信号强度
        regime = market_behavior.get('market_regime', {})
        state = regime.get('state', 'normal')
        
        # 基础时间范围
        if state == 'squeeze':
            base_hours = 4
        elif state == 'breakout':
            base_hours = 2
        else:
            base_hours = 6
            
        # 强度调整
        if strength > 80:
            time_multiplier = 0.5
        elif strength > 60:
            time_multiplier = 0.75
        else:
            time_multiplier = 1.0
            
        hours = base_hours * time_multiplier
        
        # 返回时间范围
        if hours < 2:
            return "0-2h"
        elif hours < 4:
            return "2-4h"
        elif hours < 8:
            return "4-8h"
        else:
            return "8-24h"
    
    def _assess_risk_factors(self, asset: str, market_behavior: Dict,
                           market_data: pd.DataFrame) -> List[str]:
        """评估风险因素"""
        risk_factors = []
        
        # 1. 流动性风险
        asset_data = market_data[market_data['symbol'] == asset]
        if not asset_data.empty and 'volume' in asset_data.columns:
            recent_volume = asset_data['volume'].tail(10).mean()
            avg_volume = asset_data['volume'].mean()
            
            if recent_volume < avg_volume * self.config['signal_generation']['risk_assessment']['low_liquidity_threshold']:
                risk_factors.append("低流动性")
                
        # 2. 时间风险
        now = datetime.utcnow()
        if now.weekday() >= 5:  # 周末
            risk_factors.append("周末效应")
        elif now.hour < 8 or now.hour > 20:  # 非主要交易时段
            risk_factors.append("非主要交易时段")
            
        # 3. 市场状态风险
        regime = market_behavior.get('market_regime', {})
        if regime.get('state') == 'consolidation':
            risk_factors.append("区间震荡")
            
        # 4. 背离风险
        divergences = market_behavior.get('divergences', [])
        asset_divs = [d for d in divergences if d.symbol == asset]
        if asset_divs:
            risk_factors.append("存在背离信号")
            
        return risk_factors[:3]  # 最多返回3个
    
    def _determine_signal_type(self, scores: Dict[str, float]) -> str:
        """确定信号类型"""
        # 找出最高分的维度
        max_dimension = max(scores.items(), key=lambda x: x[1])[0]
        
        if max_dimension == 'gamma_pressure':
            return "GAMMA_SQUEEZE"
        elif max_dimension == 'market_momentum':
            return "MOMENTUM_BREAKOUT"
        else:
            return "TECHNICAL_SETUP"
    
    def _extract_gamma_metrics(self, asset: str, gamma_analysis: Dict) -> Dict:
        """提取gamma相关指标"""
        metrics = {}
        
        gamma_dist = gamma_analysis.get('gamma_distribution', {}).get(asset, {})
        if gamma_dist:
            metrics['total_gamma_exposure'] = gamma_dist.get('total_exposure', 0)
            metrics['net_gamma_exposure'] = gamma_dist.get('net_exposure', 0)
            metrics['gamma_concentration'] = gamma_dist.get('concentration', 0)
            
        dealer_pos = gamma_analysis.get('dealer_position', {}).get(asset, {})
        if dealer_pos:
            metrics['dealer_net_delta'] = dealer_pos.get('net_delta', 0)
            metrics['dealer_position_score'] = dealer_pos.get('position_score', 0)
            
        return metrics
    
    def _extract_market_metrics(self, asset: str, market_behavior: Dict) -> Dict:
        """提取市场相关指标"""
        metrics = {}
        
        # 扫单统计
        sweeps = market_behavior.get('sweep_orders', [])
        asset_sweeps = [s for s in sweeps if s.symbol == asset]
        metrics['sweep_count'] = len(asset_sweeps)
        metrics['sweep_intensity'] = np.mean([s.anomaly_score for s in asset_sweeps]) if asset_sweeps else 0
        
        # 异常分数
        anomaly_scores = market_behavior.get('anomaly_scores', {})
        metrics['anomaly_score'] = anomaly_scores.get(asset, 0)
        
        return metrics
    
    def _update_signal_history(self, asset: str, signal: TradingSignal):
        """更新信号历史"""
        self.signal_history[asset].append(signal)
        
        # 保持历史长度
        max_history = 100
        if len(self.signal_history[asset]) > max_history:
            self.signal_history[asset] = self.signal_history[asset][-max_history:]
    
    def set_manual_score(self, asset: str, score: float):
        """设置手动评分"""
        if 0 <= score <= 100:
            self.manual_overrides[asset] = score
            logger.info(f"Manual score set for {asset}: {score}")
        else:
            logger.error(f"Invalid manual score: {score}. Must be between 0 and 100.")
    
    def clear_manual_scores(self):
        """清除所有手动评分"""
        self.manual_overrides.clear()
        logger.info("All manual scores cleared")
    
    def get_signal_history(self, asset: Optional[str] = None) -> List[TradingSignal]:
        """获取信号历史"""
        if asset:
            return self.signal_history.get(asset, [])
        else:
            all_signals = []
            for signals in self.signal_history.values():
                all_signals.extend(signals)
            return sorted(all_signals, key=lambda s: s.timestamp, reverse=True)
        
    def record_performance(self, accuracy: float, metadata: Dict[str, Any] = None):
        """记录性能表现"""
        performance_record = {
            'timestamp': datetime.utcnow(),
            'accuracy': accuracy,
            'config_snapshot': copy.deepcopy(self.config),
            'metadata': metadata or {}
        }
        
        self.performance_history.append(performance_record)
        
        # 保持历史长度
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_adaptation_report(self) -> Dict[str, Any]:
        """生成自适应报告"""
        return {
            'parameter_changes': len(self.parameter_history),
            'current_config': self.config,
            'base_config': self.base_config,
            'performance_trend': self._calculate_performance_trend(),
            'parameter_effectiveness': self.get_parameter_effectiveness(),
            'suggested_adjustments': self.suggest_parameter_adjustments(),
            'learning_status': {
                'enabled': self.learning_enabled,
                'last_update': self.last_update,
                'total_signals': sum(len(signals) for signals in self.signal_history.values())
            }
        }
    
    def _calculate_performance_trend(self) -> Dict[str, float]:
        """计算性能趋势"""
        if len(self.performance_history) < 10:
            return {'trend': 0.0, 'confidence': 0.0}
        
        recent_accuracy = [p['accuracy'] for p in self.performance_history[-10:]]
        older_accuracy = [p['accuracy'] for p in self.performance_history[-20:-10]] if len(self.performance_history) >= 20 else []
        
        if not older_accuracy:
            return {'trend': 0.0, 'confidence': 0.0}
        
        recent_avg = np.mean(recent_accuracy)
        older_avg = np.mean(older_accuracy)
        
        trend = recent_avg - older_avg
        confidence = min(len(self.performance_history) / 100, 1.0)
        
        return {'trend': trend, 'confidence': confidence}


class GammaPressureScorer:
    """Gamma压力评分器"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def calculate(self, asset: str, gamma_analysis: Dict) -> float:
        """计算gamma压力分数"""
        score_components = []
        
        # 1. Gamma墙接近度
        wall_score = self._calculate_wall_proximity_score(asset, gamma_analysis)
        score_components.append(wall_score * self.config['wall_proximity_weight'])
        
        # 2. 对冲流强度
        hedge_score = self._calculate_hedge_flow_score(asset, gamma_analysis)
        score_components.append(hedge_score * self.config['hedge_flow_weight'])
        
        # 3. Gamma集中度
        concentration_score = self._calculate_concentration_score(asset, gamma_analysis)
        score_components.append(concentration_score * self.config['concentration_weight'])
        
        # 4. 做市商头寸
        dealer_score = self._calculate_dealer_position_score(asset, gamma_analysis)
        score_components.append(dealer_score * self.config['dealer_position_weight'])
        
        return min(sum(score_components), 100)
    
    def _calculate_wall_proximity_score(self, asset: str, gamma_analysis: Dict) -> float:
        """计算gamma墙接近度分数"""
        indicators = gamma_analysis.get('pressure_indicators', {}).get(asset, {})
        
        if not indicators:
            return 0
            
        nearest_distance = indicators.get('nearest_wall_distance')
        if nearest_distance is None:
            return 0
            
        # 距离越近，分数越高
        if nearest_distance < 0.5:  # 0.5%以内
            return 100
        elif nearest_distance < 1.0:  # 1%以内
            return 80
        elif nearest_distance < 2.0:  # 2%以内
            return 60
        elif nearest_distance < 5.0:  # 5%以内
            return 40
        else:
            return 20
    
    def _calculate_hedge_flow_score(self, asset: str, gamma_analysis: Dict) -> float:
        """计算对冲流强度分数"""
        hedge_flows = gamma_analysis.get('hedge_flows', [])
        
        if not hedge_flows:
            return 0
            
        # 统计强度
        avg_intensity = np.mean([f.intensity for f in hedge_flows])
        
        # 转换为0-100分数
        return min(avg_intensity * 100, 100)
    
    def _calculate_concentration_score(self, asset: str, gamma_analysis: Dict) -> float:
        """计算gamma集中度分数"""
        gamma_dist = gamma_analysis.get('gamma_distribution', {}).get(asset, {})
        
        if not gamma_dist:
            return 0
            
        concentration = gamma_dist.get('concentration', 0)
        
        # 集中度越高，分数越高
        return min(concentration * 100, 100)
    
    def _calculate_dealer_position_score(self, asset: str, gamma_analysis: Dict) -> float:
        """计算做市商头寸分数"""
        dealer_pos = gamma_analysis.get('dealer_position', {}).get(asset, {})
        
        if not dealer_pos:
            return 0
            
        # 净空头程度
        position_score = abs(dealer_pos.get('position_score', 0))
        
        # 流动性失衡
        flow_imbalance = abs(dealer_pos.get('flow_imbalance', 0))
        
        # 综合评分
        return min((position_score * 50 + flow_imbalance * 50), 100)


class MarketMomentumScorer:
    """市场动量评分器"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def calculate(self, asset: str, market_behavior: Dict, 
                 market_data: pd.DataFrame) -> float:
        """计算市场动量分数"""
        score_components = []
        
        # 1. 扫单强度
        sweep_score = self._calculate_sweep_score(asset, market_behavior)
        score_components.append(sweep_score * self.config['sweep_weight'])
        
        # 2. 背离信号
        divergence_score = self._calculate_divergence_score(asset, market_behavior)
        score_components.append(divergence_score * self.config['divergence_weight'])
        
        # 3. 跨市场联动
        cross_market_score = self._calculate_cross_market_score(asset, market_behavior)
        score_components.append(cross_market_score * self.config['cross_market_weight'])
        
        return min(sum(score_components), 100)
    
    def _calculate_sweep_score(self, asset: str, market_behavior: Dict) -> float:
        """计算扫单分数"""
        sweeps = market_behavior.get('sweep_orders', [])
        asset_sweeps = [s for s in sweeps if s.symbol == asset]
        
        if not asset_sweeps:
            return 0
            
        # 考虑频率和强度
        recent_count = len(asset_sweeps)
        avg_anomaly = np.mean([s.anomaly_score for s in asset_sweeps])
        
        # 频率分数（最近10次扫单满分）
        frequency_score = min(recent_count / 10 * 100, 100)
        
        # 强度分数
        intensity_score = avg_anomaly * 100
        
        return (frequency_score + intensity_score) / 2
    
    def _calculate_divergence_score(self, asset: str, market_behavior: Dict) -> float:
        """计算背离分数"""
        divergences = market_behavior.get('divergences', [])
        asset_divs = [d for d in divergences if d.symbol == asset]
        
        if not asset_divs:
            return 0
            
        # 最强背离
        max_strength = max(d.strength for d in asset_divs)
        
        # 持续时间
        max_duration = max(d.duration for d in asset_divs)
        duration_score = min(max_duration / 10 * 100, 100)
        
        return (max_strength * 100 + duration_score) / 2
    
    def _calculate_cross_market_score(self, asset: str, market_behavior: Dict) -> float:
        """计算跨市场联动分数"""
        cross_signals = market_behavior.get('cross_market_signals', [])
        
        # 作为领先市场的信号
        lead_signals = [s for s in cross_signals if s.lead_market == asset]
        
        if not lead_signals:
            return 0
            
        # 平均信号强度
        avg_strength = np.mean([s.signal_strength for s in lead_signals])
        
        return min(avg_strength * 100, 100)


class TechnicalConfirmationScorer:
    """技术面确认评分器"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def calculate(self, asset: str, market_data: pd.DataFrame) -> float:
        """计算技术面确认分数"""
        asset_data = market_data[
            (market_data['symbol'] == asset) & 
            (market_data['data_type'] == 'spot')
        ]
        if asset_data.empty or len(asset_data) < 20:
            return 0
            
        score_components = []
        
        # 1. 支撑阻力位
        sr_score = self._calculate_support_resistance_score(asset_data)
        score_components.append(sr_score * self.config['support_resistance_weight'])
        
        # 2. 趋势一致性
        trend_score = self._calculate_trend_alignment_score(asset_data)
        score_components.append(trend_score * self.config['trend_alignment_weight'])
        
        # 3. 成交量形态
        volume_score = self._calculate_volume_profile_score(asset_data)
        score_components.append(volume_score * self.config['volume_profile_weight'])
        
        return min(sum(score_components), 100)
    
    def _calculate_support_resistance_score(self, data: pd.DataFrame) -> float:
        """计算支撑阻力位分数"""
        prices = data['price'].values
        current_price = prices[-1]
        
        # 识别近期高低点
        highs = []
        lows = []
        
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                highs.append(prices[i])
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                lows.append(prices[i])
                
        if not highs or not lows:
            return 50  # 中性分数
            
        # 检查是否接近关键位
        tolerance = self.config['key_level_tolerance']
        
        # 接近支撑位
        nearest_support = max([l for l in lows if l < current_price], default=0)
        if nearest_support and (current_price - nearest_support) / current_price < tolerance:
            return 80  # 接近支撑，看涨
            
        # 接近阻力位
        nearest_resistance = min([h for h in highs if h > current_price], default=float('inf'))
        if nearest_resistance != float('inf') and (nearest_resistance - current_price) / current_price < tolerance:
            return 20  # 接近阻力，看跌
            
        return 50
    
    def _calculate_trend_alignment_score(self, data: pd.DataFrame) -> float:
        """计算趋势一致性分数"""
        prices = data['price'].values
        # print(prices)
        # 计算不同周期的趋势
        trends = []
        for period in [5, 10, 20]:
            if len(prices) >= period:
                trend = (prices[-1] - prices[-period]) / prices[-period]
                trends.append(trend)
                
        if not trends:
            return 50
            
        # 所有趋势同向得高分
        if all(t > 0 for t in trends):
            return 80  # 上升趋势
        elif all(t < 0 for t in trends):
            return 80  # 下降趋势（也是确认）
        else:
            return 40  # 趋势不一致
    
    def _calculate_volume_profile_score(self, data: pd.DataFrame) -> float:
        """计算成交量形态分数"""
        if 'volume' not in data.columns:
            return 50
            
        volumes = data['volume'].values
        prices = data['price'].values
        # print(volumes)
        # print(prices)
        if len(volumes) < 5:
            return 50
            
        # 价量配合
        recent_price_change = (prices[-1] - prices[-5]) / prices[-5]
        recent_volume_change = (volumes[-1] - np.mean(volumes[-5:])) / np.mean(volumes[-5:])
        
        # 价涨量增或价跌量增都是确认信号
        if (recent_price_change > 0 and recent_volume_change > 0) or \
           (recent_price_change < 0 and recent_volume_change > 0):
            return 80
        else:
            return 40