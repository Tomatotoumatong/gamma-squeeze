"""
GammaPressureAnalyzer - Gamma压力分析模块
用于gamma squeeze信号捕捉系统的模式识别层
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy.stats import linregress

logger = logging.getLogger(__name__)

@dataclass
class GammaWall:
    """Gamma墙数据结构"""
    strike: float
    gamma_exposure: float
    position: str  # 'above' or 'below'
    distance_pct: float
    strength: float  # 0-1标准化强度

@dataclass
class HedgeFlow:
    """对冲流数据结构"""
    direction: str  # 'buy' or 'sell'
    intensity: float  # 0-1标准化强度
    trigger_price: float
    estimated_volume: float

class GammaPressureAnalyzer:
    """Gamma压力分析器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.gamma_history = defaultdict(list)
        self.price_history = defaultdict(list)
        self.dealer_position_tracker = DealerPositionTracker()
        
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'wall_percentile': 90,  # Gamma墙识别阈值百分位
            'history_window': 100,  # 历史窗口大小
            'gamma_decay_factor': 0.95,  # 历史gamma衰减因子
            'hedge_flow_threshold': 0.7,  # 对冲流触发阈值
            'learning_params': {  # 预留学习参数
                'adaptive_threshold': True,
                'feature_extraction': 'auto'
            }
        }
    
    def analyze(self, option_data: pd.DataFrame, spot_data: pd.DataFrame) -> Dict[str, Any]:
        """主分析方法"""
        result = {
            'timestamp': datetime.utcnow(),
            'gamma_distribution': {},
            'gamma_walls': [],
            'dealer_position': {},
            'hedge_flows': [],
            'pressure_indicators': {},
            'raw_data': {}
        }
        
        # 按标的资产分组分析
        for symbol in option_data['symbol'].unique():
            symbol_options = option_data[option_data['symbol'] == symbol]
            symbol_spot = spot_data[spot_data['symbol'] == symbol]
            
            if symbol_spot.empty:
                continue
                
            spot_price = symbol_spot.iloc[-1]['price']
            
            # 1. 计算Gamma分布
            gamma_dist = self._calculate_gamma_distribution(symbol_options, spot_price)
            result['gamma_distribution'][symbol] = gamma_dist
            
            # 2. 识别Gamma墙
            walls = self._identify_gamma_walls(gamma_dist, spot_price)
            result['gamma_walls'].extend(walls)
            
            # 3. 估算做市商头寸
            dealer_pos = self.dealer_position_tracker.estimate_position(
                symbol_options, spot_price, gamma_dist
            )
            result['dealer_position'][symbol] = dealer_pos
            
            # 4. 计算对冲流
            hedge_flows = self._calculate_hedge_flows(
                gamma_dist, dealer_pos, spot_price, symbol
            )
            result['hedge_flows'].extend(hedge_flows)
            
            # 5. 计算压力指标
            indicators = self._calculate_pressure_indicators(
                gamma_dist, walls, spot_price, symbol
            )
            result['pressure_indicators'][symbol] = indicators
            
            # 6. 保存原始数据供学习使用
            result['raw_data'][symbol] = {
                'gamma_profile': gamma_dist,
                'option_snapshot': symbol_options.to_dict('records'),
                'spot_snapshot': symbol_spot.to_dict('records')
            }
            
        return result
    
    def _calculate_gamma_distribution(self, options: pd.DataFrame, spot: float) -> Dict[str, Any]:
        """计算Gamma分布"""
        # 按行权价分组
        gamma_by_strike = defaultdict(float)
        
        for _, opt in options.iterrows():
            strike = opt.get('strike')
            if not strike:
                continue
                
            # 计算单位gamma
            unit_gamma = self._calculate_unit_gamma(
                spot, strike, 
                opt.get('expiry', 0),
                opt.get('iv', 0.5),
                opt.get('type', 'call')
            )
            
            price_normalized_gamma = unit_gamma * spot / 100
            
            # 加权gamma = 归一化gamma * 未平仓量
            weighted_gamma = price_normalized_gamma * opt.get('open_interest', 0)
            
            # 考虑做市商通常是期权的净卖方
            gamma_by_strike[strike] -= weighted_gamma
            
        # 转换为列表并排序
        gamma_profile = sorted([
            {'strike': k, 'gamma_exposure': v}
            for k, v in gamma_by_strike.items()
        ], key=lambda x: x['strike'])
        
        # 计算总gamma敞口
        total_gamma = sum(abs(g['gamma_exposure']) for g in gamma_profile)
        
        # 归一化gamma值
        if total_gamma > 0:
            for g in gamma_profile:
                g['normalized_gamma'] = g['gamma_exposure'] / total_gamma
                
        return {
            'profile': gamma_profile,
            'total_exposure': total_gamma,
            'net_exposure': sum(g['gamma_exposure'] for g in gamma_profile),
            'concentration': self._calculate_concentration(gamma_profile)
        }
    
    def _calculate_unit_gamma(self, S: float, K: float, T: float, 
                             sigma: float, option_type: str) -> float:
        """计算单位Gamma"""
        try:
            from scipy.stats import norm
            import time
            current_time_ms = time.time() * 1000
            remaining_time_ms = T - current_time_ms
            T_years = remaining_time_ms / (365 * 24 * 60 * 60 * 1000)
            
            if T_years <= 0 or sigma <= 0:
                return 0.0
                
            d1 = (np.log(S / K) + 0.5 * sigma ** 2 * T_years) / (sigma * np.sqrt(T_years))
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T_years))
            
            return gamma
        except:
            return 0.0
    
    def _identify_gamma_walls(self, gamma_dist: Dict, spot: float) -> List[GammaWall]:
        """识别Gamma墙"""
        walls = []
        profile = gamma_dist['profile']
        
        if not profile:
            return walls
            
        # 计算gamma绝对值用于识别高密度区域
        gamma_abs = [abs(g['gamma_exposure']) for g in profile]
        threshold = np.percentile(gamma_abs, self.config['wall_percentile'])
        
        # 识别超过阈值的行权价
        for item in profile:
            if abs(item['gamma_exposure']) >= threshold:
                strike = item['strike']
                distance_pct = (strike - spot) / spot * 100
                
                wall = GammaWall(
                    strike=strike,
                    gamma_exposure=item['gamma_exposure'],
                    position='above' if strike > spot else 'below',
                    distance_pct=abs(distance_pct),
                    strength=min(abs(item['gamma_exposure']) / max(gamma_abs), 1.0)
                )
                walls.append(wall)
                
        # 按距离排序
        walls.sort(key=lambda x: x.distance_pct)
        
        return walls
    
    def _calculate_hedge_flows(self, gamma_dist: Dict, dealer_pos: Dict, 
                              spot: float, symbol: str) -> List[HedgeFlow]:
        """计算潜在对冲流"""
        flows = []
        profile = gamma_dist['profile']
        
        if not profile:
            return flows
            
        # 获取最近的gamma墙
        strikes = [g['strike'] for g in profile]
        gammas = [g['gamma_exposure'] for g in profile]
        
        # 创建插值函数
        if len(strikes) > 2:
            gamma_func = interp1d(strikes, gammas, kind='linear', 
                                 bounds_error=False, fill_value=0)
            
            # 计算当前位置的gamma梯度
            delta_price = spot * 0.001  # 0.1%价格变动
            gamma_gradient = (gamma_func(spot + delta_price) - 
                            gamma_func(spot - delta_price)) / (2 * delta_price)
            
            # 估算对冲需求
            if abs(gamma_gradient) > self.config['hedge_flow_threshold']:
                # 价格上涨时的对冲流
                if gamma_gradient < 0:  # 负gamma，价格上涨需要买入对冲
                    flow = HedgeFlow(
                        direction='buy',
                        intensity=min(abs(gamma_gradient), 1.0),
                        trigger_price=spot * 1.005,  # 0.5%触发点
                        estimated_volume=abs(dealer_pos.get('net_delta', 0) * gamma_gradient * spot * 0.01)
                    )
                    flows.append(flow)
                else:  # 正gamma，价格上涨需要卖出对冲
                    flow = HedgeFlow(
                        direction='sell',
                        intensity=min(abs(gamma_gradient), 1.0),
                        trigger_price=spot * 1.005,
                        estimated_volume=abs(dealer_pos.get('net_delta', 0) * gamma_gradient * spot * 0.01)
                    )
                    flows.append(flow)
                    
        return flows
    
    def _calculate_pressure_indicators(self, gamma_dist: Dict, walls: List[GammaWall],
                                     spot: float, symbol: str) -> Dict[str, float]:
        """计算压力指标"""
        indicators = {
            'nearest_wall_distance': None,
            'gamma_change_rate': 0.0,
            'hedge_pressure': 0.0,
            'gamma_concentration': gamma_dist.get('concentration', 0),
            'adaptive_features': {}  # 预留给学习模块
        }
        
        # 1. 最近Gamma墙距离
        if walls:
            indicators['nearest_wall_distance'] = walls[0].distance_pct
            
        # 2. Gamma变化率
        self._update_gamma_history(symbol, gamma_dist['total_exposure'])
        indicators['gamma_change_rate'] = self._calculate_gamma_change_rate(symbol)
        
        # 3. 对冲压力综合指标
        indicators['hedge_pressure'] = self._calculate_hedge_pressure(
            gamma_dist, walls, spot
        )
        
        # 4. 自适应特征提取（为学习模块预留）
        feature_mode = self.config.get('learning_params', {}).get('feature_extraction', 'manual')
        if feature_mode == 'auto':
            indicators['adaptive_features'] = self._extract_adaptive_features(
                gamma_dist, walls, spot
            )
            
        return indicators
    
    def _update_gamma_history(self, symbol: str, total_gamma: float):
        """更新gamma历史"""
        self.gamma_history[symbol].append({
            'timestamp': datetime.utcnow(),
            'value': total_gamma
        })
        
        # 保持窗口大小
        if len(self.gamma_history[symbol]) > self.config['history_window']:
            self.gamma_history[symbol].pop(0)
    
    def _calculate_gamma_change_rate(self, symbol: str) -> float:
        """计算gamma变化率"""
        history = self.gamma_history[symbol]
        if len(history) < 3:
            return 0.0
            
        # 提取时间序列
        times = [(h['timestamp'] - history[0]['timestamp']).total_seconds() 
                for h in history]
        values = [h['value'] for h in history]
        
        # 线性回归计算斜率
        if len(times) > 1:
            slope, _, _, _, _ = linregress(times, values)
            return slope
        return 0.0
    
    def _calculate_concentration(self, profile: List[Dict]) -> float:
        """计算gamma集中度（基尼系数）"""
        if not profile:
            return 0.0
            
        gammas = sorted([abs(g['gamma_exposure']) for g in profile])
        n = len(gammas)
        
        if n == 0 or sum(gammas) == 0:
            return 0.0
            
        # 计算基尼系数
        cumsum = np.cumsum(gammas)
        return (2 * sum((i + 1) * g for i, g in enumerate(gammas))) / (n * sum(gammas)) - (n + 1) / n
    
    def _calculate_hedge_pressure(self, gamma_dist: Dict, walls: List[GammaWall], 
                                 spot: float) -> float:
        """计算综合对冲压力"""
        pressure = 0.0
        
        # 考虑gamma敞口
        if gamma_dist['total_exposure'] > 0:
            pressure += min(gamma_dist['net_exposure'] / gamma_dist['total_exposure'], 1.0) * 0.4
            
        # 考虑最近gamma墙
        if walls:
            nearest_wall = walls[0]
            # 距离越近，压力越大
            distance_factor = 1 / (1 + nearest_wall.distance_pct / 100)
            pressure += distance_factor * nearest_wall.strength * 0.4
            
        # 考虑gamma集中度
        pressure += gamma_dist.get('concentration', 0) * 0.2
        
        return min(pressure, 1.0)
    
    def _extract_adaptive_features(self, gamma_dist: Dict, walls: List[GammaWall],
                                  spot: float) -> Dict[str, float]:
        """提取自适应特征（学习模块使用）"""
        features = {}
        
        # 基础统计特征
        profile = gamma_dist['profile']
        if profile:
            gammas = [g['gamma_exposure'] for g in profile]
            features['gamma_skew'] = float(np.mean(gammas)) if gammas else 0
            features['gamma_kurtosis'] = float(np.std(gammas)) if len(gammas) > 1 else 0
            
        # 墙体特征
        if walls:
            features['wall_count'] = len(walls)
            features['wall_asymmetry'] = sum(1 for w in walls if w.position == 'above') / len(walls)
            
        return features


class DealerPositionTracker:
    """做市商头寸追踪器"""
    
    def __init__(self):
        self.position_history = defaultdict(list)
        self.flow_estimator = FlowEstimator()
        
    def estimate_position(self, options: pd.DataFrame, spot: float, 
                         gamma_dist: Dict) -> Dict[str, float]:
        """估算做市商净头寸"""
        position = {
            'net_delta': 0.0,
            'net_gamma': 0.0,
            'net_vega': 0.0,
            'position_score': 0.0,  # -1到1，负值表示net short
            'flow_imbalance': 0.0
        }
        
        # 计算净Greeks敞口
        for _, opt in options.iterrows():
            # 简化假设：做市商是期权的净卖方
            # 实际中需要结合订单流数据
            oi = opt.get('open_interest', 0)
            volume = opt.get('volume', 0)
            
            # 使用成交量/持仓量比率估算做市商参与度
            if oi > 0:
                dealer_ratio = min(volume / oi, 1.0) * 0.7  # 假设做市商占70%成交量
            else:
                dealer_ratio = 0.5
                
            # 估算Greeks（简化计算）
            delta = self._estimate_delta(spot, opt)
            gamma = gamma_dist['profile'][0]['gamma_exposure'] if gamma_dist['profile'] else 0
            
            position['net_delta'] -= delta * oi * dealer_ratio
            position['net_gamma'] += gamma  # 已在gamma_dist中处理过符号
            
        # 计算持仓偏向分数
        if options['open_interest'].sum() > 0:
            position['position_score'] = -position['net_delta'] / (spot * options['open_interest'].sum())
            
        # 估算流动性失衡
        position['flow_imbalance'] = self.flow_estimator.estimate_imbalance(options)
        
        return position
    
    def _estimate_delta(self, spot: float, option: pd.Series) -> float:
        """估算期权Delta"""
        strike = option.get('strike', spot)
        moneyness = spot / strike
        
        # 简化的delta估算
        if option.get('type') == 'call':
            if moneyness > 1.1:
                return 0.9
            elif moneyness < 0.9:
                return 0.1
            else:
                return 0.5 + (moneyness - 1) * 2
        else:  # put
            if moneyness > 1.1:
                return -0.1
            elif moneyness < 0.9:
                return -0.9
            else:
                return -0.5 + (moneyness - 1) * 2


class FlowEstimator:
    """订单流估算器"""
    
    def estimate_imbalance(self, options: pd.DataFrame) -> float:
        """估算订单流失衡"""
        if options.empty:
            return 0.0
            
        # 使用价差和成交量估算
        total_imbalance = 0.0
        total_volume = 0.0
        
        for _, opt in options.iterrows():
            bid = opt.get('bid', 0)
            ask = opt.get('ask', 0)
            volume = opt.get('volume', 0)
            
            if ask > 0 and bid > 0:
                # 价差越大，失衡可能越严重
                spread_ratio = (ask - bid) / ((ask + bid) / 2)
                imbalance = spread_ratio * volume
                
                total_imbalance += imbalance
                total_volume += volume
                
        return total_imbalance / total_volume if total_volume > 0 else 0.0