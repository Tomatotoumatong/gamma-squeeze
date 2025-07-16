"""
PerformanceTracker - Enhanced version with improved data handling
Fixes data validation issues and preserves rich behavioral information
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
import logging
import json
import os
import math
from collections import defaultdict, deque
import asyncio
from SignalEvaluator import TradingSignal

logger = logging.getLogger(__name__)

@dataclass
class EnhancedMarketSnapshot:
    """Enhanced market snapshot with validation and rich behavioral data"""
    asset: str
    timestamp: datetime
    
    # Validated price data
    price: float
    bid: float
    ask: float
    spread: float
    
    # Microstructure
    orderbook_depth: Dict[str, float]
    orderbook_imbalance: float
    recent_trades: List[Dict[str, Any]]
    
    # Rich behavioral data (NEW)
    sweep_orders: List[Dict[str, Any]] = field(default_factory=list)
    divergences: List[Dict[str, Any]] = field(default_factory=list)
    cross_market_signals: List[Dict[str, Any]] = field(default_factory=list)
    
    # Gamma structure
    gamma_distribution: Dict[str, float] = field(default_factory=dict)
    nearest_gamma_wall: Dict[str, Any] = field(default_factory=dict)
    dealer_position: Dict[str, Any] = field(default_factory=dict)
    
    # Market state
    volatility_regime: str = 'normal'
    liquidity_score: float = 0.5
    market_regime: str = 'normal'
    anomaly_score: float = 0.0
    
    def __post_init__(self):
        """Validate data on creation"""
        # Ensure valid prices
        if self.price <= 0:
            logger.warning(f"Invalid price {self.price}, setting to bid/ask midpoint")
            self.price = (self.bid + self.ask) / 2 if self.bid > 0 and self.ask > 0 else 1.0
            
        # Ensure valid spread
        if math.isnan(self.spread) or self.spread < 0:
            self.spread = self.ask - self.bid if self.ask > self.bid else 0.0
            
        # Validate orderbook imbalance
        if math.isnan(self.orderbook_imbalance):
            self.orderbook_imbalance = 0.0

@dataclass
class DecisionSnapshot:
    """Enhanced decision snapshot with rich behavioral context"""
    timestamp: datetime
    decision_type: str
    assets_analyzed: List[str]
    
    # Enhanced market snapshots
    market_snapshots: Dict[str, EnhancedMarketSnapshot]
    
    # Preserved behavioral details (NEW)
    behavior_summary: Dict[str, Any]
    
    # Decision factors
    gamma_metrics: Dict[str, Any]
    scores: Dict[str, float]
    
    # Results
    signal_generated: Optional['SignalPerformance'] = None
    suppression_reason: Optional[str] = None
    counterfactual_data: Dict[str, Any] = field(default_factory=dict)

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
    market_snapshot: EnhancedMarketSnapshot
    
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
    """Enhanced performance tracker with improved data handling"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # Data storage
        self.signal_db_path = self.config['signal_db_path']
        self.decision_db_path = self.config['decision_db_path']
        
        # Active tracking
        self.active_signals: Dict[str, SignalPerformance] = {}
        self.decision_history: deque = deque(maxlen=10000)
        
        # Enhanced caches (NEW)
        self.behavior_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.gamma_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Data fetchers
        self.price_fetcher = None
        self.market_data_fetcher = None
        
        # Initialize
        self._ensure_db_exists()
        self._load_active_signals()
        
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'signal_db_path': 'signal_performance_enhanced.csv',
            'decision_db_path': 'decision_history.csv',
            'check_intervals': [5/60, 15/60, 30/60, 1, 2, 4, 8, 24],
            'update_interval': 300,
            'report_interval': 1800,
            'expected_move_ranges': {
                "1-2%": (1, 2),
                "2-5%": (2, 5),
                "5-10%": (5, 10),
                "10%+": (10, 20)
            }
        }
    
    async def capture_enhanced_market_snapshot(self, asset: str,
                                             gamma_analysis: Dict[str, Any],
                                             market_behavior: Dict[str, Any],
                                             market_data: pd.DataFrame) -> Optional[EnhancedMarketSnapshot]:
        """Capture enhanced market snapshot with validation and rich data"""
        try:
            # Get current market data
            asset_data = market_data[market_data['symbol'] == asset]
            if asset_data.empty:
                return None
                
            # Extract latest spot data
            spot_data = asset_data[asset_data['data_type'] == 'spot']
            if spot_data.empty:
                return None
                
            latest = spot_data.iloc[-1]
            
            # Extract orderbook data
            ob_data = asset_data[asset_data['data_type'] == 'orderbook']
            orderbook_info = self._extract_orderbook_info(ob_data)
            
            # Extract behavioral data (NEW - preserving rich information)
            sweep_orders = self._extract_sweep_details(asset, market_behavior)
            divergences = self._extract_divergence_details(asset, market_behavior)
            cross_signals = self._extract_cross_market_details(asset, market_behavior)
            
            # Extract gamma structure
            gamma_info = self._extract_enhanced_gamma_info(asset, gamma_analysis)
            
            # Get anomaly score
            anomaly_scores = market_behavior.get('anomaly_scores', {})
            anomaly_score = anomaly_scores.get(asset, 0.0)
            
            # Create enhanced snapshot
            snapshot = EnhancedMarketSnapshot(
                asset=asset,
                timestamp=datetime.utcnow(),
                price=float(latest.get('price', 0)),
                bid=float(latest.get('bid', 0)),
                ask=float(latest.get('ask', 0)),
                spread=float(latest.get('ask', 0)) - float(latest.get('bid', 0)),
                orderbook_depth=orderbook_info['depth'],
                orderbook_imbalance=orderbook_info['imbalance'],
                recent_trades=[],  # TODO: extract from market_data if available
                sweep_orders=sweep_orders,
                divergences=divergences,
                cross_market_signals=cross_signals,
                gamma_distribution=gamma_info['distribution'],
                nearest_gamma_wall=gamma_info['nearest_wall'],
                dealer_position=gamma_info['dealer_position'],
                volatility_regime=self._determine_volatility_regime(asset_data),
                liquidity_score=self._calculate_liquidity_score(asset_data),
                market_regime=market_behavior.get('market_regime', {}).get('state', 'normal'),
                anomaly_score=anomaly_score
            )
            
            # Cache the enhanced data
            self.behavior_cache[asset].append({
                'timestamp': snapshot.timestamp,
                'sweeps': sweep_orders,
                'divergences': divergences,
                'anomaly_score': anomaly_score
            })
            
            self.gamma_cache[asset].append({
                'timestamp': snapshot.timestamp,
                'gamma_dist': gamma_info['distribution'],
                'dealer_position': gamma_info['dealer_position']
            })
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error capturing enhanced snapshot for {asset}: {e}")
            return None
    
    def _extract_sweep_details(self, asset: str, market_behavior: Dict) -> List[Dict[str, Any]]:
        """Extract detailed sweep order information"""
        all_sweeps = market_behavior.get('sweep_orders', [])
        asset_sweeps = []
        
        for sweep in all_sweeps:
            if hasattr(sweep, 'symbol') and sweep.symbol == asset:
                asset_sweeps.append({
                    'timestamp': sweep.timestamp,
                    'side': sweep.side,
                    'volume': sweep.volume,
                    'price': sweep.price,
                    'frequency': sweep.frequency,
                    'anomaly_score': sweep.anomaly_score
                })
        
        return asset_sweeps
    
    def _extract_divergence_details(self, asset: str, market_behavior: Dict) -> List[Dict[str, Any]]:
        """Extract detailed divergence information"""
        all_divergences = market_behavior.get('divergences', [])
        asset_divergences = []
        
        for div in all_divergences:
            if hasattr(div, 'symbol') and div.symbol == asset:
                asset_divergences.append({
                    'type': div.divergence_type,
                    'strength': div.strength,
                    'duration': div.duration,
                    'details': div.details
                })
        
        return asset_divergences
    
    def _extract_cross_market_details(self, asset: str, market_behavior: Dict) -> List[Dict[str, Any]]:
        """Extract cross-market signal details"""
        all_signals = market_behavior.get('cross_market_signals', [])
        asset_signals = []
        
        for signal in all_signals:
            if hasattr(signal, 'lead_market') and signal.lead_market == asset:
                asset_signals.append({
                    'role': 'lead',
                    'lag_market': signal.lag_market,
                    'correlation': signal.correlation,
                    'lag_time': signal.lag_time,
                    'signal_strength': signal.signal_strength
                })
            elif hasattr(signal, 'lag_market') and signal.lag_market == asset:
                asset_signals.append({
                    'role': 'lag',
                    'lead_market': signal.lead_market,
                    'correlation': signal.correlation,
                    'lag_time': signal.lag_time,
                    'signal_strength': signal.signal_strength
                })
        
        return asset_signals
    
    def _extract_enhanced_gamma_info(self, asset: str, gamma_analysis: Dict) -> Dict[str, Any]:
        """Extract enhanced gamma information"""
        result = {
            'distribution': {},
            'nearest_wall': {},
            'dealer_position': {}
        }
        
        # Handle asset name mapping
        symbol_map = {'BTCUSDT': 'BTC', 'ETHUSDT': 'ETH'}
        gamma_symbol = symbol_map.get(asset, asset)
        
        # Gamma distribution
        gamma_dist = gamma_analysis.get('gamma_distribution', {}).get(gamma_symbol, {})
        if gamma_dist and 'profile' in gamma_dist:
            for item in gamma_dist['profile']:
                strike = item.get('strike')
                if strike:
                    result['distribution'][str(strike)] = item.get('gamma_exposure', 0)
        
        # Nearest gamma wall
        walls = gamma_analysis.get('gamma_walls', {}).get(gamma_symbol, [])
        if walls and hasattr(walls[0], 'strike'):
            wall = walls[0]
            result['nearest_wall'] = {
                'strike': wall.strike,
                'gamma_exposure': wall.gamma_exposure,
                'distance_pct': wall.distance_pct,
                'position': wall.position,
                'strength': wall.strength
            }
        
        # Dealer position
        dealer_pos = gamma_analysis.get('dealer_position', {}).get(gamma_symbol, {})
        if dealer_pos:
            result['dealer_position'] = {
                'net_delta': dealer_pos.get('net_delta', 0),
                'net_gamma': dealer_pos.get('net_gamma', 0),
                'position_score': dealer_pos.get('position_score', 0),
                'flow_imbalance': dealer_pos.get('flow_imbalance', 0)
            }
        
        return result
    
    def _extract_orderbook_info(self, ob_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract orderbook information with validation"""
        result = {
            'depth': {},
            'imbalance': 0.0
        }
        
        if ob_data.empty:
            return result
        
        latest_ob = ob_data.iloc[-1]
        
        # Extract depth
        bid_volume = latest_ob.get('bid_volume', 0)
        ask_volume = latest_ob.get('ask_volume', 0)
        
        result['depth'] = {
            'bid_volume': float(bid_volume) if not math.isnan(bid_volume) else 0.0,
            'ask_volume': float(ask_volume) if not math.isnan(ask_volume) else 0.0
        }
        
        # Calculate imbalance with validation
        total_volume = result['depth']['bid_volume'] + result['depth']['ask_volume']
        if total_volume > 0:
            result['imbalance'] = (result['depth']['bid_volume'] - result['depth']['ask_volume']) / total_volume
        
        return result
    
    def _determine_volatility_regime(self, asset_data: pd.DataFrame) -> str:
        """Determine volatility regime"""
        if len(asset_data) < 20:
            return 'normal'
        
        prices = asset_data[asset_data['data_type'] == 'spot']['price'].values
        if len(prices) < 20:
            return 'normal'
        
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        # Simple regime classification
        if volatility < 0.01:
            return 'low'
        elif volatility > 0.03:
            return 'high'
        else:
            return 'normal'
    
    def _calculate_liquidity_score(self, asset_data: pd.DataFrame) -> float:
        """Calculate liquidity score"""
        spot_data = asset_data[asset_data['data_type'] == 'spot']
        if spot_data.empty:
            return 0.5
        
        # Use volume and spread as liquidity indicators
        volumes = spot_data['volume'].values
        if len(volumes) == 0:
            return 0.5
        
        # Normalize volume (simple approach)
        avg_volume = np.mean(volumes)
        recent_volume = volumes[-1] if len(volumes) > 0 else avg_volume
        
        volume_ratio = min(recent_volume / (avg_volume + 1e-8), 2.0)
        
        # Spread component
        if 'bid' in spot_data.columns and 'ask' in spot_data.columns:
            spreads = (spot_data['ask'] - spot_data['bid']) / spot_data['price']
            avg_spread = spreads.mean()
            spread_score = 1.0 / (1.0 + avg_spread * 100)  # Lower spread = higher score
        else:
            spread_score = 0.5
        
        # Combine scores
        return (volume_ratio * 0.6 + spread_score * 0.4) / 2
    
    async def record_decision_enhanced(self, assets_analyzed: List[str],
                                     gamma_analysis: Dict[str, Any],
                                     market_behavior: Dict[str, Any],
                                     market_data: pd.DataFrame,
                                     scores: Dict[str, Dict[str, float]],
                                     signals_generated: List[TradingSignal],
                                     suppressed_signals: Dict[str, str]):
        """Record decision with enhanced market context"""
        timestamp = datetime.utcnow()
        
        # Capture enhanced market snapshots
        market_snapshots = {}
        for asset in assets_analyzed:
            snapshot = await self.capture_enhanced_market_snapshot(
                asset, gamma_analysis, market_behavior, market_data
            )
            if snapshot:
                market_snapshots[asset] = snapshot
        
        # Create behavior summary preserving rich information
        behavior_summary = {
            'total_sweeps': len(market_behavior.get('sweep_orders', [])),
            'sweep_breakdown': self._summarize_sweeps(market_behavior),
            'divergence_types': self._summarize_divergences(market_behavior),
            'cross_market_dynamics': self._summarize_cross_market(market_behavior),
            'regime_confidence': market_behavior.get('market_regime', {}).get('confidence', 0)
        }
        
        # Create enhanced decision snapshot
        decision = DecisionSnapshot(
            timestamp=timestamp,
            decision_type='signal_generated' if signals_generated else 'no_signal',
            assets_analyzed=assets_analyzed,
            market_snapshots=market_snapshots,
            behavior_summary=behavior_summary,
            gamma_metrics=self._extract_gamma_metrics(gamma_analysis),
            scores=scores,
            suppression_reason=json.dumps(suppressed_signals) if suppressed_signals else None
        )
        
        # Process signals
        for signal in signals_generated:
            if signal.asset in market_snapshots:
                perf = self._create_signal_performance(signal, market_snapshots[signal.asset])
                decision.signal_generated = perf
                self.active_signals[perf.signal_id] = perf
                self._save_signal(perf)
        
        # Generate counterfactual data
        decision.counterfactual_data = await self._generate_enhanced_counterfactual_data(
            assets_analyzed, market_snapshots, scores, behavior_summary
        )
        
        # Save decision
        self.decision_history.append(decision)
        self._save_enhanced_decision(decision)
    
    def _summarize_sweeps(self, market_behavior: Dict) -> Dict[str, Any]:
        """Summarize sweep orders preserving key information"""
        sweeps = market_behavior.get('sweep_orders', [])
        if not sweeps:
            return {}
        
        buy_sweeps = [s for s in sweeps if hasattr(s, 'side') and s.side == 'buy']
        sell_sweeps = [s for s in sweeps if hasattr(s, 'side') and s.side == 'sell']
        
        return {
            'buy_count': len(buy_sweeps),
            'sell_count': len(sell_sweeps),
            'buy_volume': sum(s.volume for s in buy_sweeps),
            'sell_volume': sum(s.volume for s in sell_sweeps),
            'max_anomaly_score': max((s.anomaly_score for s in sweeps), default=0),
            'avg_frequency': np.mean([s.frequency for s in sweeps]) if sweeps else 0
        }
    
    def _summarize_divergences(self, market_behavior: Dict) -> Dict[str, int]:
        """Summarize divergences by type"""
        divergences = market_behavior.get('divergences', [])
        type_counts = defaultdict(int)
        
        for div in divergences:
            if hasattr(div, 'divergence_type'):
                type_counts[div.divergence_type] += 1
        
        return dict(type_counts)
    
    def _summarize_cross_market(self, market_behavior: Dict) -> Dict[str, Any]:
        """Summarize cross-market dynamics"""
        signals = market_behavior.get('cross_market_signals', [])
        if not signals:
            return {}
        
        return {
            'signal_count': len(signals),
            'avg_correlation': np.mean([s.correlation for s in signals]),
            'max_lag_time': max((s.lag_time for s in signals), default=0),
            'lead_markets': list(set(s.lead_market for s in signals if hasattr(s, 'lead_market')))
        }
    
    def track_signal_with_context(self, signal: TradingSignal, context: Dict[str, Any]):
        """Track signal with enhanced context handling"""
        # Validate context data
        validated_context = {
            'current_price': float(context.get('current_price', 0)),
            'spread': float(context.get('spread', 0)) if not math.isnan(context.get('spread', 0)) else 0.0,
            'ob_imbalance': float(context.get('ob_imbalance', 0)) if not math.isnan(context.get('ob_imbalance', 0)) else 0.0,
            'parameter_version': context.get('parameter_version', 0),
            'learning_active': context.get('learning_active', False)
        }
        
        # Get cached behavioral data if available
        behavior_data = {}
        if signal.asset in self.behavior_cache and self.behavior_cache[signal.asset]:
            recent_behavior = self.behavior_cache[signal.asset][-1]
            behavior_data = {
                'recent_sweeps': len(recent_behavior.get('sweeps', [])),
                'sweep_intensity': np.mean([s['anomaly_score'] for s in recent_behavior.get('sweeps', [])]) if recent_behavior.get('sweeps') else 0,
                'divergence_active': len(recent_behavior.get('divergences', [])) > 0,
                'anomaly_score': recent_behavior.get('anomaly_score', 0)
            }
        
        # Create simple snapshot for compatibility
        market_snapshot = EnhancedMarketSnapshot(
            asset=signal.asset,
            timestamp=datetime.utcnow(),
            price=validated_context['current_price'],
            bid=validated_context['current_price'] - validated_context['spread']/2,
            ask=validated_context['current_price'] + validated_context['spread']/2,
            spread=validated_context['spread'],
            orderbook_depth={},
            orderbook_imbalance=validated_context['ob_imbalance'],
            recent_trades=[],
            gamma_distribution={},
            iv_surface={},
            put_call_skew=0,
            nearest_gamma_wall={},
            trend_strength=0,
            support_levels=[],
            resistance_levels=[],
            momentum_indicators={},
            volatility_regime='normal',
            liquidity_score=0.5,
            market_regime='normal'
        )
        
        # Create performance record
        perf = self._create_signal_performance(signal, market_snapshot)
        
        # Enhanced metadata
        perf.metadata.update({
            'market_context': validated_context,
            'behavior_context': behavior_data,
            'entry_conditions': {
                'spread_bp': validated_context['spread'] / validated_context['current_price'] * 10000 if validated_context['current_price'] > 0 else 0,
                'orderbook_imbalance': validated_context['ob_imbalance'],
                'parameter_version': validated_context['parameter_version']
            }
        })
        
        # Save
        self.active_signals[perf.signal_id] = perf
        self._save_signal(perf)
        
        logger.info(f"Tracking signal {perf.signal_id} with validated context")
    
    async def _generate_enhanced_counterfactual_data(self, assets: List[str],
                                                   snapshots: Dict[str, EnhancedMarketSnapshot],
                                                   scores: Dict[str, Dict[str, float]],
                                                   behavior_summary: Dict) -> Dict[str, Any]:
        """Generate counterfactual analysis with behavioral context"""
        counterfactual = {}
        
        for asset in assets:
            if asset not in snapshots or asset not in scores:
                continue
            
            snapshot = snapshots[asset]
            asset_scores = scores[asset]
            
            # Calculate potential with behavioral factors
            potential_strength = np.mean(list(asset_scores.values()))
            
            # Adjust for behavioral signals
            behavior_boost = 0
            if len(snapshot.sweep_orders) > 0:
                behavior_boost += 0.1
            if len(snapshot.divergences) > 0:
                behavior_boost += 0.1
            if snapshot.anomaly_score > 0.7:
                behavior_boost += 0.1
            
            adjusted_strength = min(potential_strength + behavior_boost * 20, 100)
            
            # Estimate move based on regime and behavior
            if snapshot.market_regime == 'squeeze' and len(snapshot.sweep_orders) > 2:
                potential_move = adjusted_strength * 0.04  # 4% base for squeeze with sweeps
            elif snapshot.market_regime == 'breakout':
                potential_move = adjusted_strength * 0.05  # 5% for breakout
            else:
                potential_move = adjusted_strength * 0.02  # 2% normal
            
            counterfactual[asset] = {
                'potential_signal_strength': adjusted_strength,
                'expected_move_percent': potential_move * 100,
                'market_favorability': self._assess_enhanced_market_favorability(snapshot),
                'behavioral_support': behavior_boost,
                'missed_opportunity_score': self._calculate_enhanced_missed_opportunity_score(
                    adjusted_strength, snapshot, behavior_summary
                )
            }
        
        return counterfactual
    
    def _assess_enhanced_market_favorability(self, snapshot: EnhancedMarketSnapshot) -> float:
        """Assess market favorability with behavioral factors"""
        score = 0.5
        
        # Base factors
        score += snapshot.liquidity_score * 0.2
        
        # Behavioral factors
        if len(snapshot.sweep_orders) > 0:
            score += 0.1
        if snapshot.anomaly_score > 0.5:
            score += snapshot.anomaly_score * 0.1
        
        # Gamma factors
        if snapshot.nearest_gamma_wall and snapshot.nearest_gamma_wall.get('distance_pct', 100) < 2:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_enhanced_missed_opportunity_score(self, potential_strength: float,
                                                   snapshot: EnhancedMarketSnapshot,
                                                   behavior_summary: Dict) -> float:
        """Calculate missed opportunity with behavioral context"""
        favorability = self._assess_enhanced_market_favorability(snapshot)
        
        # Strong signal with supportive behavior
        if potential_strength > 70 and favorability > 0.7:
            base_score = (potential_strength / 100) * favorability
            
            # Boost for strong behavioral signals
            if behavior_summary.get('total_sweeps', 0) > 5:
                base_score *= 1.2
            
            return min(base_score, 1.0)
        
        return 0.0
    
    def _save_enhanced_decision(self, decision: DecisionSnapshot):
        """Save enhanced decision with behavioral data"""
        try:
            df = pd.read_csv(self.decision_db_path)
            
            # Prepare rich data for storage
            behavior_json = json.dumps(decision.behavior_summary)
            
            # Extract key snapshot data for each asset
            snapshot_summary = {}
            for asset, snapshot in decision.market_snapshots.items():
                snapshot_summary[asset] = {
                    'price': snapshot.price,
                    'spread': snapshot.spread,
                    'liquidity_score': snapshot.liquidity_score,
                    'anomaly_score': snapshot.anomaly_score,
                    'sweep_count': len(snapshot.sweep_orders),
                    'divergence_count': len(snapshot.divergences)
                }
            
            data = {
                'timestamp': decision.timestamp,
                'decision_type': decision.decision_type,
                'assets_analyzed': json.dumps(decision.assets_analyzed),
                'signal_generated': decision.signal_generated.signal_id if decision.signal_generated else None,
                'suppression_reason': decision.suppression_reason,
                'gamma_metrics': json.dumps(decision.gamma_metrics),
                'behavior_summary': behavior_json,  # Rich behavioral data
                'scores': json.dumps(decision.scores),
                'market_snapshot': json.dumps(snapshot_summary),
                'counterfactual_data': json.dumps(decision.counterfactual_data)
            }
            
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
            df.to_csv(self.decision_db_path, index=False)
            
        except Exception as e:
            logger.error(f"Error saving enhanced decision: {e}")
    
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

    
    def track_signal(self, signal: TradingSignal, initial_price: float):
        context = {'current_price': initial_price}
        self.track_signal_with_context(signal, context)
    
    def _create_signal_performance(self, signal: TradingSignal,
                                 market_snapshot: Optional[EnhancedMarketSnapshot],
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
#
#                        logger.info(f"Updated {interval_str} price for {signal_id}: "
#                                  f"{current_price} ({returns:+.2f}%)")
                
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
                                          snapshots: Dict[str,EnhancedMarketSnapshot],
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
    
    def _assess_market_favorability(self, snapshot: EnhancedMarketSnapshot) -> float:
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
                                          snapshot: EnhancedMarketSnapshot) -> float:
        """计算错失机会分数"""
        # 基于信号强度和市场有利程度
        favorability = self._assess_market_favorability(snapshot)
        
        # 只有当信号强度和市场条件都好时，才算真正的错失机会
        if potential_strength > 70 and favorability > 0.7:
            return (potential_strength / 100) * favorability
        
        return 0.0
    
    def _get_latest_market_snapshot(self, asset: str) -> Optional[EnhancedMarketSnapshot]:
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
