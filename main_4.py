#!/usr/bin/env python3
"""
Gamma Squeeze Signal System - Self-Evolving Edition with Continuous Learning
"""

import asyncio
import logging
import sys
import signal
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any
from colorama import init, Fore, Style
import json
import os

# Import system modules
from UnifiedDataCollector import UnifiedDataCollector, DataType
from GammaPressureAnalyzer import GammaPressureAnalyzer
from MarketBehaviorDetector import MarketBehaviorDetector
from SignalEvaluator import SignalEvaluator, TradingSignal
from PerformanceTracker import PerformanceTracker, SignalPerformance
from AdaptiveLearner import EnhancedAdaptiveLearner

# Initialize colorama
init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_output/gamma_squeeze_adaptive.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AdaptiveGammaSqueezeSystem:
    """Self-Evolving Gamma Squeeze Signal System with Continuous Decision Monitoring"""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config
        
        # Core components
        self.collector = None
        self.gamma_analyzer = None
        self.behavior_detector = None
        self.signal_evaluator = None
        self.performance_tracker = None
        self.adaptive_learner = None
        
        # System state
        self.running = False
        self.analysis_results = []
        self.behavior_results = []
        self.generated_signals = []
        
        # Continuous decision tracking
        self.last_decision_time = datetime.utcnow()
        self.continuous_decision_count = 0
        
        # Learning state
        self.last_learning_time = datetime.utcnow()
        self.learning_cycle_count = 0
        self.parameter_version = 0
        
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Setup signal handlers"""
        def signal_handler(sig, frame):
            logger.info("\n Received interrupt signal, shutting down gracefully...")
            asyncio.create_task(self.shutdown())
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    async def initialize(self):
        """Initialize system with adaptive learning"""
        logger.info("=" * 80)
        logger.info(" Initializing Self-Evolving Gamma Squeeze Signal System")
        logger.info("   ✓ Data Collection")
        logger.info("   ✓ Pattern Recognition") 
        logger.info("   ✓ Signal Generation")
        logger.info("   ✓ Performance Tracking")
        logger.info("   ✓ Continuous Decision Monitoring")
        logger.info("   ✓ Adaptive Learning Engine")
        logger.info("=" * 80)
        
        # Initialize components
        self.collector = UnifiedDataCollector(self.config['data_collection'])
        await self.collector.initialize()
        
        self.gamma_analyzer = GammaPressureAnalyzer(self.config['gamma_analysis'])
        self.behavior_detector = MarketBehaviorDetector(self.config['market_behavior'])
        self.signal_evaluator = SignalEvaluator(self.config['signal_generation'])
        self.performance_tracker = PerformanceTracker(self.config['performance_tracking'])
        
        # Initialize adaptive learner
        self.adaptive_learner = EnhancedAdaptiveLearner(self.config['adaptive_learning'])
        
        # Set up data fetchers
        self.performance_tracker.set_price_fetcher(self._get_current_price)
        self.performance_tracker.set_market_data_fetcher(self._get_market_data)
        
        logger.info(" ✓ All components initialized successfully")
        logger.info(f" Continuous Learning: ENABLED")
        
    async def start(self):
        """Start system with continuous monitoring"""
        logger.info("\n Starting Self-Evolving System...")
        self.running = True
        
        await self.collector.start()
        await asyncio.sleep(15)  # Initial data accumulation
        
        tasks = [
            asyncio.create_task(self._monitor_loop()),
            asyncio.create_task(self._gamma_analysis_loop()),
            asyncio.create_task(self._behavior_detection_loop()),
            asyncio.create_task(self._signal_generation_loop()),
            asyncio.create_task(self._performance_update_loop()),
            asyncio.create_task(self._continuous_decision_loop()),  # New: continuous monitoring
            asyncio.create_task(self._adaptive_learning_loop()),
            asyncio.create_task(self._learning_report_loop())
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _continuous_decision_loop(self):
        """Continuously evaluate decisions every 5 minutes"""
        while self.running:
            try:
                await asyncio.sleep(self.config['adaptive_learning']['continuous_decision_interval'])
                
                # Evaluate current decision state
                continuous_decision = await self._evaluate_continuous_decision()
                
                # Record in adaptive learner
                self.adaptive_learner.record_continuous_decision(continuous_decision)
                
                # Increment counter
                self.continuous_decision_count += 1
                
                # Log progress
                if self.continuous_decision_count % 12 == 0:  # Every hour
                    logger.info(f"Continuous decisions recorded: {self.continuous_decision_count}")
                    
            except Exception as e:
                logger.error(f"Error in continuous decision loop: {e}", exc_info=True)
                
    async def _evaluate_continuous_decision(self) -> Dict:
        """Evaluate current decision state (signal or no signal)"""
        decision = {
            'timestamp': datetime.utcnow(),
            'decision_type': 'continuous_evaluation',
            'signal_generated': False,
            'market_snapshot': {},
            'scores': {},
            'suppressed_signals': {},
            'missed_opportunity': None,
            'avoided_bad_signal': False,
            'config_snapshot': self._get_current_config()
        }
        
        # Get current market state
        if self.analysis_results and self.behavior_results:
            latest_gamma = self.analysis_results[-1]
            latest_behavior = self.behavior_results[-1]
            market_data = self.collector.get_latest_data(window_seconds=300)
            
            # Extract analyzed assets
            assets = set()
            if latest_gamma.get('gamma_distribution'):
                assets.update(latest_gamma['gamma_distribution'].keys())
                
            # Calculate scores for each asset
            for asset in assets:
                try:
                    asset_scores = self.signal_evaluator._calculate_scores(
                        asset, latest_gamma, latest_behavior, market_data
                    )
                    decision['scores'][asset] = asset_scores
                    
                    # Check if signal was suppressed
                    should_generate = self.signal_evaluator._should_generate_signal(asset, asset_scores)
                    if not should_generate:
                        decision['suppressed_signals'][asset] = self._analyze_suppression_reason(
                            asset, asset_scores
                        )
                except Exception as e:
                    logger.error(f"Error calculating scores for {asset}: {e}")
            
            # Record market snapshot
            decision['market_snapshot'] = await self._capture_market_snapshot()
            
            # Store behavior metrics for learning
            decision['behavior_metrics'] = {
                'market_regime': latest_behavior.get('market_regime', {}).get('state', 'normal'),
                'sweep_count': len(latest_behavior.get('sweep_orders', [])),
                'divergence_count': len(latest_behavior.get('divergences', [])),
                'anomaly_scores': latest_behavior.get('anomaly_scores', {})
            }
            
            # Check for recent signals
            recent_signals = [s for s in self.generated_signals 
                            if (datetime.utcnow() - s.timestamp).total_seconds() < 3600]
            
            if recent_signals:
                decision['signal_generated'] = True
                decision['signal_id'] = f"{recent_signals[-1].asset}_{recent_signals[-1].timestamp.strftime('%Y%m%d_%H%M%S')}"
                
                # Initial feedback based on immediate price movement
                signal = recent_signals[-1]
                current_price = await self._get_current_price(signal.asset)
                if current_price:
                    price_change = (current_price - signal.metadata.get('initial_price', current_price)) / signal.metadata.get('initial_price', 1)
                    if signal.direction == 'BULLISH':
                        decision['initial_feedback'] = min(price_change * 10, 1.0)
                    else:
                        decision['initial_feedback'] = min(-price_change * 10, 1.0)
            
            # Analyze missed opportunities
            decision['missed_opportunity'] = self._analyze_missed_opportunity(
                decision['scores'], decision['suppressed_signals']
            )
            
            # Check if we avoided a bad signal
            decision['avoided_bad_signal'] = self._check_avoided_bad_signal(
                decision['suppressed_signals'], market_data
            )
            
        return decision
    
    def _analyze_suppression_reason(self, asset: str, scores: Dict[str, float]) -> str:
        """Analyze why a signal was suppressed"""
        reasons = []
        
        composite_score = np.mean(list(scores.values()))
        
        if composite_score < self.config['signal_generation']['min_strength']:
            reasons.append(f"Low composite score: {composite_score:.1f}")
            
        if not self.signal_evaluator._check_cooldown(asset):
            reasons.append("In cooldown period")
            
        high_threshold = self.config['gamma_analysis']['wall_percentile']
        if not any(score >= high_threshold * 0.9 for score in scores.values()):
            reasons.append("No dimension reached high threshold")
            
        return " | ".join(reasons) if reasons else "Unknown"
    
    async def _capture_market_snapshot(self) -> Dict[str, Any]:
        """Capture current market snapshot"""
        snapshot = {
            'timestamp': datetime.utcnow(),
            'prices': {},
            'volumes': {},
            'spreads': {}
        }
        
        df = self.collector.get_latest_data(window_seconds=60)
        if not df.empty:
            spot_data = df[df['data_type'] == 'spot']
            for symbol in spot_data['symbol'].unique():
                symbol_data = spot_data[spot_data['symbol'] == symbol]
                if not symbol_data.empty:
                    latest = symbol_data.iloc[-1]
                    snapshot['prices'][symbol] = latest.get('price', 0)
                    snapshot['volumes'][symbol] = latest.get('volume', 0)
                    snapshot['spreads'][symbol] = latest.get('ask', 0) - latest.get('bid', 0)
                    
        return snapshot
    
    def _analyze_missed_opportunity(self, scores: Dict, suppressed_signals: Dict) -> Optional[Dict]:
        """Analyze if we missed a good opportunity"""
        for asset, asset_scores in scores.items():
            composite_score = np.mean(list(asset_scores.values()))
            
            # High score but suppressed
            if composite_score > 70 and asset in suppressed_signals:
                return {
                    'asset': asset,
                    'composite_score': composite_score,
                    'reason': suppressed_signals[asset],
                    'magnitude': (composite_score - 70) / 30  # 0 to 1 scale
                }
        return None
    
    def _check_avoided_bad_signal(self, suppressed_signals: Dict, market_data: pd.DataFrame) -> bool:
        """Check if we correctly avoided a bad signal"""
        if not suppressed_signals:
            return False
            
        # Simple heuristic: if we suppressed signals during low liquidity or high volatility
        recent_volumes = market_data[market_data['data_type'] == 'spot']['volume'].values
        if len(recent_volumes) > 0:
            avg_volume = np.mean(recent_volumes)
            recent_volume = recent_volumes[-1] if len(recent_volumes) > 0 else avg_volume
            
            # Low liquidity condition
            if recent_volume < avg_volume * 0.5:
                return True
                
        return False
    
    def _get_current_config(self) -> Dict:
        """Get current configuration snapshot"""
        return {
            'gamma_analysis': self.gamma_analyzer.config,
            'market_behavior': self.behavior_detector.config,
            'signal_generation': self.signal_evaluator.config
        }
        
    async def _adaptive_learning_loop(self):
        """Main adaptive learning loop"""
        while self.running:
            try:
                await asyncio.sleep(self.config['adaptive_learning']['learning_interval'])
                
                if not self.config['adaptive_learning']['enabled']:
                    continue
                
                # Perform learning from continuous decisions
                learning_result = self.adaptive_learner.learn_from_continuous_decisions()
                
                if learning_result and 'adjustments' in learning_result:
                    await self._apply_parameter_adjustments(learning_result)
                    
            except Exception as e:
                logger.error(f"Error in adaptive learning loop: {e}", exc_info=True)
                
    async def _apply_parameter_adjustments(self, learning_result: Dict):
        """Apply parameter adjustments from learning"""
        adjustments = learning_result.get('adjustments', {})
        
        if not adjustments:
            logger.info("No parameter adjustments needed")
            return
            
        logger.info(f"\n🔧 Applying {len(adjustments)} parameter adjustments")
        logger.info(f"Market Regime: {learning_result.get('regime', 'unknown')}")
        
        success_count = 0
        
        for param_path, adjustment in adjustments.items():
            try:
                # Parse parameter path
                parts = param_path.split('.')
                
                # Apply to appropriate module
                if parts[0] == 'market_behavior' and hasattr(self.behavior_detector, 'update_parameters'):
                    update_dict = self._build_nested_dict('.'.join(parts[1:]), adjustment)
                    if self.behavior_detector.update_parameters(update_dict):
                        success_count += 1
                        logger.info(f"✓ Updated {param_path}: {adjustment:.3f}")
                        
                elif parts[0] == 'gamma_analysis' and hasattr(self.gamma_analyzer, 'update_parameters'):
                    update_dict = self._build_nested_dict('.'.join(parts[1:]), adjustment)
                    if self.gamma_analyzer.update_parameters(update_dict):
                        success_count += 1
                        logger.info(f"✓ Updated {param_path}: {adjustment:.3f}")
                        
                elif parts[0] == 'signal_generation':
                    update_dict = self._build_nested_dict('.'.join(parts[1:]), adjustment)
                    if self.signal_evaluator.update_parameters(update_dict):
                        success_count += 1
                        logger.info(f"✓ Updated {param_path}: {adjustment:.3f}")
                        
            except Exception as e:
                logger.error(f"Failed to apply adjustment {param_path}: {e}")
                
        self.parameter_version += 1
        self.learning_cycle_count += 1
        
        logger.info(f"Successfully applied {success_count}/{len(adjustments)} adjustments")
        logger.info(f"Parameter version: v{self.parameter_version}")
        
        # Print learning decision details
        if 'learning_decision' in learning_result:
            decision = learning_result['learning_decision']
            logger.info(f"Expected improvement: {decision.expected_improvement:+.2%}")
            
    def _build_nested_dict(self, path: str, value: Any) -> Dict:
        """Build nested dictionary from dot-separated path"""
        parts = path.split('.')
        result = {}
        current = result
        
        for i, part in enumerate(parts[:-1]):
            current[part] = {}
            current = current[part]
            
        current[parts[-1]] = value
        
        return result
        
    async def _learning_report_loop(self):
        """Generate periodic learning reports"""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                report = self.adaptive_learner.get_learning_report()
                self._print_learning_report(report)
                
            except Exception as e:
                logger.error(f"Error in learning report loop: {e}", exc_info=True)
                
    def _print_learning_report(self, report: Dict):
        """Print comprehensive learning report"""
        logger.info(f"\n{Fore.MAGENTA} ADAPTIVE LEARNING REPORT{Style.RESET_ALL}")
        logger.info("=" * 80)
        
        logger.info(f"\nContinuous Decision Statistics:")
        logger.info(f"  Total Decisions: {report['total_continuous_decisions']}")
        logger.info(f"  Learning Cycles: {report['total_learning_cycles']}")
        
        # Parameter adjustments
        if report['parameter_adjustments']:
            logger.info(f"\nParameter Adjustments Summary:")
            for param, info in report['parameter_adjustments'].items():
                if info['total_adjustments'] > 0:
                    logger.info(f"  {param}:")
                    logger.info(f"    Adjustments: {info['total_adjustments']}")
                    logger.info(f"    Net Change: {info['net_change']:+.3f}")
                    logger.info(f"    Avg Adjustment: {info['avg_adjustment']:+.3f}")
                    
        # Performance trend
        trend = report['performance_trend']
        if trend['confidence'] > 0:
            logger.info(f"\nPerformance Trend:")
            logger.info(f"  Trend: {trend['trend']:+.2%}")
            logger.info(f"  Confidence: {trend['confidence']:.2f}")
            logger.info(f"  Current Avg: {trend.get('current_avg', 0):.2f}")
            
        # Learning effectiveness
        effectiveness = report['learning_effectiveness']
        if effectiveness['confidence'] > 0:
            logger.info(f"\nLearning Effectiveness:")
            logger.info(f"  Avg Improvement: {effectiveness['avg_improvement']:+.2%}")
            logger.info(f"  Positive Rate: {effectiveness['positive_rate']:.1%}")
            
        logger.info("=" * 80)
        
    async def _gamma_analysis_loop(self):
        """Gamma analysis loop"""
        while self.running:
            try:
                await asyncio.sleep(self.config['gamma_analysis']['interval'])
                
                option_data, spot_data = await self._prepare_analysis_data()
                
                if option_data.empty or spot_data.empty:
                    continue
                
                analysis_result = self.gamma_analyzer.analyze(option_data, spot_data)
                self.analysis_results.append(analysis_result)
                
                if len(self.analysis_results) > 100:
                    self.analysis_results = self.analysis_results[-100:]
                    
            except Exception as e:
                logger.error(f"Error in gamma analysis: {e}", exc_info=True)
                
    async def _behavior_detection_loop(self):
        """Behavior detection loop"""
        while self.running:
            try:
                await asyncio.sleep(self.config['market_behavior']['interval'])
                
                market_data = self.collector.get_latest_data(window_seconds=300)
                
                if market_data.empty:
                    continue
                
                behavior_result = self.behavior_detector.detect(market_data)
                self.behavior_results.append(behavior_result)
                
                if len(self.behavior_results) > 100:
                    self.behavior_results = self.behavior_results[-100:]
                    
            except Exception as e:
                logger.error(f"Error in behavior detection: {e}", exc_info=True)
                
    async def _signal_generation_loop(self):
        """Signal generation with learning feedback"""
        while self.running:
            try:
                await asyncio.sleep(self.config['signal_generation']['interval'])
                
                if not self.analysis_results or not self.behavior_results:
                    continue
                    
                latest_gamma = self.analysis_results[-1]
                latest_behavior = self.behavior_results[-1]
                market_data = self.collector.get_latest_data(window_seconds=300)
                
                # Generate signals
                signals = self.signal_evaluator.evaluate(
                    latest_gamma, latest_behavior, market_data
                )
                
                # Track signals with context
                for signal in signals:
                    self.generated_signals.append(signal)
                    
                    # Add metadata
                    signal.metadata['parameter_version'] = self.parameter_version
                    signal.metadata['learning_cycle'] = self.learning_cycle_count
                    signal.metadata['initial_price'] = await self._get_current_price(signal.asset)
                    
                    # Track signal
                    market_snapshot = await self._capture_market_snapshot_for_signal(
                        signal.asset, latest_gamma, latest_behavior
                    )
                    
                    self.performance_tracker.track_signal_with_context(signal, {
                        'current_price': signal.metadata['initial_price'],
                        'spread': market_snapshot.get('spread', 0),
                        'ob_imbalance': market_snapshot.get('orderbook_imbalance', 0),
                        'parameter_version': self.parameter_version,
                        'learning_active': True
                    })
                    
                    self._print_signal(signal)
                    
            except Exception as e:
                logger.error(f"Error in signal generation: {e}", exc_info=True)
                
    def _print_signal(self, signal: TradingSignal):
        """Print signal with learning context"""
        logger.info(f"\n{Fore.GREEN}!!!!! SIGNAL GENERATED !!!!! (v{self.parameter_version}){Style.RESET_ALL}")
        logger.info(f"Asset: {signal.asset} | Direction: {signal.direction}")
        logger.info(f"Type: {signal.signal_type}")
        logger.info(f"Strength: {signal.strength} | Confidence: {signal.confidence:.2f}")
        logger.info(f"Expected Move: {signal.expected_move} | Time Horizon: {signal.time_horizon}")
        logger.info(f"Key Levels: {', '.join(f'{level:.0f}' for level in signal.key_levels[:3])}")
        if signal.risk_factors:
            logger.info(f"Risk Factors: {', '.join(signal.risk_factors)}")
        logger.info(f"Learning Context: Cycle {self.learning_cycle_count}")
        
    async def _performance_update_loop(self):
        """Update performance metrics"""
        while self.running:
            try:
                await asyncio.sleep(self.config['performance_tracking']['update_interval'])
                
                await self.performance_tracker.update_prices()
                
            except Exception as e:
                logger.error(f"Error in performance update: {e}", exc_info=True)
                
    async def _monitor_loop(self):
        """Enhanced monitor with learning status"""
        while self.running:
            try:
                await asyncio.sleep(self.config['display_interval'])
                
                status_parts = []
                
                # Data status
                df = self.collector.get_latest_data(window_seconds=60)
                if not df.empty:
                    status_parts.append(f"Data: {len(df)}")
                    
                # Signals
                active_signals = len(self.performance_tracker.active_signals)
                total_signals = len(self.generated_signals)
                status_parts.append(f"Signals: {active_signals}/{total_signals}")
                
                # Continuous decisions
                status_parts.append(f"Decisions: {self.continuous_decision_count}")
                
                # Learning status
                status_parts.append(f"v{self.parameter_version}")
                
                # Latest adjustment
                learner_decisions = self.adaptive_learner.continuous_decisions
                if learner_decisions:
                    avg_quality = np.mean([d.get('decision_quality', 0.5) for d in list(learner_decisions)[-20:]])
                    status_parts.append(f"Quality: {avg_quality:.2f}")
                    
                status_line = " | ".join(status_parts)
                timestamp = datetime.now().strftime("%H:%M:%S")
                logger.info(f"[{timestamp}] - {status_line}")
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                
    from typing import Tuple
    async def _prepare_analysis_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare analysis data"""
        df = self.collector.get_latest_data(window_seconds=120)
        
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        option_mask = df['data_type'] == 'option'
        option_data = df[option_mask].copy()
        
        symbol_map = {'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT'}
        
        spot_mask = df['data_type'] == 'spot'
        spot_data = df[spot_mask].copy()
        
        if not option_data.empty:
            option_data['mapped_symbol'] = option_data['symbol'].map(symbol_map)
            option_data['symbol'] = option_data['mapped_symbol']
            option_data = option_data.drop('mapped_symbol', axis=1)
            
            if 'iv' in option_data.columns:
                option_data['iv'] = option_data['iv'] / 100.0
                
        return option_data, spot_data
        
    async def _get_current_price(self, asset: str) -> Optional[float]:
        """Get current price"""
        try:
            df = self.collector.get_latest_data(window_seconds=30)
            if df.empty:
                return None
                
            spot_df = df[(df['symbol'] == asset) & (df['data_type'] == 'spot')]
            if spot_df.empty:
                return None
                
            return float(spot_df.iloc[-1]['price'])
            
        except Exception as e:
            logger.error(f"Error getting price for {asset}: {e}")
            return None
            
    async def _get_market_data(self, asset: str) -> Dict:
        """Get comprehensive market data for asset"""
        try:
            df = self.collector.get_latest_data(window_seconds=300)
            asset_data = df[df['symbol'] == asset]
            
            if asset_data.empty:
                return {}
                
            prices = asset_data[asset_data['data_type'] == 'spot']['price'].values
            volumes = asset_data[asset_data['data_type'] == 'spot']['volume'].values
            
            market_data = {
                'price': prices[-1] if len(prices) > 0 else 0,
                'price_change_5m': ((prices[-1] - prices[0]) / prices[0] * 100) if len(prices) > 1 else 0,
                'volatility': np.std(prices) / np.mean(prices) if len(prices) > 1 else 0,
                'volume_mean': np.mean(volumes) if len(volumes) > 0 else 0,
                'volume_surge': volumes[-1] / np.mean(volumes) if len(volumes) > 1 and np.mean(volumes) > 0 else 1
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {}
            
    async def _capture_market_snapshot_for_signal(self, asset: str, 
                                                gamma_analysis: Dict,
                                                behavior_analysis: Dict) -> Dict:
        """Capture market snapshot"""
        snapshot = {}
        
        try:
            df = self.collector.get_latest_data(window_seconds=60)
            asset_data = df[df['symbol'] == asset]
            
            if not asset_data.empty:
                latest = asset_data.iloc[-1]
                snapshot['price'] = latest.get('price', 0)
                snapshot['bid'] = latest.get('bid', 0)
                snapshot['ask'] = latest.get('ask', 0)
                snapshot['spread'] = snapshot['ask'] - snapshot['bid']
                snapshot['volume'] = latest.get('volume', 0)
                
            ob_data = df[(df['symbol'] == asset) & (df['data_type'] == 'orderbook')]
            if not ob_data.empty:
                latest_ob = ob_data.iloc[-1]
                snapshot['bid_depth'] = latest_ob.get('bid_volume', 0)
                snapshot['ask_depth'] = latest_ob.get('ask_volume', 0)
                snapshot['orderbook_imbalance'] = (
                    (snapshot['bid_depth'] - snapshot['ask_depth']) / 
                    (snapshot['bid_depth'] + snapshot['ask_depth'])
                    if (snapshot['bid_depth'] + snapshot['ask_depth']) > 0 else 0
                )
                
        except Exception as e:
            logger.error(f"Error capturing market snapshot: {e}")
            
        return snapshot
        
    async def shutdown(self):
        """Shutdown system"""
        logger.info("\n Shutting down Adaptive System...")
        self.running = False
        
        if self.adaptive_learner:
            # Save learning report
            logger.info(" Saving learning report...")
            learning_report = self.adaptive_learner.get_learning_report()
            
            report_path = f'test_output/learning_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(report_path, 'w') as f:
                json.dump(learning_report, f, indent=2, default=str)
            logger.info(f"Learning report saved to {report_path}")
                
        if self.collector:
            logger.info(" Exporting data...")
            data_path = f'test_output/gamma_data_adaptive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            self.collector.export_data(data_path)
            
            await self.collector.stop()
            
        logger.info(" ✓ Shutdown complete")

async def main():
    """Main function"""
    config = {
        'data_collection': {
            'deribit': {
                'enabled': True,
                'symbols': ['BTC', 'ETH'],
                'interval': 30
            },
            'binance': {
                'enabled': True,
                'symbols': ['BTCUSDT', 'ETHUSDT'],
                'interval': 1
            },
            'buffer_size': 2000,
            'export_interval': 300
        },
        'gamma_analysis': {
            'interval': 60,
            'wall_percentile': 90,
            'history_window': 100,
            'gamma_decay_factor': 0.95,
            'hedge_flow_threshold': 0.7,
        },
        'market_behavior': {
            'interval': 30,
            'order_flow': {
                'sweep_threshold': 2.5,
                'frequency_window': 60
            },
            'divergence': {
                'lookback_period': 20,
                'significance_level': 0.05,
                'min_duration': 3
            },
            'cross_market': {
                'correlation_threshold': 0.7,
                'max_lag': 300,
                'min_observations': 100
            },
            'learning_params': {
                'enable_ml': True,
                'update_frequency': 3600
            }
        },
        'signal_generation': {
            'interval': 60,
            'min_strength': 60,
            'min_confidence': 0.6,
            'signal_cooldown': 300
        },
        'performance_tracking': {
            'signal_db_path': 'test_output/signal_performance_adaptive.csv',
            'decision_db_path': 'test_output/decision_history_adaptive.csv',
            'check_intervals': [5/60, 15/60, 30/60, 1, 2, 4, 8, 24],
            'update_interval': 300,
            'report_interval': 1800,
            'continuous_evaluation_interval': 300
        },
        'adaptive_learning': {
            'enabled': True,
            'learning_rate': 0.1,
            'learning_interval': 1800,  # 30 minutes for testing
            'min_decisions_for_learning': 20,
            'continuous_decision_interval': 300,  # 5 minutes
            'parameter_bounds': {
                'market_behavior.order_flow.sweep_threshold': (2.0, 4.0),
                'market_behavior.divergence.min_duration': (2, 10),
                'market_behavior.divergence.lookback_period': (10, 30),
                'gamma_analysis.wall_percentile': (70, 95),
                'gamma_analysis.hedge_flow_threshold': (0.5, 0.9),
                'signal_generation.min_strength': (40, 70),
                'signal_generation.min_confidence': (0.3, 0.8),
            }
        },
        'display_interval': 30,
        'debug_mode': True
    }
    
    system = AdaptiveGammaSqueezeSystem(config)
    
    try:
        await system.initialize()
        await system.start()
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
    finally:
        await system.shutdown()

if __name__ == "__main__":
    logger.info(f"{Fore.GREEN} Self-Evolving Gamma Squeeze Signal System{Style.RESET_ALL}")
    logger.info(f"{Fore.YELLOW}   ✓ Continuous Decision Monitoring{Style.RESET_ALL}")
    logger.info(f"{Fore.YELLOW}   ✓ Adaptive Parameter Optimization{Style.RESET_ALL}")
    logger.info(f"{Fore.YELLOW}   ✓ Real-time Learning from Market Feedback{Style.RESET_ALL}")
    logger.info(f"{Fore.YELLOW}   ✓ Performance-Based Evolution{Style.RESET_ALL}")
    logger.info("=" * 80)
    logger.info("Key Features:")
    logger.info("  • Records market decisions every 5 minutes")
    logger.info("  • Learns from both signals and non-signals")
    logger.info("  • Adjusts parameters based on decision quality")
    logger.info("  • Tracks missed opportunities and avoided bad signals")
    logger.info("  • Provides comprehensive learning reports")
    logger.info("=" * 80)
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 80)
    
    import os
    os.makedirs('test_output', exist_ok=True)
    
    asyncio.run(main())
