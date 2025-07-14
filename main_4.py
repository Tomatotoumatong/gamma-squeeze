#!/usr/bin/env python3
"""
Gamma Squeeze Signal System - Self-Evolving Edition
Integrates AdaptiveLearner for continuous parameter optimization
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
    """Self-Evolving Gamma Squeeze Signal System"""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or self._default_config()
        
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
        self.decision_history = []
        
        # Learning state
        self.last_learning_time = datetime.utcnow()
        self.learning_cycle_count = 0
        self.parameter_version = 0
        
        # Performance metrics
        self.performance_stats = {}
        self.learning_effectiveness = {}
        
        self._setup_signal_handlers()
        
    def _default_config(self):
        """Default configuration with learning parameters"""
        return {
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
                    'sweep_threshold': 3.0,
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
                'decision_interval': 300
            },
            'adaptive_learning': {
                'enabled': True,
                'learning_interval': 3600,  # Learn every hour
                'min_decisions_for_learning': 20,
                'performance_threshold': 0.6,  # Trigger learning if performance < 60%
                'learning_rate': 0.1,
                'parameter_bounds': {
                    'gamma_pressure.thresholds.critical': (70, 95),
                    'gamma_pressure.thresholds.high': (50, 80),
                    'gamma_pressure.wall_proximity_weight': (0.1, 0.5),
                    'gamma_pressure.hedge_flow_weight': (0.1, 0.5),
                    'market_momentum.sweep_weight': (0.2, 0.6),
                    'market_momentum.divergence_weight': (0.1, 0.5),
                    'signal_generation.min_strength': (40, 70),
                    'signal_generation.min_confidence': (0.3, 0.8),
                },
                'state_file': 'adaptive_learner_state.json'
            },
            'display_interval': 30,
            'debug_mode': True
        }
        
    def _setup_signal_handlers(self):
        """Setup signal handlers"""
        def signal_handler(sig, frame):
            logger.info("\n⚠️ Received interrupt signal, shutting down gracefully...")
            asyncio.create_task(self.shutdown())
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    async def initialize(self):
        """Initialize system with adaptive learning"""
        logger.info("=" * 80)
        logger.info("🚀 Initializing Self-Evolving Gamma Squeeze Signal System")
        logger.info("   ✓ Data Collection")
        logger.info("   ✓ Pattern Recognition") 
        logger.info("   ✓ Signal Generation")
        logger.info("   ✓ Performance Tracking")
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
        self.adaptive_learner.set_performance_tracker(self.performance_tracker)
        
        # Set up data fetchers
        self.performance_tracker.set_price_fetcher(self._get_current_price)
        self.performance_tracker.set_market_data_fetcher(self._get_market_data)
        
        logger.info("✅ All components initialized successfully")
        logger.info(f"🧠 Adaptive Learning: {'ENABLED' if self.config['adaptive_learning']['enabled'] else 'DISABLED'}")
        
    async def start(self):
        """Start system with learning loops"""
        logger.info("\n📊 Starting Self-Evolving System...")
        self.running = True
        
        await self.collector.start()
        await asyncio.sleep(15)  # Initial data accumulation
        
        tasks = [
            asyncio.create_task(self._monitor_loop()),
            asyncio.create_task(self._gamma_analysis_loop()),
            asyncio.create_task(self._behavior_detection_loop()),
            asyncio.create_task(self._signal_generation_loop()),
            asyncio.create_task(self._performance_update_loop()),
            asyncio.create_task(self._performance_report_loop()),
            asyncio.create_task(self._decision_recording_loop()),
            asyncio.create_task(self._adaptive_learning_loop()),  # New: Learning loop
            asyncio.create_task(self._learning_report_loop())     # New: Learning reports
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _adaptive_learning_loop(self):
        """Main adaptive learning loop"""
        while self.running:
            try:
                await asyncio.sleep(self.config['adaptive_learning']['learning_interval'])
                
                if not self.config['adaptive_learning']['enabled']:
                    continue
                
                # Check if enough data for learning
                if len(self.decision_history) < self.config['adaptive_learning']['min_decisions_for_learning']:
                    logger.info(f"🧠 Not enough decisions for learning: {len(self.decision_history)}")
                    continue
                
                # Perform learning
                await self._perform_adaptive_learning()
                
            except Exception as e:
                logger.error(f"Error in adaptive learning loop: {e}", exc_info=True)
                
    async def _perform_adaptive_learning(self):
        """Perform adaptive learning cycle"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🧠 ADAPTIVE LEARNING CYCLE #{self.learning_cycle_count + 1}")
        logger.info(f"{'='*60}")
        
        try:
            # Get recent performance stats
            performance_stats = self.performance_tracker.get_performance_stats(lookback_days=7)
            
            if not performance_stats:
                logger.info("No performance data available for learning")
                return
            
            # Get current configurations
            current_configs = {
                'gamma_analysis': self.gamma_analyzer.get_current_config() if hasattr(self.gamma_analyzer, 'get_current_config') else self.config['gamma_analysis'],
                'market_behavior': self.behavior_detector.get_current_config() if hasattr(self.behavior_detector, 'get_current_config') else self.config['market_behavior'],
                'signal_generation': self.signal_evaluator.config
            }
            
            # Prepare decision history with performance scores
            enriched_decisions = self._enrich_decision_history()
            
            # Learn from decisions
            learning_result = self.adaptive_learner.learn_from_decisions(
                enriched_decisions,
                performance_stats,
                current_configs
            )
            
            if learning_result and 'adjustments' in learning_result:
                # Apply parameter adjustments
                await self._apply_parameter_adjustments(learning_result['adjustments'])
                
                # Update learning metrics
                self.learning_effectiveness = {
                    'cycle': self.learning_cycle_count + 1,
                    'timestamp': datetime.utcnow(),
                    'adjustments_made': len(learning_result['adjustments']),
                    'regime': learning_result.get('regime', 'unknown'),
                    'confidence': learning_result.get('confidence', 0),
                    'expected_improvement': learning_result['learning_decision'].expected_improvement
                }
                
                self.learning_cycle_count += 1
                self.parameter_version += 1
                
                # Print learning summary
                self._print_learning_summary(learning_result)
            else:
                logger.info("🧠 No parameter adjustments needed")
                
        except Exception as e:
            logger.error(f"Error in adaptive learning: {e}", exc_info=True)
            
    def _enrich_decision_history(self) -> List[Dict]:
        """Enrich decision history with performance scores"""
        enriched = []
        
        for i, decision in enumerate(self.decision_history[-100:]):  # Last 100 decisions
            enriched_decision = {
                'timestamp': decision['timestamp'],
                'gamma_metrics': decision.get('gamma_metrics', {}),
                'behavior_metrics': decision.get('behavior_metrics', {}),
                'config_used': decision.get('config_snapshot', {}),
                'performance_score': 0.5  # Default neutral score
            }
            
            # If a signal was generated, get its performance
            if 'signal_id' in decision and decision['signal_id']:
                signal_perf = self._get_signal_performance(decision['signal_id'])
                if signal_perf:
                    # Calculate composite performance score
                    enriched_decision['performance_score'] = self._calculate_composite_score(signal_perf)
            
            enriched.append(enriched_decision)
            
        return enriched
        
    def _get_signal_performance(self, signal_id: str) -> Optional[Dict]:
        """Get performance data for a signal"""
        try:
            df = pd.read_csv(self.config['performance_tracking']['signal_db_path'])
            signal_data = df[df['signal_id'] == signal_id]
            
            if not signal_data.empty:
                return signal_data.iloc[0].to_dict()
                
        except Exception as e:
            logger.error(f"Error getting signal performance: {e}")
            
        return None
        
    def _calculate_composite_score(self, signal_perf: Dict) -> float:
        """Calculate composite performance score"""
        scores = []
        
        # Direction score (most important)
        if pd.notna(signal_perf.get('direction_score')):
            scores.append(signal_perf['direction_score'] * 0.4)
            
        # Timing score
        if pd.notna(signal_perf.get('timing_score')):
            scores.append(signal_perf['timing_score'] * 0.3)
            
        # Persistence score
        if pd.notna(signal_perf.get('persistence_score')):
            scores.append(signal_perf['persistence_score'] * 0.2)
            
        # Robustness score
        if pd.notna(signal_perf.get('robustness_score')):
            scores.append(signal_perf['robustness_score'] * 0.1)
            
        return sum(scores) if scores else 0.5
        
    async def _apply_parameter_adjustments(self, adjustments: Dict[str, float]):
        """Apply parameter adjustments to components"""
        logger.info(f"\n🔧 Applying {len(adjustments)} parameter adjustments")
        
        success_count = 0
        
        for param_path, adjustment in adjustments.items():
            try:
                # Parse parameter path
                parts = param_path.split('.')
                module = parts[0]
                
                # Apply to appropriate module
                if module == 'gamma_pressure' and len(parts) > 1:
                    # Update gamma analyzer
                    update_dict = self._build_nested_dict(param_path, adjustment)
                    if self.gamma_analyzer.update_parameters(update_dict):
                        success_count += 1
                        logger.info(f"✓ Updated {param_path}: {adjustment:.3f}")
                        
                elif module == 'market_momentum' and len(parts) > 1:
                    # Update behavior detector
                    update_dict = self._build_nested_dict(param_path, adjustment)
                    if self.behavior_detector.update_parameters(update_dict):
                        success_count += 1
                        logger.info(f"✓ Updated {param_path}: {adjustment:.3f}")
                        
                elif module == 'signal_generation' and len(parts) > 1:
                    # Update signal evaluator
                    clean_path = '.'.join(parts[1:])  # Remove 'signal_generation' prefix
                    update_dict = self._build_nested_dict(clean_path, adjustment)
                    if self.signal_evaluator.update_parameters(update_dict, f"Adaptive Learning Cycle {self.learning_cycle_count}"):
                        success_count += 1
                        logger.info(f"✓ Updated {param_path}: {adjustment:.3f}")
                        
            except Exception as e:
                logger.error(f"Failed to apply adjustment {param_path}: {e}")
                
        logger.info(f"Successfully applied {success_count}/{len(adjustments)} adjustments")
        
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
        
    def _print_learning_summary(self, learning_result: Dict):
        """Print learning cycle summary"""
        print(f"\n{Fore.CYAN}🧠 LEARNING SUMMARY{Style.RESET_ALL}")
        print(f"Regime: {learning_result.get('regime', 'unknown')}")
        print(f"Confidence: {learning_result.get('confidence', 0):.2f}")
        print(f"\nAdjustments:")
        
        for param, value in learning_result['adjustments'].items():
            print(f"  {param}: {value:+.3f}")
            
        decision = learning_result['learning_decision']
        print(f"\nExpected Improvement: {decision.expected_improvement:+.2%}")
        print(f"Decision Basis: {decision.decision_basis}")
        print(f"Exploration Factor: {decision.exploration_factor:.2f}")
        
    async def _learning_report_loop(self):
        """Generate periodic learning reports"""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                if self.adaptive_learner:
                    report = self.adaptive_learner.get_learning_report()
                    self._print_learning_report(report)
                    
            except Exception as e:
                logger.error(f"Error in learning report loop: {e}", exc_info=True)
                
    def _print_learning_report(self, report: Dict):
        """Print comprehensive learning report"""
        print(f"\n{Fore.MAGENTA}📚 ADAPTIVE LEARNING REPORT{Style.RESET_ALL}")
        print("=" * 80)
        
        print(f"\nLearning Statistics:")
        print(f"  Total Learning Decisions: {report['total_learning_decisions']}")
        print(f"  Current Regime: {report['current_regime']}")
        
        # Regime distribution
        print(f"\nRegime Distribution:")
        for regime, pct in report['regime_distribution'].items():
            print(f"  {regime}: {pct:.1%}")
            
        # Parameter evolution
        print(f"\nParameter Evolution:")
        for param, info in report['parameter_evolution'].items():
            if info['total_adjustments'] > 0:
                print(f"  {param}:")
                print(f"    Net Change: {info['net_change']:+.3f}")
                print(f"    Volatility: {info['volatility']:.3f}")
                print(f"    Trend: {info['trend']}")
                
        # Learning effectiveness
        effectiveness = report['learning_effectiveness']
        if effectiveness.get('confidence', 0) > 0:
            print(f"\nLearning Effectiveness:")
            print(f"  Performance Change: {effectiveness['effectiveness']:+.2%}")
            print(f"  Confidence: {effectiveness['confidence']:.2f}")
            print(f"  Improvement Rate: {effectiveness['improvement_rate']:.1%}")
            
        # Recommendations
        if report['recommendations']['high_uncertainty_params']:
            print(f"\n⚠️ High Uncertainty Parameters:")
            for param_info in report['recommendations']['high_uncertainty_params']:
                print(f"  {param_info['parameter']} (variance: {param_info['variance']:.3f})")
                
        print("=" * 80)
        
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
                
                # Track signals with learning context
                for signal in signals:
                    self.generated_signals.append(signal)
                    
                    # Add learning metadata
                    signal.metadata['adaptive_context'] = {
                        'parameter_version': self.parameter_version,
                        'learning_cycle': self.learning_cycle_count,
                        'regime': self.learning_effectiveness.get('regime', 'unknown')
                    }
                    
                    # Track signal
                    market_snapshot = await self._capture_market_snapshot_for_signal(
                        signal.asset, latest_gamma, latest_behavior
                    )
                    
                    self.performance_tracker.track_signal_with_context(signal, {
                        'current_price': await self._get_current_price(signal.asset),
                        'spread': market_snapshot.get('spread', 0),
                        'ob_imbalance': market_snapshot.get('orderbook_imbalance', 0),
                        'parameter_version': self.parameter_version,
                        'learning_active': self.config['adaptive_learning']['enabled']
                    })
                    
                    self._print_signal_with_learning_context(signal)
                    
            except Exception as e:
                logger.error(f"Error in signal generation: {e}", exc_info=True)
                
    def _print_signal_with_learning_context(self, signal: TradingSignal):
        """Print signal with learning context"""
        print(f"\n{Fore.GREEN}🎯 SIGNAL GENERATED (v{self.parameter_version}){Style.RESET_ALL}")
        print(f"Asset: {signal.asset} | Direction: {signal.direction}")
        print(f"Strength: {signal.strength} | Confidence: {signal.confidence}")
        
        if 'adaptive_context' in signal.metadata:
            ctx = signal.metadata['adaptive_context']
            print(f"Learning Context:")
            print(f"  Parameter Version: {ctx['parameter_version']}")
            print(f"  Learning Cycle: {ctx['learning_cycle']}")
            print(f"  Market Regime: {ctx['regime']}")
            
    async def _decision_recording_loop(self):
        """Record decisions with learning context"""
        while self.running:
            try:
                await asyncio.sleep(self.config['performance_tracking']['decision_interval'])
                
                decision = await self._record_current_decision()
                if decision:
                    # Add learning context
                    decision['parameter_version'] = self.parameter_version
                    decision['learning_cycle'] = self.learning_cycle_count
                    decision['config_snapshot'] = {
                        'gamma_analysis': self.gamma_analyzer.config if hasattr(self.gamma_analyzer, 'config') else {},
                        'market_behavior': self.behavior_detector.config if hasattr(self.behavior_detector, 'config') else {},
                        'signal_generation': self.signal_evaluator.config
                    }
                    
                    self.decision_history.append(decision)
                    
                    # Keep history size manageable
                    if len(self.decision_history) > 1000:
                        self.decision_history = self.decision_history[-1000:]
                        
            except Exception as e:
                logger.error(f"Error in decision recording: {e}", exc_info=True)
                
    async def _record_current_decision(self) -> Optional[Dict]:
        """Record current decision state"""
        if not self.analysis_results or not self.behavior_results:
            return None
            
        latest_gamma = self.analysis_results[-1]
        latest_behavior = self.behavior_results[-1]
        
        decision = {
            'timestamp': datetime.utcnow(),
            'gamma_metrics': self._extract_gamma_metrics(latest_gamma),
            'behavior_metrics': self._extract_behavior_metrics(latest_behavior)
        }
        
        return decision
        
    def _extract_gamma_metrics(self, gamma_analysis: Dict) -> Dict:
        """Extract key gamma metrics"""
        metrics = {}
        
        for asset, dist in gamma_analysis.get('gamma_distribution', {}).items():
            metrics[f'{asset}_total_gamma'] = dist.get('total_exposure', 0)
            metrics[f'{asset}_concentration'] = dist.get('concentration', 0)
            
        metrics['total_walls'] = len(gamma_analysis.get('gamma_walls', []))
        
        return metrics
        
    def _extract_behavior_metrics(self, behavior_analysis: Dict) -> Dict:
        """Extract key behavior metrics"""
        metrics = {}
        metrics['sweep_count'] = len(behavior_analysis.get('sweep_orders', []))
        metrics['divergence_count'] = len(behavior_analysis.get('divergences', []))
        metrics['market_regime'] = behavior_analysis.get('market_regime', {}).get('state', 'normal')
        raw_metrics = behavior_analysis.get('raw_metrics', {})
        if 'feature_matrix' in raw_metrics and raw_metrics['feature_matrix'] is not None:
            metrics['feature_matrix'] = raw_metrics['feature_matrix'].tolist()
        
        return metrics
        
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
                
                # Learning status
                if self.config['adaptive_learning']['enabled']:
                    status_parts.append(f"Learn: v{self.parameter_version}")
                    
                    if self.learning_effectiveness:
                        expected_imp = self.learning_effectiveness.get('expected_improvement', 0)
                        status_parts.append(f"Exp.Δ: {expected_imp:+.1%}")
                        
                # Performance
                if self.performance_stats:
                    composite = self.performance_stats.get('composite_score', 0)
                    status_parts.append(f"Perf: {composite:.2f}")
                    
                status_line = " | ".join(status_parts)
                print(f"\r🔄 {status_line}", end='', flush=True)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                
    # [Include all other necessary methods from main_3.py with minimal modifications]
    # _gamma_analysis_loop, _behavior_detection_loop, _performance_update_loop, etc.
    # These remain largely the same as main_3.py
    
    async def _prepare_analysis_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                
    async def _performance_update_loop(self):
        """Update performance metrics"""
        while self.running:
            try:
                await asyncio.sleep(self.config['performance_tracking']['update_interval'])
                
                await self.performance_tracker.update_prices()
                
                self.performance_stats = self.performance_tracker.get_performance_stats(
                    lookback_days=7
                )
                
            except Exception as e:
                logger.error(f"Error in performance update: {e}", exc_info=True)
                
    async def _performance_report_loop(self):
        """Performance reporting"""
        while self.running:
            try:
                await asyncio.sleep(self.config['performance_tracking']['report_interval'])
                
                if self.performance_stats:
                    self._print_performance_report()
                    
            except Exception as e:
                logger.error(f"Error in performance report: {e}", exc_info=True)
                
    def _print_performance_report(self):
        """Print performance report"""
        stats = self.performance_stats
        
        print(f"\n{Fore.YELLOW}📊 PERFORMANCE REPORT{Style.RESET_ALL}")
        print("=" * 60)
        
        print(f"Total Signals: {stats.get('total_signals', 0)}")
        print(f"Composite Score: {stats.get('composite_score', 0):.2f}")
        
        if self.learning_effectiveness:
            print(f"\nLearning Status:")
            print(f"  Parameter Version: v{self.parameter_version}")
            print(f"  Learning Cycles: {self.learning_cycle_count}")
            print(f"  Last Adjustment: {self.learning_effectiveness.get('adjustments_made', 0)} params")
            
        print("=" * 60)
        
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
        logger.info("\n🛑 Shutting down Adaptive System...")
        self.running = False
        
        if self.adaptive_learner:
            # Save learning state
            logger.info("💾 Saving learning state...")
            learning_report = self.adaptive_learner.get_learning_report()
            
            with open(f'test_output/learning_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
                json.dump(learning_report, f, indent=2)
                
        if self.collector:
            logger.info("💾 Exporting data...")
            self.collector.export_data(f'test_output/gamma_data_adaptive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            
            await self.collector.stop()
            
        logger.info("✅ Shutdown complete")

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
                'sweep_threshold': 3.0,
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
                'update_frequency': 3600  # 模型更新频率（秒）
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
            'decision_interval': 300
        },
        'adaptive_learning': {
            'enabled': True,
            'learning_interval': 1800,  # 30 minutes for testing
            'min_decisions_for_learning': 10,  # Lower for testing
            'performance_threshold': 0.6,
            'learning_rate': 0.1,
            'parameter_bounds': {
                'gamma_pressure.thresholds.critical': (70, 95),
                'gamma_pressure.thresholds.high': (50, 80),
                'gamma_pressure.wall_proximity_weight': (0.1, 0.5),
                'gamma_pressure.hedge_flow_weight': (0.1, 0.5),
                'market_momentum.sweep_weight': (0.2, 0.6),
                'market_momentum.divergence_weight': (0.1, 0.5),
                'signal_generation.min_strength': (40, 70),
                'signal_generation.min_confidence': (0.3, 0.8),
            },
            'state_file': 'adaptive_learner_state.json'
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
    print(f"{Fore.GREEN}🚀 Self-Evolving Gamma Squeeze Signal System{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}   ✓ Continuous Adaptive Learning{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}   ✓ Dynamic Parameter Optimization{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}   ✓ Market Regime Recognition{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}   ✓ Performance-Based Evolution{Style.RESET_ALL}")
    print("=" * 80)
    print("Features:")
    print("  • Learns from signal performance every hour")
    print("  • Adjusts parameters based on market regimes")
    print("  • Tracks parameter evolution and effectiveness")
    print("  • Provides learning reports and recommendations")
    print("  • Continuously improves signal accuracy")
    print("=" * 80)
    print("Press Ctrl+C to stop")
    print("=" * 80)
    
    import os
    os.makedirs('test_output', exist_ok=True)
    
    asyncio.run(main())