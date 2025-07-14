#!/usr/bin/env python3
"""
Gamma Squeeze Signal System - Enhanced Performance Tracking
Modified to utilize the enhanced PerformanceTracker capabilities
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

# Import system modules
from UnifiedDataCollector import UnifiedDataCollector, DataType
from GammaPressureAnalyzer import GammaPressureAnalyzer
from MarketBehaviorDetector import MarketBehaviorDetector
from SignalEvaluator import SignalEvaluator, TradingSignal
from PerformanceTracker import PerformanceTracker, SignalPerformance

# Initialize colorama
init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_output/gamma_squeeze_system_enhanced.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class GammaSqueezeSystem:
    """Enhanced Gamma Squeeze Signal System with Decision Tracking"""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or self._default_config()
        self.collector = None
        self.gamma_analyzer = None
        self.behavior_detector = None
        self.signal_evaluator = None
        self.performance_tracker = None
        self.running = False
        self.analysis_results = []
        self.behavior_results = []
        self.generated_signals = []
        self.performance_stats = {}
        self.last_decision_time = datetime.utcnow()
        self._setup_signal_handlers()
        
    def _default_config(self):
        """Default configuration with enhanced tracking"""
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
                'wall_percentile': 80,
                'history_window': 100
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
                }
            },
            'signal_generation': {
                'interval': 60,
                'min_strength': 50,
                'min_confidence': 0.5
            },
            'performance_tracking': {
                'signal_db_path': 'test_output/signal_performance_enhanced.csv',
                'decision_db_path': 'test_output/decision_history.csv',
                'check_intervals': [5/60, 15/60, 30/60, 1, 2, 4, 8, 24],
                'update_interval': 300,
                'report_interval': 1800,
                'decision_interval': 300  # Record decision every 5 minutes
            },
            'display_interval': 30,
            'debug_mode': True,
            'enhanced_tracking': True  # Enable enhanced features
        }
        
    def _setup_signal_handlers(self):
        """Setup signal handlers"""
        def signal_handler(sig, frame):
            logger.info("\n⚠️ Received interrupt signal, shutting down gracefully...")
            asyncio.create_task(self.shutdown())
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    async def initialize(self):
        """Initialize system"""
        logger.info("=" * 80)
        logger.info("🚀 Initializing Enhanced Gamma Squeeze Signal System")
        logger.info("   ✓ Data Collection")
        logger.info("   ✓ Pattern Recognition")
        logger.info("   ✓ Signal Generation")
        logger.info("   ✓ Enhanced Performance Tracking with Decision Recording")
        logger.info("=" * 80)
        
        # Create components
        self.collector = UnifiedDataCollector(self.config['data_collection'])
        await self.collector.initialize()
        
        self.gamma_analyzer = GammaPressureAnalyzer(self.config['gamma_analysis'])
        self.behavior_detector = MarketBehaviorDetector(self.config['market_behavior'])
        self.signal_evaluator = SignalEvaluator(self.config['signal_generation'])
        self.performance_tracker = PerformanceTracker(self.config['performance_tracking'])
        
        # Set up data fetchers for performance tracker
        self.performance_tracker.set_price_fetcher(self._get_current_price)
        self.performance_tracker.set_market_data_fetcher(self._get_market_data)
        
        logger.info("✅ All components initialized successfully")
        
    async def start(self):
        """Start system"""
        logger.info("\n📊 Starting Enhanced System...")
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
            asyncio.create_task(self._decision_recording_loop())  # New task
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _decision_recording_loop(self):
        """Continuously record decisions at regular intervals"""
        while self.running:
            try:
                await asyncio.sleep(self.config['performance_tracking']['decision_interval'])
                
                # Record current decision state
                await self._record_current_decision()
                
            except Exception as e:
                logger.error(f"Error in decision recording: {e}", exc_info=True)
                
    async def _record_current_decision(self):
        """Record the current decision state"""
        if not self.analysis_results or not self.behavior_results:
            return
            
        latest_gamma = self.analysis_results[-1]
        latest_behavior = self.behavior_results[-1]
        market_data = self.collector.get_latest_data(window_seconds=300)
        
        # Get analyzed assets
        assets = set()
        if latest_gamma.get('gamma_distribution'):
            assets.update(latest_gamma['gamma_distribution'].keys())
            
        # Calculate scores for each asset
        scores = {}
        suppressed_signals = {}
        
        for asset in assets:
            try:
                asset_scores = self.signal_evaluator._calculate_scores(
                    asset, latest_gamma, latest_behavior, market_data
                )
                scores[asset] = asset_scores
                
                # Check if signal was suppressed
                should_generate = self.signal_evaluator._should_generate_signal(asset, asset_scores)
                if not should_generate:
                    suppressed_signals[asset] = self._analyze_suppression_reason(
                        asset, asset_scores, latest_gamma, latest_behavior
                    )
                    
            except Exception as e:
                logger.error(f"Error calculating scores for {asset}: {e}")
                
        # Record the decision
        await self.performance_tracker.record_decision(
            assets_analyzed=list(assets),
            gamma_analysis=latest_gamma,
            market_behavior=latest_behavior,
            scores=scores,
            signals_generated=[],  # Will be filled if signals are generated
            suppressed_signals=suppressed_signals
        )
        
        # Debug output
        if self.config['enhanced_tracking']:
            self._print_decision_record(assets, scores, suppressed_signals)
            
    def _analyze_suppression_reason(self, asset: str, scores: Dict[str, float],
                                   gamma_analysis: Dict, behavior_analysis: Dict) -> str:
        """Analyze why a signal was suppressed"""
        reasons = []
        
        # Check score thresholds
        composite_score = np.mean(list(scores.values()))
        if composite_score < self.config['signal_generation']['min_strength']:
            reasons.append(f"Low composite score: {composite_score:.1f}")
            
        # Check cooldown
        if not self.signal_evaluator._check_cooldown(asset):
            reasons.append("In cooldown period")
            
        # Check individual dimensions
        high_threshold = self.config['gamma_analysis'].get('wall_percentile', 80)
        low_scoring_dims = [dim for dim, score in scores.items() if score < high_threshold]
        if low_scoring_dims:
            reasons.append(f"Low scores in: {', '.join(low_scoring_dims)}")
            
        return " | ".join(reasons) if reasons else "Unknown"
        
    async def _signal_generation_loop(self):
        """Signal generation with enhanced tracking"""
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
                
                # Enhanced tracking for generated signals
                for signal in signals:
                    self.generated_signals.append(signal)
                    
                    # Get comprehensive market snapshot
                    market_snapshot = await self._capture_market_snapshot_for_signal(
                        signal.asset, latest_gamma, latest_behavior
                    )
                    
                    # Track with context
                    self.performance_tracker.track_signal_with_context(signal, {
                        'current_price': await self._get_current_price(signal.asset),
                        'spread': market_snapshot.get('spread', 0),
                        'ob_imbalance': market_snapshot.get('orderbook_imbalance', 0),
                        'gamma_concentration': self._get_gamma_concentration(signal.asset, latest_gamma),
                        'nearest_strike_distance': self._get_nearest_strike_distance(signal.asset, latest_gamma),
                        'strategy': signal.signal_type,
                        'confidence_breakdown': signal.metadata.get('scores', {}),
                        'decision_time': (datetime.utcnow() - signal.timestamp).total_seconds() * 1000
                    })
                    
                    self._print_enhanced_signal_info(signal, market_snapshot)
                    
            except Exception as e:
                logger.error(f"Error in signal generation: {e}", exc_info=True)
                
    async def _capture_market_snapshot_for_signal(self, asset: str, 
                                                gamma_analysis: Dict,
                                                behavior_analysis: Dict) -> Dict:
        """Capture comprehensive market snapshot"""
        snapshot = {}
        
        try:
            # Get current market data
            df = self.collector.get_latest_data(window_seconds=60)
            asset_data = df[df['symbol'] == asset]
            
            if not asset_data.empty:
                latest = asset_data.iloc[-1]
                snapshot['price'] = latest.get('price', 0)
                snapshot['bid'] = latest.get('bid', 0)
                snapshot['ask'] = latest.get('ask', 0)
                snapshot['spread'] = snapshot['ask'] - snapshot['bid']
                snapshot['volume'] = latest.get('volume', 0)
                
            # Get orderbook data
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
                
            # Get behavior metrics
            snapshot['sweep_count'] = len(behavior_analysis.get('sweep_orders', []))
            snapshot['divergence_count'] = len(behavior_analysis.get('divergences', []))
            snapshot['market_regime'] = behavior_analysis.get('market_regime', {}).get('state', 'normal')
            
        except Exception as e:
            logger.error(f"Error capturing market snapshot: {e}")
            
        return snapshot
        
    def _get_gamma_concentration(self, asset: str, gamma_analysis: Dict) -> float:
        """Get gamma concentration for asset"""
        gamma_dist = gamma_analysis.get('gamma_distribution', {}).get(asset, {})
        return gamma_dist.get('concentration', 0)
        
    def _get_nearest_strike_distance(self, asset: str, gamma_analysis: Dict) -> float:
        """Get distance to nearest gamma wall"""
        indicators = gamma_analysis.get('pressure_indicators', {}).get(asset, {})
        return indicators.get('nearest_wall_distance', 100)
        
    async def _get_market_data(self, asset: str) -> Dict:
        """Get comprehensive market data for asset"""
        try:
            df = self.collector.get_latest_data(window_seconds=300)
            asset_data = df[df['symbol'] == asset]
            
            if asset_data.empty:
                return {}
                
            # Calculate various metrics
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
            
    def _print_decision_record(self, assets: set, scores: Dict, suppressed: Dict):
        """Print decision recording debug info"""
        print(f"\n{Fore.CYAN}📸 DECISION SNAPSHOT - {datetime.utcnow().strftime('%H:%M:%S')}{Style.RESET_ALL}")
        print(f"Assets analyzed: {', '.join(assets)}")
        
        for asset in assets:
            if asset in scores:
                asset_scores = scores[asset]
                composite = np.mean(list(asset_scores.values()))
                print(f"\n{asset}:")
                print(f"  Composite Score: {composite:.1f}")
                print(f"  Components: G:{asset_scores.get('gamma_pressure', 0):.0f} "
                      f"M:{asset_scores.get('market_momentum', 0):.0f} "
                      f"T:{asset_scores.get('technical', 0):.0f}")
                      
                if asset in suppressed:
                    print(f"  {Fore.YELLOW}Suppressed: {suppressed[asset]}{Style.RESET_ALL}")
                    
    def _print_enhanced_signal_info(self, signal: TradingSignal, snapshot: Dict):
        """Print enhanced signal information"""
        print(f"\n{Fore.GREEN}🎯 ENHANCED SIGNAL TRACKING{Style.RESET_ALL}")
        print(f"Asset: {signal.asset} | Direction: {signal.direction}")
        print(f"Market Context:")
        print(f"  Spread: ${snapshot.get('spread', 0):.2f}")
        print(f"  OB Imbalance: {snapshot.get('orderbook_imbalance', 0):.2%}")
        print(f"  Sweep Activity: {snapshot.get('sweep_count', 0)} sweeps")
        print(f"  Market Regime: {snapshot.get('market_regime', 'unknown')}")
        
    async def _performance_update_loop(self):
        """Update performance with enhanced metrics"""
        while self.running:
            try:
                await asyncio.sleep(self.config['performance_tracking']['update_interval'])
                
                # Update prices for all time intervals
                await self.performance_tracker.update_prices()
                
                # Get enhanced statistics
                self.performance_stats = self.performance_tracker.get_performance_stats(
                    lookback_days=7
                )
                
                # Print active signals with enhanced metrics
                if self.config['enhanced_tracking'] and self.performance_tracker.active_signals:
                    self._print_enhanced_active_signals()
                    
            except Exception as e:
                logger.error(f"Error in performance update: {e}", exc_info=True)
                
    def _print_enhanced_active_signals(self):
        """Print enhanced active signal information"""
        print(f"\n{Fore.BLUE}📈 ENHANCED ACTIVE SIGNALS UPDATE{Style.RESET_ALL}")
        print("─" * 80)
        
        for signal_id, perf in self.performance_tracker.active_signals.items():
            elapsed = (datetime.utcnow() - perf.signal_timestamp).total_seconds() / 3600
            
            print(f"\n{perf.asset} ({perf.direction}):")
            print(f"  Elapsed: {elapsed:.1f}h | Initial: ${perf.initial_price:.2f}")
            
            # Print multi-timeframe performance
            for interval in ['5m', '15m', '30m', '1h', '2h', '4h', '8h']:
                price_attr = f'price_{interval}'
                return_attr = f'return_{interval}'
                
                price = getattr(perf, price_attr, None)
                returns = getattr(perf, return_attr, None)
                
                if price is not None:
                    return_color = Fore.GREEN if returns > 0 else Fore.RED
                    print(f"  {interval}: ${price:.2f} ({return_color}{returns:+.2f}%{Style.RESET_ALL})")
                    
            # Print path metrics if available
            if perf.path_metrics:
                print(f"  Path Metrics:")
                print(f"    Max Favorable: {perf.path_metrics.max_favorable_move:.2f}%")
                print(f"    Max Adverse: {perf.path_metrics.max_adverse_move:.2f}%")
                
        print("─" * 80)
        
    async def _performance_report_loop(self):
        """Enhanced performance reporting"""
        while self.running:
            try:
                await asyncio.sleep(self.config['performance_tracking']['report_interval'])
                
                if self.performance_stats:
                    self._print_enhanced_performance_report()
                    
            except Exception as e:
                logger.error(f"Error in performance report: {e}", exc_info=True)
                
    def _print_enhanced_performance_report(self):
        """Print enhanced performance report"""
        stats = self.performance_stats
        
        print(f"\n{Fore.MAGENTA}📊 ENHANCED PERFORMANCE REPORT{Style.RESET_ALL}")
        print("=" * 80)
        
        # Overall metrics
        print(f"\n{Fore.YELLOW}Overall Performance:{Style.RESET_ALL}")
        print(f"  Total Signals: {stats.get('total_signals', 0)}")
        print(f"  Avg Direction Score: {stats.get('avg_direction_score', 0):.2f}")
        print(f"  Avg Timing Score: {stats.get('avg_timing_score', 0):.2f}")
        print(f"  Avg Persistence Score: {stats.get('avg_persistence_score', 0):.2f}")
        print(f"  Composite Score: {stats.get('composite_score', 0):.2f}")
        
        # Failure patterns
        failures = stats.get('failure_patterns', {})
        if failures:
            print(f"\n{Fore.RED}Failure Patterns:{Style.RESET_ALL}")
            for pattern, count in failures.items():
                print(f"  {pattern}: {count}")
                
        # Success patterns
        success = stats.get('success_patterns', {})
        if success:
            print(f"\n{Fore.GREEN}Success Patterns:{Style.RESET_ALL}")
            characteristics = success.get('high_performer_characteristics', {})
            if characteristics:
                print(f"  Avg Strength: {characteristics.get('avg_strength', 0):.1f}")
                print(f"  Avg Confidence: {characteristics.get('avg_confidence', 0):.1%}")
                print(f"  Best Timeframe: {characteristics.get('common_timeframe', 'N/A')}")
                
        # Missed opportunities
        missed = stats.get('missed_opportunities', {})
        if missed.get('total_missed', 0) > 0:
            print(f"\n{Fore.YELLOW}Missed Opportunities:{Style.RESET_ALL}")
            print(f"  Total: {missed['total_missed']}")
            print(f"  Avg Score: {missed.get('avg_opportunity_score', 0):.2f}")
            
        print("=" * 80)
        
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
            
    async def _monitor_loop(self):
        """Monitor loop with enhanced status"""
        while self.running:
            try:
                await asyncio.sleep(self.config['display_interval'])
                
                status_parts = []
                
                df = self.collector.get_latest_data(window_seconds=60)
                if not df.empty:
                    status_parts.append(f"Data: {len(df)}")
                
                if self.analysis_results:
                    walls = len(self.analysis_results[-1].get('gamma_walls', []))
                    status_parts.append(f"Walls: {walls}")
                
                active_signals = len(self.performance_tracker.active_signals)
                total_signals = len(self.generated_signals)
                status_parts.append(f"Signals: {active_signals}/{total_signals}")
                
                # Add decision count
                decision_count = len(self.performance_tracker.decision_history)
                status_parts.append(f"Decisions: {decision_count}")
                
                if self.performance_stats:
                    composite = self.performance_stats.get('composite_score', 0)
                    status_parts.append(f"Score: {composite:.2f}")
                
                status_line = " | ".join(status_parts)
                print(f"\r📊 {status_line}", end='', flush=True)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                
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
                
    async def shutdown(self):
        """Shutdown system"""
        logger.info("\n🛑 Shutting down Enhanced System...")
        self.running = False
        
        if self.collector:
            logger.info("💾 Exporting data...")
            self.collector.export_data(f'test_output/gamma_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            
            if self.generated_signals:
                self._export_signals()
                
            if self.performance_stats:
                self._export_enhanced_performance_report()
            
            await self.collector.stop()
            
        logger.info("✅ Shutdown complete")
        
    def _export_enhanced_performance_report(self):
        """Export enhanced performance report"""
        try:
            filename = f'test_output/enhanced_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            
            report = {
                'generation_time': datetime.now().isoformat(),
                'statistics': self.performance_stats,
                'active_signals': {},
                'decision_summary': {
                    'total_decisions': len(self.performance_tracker.decision_history),
                    'decisions_with_signals': sum(1 for d in self.performance_tracker.decision_history 
                                                if d.signal_generated),
                    'suppressed_signals': sum(1 for d in self.performance_tracker.decision_history 
                                            if d.suppression_reason)
                }
            }
            
            for signal_id, perf in self.performance_tracker.active_signals.items():
                report['active_signals'][signal_id] = {
                    'asset': perf.asset,
                    'direction': perf.direction,
                    'initial_price': perf.initial_price,
                    'signal_timestamp': perf.signal_timestamp.isoformat(),
                    'elapsed_hours': (datetime.utcnow() - perf.signal_timestamp).total_seconds() / 3600,
                    'multi_timeframe_returns': {
                        interval: getattr(perf, f'return_{interval}', None)
                        for interval in ['5m', '15m', '30m', '1h', '2h', '4h', '8h']
                    }
                }
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"📊 Enhanced report exported to {filename}")
        except Exception as e:
            logger.error(f"Error exporting enhanced report: {e}")
            
    def _export_signals(self):
        """Export signals"""
        try:
            filename = f'test_output/signals_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            
            signal_data = []
            for signal in self.generated_signals[-50:]:
                signal_dict = {
                    'timestamp': signal.timestamp.isoformat(),
                    'asset': signal.asset,
                    'signal_type': signal.signal_type,
                    'direction': signal.direction,
                    'strength': signal.strength,
                    'confidence': signal.confidence,
                    'expected_move': signal.expected_move,
                    'time_horizon': signal.time_horizon,
                    'key_levels': signal.key_levels,
                    'risk_factors': signal.risk_factors,
                    'metadata': signal.metadata
                }
                signal_data.append(signal_dict)
            
            with open(filename, 'w') as f:
                json.dump(signal_data, f, indent=2)
            
            logger.info(f"📊 Signals exported to {filename}")
        except Exception as e:
            logger.error(f"Error exporting signals: {e}")

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
                'min_duration': 2
            },
            'cross_market': {
                'correlation_threshold': 0.7,
                'max_lag': 300,
                'min_observations': 100
            },
            'learning_params': {
                'enable_ml': False
            }
        },
        'signal_generation': {
            'interval': 60,
            'min_strength': 60,
            'min_confidence': 0.6,
            'signal_cooldown': 300
        },
        'performance_tracking': {
            'signal_db_path': 'test_output/signal_performance_enhanced.csv',
            'decision_db_path': 'test_output/decision_history.csv',
            'check_intervals': [5/60, 15/60, 30/60, 1, 2, 4, 8, 24],
            'update_interval': 300,
            'report_interval': 1800,
            'decision_interval': 300
        },
        'display_interval': 30,
        'debug_mode': True,
        'enhanced_tracking': True
    }
    
    system = GammaSqueezeSystem(config)
    
    try:
        await system.initialize()
        await system.start()
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
    finally:
        await system.shutdown()

if __name__ == "__main__":
    print(f"{Fore.GREEN}🚀 Enhanced Gamma Squeeze Signal System{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}   ✓ Continuous Decision Recording{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}   ✓ Multi-Timeframe Performance Tracking{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}   ✓ Market Context Capture{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}   ✓ Counterfactual Analysis{Style.RESET_ALL}")
    print("=" * 80)
    print("Features:")
    print("  • Records decisions every 5 minutes (including non-signals)")
    print("  • Tracks performance at 5m, 15m, 30m, 1h, 2h, 4h, 8h intervals")
    print("  • Captures comprehensive market context for each signal")
    print("  • Analyzes why signals were suppressed")
    print("  • Generates enhanced performance reports")
    print("=" * 80)
    print("Press Ctrl+C to stop")
    print("=" * 80)
    
    import os
    os.makedirs('test_output', exist_ok=True)
    
    asyncio.run(main())