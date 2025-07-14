#!/usr/bin/env python3
"""
Gamma Squeeze Signal System - Main Entry Point
阶段1: 数据感知层实现 ✓
阶段2: 模式识别层 - GammaPressureAnalyzer集成 ✓
阶段3: 信号生成层 - MarketBehaviorDetector + SignalEvaluator ✓
阶段4: 性能跟踪层 - PerformanceTracker集成 (Active)
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

# 导入系统模块
from UnifiedDataCollector import UnifiedDataCollector, DataType
from GammaPressureAnalyzer import GammaPressureAnalyzer
from MarketBehaviorDetector import MarketBehaviorDetector
from SignalEvaluator import SignalEvaluator, TradingSignal
from PerformanceTracker import PerformanceTracker, SignalPerformance

# 初始化colorama
init()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_output/gamma_squeeze_system_p4.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class GammaSqueezeSystem:
    """Gamma Squeeze信号系统主类 - Phase 4"""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or self._default_config()
        self.collector = None
        self.gamma_analyzer = None
        self.behavior_detector = None
        self.signal_evaluator = None
        self.performance_tracker = None  # 新增
        self.running = False
        self.analysis_results = []
        self.behavior_results = []
        self.generated_signals = []
        self.performance_stats = {}  # 新增
        self._setup_signal_handlers()
        
    def _default_config(self):
        """默认配置"""
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
                'db_path': 'test_output/signal_performance.csv',
                'signal_db_path': 'test_output/signal_performance_enhanced.csv',  # 新增
                'decision_db_path': 'test_output/decision_history.csv',  # 新增
                'check_intervals': [1, 2, 4, 8],
                'update_interval': 300,
                'report_interval': 1800,
                'decision_interval': 60  # 新增：决策记录间隔
            },
            'display_interval': 30,
            'debug_mode': True,
            'phase4_debug': True  # Phase 4专用调试
        }
        
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(sig, frame):
            logger.info("\n⚠️ Received interrupt signal, shutting down gracefully...")
            asyncio.create_task(self.shutdown())
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    async def initialize(self):
        """初始化系统"""
        logger.info("=" * 80)
        logger.info("🚀 Initializing Gamma Squeeze Signal System")
        logger.info("   Phase 1: Data Collection ✓")
        logger.info("   Phase 2: Pattern Recognition ✓")
        logger.info("   Phase 3: Signal Generation ✓")
        logger.info("   Phase 4: Performance Tracking (Active)")
        logger.info("=" * 80)
        
        # 创建数据采集器
        self.collector = UnifiedDataCollector(self.config['data_collection'])
        await self.collector.initialize()
        
        # 创建分析组件
        self.gamma_analyzer = GammaPressureAnalyzer(self.config['gamma_analysis'])
        self.behavior_detector = MarketBehaviorDetector(self.config['market_behavior'])
        self.signal_evaluator = SignalEvaluator(self.config['signal_generation'])
        
        # 创建性能跟踪器
        self.performance_tracker = PerformanceTracker(self.config['performance_tracking'])
        
        # 设置价格获取器
        self.performance_tracker.set_price_fetcher(self._get_current_price)
        
        logger.info("✅ All system components initialized successfully")
        
    async def start(self):
        """启动系统"""
        logger.info("\n📊 Starting system with Performance Tracking...")
        self.running = True
        
        # 启动数据采集
        await self.collector.start()
        
        # 等待初始数据积累
        logger.info("⏳ Waiting for initial data accumulation...")
        await asyncio.sleep(15)
        
        # 启动各个任务
        tasks = [
            asyncio.create_task(self._monitor_loop()),
            asyncio.create_task(self._gamma_analysis_loop()),
            asyncio.create_task(self._behavior_detection_loop()),
            asyncio.create_task(self._signal_generation_loop()),
            asyncio.create_task(self._performance_update_loop()),  # 新增
            asyncio.create_task(self._performance_report_loop())   # 新增
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _monitor_loop(self):
        """监控循环 - 简化输出，添加性能指标"""
        while self.running:
            try:
                await asyncio.sleep(self.config['display_interval'])
                
                # 构建状态行
                status_parts = []
                
                # 数据状态
                df = self.collector.get_latest_data(window_seconds=60)
                if not df.empty:
                    status_parts.append(f"Data: {len(df)}")
                
                # 分析状态
                if self.analysis_results:
                    walls = len(self.analysis_results[-1].get('gamma_walls', []))
                    status_parts.append(f"Walls: {walls}")
                
                # 信号状态
                active_signals = len(self.performance_tracker.active_signals)
                total_signals = len(self.generated_signals)
                status_parts.append(f"Signals: {active_signals}/{total_signals}")
                
                # 性能状态
                if self.performance_stats:
                    accuracy = self.performance_stats.get('direction_accuracy', 0)
                    status_parts.append(f"Acc: {accuracy:.1%}")
                
                # 打印状态行
                status_line = " | ".join(status_parts)
                print(f"\r📊 {status_line}", end='', flush=True)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                
    async def _gamma_analysis_loop(self):
        """Gamma分析循环"""
        while self.running:
            try:
                await asyncio.sleep(self.config['gamma_analysis']['interval'])
                
                option_data, spot_data = await self._prepare_analysis_data()
                
                if option_data.empty or spot_data.empty:
                    continue
                
                # 执行分析
                analysis_result = self.gamma_analyzer.analyze(option_data, spot_data)
                self.analysis_results.append(analysis_result)
                
                # 保持结果列表大小
                if len(self.analysis_results) > 100:
                    self.analysis_results = self.analysis_results[-100:]
                    
            except Exception as e:
                logger.error(f"Error in gamma analysis: {e}", exc_info=True)
                
    async def _behavior_detection_loop(self):
        """市场行为检测循环"""
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
        """信号生成循环"""
        last_decision_time = datetime.utcnow()  # 新增
        
        while self.running:
            try:
                await asyncio.sleep(self.config['signal_generation']['interval'])
                
                if not self.analysis_results or not self.behavior_results:
                    continue
                
                latest_gamma = self.analysis_results[-1]
                latest_behavior = self.behavior_results[-1]
                market_data = self.collector.get_latest_data(window_seconds=300)
                
                # 生成信号
                signals = self.signal_evaluator.evaluate(
                    latest_gamma, latest_behavior, market_data
                )
                
                # 记录决策（新增）
                current_time = datetime.utcnow()
                if (current_time - last_decision_time).total_seconds() >= self.config['performance_tracking']['decision_interval']:
                    # 获取评估的资产列表
                    assets = set()
                    if latest_gamma.get('gamma_distribution'):
                        assets.update(latest_gamma['gamma_distribution'].keys())
                    
                    # 获取评分（如果signal_evaluator有这个方法）
                    scores = {}
                    for asset in assets:
                        scores[asset] = self.signal_evaluator._calculate_scores(
                            asset, latest_gamma, latest_behavior, market_data
                        )
                    
                    # 记录决策
                    await self.performance_tracker.record_decision(
                        assets_analyzed=list(assets),
                        gamma_analysis=latest_gamma,
                        market_behavior=latest_behavior,
                        scores=scores,
                        signals_generated=signals,
                        suppressed_signals={}
                    )
                    last_decision_time = current_time
                
                # 处理新信号（原有代码）
                for signal in signals:
                    self.generated_signals.append(signal)
                    current_price = await self._get_current_price(signal.asset)
                    if current_price:
                        self.performance_tracker.track_signal(signal, current_price)
                        if self.config['phase4_debug']:
                            self._print_signal_tracking_started(signal, current_price)
                    self._print_signal_summary(signal)
                    
            except Exception as e:
                logger.error(f"Error in signal generation: {e}", exc_info=True)
                
    async def _performance_update_loop(self):
        """性能更新循环"""
        while self.running:
            try:
                await asyncio.sleep(self.config['performance_tracking']['update_interval'])
                
                # 更新活跃信号的价格
                await self.performance_tracker.update_prices()
                
                # 获取最新统计
                self.performance_stats = self.performance_tracker.get_performance_stats(
                    lookback_days=7
                )
                
                # Phase 4调试输出
                if self.config['phase4_debug'] and self.performance_tracker.active_signals:
                    self._print_active_signals_update()
                    
            except Exception as e:
                logger.error(f"Error in performance update: {e}", exc_info=True)
                
    async def _performance_report_loop(self):
        """性能报告循环"""
        while self.running:
            try:
                await asyncio.sleep(self.config['performance_tracking']['report_interval'])
                
                # 生成并打印性能报告
                if self.performance_stats:
                    self._print_performance_report()
                    
            except Exception as e:
                logger.error(f"Error in performance report: {e}", exc_info=True)
                
    async def _prepare_analysis_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """准备分析所需的数据格式"""
        df = self.collector.get_latest_data(window_seconds=120)
        
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # 准备期权数据
        option_mask = df['data_type'] == 'option'
        option_data = df[option_mask].copy()
        
        # 映射symbol
        symbol_map = {'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT'}
        
        # 准备现货数据
        spot_mask = df['data_type'] == 'spot'
        spot_data = df[spot_mask].copy()
        
        # 调整期权数据symbol
        if not option_data.empty:
            option_data['mapped_symbol'] = option_data['symbol'].map(symbol_map)
            option_data['symbol'] = option_data['mapped_symbol']
            option_data = option_data.drop('mapped_symbol', axis=1)
            
            if 'iv' in option_data.columns:
                option_data['iv'] = option_data['iv'] / 100.0
                
        return option_data, spot_data
        
    async def _get_current_price(self, asset: str) -> Optional[float]:
        """获取当前价格"""
        try:
            df = self.collector.get_latest_data(window_seconds=30)
            if df.empty:
                return None
                
            # 筛选现货数据
            spot_df = df[(df['symbol'] == asset) & (df['data_type'] == 'spot')]
            if spot_df.empty:
                return None
                
            # 返回最新价格
            return float(spot_df.iloc[-1]['price'])
            
        except Exception as e:
            logger.error(f"Error getting price for {asset}: {e}")
            return None
            
    def _print_signal_tracking_started(self, signal: TradingSignal, initial_price: float):
        """打印信号跟踪开始信息"""
        print(f"\n\n{Fore.CYAN}📍 SIGNAL TRACKING STARTED:{Style.RESET_ALL}")
        print(f"   Asset: {signal.asset}")
        print(f"   Initial Price: ${initial_price:,.2f}")
        print(f"   Direction: {signal.direction}")
        print(f"   Expected Move: {signal.expected_move}")
        print(f"   Time Horizon: {signal.time_horizon}")
        print(f"   Tracking intervals: {self.config['performance_tracking']['check_intervals']}h")
        
    def _print_signal_summary(self, signal: TradingSignal):
        """打印信号摘要（简化版）"""
        direction_arrow = "↑" if signal.direction == "BULLISH" else "↓"
        direction_color = Fore.GREEN if signal.direction == "BULLISH" else Fore.RED
        
        print(f"\n{Fore.YELLOW}⚡ SIGNAL:{Style.RESET_ALL} {signal.asset} "
              f"{direction_color}{signal.direction} {direction_arrow}{Style.RESET_ALL} "
              f"| Strength: {signal.strength} | Confidence: {signal.confidence:.1%} "
              f"| Expected: {signal.expected_move} in {signal.time_horizon}")
        
    def _print_active_signals_update(self):
        """打印活跃信号更新"""
        print(f"\n\n{Fore.BLUE}📈 ACTIVE SIGNALS UPDATE:{Style.RESET_ALL}")
        print("─" * 80)
        
        for signal_id, perf in self.performance_tracker.active_signals.items():
            elapsed = (datetime.utcnow() - perf.signal_timestamp).total_seconds() / 3600
            
            print(f"\n{perf.asset} ({perf.direction}):")
            print(f"  Elapsed: {elapsed:.1f}h | Initial: ${perf.initial_price:.2f}")
            
            # 打印已记录的价格
            for interval in [1, 2, 4, 8]:
                price_attr = f'price_{interval}h'
                move_attr = f'actual_move_{interval}h'
                
                price = getattr(perf, price_attr)
                move = getattr(perf, move_attr)
                
                if price is not None:
                    move_color = Fore.GREEN if move > 0 else Fore.RED
                    print(f"  {interval}h: ${price:.2f} ({move_color}{move:+.2f}%{Style.RESET_ALL})")
                    
        print("─" * 80)
        
    def _print_performance_report(self):
        """打印性能报告"""
        stats = self.performance_stats
        
        print(f"\n\n{Fore.MAGENTA}📊 PERFORMANCE REPORT (7 days):{Style.RESET_ALL}")
        print("=" * 80)
        
        # 总体统计
        print(f"\n{Fore.YELLOW}Overall Statistics:{Style.RESET_ALL}")
        print(f"  Total Signals: {stats.get('total_signals', 0)}")
        print(f"  Direction Accuracy: {stats.get('direction_accuracy', 0):.1%}")
        print(f"  Magnitude Accuracy: {stats.get('avg_magnitude_accuracy', 0):.1%}")
        print(f"  Timing Accuracy: {stats.get('avg_timing_accuracy', 0):.1%}")
        
        # 按信号类型
        by_type = stats.get('by_signal_type', {})
        if by_type:
            print(f"\n{Fore.YELLOW}By Signal Type:{Style.RESET_ALL}")
            for sig_type, type_stats in by_type.items():
                print(f"  {sig_type}:")
                print(f"    Count: {type_stats['count']}")
                print(f"    Direction Acc: {type_stats['direction_accuracy']:.1%}")
                print(f"    Magnitude Acc: {type_stats['magnitude_accuracy']:.1%}")
        
        # 按资产
        by_asset = stats.get('by_asset', {})
        if by_asset:
            print(f"\n{Fore.YELLOW}By Asset:{Style.RESET_ALL}")
            for asset, asset_stats in by_asset.items():
                print(f"  {asset}:")
                print(f"    Count: {asset_stats['count']}")
                print(f"    Direction Acc: {asset_stats['direction_accuracy']:.1%}")
                
        # 失败模式
        failures = stats.get('failure_patterns', {})
        if failures:
            print(f"\n{Fore.RED}Failure Patterns:{Style.RESET_ALL}")
            for pattern, count in failures.items():
                print(f"  {pattern}: {count} occurrences")
                
        print("=" * 80)
        
    async def shutdown(self):
        """关闭系统"""
        logger.info("\n🛑 Shutting down Gamma Squeeze System...")
        self.running = False
        
        if self.collector:
            # 导出数据
            logger.info("💾 Exporting collected data...")
            self.collector.export_data(f'test_output/gamma_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            
            # 导出信号
            if self.generated_signals:
                self._export_signals()
                
            # 导出最终性能报告
            if self.performance_stats:
                self._export_performance_report()
            
            await self.collector.stop()
            
        logger.info("✅ System shutdown complete")
        
    def _export_signals(self):
        """导出生成的信号"""
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
            
    def _export_performance_report(self):
        """导出性能报告"""
        try:
            filename = f'test_output/performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            
            report = {
                'generation_time': datetime.now().isoformat(),
                'statistics': self.performance_stats,
                'active_signals': {}
            }
            
            # 添加活跃信号信息
            for signal_id, perf in self.performance_tracker.active_signals.items():
                report['active_signals'][signal_id] = {
                    'asset': perf.asset,
                    'direction': perf.direction,
                    'initial_price': perf.initial_price,
                    'signal_timestamp': perf.signal_timestamp.isoformat(),
                    'elapsed_hours': (datetime.utcnow() - perf.signal_timestamp).total_seconds() / 3600
                }
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"📊 Performance report exported to {filename}")
        except Exception as e:
            logger.error(f"Error exporting performance report: {e}")

async def main():
    """主函数"""
    # 系统配置
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
                'db_path': 'test_output/signal_performance.csv',
                'signal_db_path': 'test_output/signal_performance_enhanced.csv',  # 新增
                'decision_db_path': 'test_output/decision_history.csv',  # 新增
                'check_intervals': [1, 2, 4, 8],
                'update_interval': 300,
                'report_interval': 1800,
                'decision_interval': 60  # 新增：决策记录间隔
            },
        'display_interval': 30,
        'debug_mode': True,
        'phase4_debug': True
    }
    
    # 创建系统实例
    system = GammaSqueezeSystem(config)
    
    try:
        # 初始化
        await system.initialize()
        
        # 启动系统
        await system.start()
        
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
        
    finally:
        await system.shutdown()

if __name__ == "__main__":
    print(f"{Fore.GREEN}🚀 Gamma Squeeze Signal System - Phase 4{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}   Phase 1: Data Collection ✓{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}   Phase 2: Pattern Recognition ✓{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}   Phase 3: Signal Generation ✓{Style.RESET_ALL}")
    print(f"{Fore.CYAN}   Phase 4: Performance Tracking (Active){Style.RESET_ALL}")
    print("=" * 80)
    print("Features:")
    print("  • Real-time signal performance tracking")
    print("  • Automated price monitoring at 1h, 2h, 4h, 8h intervals")
    print("  • Performance statistics and failure pattern analysis")
    print("  • Periodic performance reports every 30 minutes")
    print("=" * 80)
    print("Press Ctrl+C to stop the system")
    print("=" * 80)
    
    # 确保输出目录存在
    import os
    os.makedirs('test_output', exist_ok=True)
    
    # 运行主程序
    asyncio.run(main())