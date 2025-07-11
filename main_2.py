#!/usr/bin/env python3
"""
Gamma Squeeze Signal System - Main Entry Point
阶段1: 数据感知层实现 ✓
阶段2: 模式识别层 - GammaPressureAnalyzer集成 ✓
阶段3: 信号生成层 - MarketBehaviorDetector + SignalEvaluator
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

# 导入系统模块
from UnifiedDataCollector import UnifiedDataCollector, DataType
from GammaPressureAnalyzer import GammaPressureAnalyzer
from MarketBehaviorDetector import MarketBehaviorDetector
from SignalEvaluator import SignalEvaluator, TradingSignal

# 初始化colorama（跨平台颜色支持）
init()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_output/gamma_squeeze_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class GammaSqueezeSystem:
    """Gamma Squeeze信号系统主类"""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or self._default_config()
        self.collector = None
        self.gamma_analyzer = None
        self.behavior_detector = None
        self.signal_evaluator = None
        self.running = False
        self.analysis_results = []
        self.behavior_results = []
        self.generated_signals = []
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
                'interval': 30,  # 30秒检测一次
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
                'interval': 60,  # 60秒评估一次信号
                'min_strength': 50,
                'min_confidence': 0.5
            },
            'display_interval': 10,
            'debug_mode': True,
            'phase3_debug': True  # 阶段3详细调试
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
        logger.info("   Phase 3: Signal Generation (Active)")
        logger.info("=" * 80)
        
        # 创建数据采集器
        self.collector = UnifiedDataCollector(self.config['data_collection'])
        await self.collector.initialize()
        
        # 创建分析组件
        self.gamma_analyzer = GammaPressureAnalyzer(self.config['gamma_analysis'])
        self.behavior_detector = MarketBehaviorDetector(self.config['market_behavior'])
        self.signal_evaluator = SignalEvaluator(self.config['signal_generation'])
        
        logger.info("✅ All system components initialized successfully")
        
    async def start(self):
        """启动系统"""
        logger.info("\n📊 Starting system...")
        self.running = True
        
        # 启动数据采集
        await self.collector.start()
        
        # 等待初始数据积累
        logger.info("⏳ Waiting for initial data accumulation (15 seconds)...")
        await asyncio.sleep(15)
        
        # 启动各个任务
        tasks = [
            asyncio.create_task(self._monitor_loop()),
            asyncio.create_task(self._gamma_analysis_loop()),
            asyncio.create_task(self._behavior_detection_loop()),
            asyncio.create_task(self._signal_generation_loop())
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _monitor_loop(self):
        """监控循环 - 简化输出"""
        while self.running:
            try:
                await asyncio.sleep(self.config['display_interval'])
                
                df = self.collector.get_latest_data(window_seconds=60)
                
                if not df.empty:
                    status_line = f"📊 Data: {len(df)} | "
                    
                    # 添加分析状态
                    if self.analysis_results:
                        status_line += f"Gamma Walls: {len(self.analysis_results[-1].get('gamma_walls', []))} | "
                    
                    # 添加行为检测状态
                    if self.behavior_results:
                        latest_behavior = self.behavior_results[-1]
                        sweep_count = len(latest_behavior.get('sweep_orders', []))
                        divergence_count = len(latest_behavior.get('divergences', []))
                        status_line += f"Sweeps: {sweep_count} | Divs: {divergence_count} | "
                    
                    # 添加信号状态
                    if self.generated_signals:
                        recent_signals = [s for s in self.generated_signals 
                                        if (datetime.utcnow() - s.timestamp).seconds < 300]
                        status_line += f"Signals(5m): {len(recent_signals)}"
                    
                    print(f"\r{status_line}", end='', flush=True)
                    
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
                
                # 执行分析但减少输出
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
                
                # 获取市场数据
                market_data = self.collector.get_latest_data(window_seconds=300)
                
                if market_data.empty:
                    continue
                
                # 执行行为检测
                behavior_result = self.behavior_detector.detect(market_data)
                self.behavior_results.append(behavior_result)
                
                # 调试输出
                if self.config['phase3_debug']:
                    self._print_behavior_detection_debug(behavior_result)
                
                # 保持结果列表大小
                if len(self.behavior_results) > 100:
                    self.behavior_results = self.behavior_results[-100:]
                    
            except Exception as e:
                logger.error(f"Error in behavior detection: {e}", exc_info=True)
                
    async def _signal_generation_loop(self):
        """信号生成循环"""
        while self.running:
            try:
                await asyncio.sleep(self.config['signal_generation']['interval'])
                
                # 需要最新的gamma分析和行为检测结果
                if not self.analysis_results or not self.behavior_results:
                    continue
                
                latest_gamma = self.analysis_results[-1]
                latest_behavior = self.behavior_results[-1]
                market_data = self.collector.get_latest_data(window_seconds=300)
                
                # 生成信号
                signals = self.signal_evaluator.evaluate(
                    latest_gamma, latest_behavior, market_data
                )
                
                # 处理新信号
                for signal in signals:
                    self.generated_signals.append(signal)
                    self._print_signal(signal)
                
                # 保持信号列表大小
                if len(self.generated_signals) > 200:
                    self.generated_signals = self.generated_signals[-200:]
                    
            except Exception as e:
                logger.error(f"Error in signal generation: {e}", exc_info=True)
                
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
        
    def _print_behavior_detection_debug(self, result: Dict[str, Any]):
        """打印市场行为检测调试信息"""
        print(f"\n\n{Fore.YELLOW}📈 MARKET BEHAVIOR DETECTION:{Style.RESET_ALL}")
        print("─" * 60)
        
        # 1. 扫单检测
        sweeps = result.get('sweep_orders', [])
        if sweeps:
            print(f"\n{Fore.GREEN}Sweep Orders Detected:{Style.RESET_ALL}")
            for sweep in sweeps[:3]:  # 最多显示3个
                print(f"  {sweep.symbol}: {sweep.side.upper()} sweep")
                print(f"    Volume: {sweep.volume:.0f} | Anomaly: {sweep.anomaly_score:.2f}")
                print(f"    Frequency: {sweep.frequency:.1f}/min")
        
        # 2. 背离检测
        divergences = result.get('divergences', [])
        if divergences:
            print(f"\n{Fore.MAGENTA}Divergences Detected:{Style.RESET_ALL}")
            for div in divergences[:3]:
                print(f"  {div.symbol}: {div.divergence_type}")
                print(f"    Strength: {div.strength:.2f} | Duration: {div.duration}")
        
        # 3. 跨市场信号
        cross_signals = result.get('cross_market_signals', [])
        if cross_signals:
            print(f"\n{Fore.BLUE}Cross-Market Signals:{Style.RESET_ALL}")
            for signal in cross_signals[:2]:
                print(f"  {signal.lead_market} → {signal.lag_market}")
                print(f"    Correlation: {signal.correlation:.2f} | Lag: {signal.lag_time:.0f}s")
        
        # 4. 市场状态
        regime = result.get('market_regime', {})
        if regime:
            print(f"\n{Fore.CYAN}Market Regime:{Style.RESET_ALL}")
            print(f"  State: {regime.get('state', 'unknown').upper()}")
            print(f"  Confidence: {regime.get('confidence', 0):.1%}")
        
        print("─" * 60)
        
    def _print_signal(self, signal: TradingSignal):
        """打印生成的信号"""
        print(f"\n\n{'='*80}")
        print(f"{Fore.RED}🚨 TRADING SIGNAL GENERATED 🚨{Style.RESET_ALL}")
        print(f"{'='*80}")
        
        print(f"\n{Fore.YELLOW}Asset:{Style.RESET_ALL} {signal.asset}")
        print(f"{Fore.YELLOW}Type:{Style.RESET_ALL} {signal.signal_type}")
        print(f"{Fore.YELLOW}Direction:{Style.RESET_ALL} ", end='')
        
        if signal.direction == 'BULLISH':
            print(f"{Fore.GREEN}{signal.direction} ↑{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}{signal.direction} ↓{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}Signal Metrics:{Style.RESET_ALL}")
        print(f"  Strength: {signal.strength}/100")
        print(f"  Confidence: {signal.confidence:.1%}")
        print(f"  Expected Move: {signal.expected_move}")
        print(f"  Time Horizon: {signal.time_horizon}")
        
        if signal.key_levels:
            print(f"\n{Fore.MAGENTA}Key Levels:{Style.RESET_ALL}")
            for level in signal.key_levels[:3]:
                print(f"  - ${level:,.0f}")
        
        if signal.risk_factors:
            print(f"\n{Fore.RED}Risk Factors:{Style.RESET_ALL}")
            for risk in signal.risk_factors:
                print(f"  ⚠️  {risk}")
        
        # 元数据
        metadata = signal.metadata
        print(f"\n{Fore.BLUE}Technical Details:{Style.RESET_ALL}")
        
        scores = metadata.get('scores', {})
        print("  Component Scores:")
        print(f"    Gamma Pressure: {scores.get('gamma_pressure', 0):.1f}")
        print(f"    Market Momentum: {scores.get('market_momentum', 0):.1f}")
        print(f"    Technical: {scores.get('technical', 0):.1f}")
        
        gamma_metrics = metadata.get('gamma_metrics', {})
        if gamma_metrics:
            print("  Gamma Metrics:")
            print(f"    Total Exposure: {gamma_metrics.get('total_gamma_exposure', 0):.2e}")
            print(f"    Dealer Position: {gamma_metrics.get('dealer_position_score', 0):.2f}")
        
        market_metrics = metadata.get('market_metrics', {})
        if market_metrics:
            print("  Market Metrics:")
            print(f"    Sweep Count: {market_metrics.get('sweep_count', 0)}")
            print(f"    Anomaly Score: {market_metrics.get('anomaly_score', 0):.2f}")
        
        print(f"\n{Fore.GREEN}Generated at: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}{Style.RESET_ALL}")
        print('='*80)
        
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
            
            await self.collector.stop()
            
        logger.info("✅ System shutdown complete")
        
    def _export_signals(self):
        """导出生成的信号"""
        try:
            import json
            filename = f'test_output/signals_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            
            signal_data = []
            for signal in self.generated_signals[-50:]:  # 最后50个信号
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
            'wall_percentile': 90,  # Gamma墙识别阈值百分位
            'history_window': 100,  # 历史窗口大小
            'gamma_decay_factor': 0.95,  # 历史gamma衰减因子
            'hedge_flow_threshold': 0.7,  # 对冲流触发阈值
        },
        'market_behavior': {
            'interval': 30,
            'order_flow': {
                'sweep_threshold': 2.0,  # 降低阈值以检测更多扫单
                'frequency_window': 60
            },
            'divergence': {
                'lookback_period': 20,
                'min_duration': 2
            },
            'cross_market': {  # 添加缺失的配置
                'correlation_threshold': 0.7,  # 相关性阈值
                'max_lag': 300,  # 最大延迟（秒）
                'min_observations': 100  # 最小观测数
            },
            'learning_params': {
                'enable_ml': False
            }
        },
        'signal_generation': {
            'interval': 60,
            'min_strength': 60,  # 提高最小强度要求
            'min_confidence': 0.6,
            'signal_cooldown': 300  # 5分钟冷却期
        },
        'display_interval': 10,
        'debug_mode': True,
        'phase3_debug': True
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
    print(f"{Fore.GREEN}🚀 Gamma Squeeze Signal System{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}   Phase 1: Data Collection ✓{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}   Phase 2: Pattern Recognition ✓{Style.RESET_ALL}")
    print(f"{Fore.CYAN}   Phase 3: Signal Generation (Active){Style.RESET_ALL}")
    print("=" * 80)
    print("Press Ctrl+C to stop the system")
    print("=" * 80)
    
    # 运行主程序
    asyncio.run(main())