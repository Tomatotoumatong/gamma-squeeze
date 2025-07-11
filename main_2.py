#!/usr/bin/env python3
"""
Gamma Squeeze Signal System - Main Entry Point
阶段1: 数据感知层实现 ✓
阶段2: 模式识别层 - GammaPressureAnalyzer集成 ✓
阶段3: 信号生成层 - SignalEvaluator集成
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
        self.generated_signals = []  # 存储生成的信号
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
                'history_window': 100,
                'gamma_decay_factor': 0.95,
                'hedge_flow_threshold': 0.6
            },
            'market_behavior': {
                'interval': 30,  # 30秒分析一次
                'order_flow': {
                    'sweep_threshold': 3.0,
                    'frequency_window': 60,
                    'volume_multiplier': 2.5
                },
                'divergence': {
                    'lookback_period': 20,
                    'significance_level': 0.05,
                    'min_duration': 3
                }
            },
            'signal_generation': {
                'interval': 60,  # 60秒评估一次信号
                'min_strength': 50,
                'min_confidence': 0.5,
                'signal_cooldown': 300
            },
            'learning_params': {
                'feature_extraction': 'manual'
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
        
        # 创建Gamma压力分析器
        self.gamma_analyzer = GammaPressureAnalyzer(self.config['gamma_analysis'])
        
        # 创建市场行为检测器
        self.behavior_detector = MarketBehaviorDetector(self.config['market_behavior'])
        
        # 创建信号评估器
        self.signal_evaluator = SignalEvaluator(self.config.get('signal_generation', {}))
        
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
        
        # 等待任务完成
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _monitor_loop(self):
        """监控循环 - 简化输出"""
        while self.running:
            try:
                await asyncio.sleep(self.config['display_interval'])
                
                # 简单状态显示
                df = self.collector.get_latest_data(window_seconds=60)
                if not df.empty:
                    status = f"📊 Data: {len(df)} | "
                    
                    # 最新信号状态
                    if self.generated_signals:
                        latest_signal = self.generated_signals[-1]
                        status += f"Latest Signal: {latest_signal.asset} {latest_signal.direction} "
                        status += f"({latest_signal.strength:.0f})"
                    else:
                        status += "No signals yet"
                    
                    print(f"\r{status}", end='', flush=True)
                    
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                
    async def _gamma_analysis_loop(self):
        """Gamma分析循环"""
        while self.running:
            try:
                await asyncio.sleep(self.config['gamma_analysis']['interval'])
                
                # 准备数据
                option_data, spot_data = await self._prepare_analysis_data()
                
                if option_data.empty or spot_data.empty:
                    continue
                
                # 执行Gamma分析（静默）
                analysis_result = self.gamma_analyzer.analyze(option_data, spot_data)
                self.analysis_results.append(analysis_result)
                
                # 只在debug模式下简要输出
                if self.config['debug_mode']:
                    self._print_gamma_summary(analysis_result)
                    
            except Exception as e:
                logger.error(f"Error in gamma analysis: {e}", exc_info=True)
                
    async def _behavior_detection_loop(self):
        """市场行为检测循环"""
        while self.running:
            try:
                await asyncio.sleep(self.config['market_behavior']['interval'])
                
                # 获取市场数据
                market_data = self.collector.get_latest_data(window_seconds=120)
                
                if market_data.empty:
                    continue
                
                # 检测市场行为
                behavior_result = self.behavior_detector.detect(market_data)
                
                # 存储结果
                if hasattr(self, 'behavior_results'):
                    self.behavior_results.append(behavior_result)
                else:
                    self.behavior_results = [behavior_result]
                
                # 调试输出
                if self.config['phase3_debug']:
                    self._print_behavior_summary(behavior_result)
                    
            except Exception as e:
                logger.error(f"Error in behavior detection: {e}", exc_info=True)
                
    async def _signal_generation_loop(self):
        """信号生成循环 - 阶段3核心"""
        while self.running:
            try:
                await asyncio.sleep(self.config['signal_generation']['interval'])
                
                # 检查是否有足够的分析数据
                if not self.analysis_results or not hasattr(self, 'behavior_results'):
                    logger.info("⏳ Waiting for analysis data...")
                    continue
                
                # 获取最新的分析结果
                latest_gamma = self.analysis_results[-1]
                latest_behavior = self.behavior_results[-1] if self.behavior_results else {}
                market_data = self.collector.get_latest_data(window_seconds=300)
                
                # 生成信号
                logger.info("\n" + "="*80)
                logger.info(f"🎯 PHASE 3: Signal Generation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info("="*80)
                
                signals = self.signal_evaluator.evaluate(
                    latest_gamma, 
                    latest_behavior,
                    market_data
                )
                
                # 处理生成的信号
                if signals:
                    for signal in signals:
                        self.generated_signals.append(signal)
                        self._print_signal_details(signal)
                        
                        # 导出信号
                        self._export_signal(signal)
                else:
                    logger.info("📌 No signals generated in this cycle")
                    
                # 打印信号统计
                self._print_signal_statistics()
                
            except Exception as e:
                logger.error(f"Error in signal generation: {e}", exc_info=True)
                
    def _print_gamma_summary(self, result: Dict[str, Any]):
        """打印Gamma分析摘要（简化版）"""
        print(f"\n{Fore.CYAN}📈 Gamma Analysis Summary:{Style.RESET_ALL}")
        
        for symbol, indicators in result.get('pressure_indicators', {}).items():
            wall_distance = indicators.get('nearest_wall_distance', 'N/A')
            hedge_pressure = indicators.get('hedge_pressure', 0)
            
            status = "🔴" if isinstance(wall_distance, float) and wall_distance < 1.0 else "🟢"
            print(f"  {symbol}: Wall Distance: {wall_distance:.2f}% {status} | "
                  f"Hedge Pressure: {hedge_pressure:.2f}")
                  
    def _print_behavior_summary(self, result: Dict[str, Any]):
        """打印市场行为摘要"""
        print(f"\n{Fore.YELLOW}🔍 Market Behavior Summary:{Style.RESET_ALL}")
        
        # 扫单统计
        sweeps = result.get('sweep_orders', [])
        if sweeps:
            print(f"  Sweep Orders: {len(sweeps)} detected")
            for sweep in sweeps[:3]:  # 最多显示3个
                print(f"    - {sweep.symbol}: {sweep.side.upper()} "
                      f"vol={sweep.volume:.0f} score={sweep.anomaly_score:.2f}")
                      
        # 市场状态
        regime = result.get('market_regime', {})
        if regime:
            print(f"  Market Regime: {regime.get('state', 'unknown').upper()} "
                  f"(confidence: {regime.get('confidence', 0):.2f})")
                  
    def _print_signal_details(self, signal: TradingSignal):
        """打印信号详情"""
        print(f"\n{Fore.GREEN}🚨 NEW TRADING SIGNAL:{Style.RESET_ALL}")
        print("─" * 60)
        
        # 信号基本信息
        print(f"{Fore.CYAN}Asset:{Style.RESET_ALL} {signal.asset}")
        print(f"{Fore.CYAN}Type:{Style.RESET_ALL} {signal.signal_type}")
        print(f"{Fore.CYAN}Direction:{Style.RESET_ALL} ", end="")
        
        if signal.direction == "BULLISH":
            print(f"{Fore.GREEN}▲ {signal.direction}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}▼ {signal.direction}{Style.RESET_ALL}")
            
        print(f"{Fore.CYAN}Strength:{Style.RESET_ALL} {signal.strength:.1f}/100")
        print(f"{Fore.CYAN}Confidence:{Style.RESET_ALL} {signal.confidence:.2f}")
        
        # 预期和时间
        print(f"\n{Fore.YELLOW}Expected Move:{Style.RESET_ALL} {signal.expected_move}")
        print(f"{Fore.YELLOW}Time Horizon:{Style.RESET_ALL} {signal.time_horizon}")
        
        # 关键价位
        if signal.key_levels:
            print(f"\n{Fore.MAGENTA}Key Levels:{Style.RESET_ALL}")
            for i, level in enumerate(signal.key_levels[:3]):
                print(f"  Level {i+1}: {level:,.2f}")
                
        # 风险因素
        if signal.risk_factors:
            print(f"\n{Fore.RED}Risk Factors:{Style.RESET_ALL}")
            for risk in signal.risk_factors:
                print(f"  ⚠️  {risk}")
                
        # 元数据摘要
        if signal.metadata:
            print(f"\n{Fore.BLUE}Technical Details:{Style.RESET_ALL}")
            scores = signal.metadata.get('scores', {})
            print(f"  Gamma Pressure Score: {scores.get('gamma_pressure', 0):.1f}")
            print(f"  Market Momentum Score: {scores.get('market_momentum', 0):.1f}")
            print(f"  Technical Score: {scores.get('technical', 0):.1f}")
            
        print("─" * 60)
        
    def _print_signal_statistics(self):
        """打印信号统计"""
        if not self.generated_signals:
            return
            
        print(f"\n{Fore.CYAN}📊 Signal Statistics:{Style.RESET_ALL}")
        
        # 按资产统计
        asset_counts = {}
        for signal in self.generated_signals[-20:]:  # 最近20个信号
            asset_counts[signal.asset] = asset_counts.get(signal.asset, 0) + 1
            
        print(f"  Recent signals (last 20):")
        for asset, count in asset_counts.items():
            print(f"    {asset}: {count}")
            
        # 方向统计
        bullish = sum(1 for s in self.generated_signals[-20:] if s.direction == "BULLISH")
        bearish = sum(1 for s in self.generated_signals[-20:] if s.direction == "BEARISH")
        print(f"  Direction: {bullish} Bullish, {bearish} Bearish")
        
    def _export_signal(self, signal: TradingSignal):
        """导出信号到文件"""
        try:
            # 转换为可序列化格式
            signal_dict = {
                'timestamp': signal.timestamp.isoformat(),
                'asset': signal.asset,
                'signal_type': signal.signal_type,
                'direction': signal.direction,
                'strength': signal.strength,
                'confidence': signal.confidence,
                'key_levels': signal.key_levels,
                'expected_move': signal.expected_move,
                'time_horizon': signal.time_horizon,
                'risk_factors': signal.risk_factors,
                'metadata': signal.metadata
            }
            
            # 写入JSON文件
            filename = f'test_output/signals/signal_{signal.timestamp.strftime("%Y%m%d_%H%M%S")}_{signal.asset}.json'
            
            import os
            os.makedirs('test_output/signals', exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump(signal_dict, f, indent=2)
                
            logger.info(f"📁 Signal exported to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting signal: {e}")
            
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
        
        # 为期权数据添加映射
        if not option_data.empty:
            option_data['mapped_symbol'] = option_data['symbol'].map(symbol_map)
            option_data['symbol'] = option_data['mapped_symbol']
            option_data = option_data.drop('mapped_symbol', axis=1)
        
            if 'iv' in option_data.columns:
                option_data['iv'] = option_data['iv'] / 100.0 

        return option_data, spot_data
        
    async def shutdown(self):
        """关闭系统"""
        logger.info("\n🛑 Shutting down Gamma Squeeze System...")
        self.running = False
        
        if self.collector:
            # 导出数据
            logger.info("💾 Exporting collected data...")
            self.collector.export_data(f'test_output/gamma_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            
            # 导出信号汇总
            if self.generated_signals:
                self._export_signal_summary()
            
            # 停止采集器
            await self.collector.stop()
            
        logger.info("✅ System shutdown complete")
        
    def _export_signal_summary(self):
        """导出信号汇总"""
        try:
            summary = {
                'total_signals': len(self.generated_signals),
                'signals': []
            }
            
            for signal in self.generated_signals:
                summary['signals'].append({
                    'timestamp': signal.timestamp.isoformat(),
                    'asset': signal.asset,
                    'direction': signal.direction,
                    'strength': signal.strength,
                    'confidence': signal.confidence,
                    'type': signal.signal_type
                })
                
            filename = f'test_output/signal_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2)
                
            logger.info(f"📊 Signal summary exported to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting signal summary: {e}")

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
            'hedge_flow_threshold': 0.7
        },
        'market_behavior':{
            'interval': 30, 
            'order_flow': {
                'sweep_threshold': 3.0,  # 扫单识别阈值（标准差）
                'frequency_window': 60,  # 频率计算窗口（秒）
                'volume_multiplier': 2.5  # 大额判定倍数
            },
            'divergence': {
                'lookback_period': 20,  # 回看周期
                'significance_level': 0.05,  # 显著性水平
                'min_duration': 3  # 最小持续周期
            },
            'cross_market': {
                'correlation_threshold': 0.7,  # 相关性阈值
                'max_lag': 300,  # 最大延迟（秒）
                'min_observations': 100  # 最小观测数
            },
        },
        'signal_generation': {
            'interval': 60,
            'min_strength': 50,
            'min_confidence': 0.5,
            'signal_cooldown': 300
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
    print(f"{Fore.CYAN}   Phase 2: Pattern Recognition ✓{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}   Phase 3: Signal Generation (Active){Style.RESET_ALL}")
    print("=" * 80)
    print("Press Ctrl+C to stop the system")
    print("=" * 80)
    
    # 运行主程序
    asyncio.run(main())