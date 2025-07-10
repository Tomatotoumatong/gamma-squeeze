#!/usr/bin/env python3
"""
Gamma Squeeze Signal System - Main Entry Point
阶段1: 数据感知层实现 ✓
阶段2: 模式识别层 - GammaPressureAnalyzer集成
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
        self.running = False
        self.analysis_results = []  # 存储分析结果
        self._setup_signal_handlers()
        
    def _default_config(self):
        """默认配置"""
        return {
            'data_collection': {
                'deribit': {
                    'enabled': True,
                    'symbols': ['BTC', 'ETH'],
                    'interval': 30  # 30秒更新一次期权数据
                },
                'binance': {
                    'enabled': True,
                    'symbols': ['BTCUSDT', 'ETHUSDT'],
                    'interval': 1   # 1秒更新一次现货数据
                },
                'buffer_size': 2000,
                'export_interval': 300  # 5分钟导出一次
            },
            'gamma_analysis': {
                'interval': 60,  # 60秒分析一次
                'wall_percentile': 80,
                'history_window': 100,
                'gamma_decay_factor': 0.95,
                'hedge_flow_threshold': 0.6
            },
            'learning_params': {
            'feature_extraction': 'manual' 
            },
            'display_interval': 10,  # 每10秒显示一次统计信息
            'debug_mode': True,      # 调试模式
            'phase2_debug': True     # 阶段2详细调试
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
        logger.info("   Phase 2: Pattern Recognition (GammaPressureAnalyzer)")
        logger.info("=" * 80)
        
        # 创建数据采集器
        self.collector = UnifiedDataCollector(self.config['data_collection'])
        await self.collector.initialize()
        
        # 创建Gamma压力分析器
        self.gamma_analyzer = GammaPressureAnalyzer(self.config['gamma_analysis'])
        
        logger.info("✅ System components initialized successfully")
        
    async def start(self):
        """启动系统"""
        logger.info("\n📊 Starting system...")
        self.running = True
        
        # 启动数据采集
        await self.collector.start()
        
        # 等待初始数据积累
        logger.info("⏳ Waiting for initial data accumulation (10 seconds)...")
        await asyncio.sleep(10)
        
        # 启动监控任务（简化输出）
        monitor_task = asyncio.create_task(self._monitor_loop())
        
        # 启动Gamma分析任务（阶段2）
        gamma_analysis_task = asyncio.create_task(self._gamma_analysis_loop())
        
        # 等待任务完成
        await asyncio.gather(monitor_task, gamma_analysis_task, return_exceptions=True)
        
    async def _monitor_loop(self):
        """监控循环 - 简化输出"""
        while self.running:
            try:
                await asyncio.sleep(self.config['display_interval'])
                
                # 获取最新数据
                df = self.collector.get_latest_data(window_seconds=60)
                
                if not df.empty:
                    # 简化的统计信息
                    type_counts = df['data_type'].value_counts()
                    total_points = len(df)
                    
                    # 简单状态行
                    status_line = f"📊 Data: {total_points} points | "
                    for data_type, count in type_counts.items():
                        status_line += f"{data_type}: {count} | "
                    
                    # 添加最新分析结果统计
                    if self.analysis_results:
                        latest_result = self.analysis_results[-1]
                        wall_count = len(latest_result.get('gamma_walls', []))
                        status_line += f"Gamma Walls: {wall_count}"
                    
                    print(f"\r{status_line}", end='', flush=True)
                    
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                
    async def _gamma_analysis_loop(self):
        """Gamma分析循环 - 阶段2核心"""
        analysis_interval = self.config['gamma_analysis']['interval']
        
        while self.running:
            try:
                await asyncio.sleep(analysis_interval)
                
                # 准备数据
                option_data, spot_data = await self._prepare_analysis_data()
                
                if option_data.empty or spot_data.empty:
                    logger.warning("⚠️ Insufficient data for gamma analysis")
                    continue
                
                # 执行Gamma分析
                logger.info("\n" + "="*80)
                logger.info(f"🔬 PHASE 2: Gamma Pressure Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info("="*80)
                
                analysis_result = self.gamma_analyzer.analyze(option_data, spot_data)
                self.analysis_results.append(analysis_result)
                
                # 详细调试输出
                if self.config['phase2_debug']:
                    self._print_gamma_analysis_debug(analysis_result)
                    
            except Exception as e:
                logger.error(f"Error in gamma analysis loop: {e}", exc_info=True)
                
    async def _prepare_analysis_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """准备分析所需的数据格式"""
        # 获取最新数据
        df = self.collector.get_latest_data(window_seconds=120)
        
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # 准备期权数据
        option_mask = df['data_type'] == 'option'
        option_data = df[option_mask].copy()
        
        # 映射symbol（BTC/ETH -> BTCUSDT/ETHUSDT的映射）
        symbol_map = {'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT'}
        
        # 准备现货数据
        spot_mask = df['data_type'] == 'spot'
        spot_data = df[spot_mask].copy()
        
        # 为期权数据添加映射后的symbol以便匹配
        if not option_data.empty:
            option_data['mapped_symbol'] = option_data['symbol'].map(symbol_map)
            option_data['symbol'] = option_data['mapped_symbol']
            option_data = option_data.drop('mapped_symbol', axis=1)
        
            if 'iv' in option_data.columns:
                option_data['iv'] = option_data['iv'] / 100.0 

        return option_data, spot_data
        
    def _print_gamma_analysis_debug(self, result: Dict[str, Any]):
        """打印Gamma分析调试信息"""
        print(f"\n{Fore.CYAN}📊 GAMMA ANALYSIS RESULTS:{Style.RESET_ALL}")
        print("─" * 60)
        
        # 1. Gamma分布
        print(f"\n{Fore.YELLOW}1. Gamma Distribution:{Style.RESET_ALL}")
        for symbol, dist in result.get('gamma_distribution', {}).items():
            print(f"\n   Symbol: {symbol}")
            print(f"   Total Gamma Exposure: {dist.get('total_exposure', 0):,.2f}")
            print(f"   Net Gamma Exposure: {dist.get('net_exposure', 0):,.2f}")
            print(f"   Concentration (Gini): {dist.get('concentration', 0):.3f}")
            
            # 打印前5个gamma点
            # profile = dist.get('profile', [])[:5]
            profile = dist.get('profile', [])
            if profile:
                print(f"   Top Gamma Points:")
                for point in profile:
                    print(f"     Strike: {point['strike']:,} | Gamma: {point['gamma_exposure']:,.5f}")
        
        # 2. Gamma墙
        print(f"\n{Fore.GREEN}2. Gamma Walls:{Style.RESET_ALL}")
        walls = result.get('gamma_walls', [])
        if walls:
            walls_above = [w for w in walls if w.position == 'above'][:5]
            walls_below = [w for w in walls if w.position == 'below'][:5]
            
            if walls_above:
                print("   Walls Above Current Price:")
                for i, wall in enumerate(walls_above):
                    print(f"     Wall #{i+1}:")
                    print(f"       Strike: {wall.strike:,}")
                    print(f"       Distance: {wall.distance_pct:.2f}%")
                    print(f"       Strength: {wall.strength:.2f}")
                    print(f"       Gamma Exposure: {wall.gamma_exposure:,.2f}")
                    
            if walls_below:
                print("   Walls Below Current Price:")
                for i, wall in enumerate(walls_below):
                    print(f"     Wall #{i+1}:")
                    print(f"       Strike: {wall.strike:,}")
                    print(f"       Distance: {wall.distance_pct:.2f}%")
                    print(f"       Strength: {wall.strength:.2f}")
                    print(f"       Gamma Exposure: {wall.gamma_exposure:,.2f}")
        else:
            print("   No significant gamma walls detected")
        
        
        # 3. 做市商头寸
        print(f"\n{Fore.MAGENTA}3. Dealer Positions:{Style.RESET_ALL}")
        for symbol, pos in result.get('dealer_position', {}).items():
            print(f"\n   Symbol: {symbol}")
            print(f"   Net Delta: {pos.get('net_delta', 0):,.2f}")
            print(f"   Net Gamma: {pos.get('net_gamma', 0):,.2f}")
            print(f"   Position Score: {pos.get('position_score', 0):.3f} ({'Short' if pos.get('position_score', 0) < 0 else 'Long'})")
            print(f"   Flow Imbalance: {pos.get('flow_imbalance', 0):.3f}")
        
        # 4. 对冲流
        print(f"\n{Fore.BLUE}4. Hedge Flows:{Style.RESET_ALL}")
        flows = result.get('hedge_flows', [])
        if flows:
            for flow in flows[:3]:  # 最多显示3个
                print(f"   Direction: {flow.direction.upper()}")
                print(f"   Intensity: {flow.intensity:.2f}")
                print(f"   Trigger Price: {flow.trigger_price:,.2f}")
                print(f"   Est. Volume: {flow.estimated_volume:,.0f}")
                print("   ---")
        else:
            print("   No significant hedge flows detected")
        
        # 5. 压力指标
        print(f"\n{Fore.RED}5. Pressure Indicators:{Style.RESET_ALL}")
        for symbol, indicators in result.get('pressure_indicators', {}).items():
            print(f"\n   Symbol: {symbol}")
            print(f"   Nearest Wall Distance: {indicators.get('nearest_wall_distance', 'N/A')}")
            print(f"   Gamma Change Rate: {indicators.get('gamma_change_rate', 0):.6f}")
            print(f"   Hedge Pressure: {indicators.get('hedge_pressure', 0):.2f}")
            print(f"   Gamma Concentration: {indicators.get('gamma_concentration', 0):.3f}")
            
            # 自适应特征
            adaptive = indicators.get('adaptive_features', {})
            if adaptive:
                print(f"   Adaptive Features:")
                for key, value in adaptive.items():
                    print(f"     {key}: {value:.3f}")
        
        print("\n" + "─" * 60)
        
        # 摘要
        print(f"\n{Fore.CYAN}📌 ANALYSIS SUMMARY:{Style.RESET_ALL}")
        wall_count = len(result.get('gamma_walls', []))
        flow_count = len(result.get('hedge_flows', []))
        symbols_analyzed = len(result.get('gamma_distribution', {}))
        
        print(f"   Symbols Analyzed: {symbols_analyzed}")
        print(f"   Gamma Walls Identified: {wall_count}")
        print(f"   Hedge Flows Detected: {flow_count}")
        
        # 警报条件
        alerts = []
        for symbol, indicators in result.get('pressure_indicators', {}).items():
            if indicators.get('hedge_pressure', 0) > 0.7:
                alerts.append(f"{symbol}: High hedge pressure ({indicators['hedge_pressure']:.2f})")
            if indicators.get('nearest_wall_distance') and indicators['nearest_wall_distance'] < 1.0:
                alerts.append(f"{symbol}: Very close to gamma wall ({indicators['nearest_wall_distance']:.2f}%)")
        
        if alerts:
            print(f"\n{Fore.RED}⚠️  ALERTS:{Style.RESET_ALL}")
            for alert in alerts:
                print(f"   - {alert}")
        
        print("\n" + "="*80)
        
    async def shutdown(self):
        """关闭系统"""
        logger.info("\n🛑 Shutting down Gamma Squeeze System...")
        self.running = False
        
        if self.collector:
            # 导出最终数据
            logger.info("💾 Exporting collected data...")
            self.collector.export_data(f'test_output/gamma_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            
            # 导出分析结果
            if self.analysis_results:
                self._export_analysis_results()
            
            # 停止采集器
            await self.collector.stop()
            
        logger.info("✅ System shutdown complete")
        
    def _export_analysis_results(self):
        """导出分析结果"""
        try:
            import json
            filename = f'test_output/gamma_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            
            # 转换为可序列化格式
            exportable_results = []
            for result in self.analysis_results[-10:]:  # 最后10个结果
                exportable = {
                    'timestamp': result['timestamp'].isoformat(),
                    'gamma_walls_count': len(result.get('gamma_walls', [])),
                    'hedge_flows_count': len(result.get('hedge_flows', [])),
                    'pressure_indicators': result.get('pressure_indicators', {})
                }
                exportable_results.append(exportable)
            
            with open(filename, 'w') as f:
                json.dump(exportable_results, f, indent=2)
            
            logger.info(f"📊 Analysis results exported to {filename}")
        except Exception as e:
            logger.error(f"Error exporting analysis results: {e}")

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
            'interval': 60,  # 每60秒分析一次
            'wall_percentile': 90,
            'history_window': 100,
            'gamma_decay_factor': 0.95,
            'hedge_flow_threshold': 0.7
        },
        'display_interval': 10,
        'debug_mode': True,
        'phase2_debug': True  # 启用阶段2详细调试
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
    print(f"{Fore.CYAN}   Phase 2: Pattern Recognition (Active){Style.RESET_ALL}")
    print("=" * 80)
    print("Press Ctrl+C to stop the system")
    print("=" * 80)
    
    # 运行主程序
    asyncio.run(main())