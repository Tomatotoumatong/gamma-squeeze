#!/usr/bin/env python3
"""
Gamma Squeeze Signal System - Main Entry Point
阶段1: 数据感知层实现
"""

import asyncio
import logging
import sys
import signal
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional
from colorama import init, Fore, Style

# 导入系统模块
from UnifiedDataCollector import UnifiedDataCollector, DataType

# 初始化colorama（跨平台颜色支持）
init()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gamma_squeeze_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class GammaSqueezeSystem:
    """Gamma Squeeze信号系统主类"""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or self._default_config()
        self.collector = None
        self.running = False
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
            'display_interval': 10,  # 每10秒显示一次统计信息
            'debug_mode': True       # 调试模式
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
        logger.info("🚀 Initializing Gamma Squeeze Signal System - Phase 1: Data Collection")
        logger.info("=" * 80)
        
        # 创建数据采集器
        self.collector = UnifiedDataCollector(self.config['data_collection'])
        await self.collector.initialize()
        
        logger.info("✅ Data collector initialized successfully")
        logger.info(f"   - Deribit: {self.config['data_collection']['deribit']}")
        logger.info(f"   - Binance: {self.config['data_collection']['binance']}")
        
    async def start(self):
        """启动系统"""
        logger.info("\n📊 Starting data collection...")
        self.running = True
        
        # 启动数据采集
        await self.collector.start()
        
        # 启动监控任务
        monitor_task = asyncio.create_task(self._monitor_loop())
        
        # 启动数据分析任务
        analysis_task = asyncio.create_task(self._analysis_loop())
        
        # 等待任务完成
        await asyncio.gather(monitor_task, analysis_task, return_exceptions=True)
        
    async def _monitor_loop(self):
        """监控循环 - 显示实时统计信息"""
        while self.running:
            try:
                await asyncio.sleep(self.config['display_interval'])
                
                # 获取最新数据
                df = self.collector.get_latest_data(window_seconds=60)
                
                if not df.empty:
                    self._display_statistics(df)
                else:
                    logger.warning("⚠️ No data collected yet...")
                    
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                
    async def _analysis_loop(self):
        """分析循环 - 实时计算Greeks和衍生指标"""
        while self.running:
            try:
                await asyncio.sleep(5)  # 每5秒分析一次
                
                df = self.collector.get_latest_data(window_seconds=30)
                
                if not df.empty:
                    self._analyze_greeks(df)
                    
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                
    def _display_statistics(self, df: pd.DataFrame):
        """显示统计信息"""
        print("\n" + "=" * 100)
        print(f"{Fore.CYAN}📊 REAL-TIME DATA STATISTICS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
        print("=" * 100)
        
        # 按数据类型分组统计
        type_counts = df['data_type'].value_counts()
        print(f"\n{Fore.GREEN}📈 Data Points Collected (Last 60s):{Style.RESET_ALL}")
        for data_type, count in type_counts.items():
            print(f"   - {data_type}: {count} points")
            
        # 现货数据统计
        spot_data = df[df['data_type'] == 'spot']
        if not spot_data.empty:
            print(f"\n{Fore.YELLOW}💰 Spot Market Summary:{Style.RESET_ALL}")
            for symbol in spot_data['symbol'].unique():
                symbol_data = spot_data[spot_data['symbol'] == symbol]
                if 'price' in symbol_data.columns:
                    latest_price = symbol_data['price'].iloc[-1]
                    price_change = ((symbol_data['price'].iloc[-1] / symbol_data['price'].iloc[0]) - 1) * 100
                    latest_volume = symbol_data['volume'].iloc[-1]
                    if 'quote_volume' in symbol_data.columns:
                        quote_volume = symbol_data['quote_volume'].iloc[-1]
                        if 'BTC' in symbol:
                            print(f"   {symbol}: ${latest_price:,.2f} ({price_change:+.2f}%) | 24h Vol: {latest_volume:,.2f} BTC (${quote_volume/1e6:,.1f}M USDT)")
                        else:
                            print(f"   {symbol}: ${latest_price:,.2f} ({price_change:+.2f}%) | 24h Vol: {latest_volume:,.0f} ETH (${quote_volume/1e6:,.1f}M USDT)")
                    else:
                        print(f"   {symbol}: ${latest_price:,.2f} ({price_change:+.2f}%) | 24h Vol: {latest_volume:,.0f}")
        # 期权数据统计
        option_data = df[df['data_type'] == 'option']
        if not option_data.empty:
            print(f"\n{Fore.MAGENTA}📊 Options Market Summary:{Style.RESET_ALL}")
            for symbol in option_data['symbol'].unique():
                symbol_options = option_data[option_data['symbol'] == symbol]
                total_oi = symbol_options['open_interest'].sum() if 'open_interest' in symbol_options.columns else 0
                avg_iv = symbol_options['iv'].mean() if 'iv' in symbol_options.columns else 0
                print(f"   {symbol}: Contracts: {len(symbol_options)} | Total OI: {total_oi:,.0f} | Avg IV: {avg_iv:.1f}%")
                
        # 订单簿数据统计
        orderbook_data = df[df['data_type'] == 'orderbook']
        if not orderbook_data.empty:
            print(f"\n{Fore.BLUE}📖 Order Book Summary:{Style.RESET_ALL}")
            for symbol in orderbook_data['symbol'].unique():
                symbol_ob = orderbook_data[orderbook_data['symbol'] == symbol]
                if 'bid_volume' in symbol_ob.columns and 'ask_volume' in symbol_ob.columns:
                    bid_vol = symbol_ob['bid_volume'].iloc[-1]
                    ask_vol = symbol_ob['ask_volume'].iloc[-1]
                    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) * 100
                    print(f"   {symbol}: Bid Vol: {bid_vol:,.0f} | Ask Vol: {ask_vol:,.0f} | Imbalance: {imbalance:+.1f}%")
                    
        # 衍生指标统计
        derived_data = df[df['data_type'] == 'derived']
        if not derived_data.empty:
            print(f"\n{Fore.RED}🔬 Derived Metrics:{Style.RESET_ALL}")
            for symbol in derived_data['symbol'].unique():
                symbol_derived = derived_data[derived_data['symbol'] == symbol]
                if 'price_acceleration' in symbol_derived.columns:
                    accel = symbol_derived['price_acceleration'].iloc[-1]
                    vol_anomaly = symbol_derived['volume_anomaly'].iloc[-1] if 'volume_anomaly' in symbol_derived.columns else 0
                    print(f"   {symbol}: Price Accel: {accel:.6f} | Vol Anomaly: {vol_anomaly:.2f}σ")
                    
        print("\n" + "=" * 100)
        
    def _analyze_greeks(self, df: pd.DataFrame):
        """分析Greeks（调试模式下显示）"""
        if not self.config['debug_mode']:
            return
            
        option_data = df[df['data_type'] == 'option']
        if option_data.empty:
            return
            
        # 只在调试模式下偶尔显示Greeks分析
        if asyncio.get_event_loop().time() % 30 < 5:  # 每30秒显示一次
            print(f"\n{Fore.CYAN}🔍 DEBUG: Greeks Analysis Sample{Style.RESET_ALL}")
            
            # 选择几个代表性期权显示
            sample_options = option_data.head(3)
            for _, opt in sample_options.iterrows():
                if all(k in opt for k in ['strike', 'iv', 'type']):
                    print(f"   {opt['instrument']}: Strike={opt['strike']} IV={opt['iv']:.1f}% Type={opt['type']}")
                    
    async def shutdown(self):
        """关闭系统"""
        logger.info("\n🛑 Shutting down Gamma Squeeze System...")
        self.running = False
        
        if self.collector:
            # 导出最终数据
            logger.info("💾 Exporting collected data...")
            self.collector.export_data(f'gamma_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            
            # 停止采集器
            await self.collector.stop()
            
        logger.info("✅ System shutdown complete")
        
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
        'display_interval': 10,
        'debug_mode': True
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
    print(f"{Fore.GREEN}🚀 Gamma Squeeze Signal System - Phase 1: Data Collection{Style.RESET_ALL}")
    print("=" * 80)
    print("Press Ctrl+C to stop the system")
    print("=" * 80)
    
    # 运行主程序
    asyncio.run(main())