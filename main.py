"""
Clash of Clans Assistant - 主入口

启动自动化助手，执行资源收集、建筑升级、捐兵、攻击等任务。
"""

import sys
import time
import signal
import logging
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from bot.orchestrator import Orchestrator, OrchestratorConfig


def setup_logging(level: str = "INFO", log_file: str = None):
    """配置日志"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


def signal_handler(sig, frame):
    """信号处理"""
    print("\n收到中断信号，正在停止...")
    sys.exit(0)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Clash of Clans Assistant - 自动化助手"
    )
    parser.add_argument(
        "--adb",
        default="adb",
        help="ADB 可执行文件路径"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="设备序列号（默认自动选择）"
    )
    parser.add_argument(
        "--templates",
        default="assets/templates",
        help="模板目录路径"
    )
    parser.add_argument(
        "--config",
        default="config/bot.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--scrcpy-server",
        default=None,
        help="scrcpy-server.jar 路径（默认读取配置文件 device.scrcpy_server_path）"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )
    parser.add_argument(
        "--log-file",
        default="logs/bot.log",
        help="日志文件路径"
    )
    
    args = parser.parse_args()
    
    # 配置日志
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("=" * 60)
    logger.info("Clash of Clans Assistant 启动")
    logger.info("=" * 60)
    
    # 读取配置文件中的设备与安全参数
    device_cfg = {}
    safety_cfg = {}
    deployer_cfg = {}
    attacker_cfg = {}
    config_path = Path(args.config)
    if config_path.exists():
        try:
            import yaml
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            device_cfg = cfg.get("device", {}) or {}
            safety_cfg = cfg.get("safety", {}) or {}
            deployer_cfg = cfg.get("deployer", {}) or {}
            attacker_cfg = cfg.get("attacker", {}) or {}
        except Exception as e:
            logger.warning(f"读取配置文件失败: {e}")
    
    scrcpy_server_path = args.scrcpy_server or device_cfg.get("scrcpy_server_path")
    click_offset = safety_cfg.get("click_offset")
    
    # 创建调度器
    config = OrchestratorConfig()
    orchestrator = Orchestrator(config)
    
    # 初始化
    logger.info("正在初始化组件...")
    if not orchestrator.initialize(
        adb_path=args.adb,
        device_serial=args.device,
        templates_dir=args.templates,
        scrcpy_server_path=scrcpy_server_path,
        click_offset=click_offset,
        deploy_config=deployer_cfg,
        attacker_config=attacker_cfg
    ):
        logger.error("初始化失败，程序退出")
        return 1
    
    logger.info("初始化完成，开始运行")
    
    # 启动
    try:
        orchestrator.start()
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"运行异常: {e}", exc_info=True)
    finally:
        orchestrator.stop()
        logger.info("程序已退出")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
