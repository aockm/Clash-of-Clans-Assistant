"""
Orchestrator - 主调度器

整合所有模块，协调任务执行。
按策略决定当前该做什么（工人空闲优先升级，兵营满优先进攻，有捐兵请求优先捐赠）。
"""

import time
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

import numpy as np

from .device.frame_buffer import FrameBufferManager
from .device.adb_manager import ADBManager
from .device.coordinate import CoordinateMapper
from .device.scrcpy_session import ScrcpySession
from .device.fallback_capture import FallbackCapture
from .vision.template_matcher import TemplateMatcher
from .vision.ocr_engine import OCREngine
from .vision.edge_detector import EdgeDetector
from .states.state_machine import GameState, StateMachine
from .states.classifier import StateClassifier
from .states.navigator import Navigator, create_default_state_graph
from .states.fallback import FallbackHandler, create_default_fallback_handler
from .features.collector import ResourceCollector
from .features.upgrader import BuildingUpgrader
from .features.donator import TroopDonator
from .features.attacker import AttackSearcher
from .features.deployer import BattleDeployer, HeroConfig
from .safety.randomizer import Randomizer
from .safety.session_control import SessionController, SessionConfig

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """调度器配置"""
    # 任务优先级
    priority_donate: int = 10
    priority_collect: int = 8
    priority_upgrade: int = 6
    priority_attack: int = 4
    
    # 功能开关
    enable_collect: bool = True
    enable_upgrade: bool = True
    enable_donate: bool = True
    enable_attack: bool = True
    
    # 循环配置
    loop_interval: float = 0.5  # 主循环间隔（秒）
    frame_timeout: float = 2.0  # 等待帧超时（秒）


class Orchestrator:
    """
    主调度器
    
    整合感知层、视觉层、状态机、业务功能层，
    实现自动化决策和执行。
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        初始化调度器
        
        Args:
            config: 调度器配置
        """
        self.config = config or OrchestratorConfig()
        
        # 核心组件（延迟初始化）
        self.frame_buffer: Optional[FrameBufferManager] = None
        self.adb: Optional[ADBManager] = None
        self.coord_mapper: Optional[CoordinateMapper] = None
        self.scrcpy: Optional[ScrcpySession] = None
        
        # 视觉组件
        self.template_matcher: Optional[TemplateMatcher] = None
        self.ocr_engine: Optional[OCREngine] = None
        self.edge_detector: Optional[EdgeDetector] = None
        
        # 状态机组件
        self.state_machine: Optional[StateMachine] = None
        self.state_classifier: Optional[StateClassifier] = None
        self.navigator: Optional[Navigator] = None
        self.fallback_handler: Optional[FallbackHandler] = None
        
        # 功能模块
        self.collector: Optional[ResourceCollector] = None
        self.upgrader: Optional[BuildingUpgrader] = None
        self.donator: Optional[TroopDonator] = None
        self.searcher: Optional[AttackSearcher] = None
        self.deployer: Optional[BattleDeployer] = None
        
        # 安全组件
        self.randomizer: Optional[Randomizer] = None
        self.session_controller: Optional[SessionController] = None
        
        # 运行状态
        self._running = False
        self._loop_count = 0
        
        logger.info("主调度器初始化完成")
    
    def initialize(
        self,
        adb_path: str = "adb",
        device_serial: Optional[str] = None,
        templates_dir: str = "assets/templates",
        scrcpy_server_path: Optional[str] = None,
        click_offset: Optional[int] = None,
        deploy_config: Optional[Dict[str, Any]] = None,
        attacker_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        初始化所有组件
        
        Args:
            adb_path: ADB 路径
            device_serial: 设备序列号
            templates_dir: 模板目录
            scrcpy_server_path: 本地 scrcpy-server.jar 路径，None 则直接使用 ADB 截图采集
            
        Returns:
            初始化成功返回 True
        """
        logger.info("开始初始化组件...")
        
        try:
            # 1. 初始化设备层
            self.adb = ADBManager(adb_path)
            if not self.adb.connect(device_serial):
                logger.error("ADB 连接失败")
                return False
            
            # 获取设备信息
            screen_size = self.adb.get_screen_size()
            if screen_size:
                device_w, device_h = screen_size
            else:
                device_w, device_h = 1080, 1920  # 默认值
            
            # 2. 初始化帧缓冲（优先 scrcpy 视频流，失败降级 ADB 截图）
            self.frame_buffer = FrameBufferManager(buffer_size=10)
            if scrcpy_server_path:
                self.scrcpy = ScrcpySession(
                    adb_path=adb_path,
                    device_serial=device_serial,
                    scrcpy_server_path=scrcpy_server_path
                )
                if self.scrcpy.start():
                    self.frame_buffer.set_source(self.scrcpy)
                else:
                    logger.warning("scrcpy 会话启动失败，降级到 ADB 截图采集")
                    self.scrcpy = None
                    self.frame_buffer.set_source(FallbackCapture(adb_path, device_serial))
            else:
                logger.info("未配置 scrcpy_server_path，使用 ADB 截图采集")
                self.frame_buffer.set_source(FallbackCapture(adb_path, device_serial))
            
            # 3. 初始化坐标映射器（使用真实视频流尺寸，而非硬编码）
            video_w, video_h = self._wait_video_size()
            if video_w == 0:
                video_w, video_h = device_w, device_h
            # adb input tap 坐标基于当前显示方向：视频流与设备自然方向横竖不一致时需交换
            if (video_w > video_h) != (device_w > device_h):
                device_w, device_h = device_h, device_w
            self.coord_mapper = CoordinateMapper(
                video_w, video_h, device_w, device_h
            )
            
            # 4. 初始化视觉组件
            self.template_matcher = TemplateMatcher(templates_dir)
            self.ocr_engine = OCREngine()
            self.edge_detector = EdgeDetector()
            
            # 5. 初始化状态机
            self.state_machine = StateMachine(GameState.MAIN_VILLAGE)
            self.state_classifier = StateClassifier(self.template_matcher)
            
            # 加载模板并注册状态检测（当前仅主村庄，其余状态待补充模板后注册）
            self.template_matcher.load_templates_from_dir()
            self.state_classifier.register_state(
                GameState.MAIN_VILLAGE,
                [
                    "main_village_battle_btn",
                    "main_village_build_btn",
                    "main_village_home_icon",
                ],
                threshold=0.8
            )
            # 攻击链状态模板均为固定 UI 位置，使用 ROI + 高阈值避免村庄画面误匹配
            self.state_classifier.register_state(
                GameState.ATTACK_MENU, ["find_match_button"],
                threshold=0.8, roi=(50, 225, 180, 80)
            )
            self.state_classifier.register_state(
                GameState.MY_ARMY, ["army_attack_button"],
                threshold=0.8, roi=(560, 290, 200, 70)
            )
            self.state_classifier.register_state(
                GameState.ENEMY_BASE, ["battle_end_button"],
                threshold=0.8, roi=(10, 260, 130, 60)
            )
            self.state_classifier.register_state(
                GameState.BATTLE_RESULTS, ["result_return_button"],
                threshold=0.85, roi=(330, 295, 150, 65)
            )
            self.navigator = Navigator(create_default_state_graph())
            self.fallback_handler = create_default_fallback_handler()
            
            # 6. 初始化功能模块
            click_callback = self._safe_click
            
            self.collector = ResourceCollector(
                self.state_machine, self.template_matcher, click_callback
            )
            self.collector.setup_default_buildings()
            
            self.upgrader = BuildingUpgrader(
                self.state_machine, self.template_matcher, 
                self.ocr_engine, click_callback
            )
            
            self.donator = TroopDonator(
                self.state_machine, self.template_matcher,
                self.ocr_engine, click_callback
            )
            
            self.searcher = AttackSearcher(
                self.state_machine, self.template_matcher,
                self.ocr_engine, click_callback
            )
            
            self.deployer = BattleDeployer(
                self.state_machine, self.template_matcher,
                self.edge_detector, click_callback,
                frame_provider=lambda: self.frame_buffer.get_latest() if self.frame_buffer else None,
                ocr=self.ocr_engine
            )
            
            # 从配置加载部署设置（兵种按兵种栏槽位动态全量部署，无需配置模板）
            if deploy_config:
                for h in deploy_config.get("heroes", []) or []:
                    self.deployer.add_hero(HeroConfig(
                        name=h.get("name", "hero"),
                        template=h.get("template", ""),
                    ))
                if deploy_config.get("deploy_zone_padding"):
                    self.deployer.deploy_zone_padding = float(deploy_config["deploy_zone_padding"])
            
            # 自动配兵配置
            if attacker_config and self.searcher:
                army_cfg = attacker_config.get("army", {}) or {}
                self.searcher.set_army_config(army_cfg.get("troops", []) or [])
            
            # 7. 初始化安全组件
            self.randomizer = Randomizer(click_offset=click_offset if click_offset is not None else 5)
            self.session_controller = SessionController()
            
            logger.info("所有组件初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            return False
    
    def _wait_video_size(self, timeout: float = 10.0) -> Tuple[int, int]:
        """
        等待 scrcpy 解码首帧，获取真实视频流尺寸
        
        Returns:
            (video_w, video_h)，不可用时返回 (0, 0)
        """
        if not self.scrcpy:
            return (0, 0)
        deadline = time.time() + timeout
        while time.time() < deadline:
            w, h = self.scrcpy.video_size
            if w > 0 and h > 0:
                return (w, h)
            time.sleep(0.2)
        logger.warning("等待视频流尺寸超时，回退到设备尺寸 1:1 映射")
        return (0, 0)
    
    def _safe_click(self, x: int, y: int) -> bool:
        """
        安全点击（带随机化）
        
        Args:
            x: 视频坐标 X
            y: 视频坐标 Y
            
        Returns:
            点击成功返回 True
        """
        if self.randomizer:
            x, y = self.randomizer.randomize_click(x, y)
        
        # 转换为设备坐标
        if self.coord_mapper:
            device_x, device_y = self.coord_mapper.video_to_device(x, y)
        else:
            device_x, device_y = x, y
        
        if self.adb:
            return self.adb.tap(device_x, device_y)
        
        logger.warning("ADB 未初始化，无法点击")
        return False
    
    def start(self) -> bool:
        """
        启动调度器
        
        Returns:
            启动成功返回 True
        """
        if self._running:
            logger.warning("调度器已在运行中")
            return False
        
        if not self.frame_buffer:
            logger.error("帧缓冲未初始化")
            return False
        
        # 启动帧缓冲
        if not self.frame_buffer.start():
            logger.error("帧缓冲启动失败")
            return False
        
        # 等待第一帧
        if not self.frame_buffer.wait_for_first_frame(timeout=5.0):
            logger.error("等待第一帧超时")
            return False
        
        self._running = True
        logger.info("调度器已启动")
        
        # 运行主循环
        self._main_loop()
        
        return True
    
    def stop(self) -> None:
        """停止调度器"""
        self._running = False
        
        if self.frame_buffer:
            self.frame_buffer.stop()
        
        if self.scrcpy:
            self.scrcpy.stop()
            self.scrcpy = None
        
        if self.session_controller:
            self.session_controller.stop()
        
        logger.info("调度器已停止")
    
    def _main_loop(self) -> None:
        """主循环"""
        logger.info("进入主循环")
        
        while self._running:
            try:
                # 检查会话控制
                if self.session_controller and not self.session_controller.should_continue():
                    logger.info("会话结束条件满足")
                    break
                
                # 获取最新帧
                frame = self.frame_buffer.get_latest_frame(
                    timeout=self.config.frame_timeout
                )
                
                if frame is None:
                    logger.warning("获取帧失败")
                    time.sleep(0.1)
                    continue
                
                # 执行一次循环
                self._loop_once(frame)
                
                # 检查是否需要休息
                if self.session_controller and self.session_controller.should_take_break():
                    self.session_controller.take_break()
                
                # 循环间隔
                time.sleep(self.config.loop_interval)
                
            except KeyboardInterrupt:
                logger.info("收到中断信号")
                break
            except Exception as e:
                logger.error(f"主循环异常: {e}")
                time.sleep(1.0)
        
        logger.info("主循环退出")
    
    def _loop_once(self, frame: np.ndarray) -> None:
        """
        执行一次主循环
        
        Args:
            frame: 当前帧
        """
        self._loop_count += 1
        
        # 1. 状态分类
        if self.state_classifier:
            result = self.state_classifier.classify(frame)
            self.state_machine.transition(result.state, result.confidence)
        
        current_state = self.state_machine.current_state
        
        # 2. 处理异常状态
        if self.fallback_handler and self.fallback_handler.should_handle(current_state):
            # 攻击流程界面切换（匹配/加载）或自动配兵期间跳过恢复，继续分发
            in_attack_flow = self.searcher and (
                self.searcher.in_transition() or self.searcher.in_army_config()
            )
            if not in_attack_flow:
                logger.info(f"处理异常状态: {current_state.name}")
                self.fallback_handler.execute_recovery(current_state)
                return
        
        # 3. 根据状态执行任务
        # 自动配兵流程优先（弹窗打开时界面分类不可靠），
        # 但已切入敌方基地/结算时放行，保证战斗部署不被拦截
        if self.searcher and self.searcher.in_army_config() and current_state not in (
            GameState.ENEMY_BASE, GameState.BATTLE_RESULTS
        ):
            self.searcher.handle_army_page(frame)
            return
        
        if current_state == GameState.MAIN_VILLAGE:
            self._handle_main_village(frame)
        
        elif current_state == GameState.CLAN_CHAT:
            self._handle_clan_chat(frame)
        
        elif current_state == GameState.ATTACK_MENU:
            self._handle_attack_menu(frame)
        
        elif current_state == GameState.MY_ARMY or (
            self.searcher and self.searcher.in_army_config()
        ):
            if self.searcher:
                self.searcher.handle_army_page(frame)
        
        elif current_state == GameState.ENEMY_BASE:
            self._handle_battle(frame)
        
        elif current_state == GameState.BATTLE_RESULTS:
            self._handle_results(frame)
    
    def _handle_main_village(self, frame: np.ndarray) -> None:
        """处理主村庄状态"""
        # 按优先级执行任务
        
        # 1. 收集资源
        if self.config.enable_collect and self.collector:
            collected = self.collector.collect(frame)
            if collected > 0:
                logger.info(f"收集了 {collected} 个资源建筑")
                return
        
        # 2. 升级建筑
        if self.config.enable_upgrade and self.upgrader:
            upgraded = self.upgrader.try_upgrade(frame)
            if upgraded:
                logger.info("执行了建筑升级")
                return
        
        # 3. 攻击流程
        if self.searcher:
            want_attack = (
                self.config.enable_attack
                and self.session_controller
                and self.session_controller.remaining_attacks > 0
            )
            if want_attack:
                self.searcher.start_search(frame)
            elif self.searcher.return_to_village(frame):
                # 不需要攻击时，若停留在攻击页面则单击右侧屏幕返回村庄
                return
    
    def _handle_clan_chat(self, frame: np.ndarray) -> None:
        """处理部落聊天状态"""
        if self.config.enable_donate and self.donator:
            donated = self.donator.run_donation_cycle(frame)
            if donated > 0:
                logger.info(f"捐赠了 {donated} 次")
    
    def _handle_attack_menu(self, frame: np.ndarray) -> None:
        """处理攻击菜单状态"""
        if self.searcher:
            self.searcher.start_search(frame)
    
    def _handle_battle(self, frame: np.ndarray) -> None:
        """处理战斗状态"""
        if self.deployer:
            self.deployer.run_battle(frame)
    
    def _handle_results(self, frame: np.ndarray) -> None:
        """处理战斗结算状态"""
        now = time.time()
        if now - getattr(self, "_last_result_ts", 0.0) > 10.0:
            self._last_result_ts = now
            if self.session_controller:
                self.session_controller.record_attack()
            if self.deployer:
                self.deployer.finish_battle()
        
        # 点击"回营"返回按钮
        time.sleep(2.0)
        result = self.template_matcher.match(frame, "result_return_button", threshold=0.8)
        if result.found:
            self._safe_click(*result.center)
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取调度器状态
        
        Returns:
            状态信息字典
        """
        return {
            "running": self._running,
            "loop_count": self._loop_count,
            "current_state": self.state_machine.current_state.name if self.state_machine else "N/A",
            "session_stats": self.session_controller.get_stats() if self.session_controller else {},
        }
    
    def __repr__(self) -> str:
        return (
            f"Orchestrator(running={self._running}, "
            f"loops={self._loop_count})"
        )
