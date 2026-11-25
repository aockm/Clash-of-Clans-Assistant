import logging
import time
from dataclasses import dataclass, field
from typing import Any

from bot.config import Config
from bot.device.adb_manager import AdbManager
from bot.device.coordinate import CoordinateMapper
from bot.device.fallback_capture import FallbackCapture
from bot.device.frame_buffer import FrameBuffer
from bot.device.scrcpy_session import ScrcpySession
from bot.features.attacker import Attacker
from bot.features.deployer import Deployer
from bot.features.finisher import Finisher
from bot.safety.randomizer import SafeActions
from bot.safety.session_control import SessionControl
from bot.states.classifier import StateClassifier
from bot.states.base import GameState
from bot.states.fallback import FallbackHandler
from bot.states.navigator import Navigator
from bot.vision.edge_detector import RedZoneEdgeDetector
from bot.vision.ocr_engine import OcrEngine
from bot.vision.template_matcher import TemplateMatcher

logger = logging.getLogger(__name__)


@dataclass
class GameContext:
    """贯穿各层的共享上下文。"""
    config: Config
    frames: FrameBuffer
    adb: AdbManager
    matcher: TemplateMatcher
    ocr: OcrEngine
    edge_detector: RedZoneEdgeDetector
    classifier: StateClassifier
    navigator: Navigator
    actions: SafeActions
    flags: dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    """主调度器：装配各层，驱动 感知→分类→决策→执行 主循环。"""

    def __init__(self, config: Config) -> None:
        logging.basicConfig(level=config.log_level,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        self.config = config

        frames = FrameBuffer(config.device.frame_buffer_size)
        adb = AdbManager(config.device.serial)
        self.mapper = CoordinateMapper.from_device_size(adb.screen_size(),
                                                        config.device.scrcpy_max_size)
        mapper = self.mapper
        matcher = TemplateMatcher(config.templates_dir)
        ocr = OcrEngine()
        classifier = StateClassifier(matcher, config.safety.confirm_frames)
        actions = SafeActions(adb, mapper, config.safety.click_jitter_px,
                              config.safety.action_delay_range)

        self.ctx = GameContext(
            config=config, frames=frames, adb=adb, matcher=matcher, ocr=ocr,
            edge_detector=RedZoneEdgeDetector(), classifier=classifier,
            navigator=Navigator(), actions=actions,
        )

        self.scrcpy = ScrcpySession(config.device.serial,
                                    config.device.scrcpy_max_size,
                                    config.device.scrcpy_bit_rate, frames)
        self.fallback_capture = FallbackCapture(adb)
        self.fallback_handler = FallbackHandler(self.ctx)
        self.session = SessionControl(config.safety.max_attacks_per_session)

        self.features = [Attacker(self.ctx), Deployer(self.ctx), Finisher(self.ctx)]

    def run(self) -> None:
        logger.info("Orchestrator starting")
        try:
            self.scrcpy.start()
            self._sync_coordinate_space(self.scrcpy.wait_first_frame(timeout=30))
        except Exception:
            logger.exception("scrcpy 启动失败，将回退到 adb screencap")
        # 已达本次攻击次数上限时，也要等这场战斗收尾、回到主村庄再退出
        while (not self.session.should_stop()
               or self.ctx.flags.get("deployed")
               or self.ctx.flags.get("await_confirm")
               or self.ctx.flags.get("needs_home")):
            try:
                self._tick()
            except Exception:
                logger.exception("Tick error")
            time.sleep(0.2)
        self.scrcpy.stop()
        logger.info("Session finished")

    def _sync_coordinate_space(self, frame) -> None:
        """用真实视频帧修正坐标空间：COC 横屏时设备像素尺寸需翻转。"""
        h, w = frame.shape[:2]
        dev_w, dev_h = self.mapper.device_size
        if (w > h) != (dev_w > dev_h):
            dev_w, dev_h = dev_h, dev_w
        self.mapper.video_size = (w, h)
        self.mapper.device_size = (dev_w, dev_h)
        logger.info("坐标空间同步完成：video=%s device=%s",
                    self.mapper.video_size, self.mapper.device_size)

    def _tick(self) -> None:
        frame = self.ctx.frames.get_latest()
        if frame is None:
            frame = self.fallback_capture.capture()
        state = self.ctx.classifier.classify(frame).state
        if state.name in ("UNKNOWN", "POPUP"):
            self.fallback_handler.on_enter(frame)
            return
        self.fallback_handler.reset()
        if state == GameState.MAIN_VILLAGE:
            self.ctx.flags["needs_home"] = False
        for feat in self.features:
            try:
                if feat.run():
                    if isinstance(feat, Deployer):
                        self.session.record_attack()
                        self.ctx.flags["needs_home"] = True
                    break
            except NotImplementedError:
                continue
