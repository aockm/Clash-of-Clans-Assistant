import logging
import time
from dataclasses import dataclass, field
from typing import Any

from bot.account_store import AccountStore
from bot.config import Config
from bot.device.adb_manager import AdbManager
from bot.device.coordinate import CoordinateMapper
from bot.device.fallback_capture import FallbackCapture
from bot.device.frame_buffer import FrameBuffer
from bot.device.scrcpy_session import ScrcpySession
from bot.device.touch_injector import ScrcpyTouch
from bot.features.attacker import Attacker
from bot.features.account_switcher import AccountSwitcher
from bot.features.deployer import Deployer
from bot.features.finisher import Finisher
from bot.features.reward import RewardCollector
from bot.features.upgrader import Upgrader
from bot.safety.randomizer import SafeActions
from bot.safety.session_control import SessionControl
from bot.states.classifier import StateClassifier
from bot.states.base import GameState
from bot.states.fallback import FallbackHandler
from bot.states.navigator import Navigator
from bot.vision.edge_detector import RedZoneEdgeDetector
from bot.vision.loot_meter import LootMeter
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
    touch: ScrcpyTouch | None = None  # scrcpy 多点触控（双指缩放）
    store: AccountStore | None = None      # 账号状态存储（Redis）
    account_name: str | None = None        # 当前登录账号名


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
            store=AccountStore(config.redis),
        )
        self.loot_meter = LootMeter()

        self.scrcpy = ScrcpySession(config.device.serial,
                                    config.device.scrcpy_max_size,
                                    config.device.scrcpy_bit_rate, frames)
        self.fallback_capture = FallbackCapture(adb)
        self.fallback_handler = FallbackHandler(self.ctx)
        self.session = SessionControl(config.safety.max_attacks_per_session)
        self.reward = RewardCollector(self.ctx)
        self.switcher = AccountSwitcher(self.ctx, self._grab_frame, self.reward)

        # 奖励优先：宝箱/胜利之星弹窗可能盖在任何画面上，先收掉再做别的
        self.features = [self.reward, Upgrader(self.ctx), Attacker(self.ctx),
                         Deployer(self.ctx), Finisher(self.ctx)]

    def _grab_frame(self):
        frame = self.ctx.frames.get_latest()
        if frame is None:
            frame = self.fallback_capture.capture()
        return frame

    def run(self) -> None:
        logger.info("Orchestrator starting")
        try:
            self.scrcpy.start()
            self._sync_coordinate_space(self.scrcpy.wait_first_frame(timeout=30))
            self.ctx.touch = ScrcpyTouch(self.ctx.adb.device.serial)
        except Exception:
            logger.exception("scrcpy 启动失败，将回退到 adb screencap")
        try:
            self._run_accounts()
        finally:
            self.scrcpy.stop()
        logger.info("Session finished")

    def _run_accounts(self) -> None:
        """多账号轮换：切号 → 更新战利品门槛 → 打满本账号场次 → 记录状态。"""
        accounts = self.config.accounts
        names: list[str | None] = [None]
        if accounts.enabled and accounts.names:
            names = list(accounts.names)

        for name in names:
            if name:
                if self.ctx.store is not None:
                    skip, reason = self.ctx.store.should_skip(
                        name, valid_hours=accounts.skip_valid_hours)
                    if skip:
                        logger.info("账号 %s：%s，跳过本轮", name, reason)
                        continue
                th = self.switcher.switch_to(name)
                if th:
                    self.config.attack.town_hall_level = th
                    self.ctx.store.set_th(name, th)
                    logger.info("账号 %s：大本营 %d 级，战利品门槛 金币=%d 圣水=%d",
                                name, th, self.config.attack.gold_threshold,
                                self.config.attack.elixir_threshold)
                else:
                    # 切号失败时不能在原账号上继续，否则门槛/战绩都会算错
                    logger.warning("切换账号 %s 失败，跳过本账号", name)
                    continue
                self.ctx.account_name = name
                self.ctx.flags.clear()

            self.session = SessionControl(self._session_attacks())
            logger.info("开始账号 %s 的进攻轮次（本账号 %d 场）",
                        name or "当前账号", self.session.max_attacks)
            self._run_session()

            if name and self.ctx.store is not None:
                self._record_account_state(name)

    def _session_attacks(self) -> tuple[int, int]:
        """账号模式使用固定场次（每个号打满即切），否则沿用随机区间。"""
        accounts = self.config.accounts
        if accounts.enabled and accounts.attacks_per_account > 0:
            n = accounts.attacks_per_account
            return n, n
        return self.config.safety.max_attacks_per_session

    def _record_account_state(self, name: str) -> None:
        """会话结束记录：资源是否已满、是否已无可升级建筑。"""
        frame = self._grab_frame()
        loot_full = False
        if frame is not None:
            gold, elixir = self.loot_meter.ratios(frame)
            ratio = self.config.accounts.loot_full_ratio
            loot_full = gold >= ratio and elixir >= ratio
            logger.info("账号 %s 资源条：金币 %.0f%% 圣水 %.0f%%（阈值 %.0f%%）",
                        name, gold * 100, elixir * 100, ratio * 100)
        no_upgrade = bool(self.ctx.flags.get("upgrade_exhausted"))
        self.ctx.store.set_idle(name, loot_full, no_upgrade)
        logger.info("账号 %s 状态已记录：资源满=%s 无可升级=%s",
                    name, loot_full, no_upgrade)

    def _run_session(self) -> None:
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

    def _upgrade_busy(self) -> bool:
        """升级流程进行中（等待列表/确认框）时，避免兜底逻辑干预。"""
        return any(self.ctx.flags.get(k) for k in
                   ("upgrade_menu_at", "upgrade_await_at", "upgrade_done"))

    def _tick(self) -> None:
        frame = self.ctx.frames.get_latest()
        if frame is None:
            frame = self.fallback_capture.capture()
        state = self.ctx.classifier.classify(frame).state
        if state.name in ("UNKNOWN", "POPUP"):
            # 升级面板/确认框切换期间会出现 UNKNOWN 帧，此时不要按弹窗兜底关闭
            if not self._upgrade_busy():
                self.fallback_handler.on_enter(frame)
            return
        self.fallback_handler.reset()
        if state == GameState.MAIN_VILLAGE:
            if time.time() < self.ctx.flags.get("reward_grace_until", 0.0):
                # 战后奖励弹窗（胜利之星/宝箱）会晚几秒才盖上来，先等它出现
                return
            self.ctx.flags.pop("reward_grace_until", None)
            self.ctx.flags["needs_home"] = False
            # 本账号场次已打完：只等回村收尾，不再开新的一轮搜索
            if (self.session.should_stop()
                    and not self.ctx.flags.get("deployed")
                    and not self.ctx.flags.get("await_confirm")):
                return
        for feat in self.features:
            try:
                if feat.run():
                    if isinstance(feat, Deployer):
                        self.session.record_attack()
                        self.ctx.flags["needs_home"] = True
                    break
            except NotImplementedError:
                continue
