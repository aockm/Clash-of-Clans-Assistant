import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

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
from bot.run_log import start_run_log
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


def next_slot_delay(cfg, now: float | None = None) -> float:
    """距离下一个启动点还有多少秒。

    启动点从当天 0 点起每 interval_minutes 分钟一个，再整体后移
    “当天日期 % 10”分钟（day_offset），避免所有机器都在整点抢同一批对手。
    """
    now = time.time() if now is None else now
    lt = time.localtime(now)
    midnight = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
    offset = (lt.tm_mday % 10) * 60 if cfg.day_offset else 0
    interval = cfg.interval_minutes * 60
    first = midnight + offset
    if now < first:
        return first - now
    steps = int((now - first) // interval) + 1
    return first + steps * interval - now


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

    HOME_BTN = (64, 520)   # 搜索页左下角“回营”按钮（视频坐标）

    def __init__(self, config: Config) -> None:
        level = logging.DEBUG if config.debug else config.log_level
        logging.basicConfig(level=level,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        if not config.debug:
            # wetest.scrcpy 库自建 logger 并固定为 DEBUG，会绕过根 logger 的级别，
            # 逐帧打印“Parsed image size”。run 模式下抬到 WARNING 静音掉。
            scrcpy_logger = logging.getLogger("wetest.scrcpy")
            scrcpy_logger.setLevel(logging.WARNING)
            for h in scrcpy_logger.handlers:
                h.setLevel(logging.WARNING)
        self.config = config
        self.run_log_path = start_run_log(level)
        logger.info("本次运行日志：%s", self.run_log_path)
        logger.debug("配置：%r", config)
        self._last_state = None
        self._last_state_log_at = 0.0

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
        self.upgrader = Upgrader(self.ctx)
        self.features = [self.reward, self.upgrader, Attacker(self.ctx),
                         Deployer(self.ctx), Finisher(self.ctx)]

        self._debug_dir = self._make_debug_dir() if config.debug else None
        self._debug_seq = 0
        self._debug_saved_at = 0.0

    @staticmethod
    def _make_debug_dir() -> Path:
        path = Path("screenshots/debug") / time.strftime("%Y%m%d_%H%M%S")
        path.mkdir(parents=True, exist_ok=True)
        logger.info("debug 模式：画面存到 %s", path)
        return path

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
            self._run_rounds()
        finally:
            self.scrcpy.stop()
        logger.info("Session finished")

    def _run_rounds(self) -> None:
        """反复跑整轮：第一次立即开始，之后每到启动点开新一轮。

        跑到启动点时如果还在跑，就让当前轮跑完，不中途重启。
        """
        while True:
            self.ctx.adb.wake()
            if self._ensure_game_running():
                self._run_accounts()
                self._maintain_finished_accounts()
            else:
                logger.error("多次尝试仍无法启动游戏，本轮跳过")
            self._release_device()
            if not self.config.schedule.enabled:
                return
            delay = next_slot_delay(self.config.schedule)
            logger.info("本轮结束，等待 %.0f 分钟后启动下一轮（%s）", delay / 60,
                        time.strftime("%H:%M", time.localtime(time.time() + delay)))
            time.sleep(delay)

    def _release_device(self) -> None:
        """本轮收工：退出游戏并锁屏，等下一个启动点再唤醒重开。"""
        pkg = self.config.device.game_package
        logger.info("本轮结束，退出游戏并锁屏")
        try:
            self.ctx.adb.force_stop_app(pkg)
            time.sleep(1.0)
            self.ctx.adb.sleep()
        except Exception:
            logger.exception("退出游戏/锁屏失败")

    def _ensure_game_running(self, timeout: float = 60.0) -> bool:
        """确保游戏在前台并已加载；未启动则拉起，加载失败则强杀重试。"""
        pkg = self.config.device.game_package
        for attempt in range(3):
            self.ctx.adb.wake()
            current = self.ctx.adb.current_package()
            if current == pkg:
                logger.info("游戏已在前台")
            else:
                logger.info("游戏未在前台（当前 %s），启动 %s（第 %d 次）",
                            current, pkg, attempt + 1)
                self.ctx.adb.launch_app(pkg)
            if self._wait_game_ready(timeout):
                return True
            logger.warning("游戏未在 %.0f 秒内就绪，强制重启重试", timeout)
            self.ctx.adb.force_stop_app(pkg)
            time.sleep(2.0)
        return False

    def _wait_game_ready(self, timeout: float) -> bool:
        """等待画面落到某个可识别状态；顺带关掉“欢迎回来”弹窗。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self._grab_frame()
            if frame is None:
                time.sleep(1.0)
                continue
            state = self.ctx.classifier.classify(frame).state
            if state.name not in ("UNKNOWN", "POPUP"):
                logger.info("游戏已就绪，状态 %s", state.name)
                return True
            m = self.ctx.matcher.match(frame, "welcome_continue_btn",
                                       threshold=0.90)
            if m:
                logger.info("关闭“欢迎回来”弹窗")
                self.ctx.actions.tap(*m.center)
                time.sleep(1.0)
                continue
            time.sleep(1.0)
        return False

    def _run_accounts(self) -> None:
        """多账号轮换：每个号都切过去读一次真实状态，再决定是否进攻。

        不再根据 Redis 里的历史状态提前跳过账号——那会导致资源状态长期不更新；
        每次都切换一次，靠 _loot_full/_record_account_state 用实时画面刷新数据。
        """
        accounts = self.config.accounts
        names: list[str | None] = [None]
        if accounts.enabled and accounts.names:
            names = list(accounts.names)

        for name in names:
            self._process_account(name, self._session_attacks())

    def _process_account(self, name: str | None,
                         attacks: tuple[int, int]) -> None:
        """切到账号（若指定）、跑指定场次、记录状态。"""
        if name:
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
                return
            self.ctx.account_name = name
            self.ctx.flags.clear()
            # 每次切号后都主动读一次在建建筑：资源已满时主循环会提前返回，
            # 不在这里读就会一直沿用上一轮的陈旧数据
            self.upgrader.refresh_building_state()

        self.session = SessionControl(attacks)
        logger.info("开始账号 %s 的进攻轮次（本账号 %d 场）",
                    name or "当前账号", self.session.max_attacks)
        self._run_session()

        if self.ctx.flags.get("loot_full_stop"):
            # 资源满了不能进攻，但正该把资源花在升级上，否则这个账号永远不升级
            self._upgrade_sweep()

        if name and self.ctx.store is not None:
            self._record_account_state(name)

    def _upgrade_sweep(self, seconds: float = 180.0) -> None:
        """资源已满时的收尾：跑升级流程，直到没得升/没空闲工人或超时。"""
        if not self.config.upgrade.enabled:
            return
        logger.info("资源已满，转入升级流程（最多 %.0f 秒）", seconds)
        deadline = time.time() + seconds
        last_state, stuck_since = None, 0.0
        while time.time() < deadline:
            flags = self.ctx.flags
            if flags.get("upgrade_exhausted") or flags.get("upgrade_nobusy_at"):
                logger.info("升级已收尾：%s", "无可升级建筑"
                            if flags.get("upgrade_exhausted") else "没有空闲工人")
                return
            frame = self.ctx.frames.get_latest()
            if frame is not None:
                state = self.ctx.classifier.classify(frame).state
                if state != last_state:
                    logger.info("升级流程状态：%s → %s",
                                getattr(last_state, "name", "-"), state.name)
                    last_state = state
                if state in (GameState.MAIN_VILLAGE, GameState.BUILDER_MENU,
                             GameState.BUILDING_PANEL):
                    stuck_since = 0.0
                else:
                    # 新建建筑行点下去会进放置模式，这里兜底退出，别干等
                    stuck_since = stuck_since or time.time()
                    if time.time() - stuck_since > 3.0:
                        logger.info("升级流程停在 %s，按返回键退出", state.name)
                        self.ctx.adb.key_back()
                        stuck_since = 0.0
            try:
                self.upgrader.run()
            except Exception:
                logger.exception("升级流程异常")
            time.sleep(0.2)
        logger.info("升级流程达到 %.0f 秒上限，结束", seconds)

    def _maintain_finished_accounts(self) -> None:
        """本轮结束后检查 Redis：有建筑已完成的账号，切过去升级并战斗一次。"""
        accounts = self.config.accounts
        if not (accounts.enabled and accounts.names and self.ctx.store is not None):
            return
        now = time.time()
        for name in accounts.names:
            finished = self.ctx.store.finished_buildings(name, now)
            if not finished:
                continue
            logger.info("账号 %s 有 %d 个建筑已完成，切过去升级并战斗一次",
                        name, len(finished))
            self._process_account(name, (1, 1))

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

    def _flags_brief(self) -> str:
        """关键流程标志的快照，日志里用来还原“为什么没下兵/没开打”。"""
        keys = ("attack_accepted", "deployed", "await_confirm", "needs_home",
                "next_attempts", "battle_zoomed", "upgrade_done",
                "upgrade_await_at", "upgrade_menu_at")
        return "{" + ", ".join(f"{k}={self.ctx.flags.get(k)}" for k in keys
                               if self.ctx.flags.get(k)) + "}"

    def _tick(self) -> None:
        frame = self.ctx.frames.get_latest()
        if frame is None:
            frame = self.fallback_capture.capture()
        state = self.ctx.classifier.classify(frame).state
        self._save_debug_frame(frame, state)
        if state != self._last_state:
            logger.info("状态切换：%s → %s｜flags=%s",
                        getattr(self._last_state, "name", "-"), state.name,
                        self._flags_brief())
            self._last_state = state
            self._last_state_log_at = time.time()
        elif time.time() - self._last_state_log_at >= 2.0:
            # 每 2 秒留一条心跳，足够还原现场又不至于把日志刷爆
            logger.debug("状态维持 %s｜flags=%s", state.name,
                         self._flags_brief())
            self._last_state_log_at = time.time()
        # “暂时无法攻击此村庄”会在搜索页出现，回村后还要靠这个标志收尾，
        # 所以放在状态分发之前、所有状态下都检查
        if self._attack_unavailable(frame):
            return
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
            if self._loot_full(frame):
                return
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

    def _attack_unavailable(self, frame) -> bool:
        """搜索页提示“暂时无法攻击此村庄”：点回营退出搜索，本号到此为止，换下一个号。

        这个号被封了可攻击状态，再搜也搜不到，硬等只会白白耗时间。
        """
        flags = self.ctx.flags
        tip = self.ctx.matcher.match(frame, "attack_blocked_tip", threshold=0.85)
        started = flags.get("attack_blocked")
        if not tip and not started:
            return False
        if not tip:
            if not flags.get("attack_blocked_done"):
                flags["attack_blocked_done"] = True
                logger.info("已离开“无法攻击此村庄”提示页，本账号结束并切换下一个")
            self.session.stop()
            # _run_session 的退出条件还看着这几个标志，这里不一起清会一直转圈，
            # 永远轮不到下一个账号
            flags["deployed"] = False
            flags["await_confirm"] = False
            flags["needs_home"] = False
            return True
        now = time.time()
        if not started:
            logger.warning("提示“暂时无法攻击此村庄”，点回营并切换到下一个账号")
            flags["attack_blocked"] = now
        if now - flags.get("attack_blocked_at", 0.0) >= 2.0:
            flags["attack_blocked_at"] = now
            self.ctx.actions.tap(*self.HOME_BTN)
        if now - flags["attack_blocked"] > 15.0:
            logger.warning("“无法攻击”提示已持续 %.0f 秒，仍结束本账号",
                           now - flags["attack_blocked"])
            self.session.stop()
            flags["deployed"] = False
            flags["await_confirm"] = False
            flags["needs_home"] = False
        return True

    def _loot_full(self, frame) -> bool:
        """金币和圣水都满了就不再进攻（留时间让援兵/工人慢慢用掉）。"""
        ratio = self.config.accounts.loot_full_ratio
        if not self.loot_meter.is_full(frame, ratio):
            return False
        if not self.ctx.flags.get("loot_full_stop"):
            gold, elixir = self.loot_meter.ratios(frame)
            self.ctx.flags["loot_full_stop"] = True
            self.session.stop()
            logger.info("资源已满（金币 %.0f%% / 圣水 %.0f%%，阈值 %.0f%%），本账号不再进攻",
                        gold * 100, elixir * 100, ratio * 100)
        return True

    def _save_debug_frame(self, frame, state: GameState) -> None:
        if self._debug_dir is None:
            return
        now = time.time()
        if now - self._debug_saved_at < 1.0:
            return
        self._debug_saved_at = now
        self._debug_seq += 1
        name = f"{self._debug_seq:05d}_{state.name}.jpg"
        cv2.imwrite(str(self._debug_dir / name), frame)
