import logging
import time

import numpy as np

from bot.states.base import GameState, StateHandler

logger = logging.getLogger(__name__)


class FallbackHandler(StateHandler):
    """处理 UNKNOWN / 弹窗等意外状态：尝试点掉弹窗；连续未知过久则重启游戏。"""

    state = GameState.POPUP

    MAX_ATTEMPTS = 5
    RESTART_AFTER = 45.0   # 连续 UNKNOWN 超过该时长（秒）则重启游戏
    RESTART_WAIT = 30.0    # 重启后等待游戏加载的时长，期间不再次重启

    # 依次尝试：奖励/提示弹窗的“继续”
    DISMISS_TEMPLATES = ("popup_continue_btn",)

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._attempts = 0
        self._unknown_since: float | None = None
        self._restart_until = 0.0

    def reset(self) -> None:
        self._attempts = 0
        self._unknown_since = None

    def on_enter(self, frame: np.ndarray) -> GameState | None:
        now = time.time()
        if now < self._restart_until:
            return None
        self._attempts += 1
        for name in self.DISMISS_TEMPLATES:
            m = self.ctx.matcher.match(frame, name, threshold=0.85)
            if m:
                self.ctx.actions.tap(*m.center)
                self._unknown_since = None
                return None
        # 加载/动画中会连续出现 UNKNOWN；超过阈值说明游戏可能崩了/被切后台，重启它
        if self._unknown_since is None:
            self._unknown_since = now
        elif now - self._unknown_since >= self.RESTART_AFTER:
            logger.warning("连续 %.0f 秒未识别出游戏画面，重启游戏",
                           now - self._unknown_since)
            self._restart_game()
            self._unknown_since = None
            self._restart_until = now + self.RESTART_WAIT
        return None

    def _restart_game(self) -> None:
        pkg = self.ctx.config.device.game_package
        self.ctx.adb.force_stop_app(pkg)
        time.sleep(1.0)
        self.ctx.adb.launch_app(pkg)
