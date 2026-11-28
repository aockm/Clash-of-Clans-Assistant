import logging
import time

import numpy as np

from bot.features.base import Feature

logger = logging.getLogger(__name__)

REWARD_GRACE = 8.0   # 战斗结算返回后，奖励弹窗可能要等加载完才出现的一段等待时间


class RewardCollector(Feature):
    """战后奖励：胜利之星奖励弹窗「领取」→ 宝箱「点击!」连点开箱 → 奖励卡「继续」。

    宝箱图案随奖励档位变化（箱子上的星星数量不同），模板相似度只有 ~0.83，
    阈值要比普通按钮放宽；奖励界面点满上限仍不消失就停下交给兜底，避免死循环。
    """

    name = "reward"

    CHEST = "loading_reward_chest"
    COLLECT = "star_bonus_collect_btn"
    # 奖励卡宽度不固定（战后宝箱卡 / 切号加载页卡），两种按钮各存一份模板
    CONTINUE = ("reward_card_continue_btn", "reward_continue_btn")

    CHEST_THRESHOLD = 0.78
    BUTTON_THRESHOLD = 0.85
    TAP_INTERVAL = 0.7      # 两次奖励点击的最小间隔
    TAP_LIMIT_SCALE = 2     # 点击上限 = reward_taps * 该倍数

    def run(self) -> bool:
        frame = self.ctx.frames.get_latest()
        if frame is None:
            return False
        return self.collect(frame)

    def collect(self, frame: np.ndarray) -> bool:
        """画面有奖励界面就点一下并返回 True（切号加载页复用同一套逻辑）。"""
        target = self._locate(frame)
        if target is None:
            self._reset()
            return False

        flags = self.ctx.flags
        now = time.time()
        if now - flags.get("reward_tap_at", 0.0) < self.TAP_INTERVAL:
            return True

        label, center = target
        limit = self.ctx.config.accounts.reward_taps * self.TAP_LIMIT_SCALE
        taps = flags.get("reward_taps", 0) + 1
        if taps > limit:
            if not flags.get("reward_stuck"):
                flags["reward_stuck"] = True
                logger.warning("奖励界面已点 %d 次仍未消失，停止点击", limit)
            return False

        flags["reward_taps"] = taps
        flags["reward_tap_at"] = now
        flags["reward_grace_until"] = now + REWARD_GRACE
        logger.info("%s，点击 %d/%d", label, taps, limit)
        self.ctx.actions.tap(*center)
        return True

    def _locate(self, frame: np.ndarray) -> tuple[str, tuple[int, int]] | None:
        chest = self.ctx.matcher.match(frame, self.CHEST,
                                       threshold=self.CHEST_THRESHOLD)
        if chest:
            return "战斗宝箱", chest.center
        for name in (self.COLLECT, *self.CONTINUE):
            m = self.ctx.matcher.match(frame, name, threshold=self.BUTTON_THRESHOLD)
            if m:
                return "奖励继续" if name in self.CONTINUE else "胜利之星奖励", m.center
        return None

    def _reset(self) -> None:
        for key in ("reward_taps", "reward_tap_at", "reward_stuck"):
            self.ctx.flags.pop(key, None)
