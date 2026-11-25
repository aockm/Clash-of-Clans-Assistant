import logging
import time

import numpy as np

from bot.features.base import Feature
from bot.states.base import GameState
from bot.vision.card_reader import occupied_cards

logger = logging.getLogger(__name__)


class Finisher(Feature):
    """战斗收尾：等兵力打光、摧毁率稳定或敌方资源长时间不动后结束战斗。

    不立刻放弃，避免兵还在路上就结算导致摧毁率偏低。
    """

    name = "finisher"

    ROI_GOLD = (86, 68, 60, 26)
    ROI_ELIXIR = (86, 100, 45, 26)
    ROI_DESTROY = (1170, 414, 46, 32)

    CONFIRM_TIMEOUT = 10.0

    def run(self) -> bool:
        frame = self.ctx.frames.get_latest()
        if frame is None:
            return False

        if self.ctx.flags.get("await_confirm"):
            return self._confirm(frame)

        if not self.ctx.flags.get("deployed"):
            return False
        if self.ctx.classifier.classify(frame).state != GameState.BATTLE:
            return False

        now = time.time()
        cfg = self.ctx.config.battle_end
        if now - self.ctx.flags.get("last_sample", 0.0) < cfg.sample_interval:
            return False
        self.ctx.flags["last_sample"] = now

        destroy = self._read_destroy(frame)
        gp = self.ctx.ocr.read_number(frame, self.ROI_GOLD) or 0
        ep = self.ctx.ocr.read_number(frame, self.ROI_ELIXIR) or 0
        loot = gp + ep

        history = self.ctx.flags.setdefault("battle_history", [])
        history.append((destroy, loot))
        del history[:-cfg.history_size]
        logger.info("战斗监控：摧毁 %d%%，剩余战利品 %d", destroy, loot)

        reason = self._end_reason(frame, history)
        if not reason:
            return False

        m = self.ctx.matcher.match(frame, "battle_end_battle_btn")
        if not m:
            return False
        logger.info("结束战斗：%s", reason)
        self.ctx.actions.tap(*m.center)
        self.ctx.flags["await_confirm"] = True
        self.ctx.flags["await_confirm_at"] = time.time()
        return True

    def _read_destroy(self, frame: np.ndarray) -> int:
        """百分号偶尔被识别成数字（50% → 505），超过 100 时去掉末位。"""
        value = self.ctx.ocr.read_number(frame, self.ROI_DESTROY) or 0
        return value // 10 if value > 100 else value

    def _end_reason(self, frame: np.ndarray,
                    history: list[tuple[int, int]]) -> str | None:
        cfg = self.ctx.config.battle_end
        if len(history) < cfg.stable_samples:
            return None

        recent = history[-cfg.stable_samples:]
        destroy_stable = len({d for d, _ in recent}) == 1
        no_troops = not occupied_cards(frame, self.ctx.config.deploy)
        if destroy_stable and no_troops:
            return "兵力耗尽且摧毁率稳定"

        if len({l for _, l in history}) == 1 and \
                len(history) * cfg.sample_interval >= cfg.stall_seconds:
            return f"敌方资源 {cfg.stall_seconds}s 无变化"
        return None

    def _confirm(self, frame: np.ndarray) -> bool:
        """结束战斗会弹“放弃?”确认框，点确定。"""
        m = self.ctx.matcher.match(frame, "popup_confirm_btn", threshold=0.85)
        if m:
            self.ctx.actions.tap(*m.center)
            logger.info("已确认放弃，等待结算")
            self.ctx.flags["await_confirm"] = False
            self.ctx.flags["deployed"] = False
            self.ctx.flags["battle_history"] = []
            return True

        if time.time() - self.ctx.flags.get("await_confirm_at", 0) > self.CONFIRM_TIMEOUT:
            logger.warning("未等到放弃确认框，取消等待")
            self.ctx.flags["await_confirm"] = False
        return True
