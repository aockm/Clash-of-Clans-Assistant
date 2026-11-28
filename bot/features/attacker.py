import logging
import time

import numpy as np

from bot.features.base import Feature
from bot.features.reward import REWARD_GRACE
from bot.states.base import GameState

logger = logging.getLogger(__name__)


class Attacker(Feature):
    """搜索对手：主村庄 → 进攻菜单 → 配兵确认 → 搜索，按资源量决定是否开战。

    状态流：MainVillage → AttackMenu → ArmySetup → FindMatch → (Battle | skip)
    """

    name = "attacker"

    ROI_GOLD = (86, 70, 62, 24)      # 对手预览页“可获得的战利品”金币数值
    ROI_ELIXIR = (86, 100, 58, 24)   # 同上，圣水数值

    def run(self) -> bool:
        frame = self.ctx.frames.get_latest()
        if frame is None:
            return False
        state = self.ctx.classifier.classify(frame).state

        if state == GameState.MAIN_VILLAGE:
            return self._open_attack_menu(frame)
        if state == GameState.ATTACK_MENU:
            return self._find_match(frame)
        if state == GameState.ARMY_SETUP:
            return self._confirm_army(frame)
        if state == GameState.FIND_MATCH:
            return self._judge_opponent(frame)
        if state == GameState.RESULTS:
            return self._return_home(frame)
        return False

    def _return_home(self, frame: np.ndarray) -> bool:
        m = self.ctx.matcher.match(frame, "results_return_home", threshold=0.9)
        if m:
            logger.info("结算页返回主页")
            # 战斗已结束（含打满摧毁率自动结算），清理战斗状态，
            # 否则 deployed 一直为真会导致主循环无法退出
            self.ctx.flags.pop("battle_zoomed", None)
            self.ctx.flags["deployed"] = False
            self.ctx.flags["await_confirm"] = False
            self.ctx.flags["battle_history"] = []
            # 回村后可能弹胜利之星奖励/宝箱，这段时间不算“已回村”
            self.ctx.flags["reward_grace_until"] = time.time() + REWARD_GRACE
            self.ctx.actions.tap(*m.center)
            return True
        return False

    def _open_attack_menu(self, frame: np.ndarray) -> bool:
        m = self.ctx.matcher.match(frame, "main_village_attack_btn")
        if m:
            logger.info("Opening attack menu")
            self.ctx.actions.tap(*m.center)
            return True
        return False

    def _find_match(self, frame: np.ndarray) -> bool:
        m = self.ctx.matcher.match(frame, "attack_menu_find_match_btn")
        if m:
            self.ctx.actions.tap(*m.center)
            return True
        return False

    def _confirm_army(self, frame: np.ndarray) -> bool:
        m = self.ctx.matcher.match(frame, "army_setup_attack_btn")
        if m:
            self.ctx.actions.tap(*m.center)
            return True
        return False

    def _judge_opponent(self, frame: np.ndarray) -> bool:
        if self.ctx.flags.get("attack_accepted"):
            return False

        gold = self.ctx.ocr.read_number(frame, self.ROI_GOLD) or 0
        elixir = self.ctx.ocr.read_number(frame, self.ROI_ELIXIR) or 0
        attack_cfg = self.ctx.config.attack
        logger.info("对手战利品：金币=%d 圣水=%d（门槛 金币%d/圣水%d，本营%d级）",
                    gold, elixir, attack_cfg.gold_threshold,
                    attack_cfg.elixir_threshold, attack_cfg.town_hall_level)

        if gold == 0 and elixir == 0:
            # 多半是 OCR 没读到（界面遮挡/动画中），不做“下一个”以免白花金币
            logger.warning("战利品读取为 0，本轮不做决策")
            return False

        if gold >= attack_cfg.gold_threshold or elixir >= attack_cfg.elixir_threshold:
            logger.info("资源达标，进入战斗")
            self.ctx.flags["attack_accepted"] = True
            return True

        attempts = self.ctx.flags.get("next_attempts", 0)
        if attempts >= attack_cfg.max_next_attempts:
            logger.warning("已换满 %d 个对手，结束搜索", attack_cfg.max_next_attempts)
            self.ctx.flags["next_attempts"] = 0
            end = self.ctx.matcher.match(frame, "battle_end_battle_btn")
            if end:
                self.ctx.actions.tap(*end.center)
            return False

        nxt = self.ctx.matcher.match(frame, "find_match_next_btn")
        if nxt:
            self.ctx.flags["next_attempts"] = attempts + 1
            self.ctx.actions.tap(*nxt.center)
            return True
        return False
