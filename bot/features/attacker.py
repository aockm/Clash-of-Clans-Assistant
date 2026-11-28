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

    # 战利品栏位置不固定（敌方资源有两种排布），交给 ocr.read_loot 按图标定位

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
            # 已在倒计时阶段开始下兵，别再点“下一个”
            if self.ctx.flags.get("deployed"):
                return False
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
            self.ctx.flags.pop("deploy_skip_logged", None)
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

        gold, elixir = self.ctx.ocr.read_loot(frame)
        gold, elixir = gold or 0, elixir or 0
        attack_cfg = self.ctx.config.attack
        logger.info("对手战利品：金币=%d 圣水=%d（门槛 金币%d/圣水%d，本营%d级）",
                    gold, elixir, attack_cfg.gold_threshold,
                    attack_cfg.elixir_threshold, attack_cfg.town_hall_level)

        if gold == 0 and elixir == 0:
            # 预览页有 30 秒倒计时，倒计时一到会自动开战；一直读不出来就别干等，
            # 否则会白白开一场一个兵都不下的战斗
            first = self.ctx.flags.setdefault("loot_unread_at", time.time())
            wait = time.time() - first
            if wait < 3.0:
                logger.warning("战利品读取为 0，先等一下再判定（%.0f 秒）", wait)
                return False
            logger.warning("战利品连续 %.0f 秒读不出来，直接换下一个对手", wait)
            self.ctx.flags.pop("loot_unread_at", None)
            return self._next_opponent(frame)
        self.ctx.flags.pop("loot_unread_at", None)

        if gold >= attack_cfg.gold_threshold or elixir >= attack_cfg.elixir_threshold:
            logger.info("资源达标，进入战斗")
            self.ctx.flags["attack_accepted"] = True
            return True

        return self._next_opponent(frame)

    def _next_opponent(self, frame: np.ndarray) -> bool:
        attack_cfg = self.ctx.config.attack
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
