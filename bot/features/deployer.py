import logging
import time

import cv2
import numpy as np

from bot.features.base import Feature
from bot.states.base import GameState
from bot.vision.card_reader import card_present, occupied_cards

logger = logging.getLogger(__name__)


class Deployer(Feature):
    """部署兵种：进入战斗后检测红区外沿，按兵种卡牌顺序逐点投放。

    流程：必要时拖动地图让底部边界露出 → 检测红区外沿 → 逐张选卡 →
    每轮对全部边界点各点一次（一次放一个兵），卡牌耗尽即切下一张。
    """

    name = "deployer"

    # 手指上移会带着地图内容一起上移，把基地从底部抬到画面中部
    RECENTER_START_Y = 320
    RECENTER_DURATION = 0.6
    RAPID_SETTLE = 0.25   # 连点一轮后等帧缓冲追上再判断卡牌是否耗尽

    def run(self) -> bool:
        frame = self.ctx.frames.get_latest()
        if frame is None or not self.ctx.flags.get("attack_accepted"):
            return False
        # 找到对手后先有 30 秒开战倒计时，倒计时期间就能下兵，不必等它走完
        if self.ctx.classifier.classify(frame).state not in (GameState.BATTLE,
                                                            GameState.FIND_MATCH):
            return False
        return self._deploy_all(frame)

    def _deploy_all(self, frame: np.ndarray) -> bool:
        cfg = self.ctx.config.deploy

        frame = self._zoom_out(frame)
        frame, points = self._recenter_until_fit(frame)
        if len(points) < 4:
            points = self._reveal_boundary()
        if len(points) < 4:
            logger.warning("未检测到红区边界，跳过部署")
            return False

        base_center = self.ctx.edge_detector.red_centroid(frame, cfg.map_bottom)
        if base_center:
            self.ctx.flags["base_center"] = base_center

        cards = occupied_cards(frame, cfg)
        if not cards:
            logger.warning("没有可用兵种卡牌")
            return False

        logger.info("开始部署：兵种 %d 张，边界点 %d 个", len(cards), len(points))
        for center in cards:
            self.ctx.actions.tap(*center)
            self._tap_until_empty(center, points, cfg.max_rounds)

        self.ctx.flags["attack_accepted"] = False
        self.ctx.flags["next_attempts"] = 0
        self.ctx.flags["deployed"] = True
        self.ctx.flags["battle_zoomed"] = False
        self.ctx.flags["battle_history"] = []
        self.ctx.flags["last_sample"] = 0.0
        return True

    def _zoom_out(self, frame: np.ndarray) -> np.ndarray:
        """进入敌方村庄先双指捏合缩小地图，确保整个村庄可见再做部署。"""
        if self.ctx.flags.get("battle_zoomed"):
            return frame
        touch = self.ctx.touch
        if touch is None:
            logger.warning("没有 scrcpy 触控通道，跳过地图缩小")
            return frame

        h, w = frame.shape[:2]
        logger.info("缩小敌方地图（双指捏合）")
        if not touch.pinch_in(w // 2, int(h * 0.55)):
            return frame
        time.sleep(0.8)
        self.ctx.flags["battle_zoomed"] = True
        latest = self.ctx.frames.get_latest()
        return latest if latest is not None else frame

    def _recenter_until_fit(self, frame: np.ndarray
                            ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """底部边界被卡牌栏遮挡时，向上拖动地图直到边界完整露出。"""
        cfg = self.ctx.config.deploy
        points = self.ctx.edge_detector.boundary_points(frame, cfg.boundary_margin,
                                                        cfg.map_bottom)
        for i in range(cfg.max_recenter):
            if not points:
                return frame, points
            overflow = max(p[1] for p in points) - (cfg.map_bottom - 20)
            if overflow <= 25:
                # 少量贴底不影响可用性，强行拖动反而会把顶部边界推出画面
                return frame, points
            dy = int(min(overflow + 30, 260))
            logger.info("底部边界超出 %dpx，向上拖动地图 %dpx（%d/%d）",
                        overflow, dy, i + 1, cfg.max_recenter)
            self.ctx.actions.swipe(640, self.RECENTER_START_Y,
                                   640, self.RECENTER_START_Y - dy,
                                   self.RECENTER_DURATION)
            time.sleep(1.2)
            frame = self.ctx.frames.get_latest()
            points = self.ctx.edge_detector.boundary_points(frame, cfg.boundary_margin,
                                                            cfg.map_bottom)
        return frame, points

    def _reveal_boundary(self) -> list[tuple[int, int]]:
        """部署过程中红区会消失：选一张还有兵的卡，再点建筑让它重新显示。"""
        cfg = self.ctx.config.deploy
        frame = self.ctx.frames.get_latest()
        cards = occupied_cards(frame, cfg)
        if not cards:
            logger.warning("没有可部署兵种，无法重新显示红区")
            return []

        self.ctx.actions.tap(*cards[0])
        target = self.ctx.flags.get("base_center")
        if target:
            logger.info("点击建筑重新显示红区：%s", target)
            self.ctx.actions.tap(*target)
            time.sleep(1.2)

        frame = self.ctx.frames.get_latest()
        return self.ctx.edge_detector.boundary_points(frame, cfg.boundary_margin,
                                                      cfg.map_bottom)

    def _tap_until_empty(self, card: tuple[int, int],
                         points: list[tuple[int, int]], max_rounds: int) -> None:
        """卡牌选定后会一直保持选中，连点部署点即可连续下兵；卡牌空了就停。"""
        for r in range(max_rounds):
            for pt in points:
                self.ctx.actions.tap(*pt, pause=False)
            time.sleep(self.RAPID_SETTLE)
            frame = self.ctx.frames.get_latest()
            if frame is None or not card_present(frame, card):
                logger.info("卡牌已耗尽，共 %d 轮", r + 1)
                return
        logger.warning("达到最大轮次 %d，卡牌仍有剩余", max_rounds)
