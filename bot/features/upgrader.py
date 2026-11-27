import logging
import time

import numpy as np

from bot.features.base import Feature
from bot.states.base import GameState

logger = logging.getLogger(__name__)


class Upgrader(Feature):
    """建筑升级：主村庄点右下角“建造” → 建筑和英雄列表 → 点列表行 → 确认升级。

    列表行从右侧货币图标动态识别（列表会随升级进度重排，不能写死坐标）。
    宝石行直接跳过；点行后没弹确认框（资源不足/该行只是提示）就换下一行；
    弹出“所有建筑工人都很忙”时关闭弹窗并暂停升级一段时间。
    """

    name = "upgrader"

    BUILD_BTN = "main_village_build_btn"
    MENU_CLOSE = "builder_menu_close"
    CONFIRM_BTN = "upgrade_confirm_btn"
    DIALOG_CLOSE = "upgrade_dialog_close"
    BUSY_TITLE = "builder_busy_dialog"
    BUSY_CLOSE = "builder_busy_close"
    CURRENCIES = (("currency_gold", "gold"),
                  ("currency_elixir", "elixir"),
                  ("currency_gem", "gem"))
    ICON_ROI = (1180, 60, 64, 490)   # 列表货币图标列（视频坐标 x, y, w, h）
    CLOSE_INTERVAL = 1.2             # 连续点关闭的最小间隔（秒）

    def run(self) -> bool:
        cfg = self.ctx.config.upgrade
        if not cfg.enabled:
            return False

        frame = self.ctx.frames.get_latest()
        if frame is None:
            return False

        state = self.ctx.classifier.classify(frame).state
        disabled = (self.ctx.flags.get("upgrade_failures", 0)
                    >= cfg.max_failures)
        if state == GameState.MAIN_VILLAGE:
            return False if disabled else self._open_menu(frame)
        if state == GameState.BUILDER_MENU:
            return self._handle_menu(frame, disabled)
        if state == GameState.BUILDING_PANEL:
            if disabled:
                return self._dismiss_dialog(frame)
            return self._handle_confirm(frame)
        return False

    def _open_menu(self, frame: np.ndarray) -> bool:
        """主村庄：点右下角“建造”，随后等列表出现。"""
        now = time.time()
        flags = self.ctx.flags
        cfg = self.ctx.config.upgrade

        menu_at = flags.get("upgrade_menu_at", 0.0)
        if menu_at and now - menu_at <= cfg.panel_timeout:
            return True
        if menu_at:
            self._fail("点击建造后未出现建筑升级列表")

        # 已回到主村庄：清理上一轮列表状态，恢复其它功能
        for key in ("upgrade_done", "upgrade_await_at",
                    "upgrade_close_at", "upgrade_tried"):
            flags.pop(key, None)

        if now - flags.get("upgrade_nobusy_at", 0.0) < cfg.nobuilder_cooldown:
            return False
        if now - flags.get("upgrade_last_try", 0.0) < cfg.cooldown:
            return False

        m = self.ctx.matcher.match(frame, self.BUILD_BTN, threshold=0.90)
        if not m:
            return False
        flags["upgrade_last_try"] = now
        flags["upgrade_menu_at"] = now
        logger.info("打开建筑升级列表")
        self.ctx.actions.tap(*m.center)
        return True

    def _handle_menu(self, frame: np.ndarray, disabled: bool) -> bool:
        """建筑和英雄列表：按顺序点一行打开升级确认框。"""
        now = time.time()
        flags = self.ctx.flags
        cfg = self.ctx.config.upgrade
        flags["upgrade_menu_at"] = 0.0

        if disabled or flags.get("upgrade_done"):
            return self._close_menu(frame, now)

        await_at = flags.get("upgrade_await_at", 0.0)
        if await_at:
            if now - await_at <= cfg.confirm_timeout:
                return True
            # 这一行没弹出确认框（资源不足/只是提示），继续试下一行
            flags.pop("upgrade_await_at", None)

        for y, kind in self._find_rows(frame):
            if kind == "gem":
                continue
            tried = flags.setdefault("upgrade_tried", [])
            if any(abs(y - t) <= 8 for t in tried):
                continue
            tried.append(y)
            flags["upgrade_await_at"] = now
            logger.info("点击建筑列表行 y=%d（%s）", y, kind)
            self.ctx.actions.tap(cfg.row_x, y)
            return True

        self._fail("列表中没有可升级的建筑")
        return self._close_menu(frame, now)

    def _handle_confirm(self, frame: np.ndarray) -> bool:
        """升级确认框；若提示工人都在忙则关掉弹窗并暂停升级。"""
        now = time.time()
        flags = self.ctx.flags
        cfg = self.ctx.config.upgrade

        if self.ctx.matcher.match(frame, self.BUSY_TITLE, threshold=0.90):
            flags["upgrade_nobusy_at"] = now
            flags["upgrade_done"] = now
            flags.pop("upgrade_await_at", None)
            logger.info("所有工人都忙，暂停升级")
            if now - flags.get("upgrade_close_at", 0.0) >= self.CLOSE_INTERVAL:
                x = self.ctx.matcher.match(frame, self.BUSY_CLOSE, threshold=0.90)
                if x:
                    flags["upgrade_close_at"] = now
                    self.ctx.actions.tap(*x.center)
            return True

        if flags.get("upgrade_done"):
            return True

        m = self.ctx.matcher.match(frame, self.CONFIRM_BTN, threshold=0.90)
        if m:
            self.ctx.actions.tap(*m.center)
            flags["upgrade_done"] = now
            flags["upgrade_failures"] = 0
            flags["upgrade_last_try"] = now
            flags.pop("upgrade_await_at", None)
            logger.info("已点击确认升级")
            return True

        if now - flags.get("upgrade_await_at", now) > cfg.confirm_timeout:
            self._fail("确认框中没有升级按钮")
            x = self.ctx.matcher.match(frame, self.DIALOG_CLOSE, threshold=0.90)
            if x:
                self.ctx.actions.tap(*x.center)
        return True

    def _find_rows(self, frame: np.ndarray) -> list[tuple[int, str]]:
        """按 y 从上到下返回列表中各行的货币图标位置与类型。"""
        rows: list[tuple[int, str]] = []
        for name, kind in self.CURRENCIES:
            for m in self.ctx.matcher.match_all(frame, name, roi=self.ICON_ROI,
                                                threshold=0.90):
                rows.append((m.center[1], kind))
        rows.sort()
        return rows

    def _close_menu(self, frame: np.ndarray, now: float) -> bool:
        flags = self.ctx.flags
        if now - flags.get("upgrade_close_at", 0.0) < self.CLOSE_INTERVAL:
            return True
        m = self.ctx.matcher.match(frame, self.MENU_CLOSE, threshold=0.90)
        if m:
            flags["upgrade_close_at"] = now
            logger.info("关闭建筑升级列表")
            self.ctx.actions.tap(*m.center)
        return True

    def _dismiss_dialog(self, frame: np.ndarray) -> bool:
        flags = self.ctx.flags
        if time.time() - flags.get("upgrade_close_at", 0.0) < self.CLOSE_INTERVAL:
            return True
        for name in (self.BUSY_CLOSE, self.DIALOG_CLOSE):
            m = self.ctx.matcher.match(frame, name, threshold=0.90)
            if m:
                flags["upgrade_close_at"] = time.time()
                self.ctx.actions.tap(*m.center)
                break
        return True

    def _fail(self, reason: str) -> None:
        flags = self.ctx.flags
        cfg = self.ctx.config.upgrade
        count = flags.get("upgrade_failures", 0) + 1
        flags["upgrade_failures"] = count
        flags["upgrade_last_try"] = time.time()
        for key in ("upgrade_menu_at", "upgrade_await_at",
                    "upgrade_close_at", "upgrade_done", "upgrade_tried"):
            flags.pop(key, None)
        logger.warning("升级失败(%d/%d)：%s", count, cfg.max_failures, reason)
