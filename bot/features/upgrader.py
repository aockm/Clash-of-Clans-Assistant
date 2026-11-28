import logging
import time

import cv2
import numpy as np

from bot.features.base import Feature
from bot.states.base import GameState

logger = logging.getLogger(__name__)


class Upgrader(Feature):
    """建筑/军队升级：主村庄点右下角“建造” → 列表 → 点行 → 确认。

    列表行从右侧货币图标动态识别（列表会随升级进度重排，不能写死坐标）。
    底部“建筑工人 x/y”胶囊表示剩余/总工人数，x 为 0 才说明没有空闲工人；
    第二个“军队升级”标签页用于法术/兵种升级，全部升满时会显示“已完成全部升级”。
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
    ICON_ROI = (1180, 60, 64, 490)     # 列表货币图标列（视频坐标 x, y, w, h）
    ARMY_ICON_ROI = (1060, 40, 220, 500)  # 军队升级页的货币图标列更靠左
    CLOCK_ICON = "upgrade_clock_icon"  # 仅在“升级中”的行上出现（可升级行是普通时钟）
    CLOCK_ROI = (1180, 24, 100, 530)   # 秒表图标列
    # 底部“建筑工人 x/y”与“军队升级 x/y”两个胶囊，x 为剩余可用数
    WORKER_CHIP_ROI = (890, 534, 88, 26)
    ARMY_TAB = (1007, 551)
    ARMY_DONE_ROI = (864, 240, 368, 160)   # 军队升级页“已完成全部升级”提示
    # 时长文字在秒表图标左侧，给足宽度让 OCR 自己找紧包围盒
    DURATION_BAND = (140, 15, 128, 30)
    CLOSE_INTERVAL = 1.2               # 连续点关闭的最小间隔（秒）
    # 新建建筑行点了会进“放置模式”：建造列表消失、只剩主村庄 HUD，
    # 画面上多出一个红底白叉的取消按钮。当前策略是不放新建筑，点取消跳过。
    PLACE_WAIT = 1.2                   # 点行后等确认框的最短时间（秒）
    PLACE_GIVE_UP = 4.0                # 找不到取消按钮就放弃这一行的时限（秒）

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
            # 放置模式下状态同样被判成主村庄，必须先识别出来，否则会误点“建造”
            flags = self.ctx.flags
            await_at = flags.get("upgrade_await_at", 0.0)
            if await_at and time.time() - await_at < self.PLACE_WAIT:
                return True      # 刚点完行，等确认框/放置画面稳定
            if time.time() - flags.get("upgrade_place_cancel_at", 0.0) < 1.5:
                return True
            if self._skip_placement(frame):
                return True
            return False if disabled else self._open_menu(frame)
        if state == GameState.BUILDER_MENU:
            return self._handle_menu(frame, disabled)
        if state == GameState.BUILDING_PANEL:
            if disabled:
                return self._dismiss_dialog(frame)
            return self._handle_confirm(frame)
        return False

    def refresh_building_state(self, timeout: float = 15.0) -> bool:
        """切号后立刻打开建造列表，把在建建筑与工人数写进账号状态。

        主循环在资源已满时会直接返回，升级流程根本不会跑，
        所以必须在这里主动读一次，保证每轮每个账号的 Redis 数据都被刷新。
        """
        name = getattr(self.ctx, "account_name", None)
        if not name or self.ctx.store is None:
            return False

        # 状态分类需要连续多帧确认，这里多试几次再判定是否在主村庄
        frame, button, state = None, None, None
        for _ in range(10):
            frame = self.ctx.frames.get_latest()
            if frame is not None:
                state = self.ctx.classifier.classify(frame).state
                if state == GameState.MAIN_VILLAGE:
                    button = self.ctx.matcher.match(frame, self.BUILD_BTN,
                                                    threshold=0.90)
                    if button:
                        break
            time.sleep(0.3)
        if frame is None:
            return False
        if state != GameState.MAIN_VILLAGE:
            logger.info("账号 %s 不在主村庄（%s），跳过建造列表刷新",
                        name, getattr(state, "name", "-"))
            return False
        if not button:
            logger.info("账号 %s 未找到建造按钮，跳过建造列表刷新", name)
            return False

        # 上一轮可能停在新建建筑的放置模式，先取消，否则后面点“建造”会点不动
        box = self._placement_cancel_btn(frame)
        if box:
            logger.info("账号 %s 停留在新建建筑放置模式，先取消", name)
            self.ctx.actions.tap(int(box[0] + box[2] / 2), int(box[1] + box[3] / 2))
            time.sleep(1.0)
            latest = self.ctx.frames.get_latest()   # 取消后画面会重画，取新帧
            if latest is not None:
                frame = latest

        self.ctx.actions.tap(*button.center)
        if not self._wait_state(GameState.BUILDER_MENU, timeout):
            logger.warning("账号 %s 点击建造后未出现列表", name)
            return False
        time.sleep(0.6)          # 等列表滚动/淡入稳定再读秒表
        frame = self.ctx.frames.get_latest()

        self._read_buildings(frame, name)
        workers = self._read_workers(frame)
        if workers:
            logger.info("账号 %s 空闲工人 %d/%d", name, workers[0], workers[1])
        else:
            logger.warning("账号 %s 工人胶囊识别失败（%s），按弹窗兜底判断",
                           name, self.WORKER_CHIP_ROI)

        if self._close_menu_now():
            logger.info("账号 %s 建造列表已关闭", name)
        return True

    def _close_menu_now(self) -> bool:
        """关掉建造列表并用状态确认真的退出了，必要时用返回键兜底。"""
        for attempt in range(2):
            frame = self.ctx.frames.get_latest()
            if frame is None:
                continue
            close = self.ctx.matcher.match(frame, self.MENU_CLOSE, threshold=0.90)
            if close:
                self.ctx.actions.tap(*close.center)
            for _ in range(8):
                time.sleep(0.3)
                frame = self.ctx.frames.get_latest()
                if frame is None:
                    continue
                if self.ctx.classifier.classify(frame).state != GameState.BUILDER_MENU:
                    return True
            logger.info("建造列表未关闭，重试（%d/2）", attempt + 1)
        self.ctx.adb.key_back()
        time.sleep(0.8)
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

        # 已回到主村庄：清理上一轮列表状态，恢复其它功能。
        # upgrade_tried 故意保留：新建建筑行点了不会弹确认框，
        # 保留已试过的行号才能在重开列表后继续试下一行，而不是原地打转。
        for key in ("upgrade_done", "upgrade_await_at",
                    "upgrade_close_at",
                    "upgrade_buildings_read", "upgrade_menu_seen_at",
                    "upgrade_army_checked"):
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

        if not flags.get("upgrade_buildings_read"):
            seen = flags.get("upgrade_menu_seen_at")
            if not seen:
                flags["upgrade_menu_seen_at"] = now
            elif now - seen >= 0.5:      # 等列表动画稳定
                flags["upgrade_buildings_read"] = True
                self._read_buildings(frame, getattr(self.ctx, "account_name", None))

        if disabled or flags.get("upgrade_done"):
            return self._close_menu(frame, now)

        # 底部工人胶囊：剩余 0 就没有空闲工人，不必再逐行试点
        workers = self._read_workers(frame)
        if workers and workers[0] == 0:
            flags["upgrade_nobusy_at"] = now
            logger.info("工人胶囊 %d/%d：没有空闲工人，暂停建筑升级",
                        workers[0], workers[1])
        else:
            await_at = flags.get("upgrade_await_at", 0.0)
            if await_at:
                if now - await_at <= cfg.confirm_timeout:
                    return True
                # 点完这一行既没弹升级确认框、列表也还在：多半是没点中。
                # 不要按返回键（可能弹“确定退出游戏”），直接换下一行
                flags.pop("upgrade_await_at", None)
                logger.info("该行没有弹出升级确认框（列表仍在），跳过 y=%d",
                            flags.get("upgrade_await_last_y", 0))
                # 新建建筑会占掉一次冷却，这里缩短，避免每行白等 20 秒
                flags["upgrade_last_try"] = now - cfg.cooldown + 2.0
                return True

            for y, kind in self._find_rows(frame):
                if kind == "gem":
                    continue
                tried = flags.setdefault("upgrade_tried", [])
                if any(abs(y - t) <= 8 for t in tried):
                    continue
                tried.append(y)
                flags["upgrade_await_at"] = now
                flags["upgrade_await_last_y"] = y
                logger.info("点击建筑列表行 y=%d（%s，已试 %d 行）", y, kind,
                            len(tried))
                self.ctx.actions.tap(cfg.row_x, y)
                return True

        # 建筑这边没得做，接着看“军队升级”页（法术/兵种，不吃建筑工人）
        if self._try_army_upgrade():
            return True

        flags["upgrade_exhausted"] = True
        return self._close_menu(frame, now)

    def _handle_confirm(self, frame: np.ndarray) -> bool:
        """升级确认框；若提示工人都在忙则关掉弹窗并暂停升级。"""
        now = time.time()
        flags = self.ctx.flags
        cfg = self.ctx.config.upgrade

        if self.ctx.matcher.match(frame, self.BUSY_TITLE, threshold=0.90):
            flags["upgrade_nobusy_at"] = now
            flags["upgrade_done"] = now
            flags["upgrade_exhausted"] = True
            flags.pop("upgrade_await_at", None)
            logger.info("所有工人都忙，暂停升级")
            if now - flags.get("upgrade_close_at", 0.0) >= self.CLOSE_INTERVAL:
                x = self.ctx.matcher.match(frame, self.BUSY_CLOSE, threshold=0.90)
                if x:
                    flags["upgrade_close_at"] = now
                    self.ctx.actions.tap(*x.center)
            return True

        if flags.get("upgrade_done"):
            # 确认过了面板还挂在屏幕上（例如“需要 N 级实验室”这类升不了的提示），
            # 一直停在这里会把整个升级流程卡死，超时后主动收起来
            if (now - flags["upgrade_done"] > cfg.confirm_timeout
                    and now - flags.get("upgrade_close_at", 0.0) >= self.CLOSE_INTERVAL):
                x = self.ctx.matcher.match(frame, self.DIALOG_CLOSE, threshold=0.90)
                if x:
                    flags["upgrade_close_at"] = now
                    logger.info("确认后面板仍未关闭，主动关闭")
                    self.ctx.actions.tap(*x.center)
            return True

        m = self.ctx.matcher.match(frame, self.CONFIRM_BTN, threshold=0.90)
        if m:
            self.ctx.actions.tap(*m.center)
            flags["upgrade_done"] = now
            flags["upgrade_failures"] = 0
            flags["upgrade_exhausted"] = False
            flags["upgrade_last_try"] = now
            flags.pop("upgrade_await_at", None)
            # 列表会因这次升级而重排，之前的行号不能再用
            flags.pop("upgrade_tried", None)
            logger.info("已点击确认升级")
            return True

        await_at = flags.get("upgrade_await_at")
        if await_at is None:
            # 不是本轮点出来的面板（比如上个会话遗留的“正在将XX升至N级”信息页），
            # 上面没有升级按钮，直接收起就行，不计入失败次数
            if now - flags.get("upgrade_close_at", 0.0) >= self.CLOSE_INTERVAL:
                x = self.ctx.matcher.match(frame, self.DIALOG_CLOSE, threshold=0.90)
                if x:
                    flags["upgrade_close_at"] = now
                    logger.info("面板没有升级按钮（多半是升级中信息页），收起")
                    self.ctx.actions.tap(*x.center)
            return True

        if now - await_at > cfg.confirm_timeout:
            # 面板打开了但升级按钮点不了：最常见是资源不够（按钮变灰，绿色模板匹配不上）。
            # 这只是这一行升不了，回列表试下一行就行，不能算失败——
            # 否则三行凑不出资源就把整个升级流程判死，后面能升的也不升了
            flags.pop("upgrade_await_at", None)
            x = self.ctx.matcher.match(frame, self.DIALOG_CLOSE, threshold=0.90)
            if x:
                flags["upgrade_close_at"] = now
                logger.info("该建筑升级按钮不可用（多半资源不足），换下一个")
                self.ctx.actions.tap(*x.center)
            else:
                self._fail("确认框既没有升级按钮也没有关闭按钮")
        return True

    def _skip_placement(self, frame: np.ndarray) -> bool:
        """点完列表行却停在主村庄 = 进了新建建筑的放置模式，点取消跳过这一行。

        放置模式下按返回键会弹“确定退出游戏”，可能把游戏关掉，所以只认那个
        红底白叉的取消按钮；找不到就过一会儿放弃，绝不去点返回键。
        """
        flags = self.ctx.flags
        now = time.time()
        await_at = flags.get("upgrade_await_at", 0.0)
        if not await_at or now - await_at < self.PLACE_WAIT:
            return False

        box = self._placement_cancel_btn(frame)
        if not box:
            if now - await_at < self.PLACE_GIVE_UP:
                return False
            flags.pop("upgrade_await_at", None)
            logger.info("点行后 %.1f 秒既无确认框也无放置按钮，跳过该行",
                        now - await_at)
            return True

        x, y, w, h = box
        self.ctx.actions.tap(int(x + w / 2), int(y + h / 2))
        flags.pop("upgrade_await_at", None)
        flags["upgrade_place_cancel_at"] = now
        # 这一行白跑一趟，缩短冷却，马上重开列表试下一行
        flags["upgrade_last_try"] = now - self.ctx.config.upgrade.cooldown + 2.0
        logger.info("新建建筑放置模式：已点取消跳过 y=%d",
                    flags.get("upgrade_await_last_y", 0))
        return True

    def _placement_cancel_btn(self, frame: np.ndarray) -> tuple | None:
        """放置模式里的红底白叉取消按钮；位置随镜头缩放浮动，只能按颜色找。"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 140, 140), (10, 255, 255))
        mask |= cv2.inRange(hsv, (170, 140, 140), (180, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        white = cv2.inRange(frame, (210, 210, 210), (255, 255, 255))
        best = None
        for c in cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(c) < 300:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if not (0.75 <= w / h <= 1.3) or not 18 <= w <= 140:
                continue
            # 按钮里是白色叉；红色网格、红点之类几乎没有白
            ratio = white[y:y + h, x:x + w].mean() / 255.0
            if not 0.08 <= ratio <= 0.32:
                continue
            if best is None or w > best[2]:
                best = (x, y, w, h)
        return best

    def _read_buildings(self, frame: np.ndarray, name: str | None) -> None:
        """读列表里在建建筑的剩余时间，换算完成时间戳写入账号状态。"""
        store = getattr(self.ctx, "store", None)
        if not name or store is None:
            return

        icons = self.ctx.matcher.match_all(frame, self.CLOCK_ICON,
                                           roi=self.CLOCK_ROI, threshold=0.85)
        if not icons:
            # 列表里没有任何“升级中”的行：清空记录，避免陈旧数据反复触发维护
            store.set_buildings(name, [])
            logger.info("账号 %s 当前没有在建建筑", name)
            return

        shot, sx, sy = self._native_shot(frame)
        now = time.time()
        buildings: list[dict] = []
        for m in icons:
            cx, cy = int(m.center[0] * sx), int(m.center[1] * sy)
            bw, bh, bwl, bhh = self.DURATION_BAND
            seconds = self.ctx.ocr.read_duration(
                shot, (cx - bw, cy - bh, bwl, bhh))
            if not seconds:
                continue
            finish = now + seconds
            buildings.append({"finish_at": round(finish, 1)})
            logger.info("在建建筑剩余 %d 秒（预计 %s 完成）", seconds,
                        time.strftime("%m-%d %H:%M", time.localtime(finish)))

        if buildings:
            store.set_buildings(name, buildings)
            logger.info("账号 %s 在建建筑 %d 座", name, len(buildings))
        else:
            # 有秒表却一个都没读出来，多为识别失败；保留旧数据比写空更安全
            logger.warning("账号 %s 有 %d 个在建行但时长识别失败，保留旧数据",
                           name, len(icons))

    def _read_workers(self, frame: np.ndarray) -> tuple[int, int] | None:
        """读底部“建筑工人 x/y”胶囊，返回（剩余, 总数）。"""
        shot, sx, sy = self._native_shot(frame)
        x, y, w, h = self.WORKER_CHIP_ROI
        return self.ctx.ocr.read_ratio(
            shot, (int(x * sx), int(y * sy), int(w * sx), int(h * sy)))

    def _army_all_done(self, frame: np.ndarray) -> bool:
        """军队升级页是否已全部升满（这时不需要再做任何升级）。"""
        if "chi_sim" not in getattr(self.ctx.ocr, "languages", set()):
            return True
        shot, sx, sy = self._native_shot(frame)
        x, y, w, h = self.ARMY_DONE_ROI
        text = self.ctx.ocr.read_text(
            shot, (int(x * sx), int(y * sy), int(w * sx), int(h * sy)),
            lang="chi_sim", scale=3, psm=6)
        compact = "".join(text.split())
        return "完成" in compact and "升级" in compact

    def _try_army_upgrade(self) -> bool:
        """切到“军队升级”标签页：全部升满就退出，否则点第一行尝试升级。

        法术/兵种升级不占用建筑工人，所以建筑工人全忙时也要继续走这一步。
        每次进菜单只试一次，避免来回切标签页。
        """
        flags = self.ctx.flags
        if flags.get("upgrade_army_checked"):
            return False
        flags["upgrade_army_checked"] = True

        self.ctx.actions.tap(*self.ARMY_TAB)
        time.sleep(0.8)
        frame = self.ctx.frames.get_latest()
        if frame is None:
            return False
        if self._army_all_done(frame):
            logger.info("军队升级页显示“已完成全部升级”，跳过")
            return False

        for y, kind in self._find_rows(frame, self.ARMY_ICON_ROI):
            if kind == "gem":
                continue
            flags["upgrade_await_at"] = time.time()
            logger.info("点击军队升级列表行 y=%d（%s）", y, kind)
            self.ctx.actions.tap(self.ctx.config.upgrade.row_x, y)
            return True
        logger.info("军队升级页没有可点击的行")
        return False

    def _find_rows(self, frame: np.ndarray,
                   roi: tuple[int, int, int, int] | None = None
                   ) -> list[tuple[int, str]]:
        """按 y 从上到下返回列表中各行的货币图标位置与类型。"""
        roi = roi or self.ICON_ROI
        rows: list[tuple[int, str]] = []
        for name, kind in self.CURRENCIES:
            for m in self.ctx.matcher.match_all(frame, name, roi=roi,
                                                threshold=0.85):
                rows.append((m.center[1], kind))
        rows.sort()
        return rows

    def _native_shot(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        """取一张设备原生分辨率截图：小字号时间/比例在视频流里会被压糊。"""
        try:
            raw = np.array(self.ctx.adb.device.screenshot())
            shot = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        except Exception:
            return frame, 1.0, 1.0
        sy = shot.shape[0] / frame.shape[0]
        sx = shot.shape[1] / frame.shape[1]
        return shot, sx, sy

    def _wait_state(self, state: GameState, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self.ctx.frames.get_latest()
            if frame is not None:
                if self.ctx.classifier.classify(frame).state == state:
                    return True
            time.sleep(0.3)
        return False

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
