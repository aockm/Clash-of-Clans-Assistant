import json
import logging
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

import numpy as np

from bot.states.base import GameState

logger = logging.getLogger(__name__)

CACHE_PATH = Path("data/account_cache.json")


class _LevelCache:
    """大本营等级缓存：{账号名: {"th": 3, "updated_at": 时间戳}}。"""

    def __init__(self, path: Path, ttl_hours: float) -> None:
        self._path = path
        self._ttl = ttl_hours * 3600
        self._data: dict[str, dict] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("账号等级缓存读取失败：%s", path)

    def get(self, name: str) -> int | None:
        entry = self._data.get(name)
        if not entry:
            return None
        if time.time() - entry.get("updated_at", 0) > self._ttl:
            return None
        return entry.get("th")

    def put(self, name: str, th: int) -> None:
        self._data[name] = {"th": th, "updated_at": time.time()}

    def save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8")
        except Exception:
            logger.exception("账号等级缓存写入失败：%s", self._path)


class AccountSwitcher:
    """切换 Supercell ID 账号，并识别/缓存大本营等级。

    路径：主村庄左上角大本营图标 → 设置页 → Supercell ID 切换按钮 →
    「切换ID」页（灰色徽标定位每行，OCR 账号名与大本营等级）→ 点目标账号 →
    等待重载回村并关闭“欢迎回来”弹窗。
    """

    TH_ICON = (69, 54)             # 主村庄左上角大本营图标
    SETTINGS_SWITCH = (813, 110)   # 设置页 Supercell ID 右侧切换按钮
    SETTINGS_CLOSE = (1084, 78)    # 设置页红色关闭 X
    SWITCH_CLOSE = (98, 50)        # 切换ID页左上角白色 X
    ROW_TAP_X = 880                # 点击账号行的 x
    LIST_SWIPE_X = 1050            # 账号列表区域内滑动的 x
    BADGE = "switch_account_badge"
    CHEST = "loading_reward_chest"  # 切号加载页的奖励宝箱（点一次给一次奖励）
    REWARD_CONTINUE = "reward_continue_btn"  # 奖励卡片上的“继续”

    # 账号行内文字相对灰色徽标中心的偏移（视频坐标）
    NAME_X0, NAME_X1 = 720, 1064
    NAME_DY0, NAME_DY1 = -30, 12
    # 等级文字“N级大本营”的位置（相对灰色徽标中心，视频坐标）
    TH_X0, TH_X1 = 736, 876
    TH_DY0, TH_DY1 = 40, 76

    def __init__(self, ctx, frame_getter: Callable[[], np.ndarray | None]) -> None:
        self.ctx = ctx
        self._grab = frame_getter
        cfg = ctx.config.accounts
        self._names = cfg.names
        self._reward_taps = cfg.reward_taps
        self._cache = _LevelCache(CACHE_PATH, cfg.th_cache_hours)

    def switch_to(self, name: str) -> int | None:
        """切到指定账号，返回该账号大本营等级；失败返回 None。"""
        if not self._ensure_village(15):
            logger.warning("当前不在主村庄，无法切换账号")
            return None

        logger.info("切换账号：%s", name)
        self.ctx.actions.tap(*self.TH_ICON)
        if not self._wait_state(GameState.SETTINGS, 8):
            logger.warning("未打开设置页")
            return None

        self.ctx.actions.tap(*self.SETTINGS_SWITCH)
        if not self._wait_state(GameState.SWITCH_ACCOUNT, 8):
            logger.warning("未打开切换ID页")
            self.ctx.actions.tap(*self.SETTINGS_CLOSE)
            return None

        # 切换ID页有淡入动画；账号多时列表要向下滚动才能看到
        target = self._locate_account(name)
        if target is None:
            logger.warning("未找到账号 %s（已尝试滚动列表）", name)
            self.ctx.actions.tap(*self.SWITCH_CLOSE)
            return None

        if target["y"] > 500 or target["y"] < 110:
            # 目标行贴边（被裁切），再滑动一次让它完整露出
            self._scroll_list(up=target["y"] > 500)
            time.sleep(0.8)
            target = self._pick(self._read_accounts(), name) or target

        self.ctx.actions.tap(self.ROW_TAP_X, target["y"])
        if self._wait_village(120, target["y"]):
            logger.info("已切换到账号 %s（大本营 %s 级）", name, target["th"])
            return target["th"]
        logger.warning("切换账号 %s 后未回到主村庄", name)
        return None

    def _ensure_village(self, timeout: float) -> bool:
        """回到主村庄；若停在设置页/切换ID页/欢迎弹窗则先退出去。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self._grab()
            if frame is not None:
                m = self.ctx.matcher.match(frame, "welcome_continue_btn",
                                           threshold=0.90)
                if m:
                    logger.info("关闭“欢迎回来”弹窗")
                    self.ctx.actions.tap(*m.center)
                    time.sleep(1.0)
                    continue
                state = self.ctx.classifier.classify(frame).state
                if state == GameState.MAIN_VILLAGE:
                    return True
                if state == GameState.SETTINGS:
                    self.ctx.actions.tap(*self.SETTINGS_CLOSE)
                elif (state == GameState.SWITCH_ACCOUNT
                      or self.ctx.matcher.match(frame, "supercell_id_logo",
                                                threshold=0.90)):
                    # 列表滚动后“切换ID”标题可能不可见，用左侧 Supercell ID 标识兜底
                    self.ctx.actions.tap(*self.SWITCH_CLOSE)
            time.sleep(0.6)
        return False

    def _wait_state(self, state: GameState, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self._grab()
            if frame is not None:
                if self.ctx.classifier.classify(frame).state == state:
                    return True
            time.sleep(0.4)
        return False

    def _read_accounts(self) -> list[dict]:
        frame = self._grab()
        if frame is None:
            return []
        badges = self.ctx.matcher.match_all(frame, self.BADGE, threshold=0.90)
        height = frame.shape[0]
        rows: list[dict] = []
        for m in sorted(badges, key=lambda b: b.center[1]):
            bc = m.center[1]
            # 滚动后首/末行可能只露出一半，跳过避免读到残缺文字
            if bc + self.NAME_DY0 < 0 or bc + self.NAME_DY1 > height:
                continue
            name = self.ctx.ocr.read_text(
                frame, (self.NAME_X0, bc + self.NAME_DY0,
                        self.NAME_X1 - self.NAME_X0,
                        self.NAME_DY1 - self.NAME_DY0))
            name = " ".join(name.split())
            if not name:
                continue
            key = self._best(self._names, name) or name
            th = self._cache.get(key)
            if th is None:
                th = self._read_th(frame, bc)
                if th:
                    self._cache.put(key, th)
                    logger.info("识别大本营等级：%s = %d 级", key, th)
            rows.append({"name": key, "th": th, "y": bc})
        self._cache.save()
        logger.info("切换ID页账号：%s", [(r["name"], r["th"]) for r in rows])
        return rows

    def _read_th(self, frame: np.ndarray, bc: int) -> int | None:
        """识别“N级大本营”的等级；有中文包时整段识别，否则退回数字 OCR。"""
        roi = (self.TH_X0, bc + self.TH_DY0,
               self.TH_X1 - self.TH_X0, self.TH_DY1 - self.TH_DY0)
        if "chi_sim" in getattr(self.ctx.ocr, "languages", set()):
            text = self.ctx.ocr.read_text(frame, roi, lang="chi_sim+eng",
                                          scale=3, psm=7)
            m = re.search(r"\d+", text)
            if m:
                return int(m.group())
        return self.ctx.ocr.read_number(
            frame, (self.TH_X0, bc + self.TH_DY0, 40,
                    self.TH_DY1 - self.TH_DY0))

    def _locate_account(self, name: str) -> dict | None:
        """在切换ID页查找账号；没找到就向下滚动列表继续找。"""
        before: list[str] = []
        for attempt in range(4):
            for _ in range(3):          # 等淡入/滚动动画结束再读
                rows = self._read_accounts()
                target = self._pick(rows, name)
                if target is not None:
                    logger.info("账号 %s 位于列表第 %d 行（y=%d）",
                                name, len(rows), target["y"])
                    return target
                if rows:
                    break
                time.sleep(0.8)
            if not rows or attempt == 3:
                break
            if [r["name"] for r in rows] == before:
                logger.info("账号列表已到底，未找到 %s", name)
                break
            before = [r["name"] for r in rows]
            logger.info("列表未找到 %s，向下滚动继续查找", name)
            self._scroll_list(up=True)
            time.sleep(0.8)
        logger.info("切换ID页（已滚动）仍未找到 %s", name)
        return None

    def _scroll_list(self, up: bool) -> None:
        """在账号列表区域滑动；up=True 表示内容上移（看下面的账号）。"""
        y1, y2 = (470, 200) if up else (200, 470)
        self.ctx.actions.swipe(self.LIST_SWIPE_X, y1,
                               self.LIST_SWIPE_X, y2, 0.35)

    def _pick(self, rows: list[dict], name: str) -> dict | None:
        norm = self._norm(name)
        for row in rows:
            if self._norm(row["name"]) == norm:
                return row
        # OCR 偶尔漏/多字母（如 MudBottleAlley → MudBotleAlley），模糊再找一次
        for row in rows:
            if SequenceMatcher(None, self._norm(row["name"]), norm).ratio() >= 0.8:
                logger.info("账号名模糊匹配：%s ≈ %s", row["name"], name)
                return row
        return None

    def _wait_village(self, timeout: float, row_y: int) -> bool:
        """等待切换后重载回村，期间处理奖励宝箱、欢迎弹窗与“已关闭页面”的情况。"""
        deadline = time.time() + timeout
        tapped_at = time.time()
        retried = False
        chest_taps = 0
        continue_taps = 0
        while time.time() < deadline:
            frame = self._grab()
            if frame is not None:
                if chest_taps < self._reward_taps:
                    chest = self.ctx.matcher.match(frame, self.CHEST,
                                                   threshold=0.85)
                    if chest:
                        chest_taps += 1
                        logger.info("加载页奖励宝箱，点击 %d/%d",
                                    chest_taps, self._reward_taps)
                        self.ctx.actions.tap(*chest.center)
                        time.sleep(0.35)
                        continue
                if continue_taps < self._reward_taps * 2:
                    btn = self.ctx.matcher.match(frame, self.REWARD_CONTINUE,
                                                 threshold=0.88)
                    if btn:
                        continue_taps += 1
                        logger.info("奖励卡片继续（第 %d 次）", continue_taps)
                        self.ctx.actions.tap(*btn.center)
                        time.sleep(0.5)
                        continue
                m = self.ctx.matcher.match(frame, "welcome_continue_btn",
                                           threshold=0.90)
                if m:
                    logger.info("关闭“欢迎回来”弹窗")
                    self.ctx.actions.tap(*m.center)
                    time.sleep(1.0)
                    continue
                state = self.ctx.classifier.classify(frame).state
                if state == GameState.MAIN_VILLAGE:
                    return True
                if state == GameState.SETTINGS:
                    # 点的是当前账号时页面只会关掉，不会重载
                    self.ctx.actions.tap(*self.SETTINGS_CLOSE)
                elif (state == GameState.SWITCH_ACCOUNT and not retried
                      and time.time() - tapped_at > 8):
                    retried = True
                    self.ctx.actions.tap(self.ROW_TAP_X, row_y)
            time.sleep(0.5)
        return False

    @staticmethod
    def _norm(text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum())

    @classmethod
    def _best(cls, candidates: list[str], text: str) -> str | None:
        """OCR 文本与配置账号名的模糊匹配，容忍识别误差。"""
        normalized = cls._norm(text)
        best, score = None, 0.0
        for cand in candidates:
            cn = cls._norm(cand)
            if not cn:
                continue
            ratio = SequenceMatcher(None, cn, normalized).ratio()
            if cn in normalized or normalized in cn:
                ratio = max(ratio, 0.99)
            if ratio > score:
                best, score = cand, ratio
        return best if score >= 0.75 else None
