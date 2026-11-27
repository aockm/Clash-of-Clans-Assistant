import json
import logging
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
    BADGE = "switch_account_badge"

    # 账号行内文字相对灰色徽标中心的偏移（视频坐标）
    NAME_X0, NAME_X1 = 720, 1064
    NAME_DY0, NAME_DY1 = -30, 12
    # 等级数字只占“N级大本营”前 1~2 个字符，ROI 过宽会把“级”误识别成数字
    TH_X0, TH_X1 = 736, 782
    TH_DY0, TH_DY1 = 40, 76

    def __init__(self, ctx, frame_getter: Callable[[], np.ndarray | None]) -> None:
        self.ctx = ctx
        self._grab = frame_getter
        cfg = ctx.config.accounts
        self._names = cfg.names
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

        rows = self._read_accounts()
        if not rows:
            logger.warning("切换ID页没有识别到账号")
            self.ctx.actions.tap(*self.SWITCH_CLOSE)
            return None

        target = self._pick(rows, name)
        if target is None:
            logger.warning("未找到账号 %s，页面账号：%s", name,
                           [r["name"] for r in rows])
            self.ctx.actions.tap(*self.SWITCH_CLOSE)
            return None

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
                elif state == GameState.SWITCH_ACCOUNT:
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
        rows: list[dict] = []
        for m in sorted(badges, key=lambda b: b.center[1]):
            bc = m.center[1]
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
                th = self.ctx.ocr.read_number(
                    frame, (self.TH_X0, bc + self.TH_DY0,
                            self.TH_X1 - self.TH_X0,
                            self.TH_DY1 - self.TH_DY0))
                if th:
                    self._cache.put(key, th)
                    logger.info("识别大本营等级：%s = %d 级", key, th)
            rows.append({"name": key, "th": th, "y": bc})
        self._cache.save()
        logger.info("切换ID页账号：%s", [(r["name"], r["th"]) for r in rows])
        return rows

    def _pick(self, rows: list[dict], name: str) -> dict | None:
        for row in rows:
            if self._norm(row["name"]) == self._norm(name):
                return row
        return None

    def _wait_village(self, timeout: float, row_y: int) -> bool:
        """等待切换后重载回村，期间处理欢迎弹窗与“已关闭页面”的情况。"""
        deadline = time.time() + timeout
        tapped_at = time.time()
        retried = False
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
