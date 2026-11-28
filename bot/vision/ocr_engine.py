import logging
import re
from collections import Counter

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)

# 例如：1天2小时 / 1小时52分钟 / 4分钟0秒 / 5秒 / 2小时0分
_DURATION_RE = re.compile(
    r"(?:(\d+)\s*天)?\s*(?:(\d+)\s*小?\s*时)?\s*(?:(\d+)\s*分\s*钟?)?\s*(?:(\d+)\s*秒)?")
_DIGITS_RE = re.compile(r"\d+")

# 单字单位 → (本单秒数, 下一级单秒数)。游戏里时长总是从大到小排列，
# 所以只要认出“最大的那个单位”，后面的位数顺序就固定了。
_UNITS = {
    "day": (86400, 3600),
    "hour": (3600, 60),
    "minute": (60, 1),
    "second": (1, 1),
}
_UNIT_ORDER = ("day", "hour", "minute", "second")

# 游戏内小字号白字黑描边，单一阈值/单一缩放都容易漏，
# 用多组“缩放×阈值×膨胀”组合分别识别再投票，实测比单次稳定得多。
_DURATION_VARIANTS = (
    (4, 130, 3), (6, 130, 3), (6, 150, 3), (8, 130, 3), (8, 150, 2),
    (10, 130, 3), (4, 150, 0), (4, 185, 0), (6, 185, 0), (4, 200, 0),
)


def parse_duration(text: str) -> int | None:
    """把“1小时52分钟”这类文字解析成秒数；无法解析返回 None。"""
    if not text:
        return None
    compact = "".join(text.split())
    m = _DURATION_RE.search(compact)
    if not m:
        return None
    days, hours, minutes, seconds = (int(v) if v else 0 for v in m.groups())
    if not any((days, hours, minutes, seconds)):
        return None
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _largest_unit(text: str) -> str | None:
    """返回文本里出现过的最大时间单位。"""
    if "天" in text:
        return "day"
    if "时" in text or "小" in text:
        return "hour"
    if "分" in text:
        return "minute"
    if "秒" in text:
        return "second"
    return None


class OcrEngine:
    """数字 OCR：灰度 → 上采样 → 二值化，白名单限定数字。"""

    CONFIG = '--psm 7 -c tessedit_char_whitelist=0123456789,'
    RATIO_CONFIG = '--psm 7 -c tessedit_char_whitelist=0123456789/'
    TEXT_THRESHOLD = 195      # 提取“亮色文字”的灰度阈值
    NUMBER_CUTS = (200, 210, 220)   # 数字 OCR 的亮色阈值投票组
    # 左上角战利品栏：金币是亮黄圆币、圣水是紫色水滴，数字紧跟在图标右侧。
    # 敌方资源栏有两种排布（带“2倍”奖励标记 / 不带），位置会整体错开，
    # 所以先用图标定位再框数字；找不到图标时退回下面这套固定 ROI。
    LOOT_AREA = (30, 30, 140, 220)   # 只在左上角这块区域找图标（x, y, w, h）
    LOOT_ICONS = {
        "gold": ((18, 150, 190), (35, 255, 255)),
        "elixir": ((125, 120, 140), (165, 255, 255)),
    }
    LOOT_ROI = {"gold": (88, 68, 78, 26), "elixir": (88, 104, 78, 26)}

    def __init__(self) -> None:
        try:
            self.languages = set(pytesseract.get_languages())
        except Exception:
            self.languages = set()
        self._warned_zh = False

    def read_number(self, frame: np.ndarray,
                    roi: tuple[int, int, int, int]) -> int | None:
        """读亮色数字（战利品/摧毁率）：多组高亮阈值各识别一次再投票。

        游戏里的数字是亮黄/纯白配深描边，但底色深浅差别很大（预览页浅绿草地、
        战斗页深色地面）。单用 OTSU 会被大面积背景带偏，实测固定高亮阈值更稳。
        """
        roi = self._clip(frame, roi)
        if roi is None:
            return None
        x, y, w, h = roi
        gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        votes: Counter[int] = Counter()
        first: int | None = None
        for cut in self.NUMBER_CUTS:
            mask = (gray > cut).astype(np.uint8) * 255
            mask = cv2.resize(mask, None, fx=6, fy=6,
                              interpolation=cv2.INTER_CUBIC)
            mask = cv2.copyMakeBorder(255 - mask, 20, 20, 20, 20,
                                      cv2.BORDER_CONSTANT, value=255)
            text = pytesseract.image_to_string(mask, config=self.CONFIG)
            digits = "".join(ch for ch in text if ch.isdigit())
            if not digits:
                continue
            value = int(digits)
            votes[value] += 1
            if first is None:
                first = value
        if not votes:
            return None
        best, count = votes.most_common(1)[0]
        return best if count >= 2 else first

    def read_loot(self, frame: np.ndarray) -> tuple[int | None, int | None]:
        """读左上角敌方资源栏，返回（金币, 圣水）。

        先按图标定位数字，能兼容“带/不带 2 倍奖励标记”两种排布；
        图标没认出来时退回固定 ROI，两个数值互不影响。
        """
        result = []
        for key in ("gold", "elixir"):
            roi = self._loot_roi(frame, key) or self.LOOT_ROI[key]
            result.append(self.read_number(frame, roi))
        return result[0], result[1]

    def _loot_roi(self, frame: np.ndarray,
                  key: str) -> tuple[int, int, int, int] | None:
        """在战利品栏里找资源图标，返回其右侧数字区域。"""
        x0, y0, w0, h0 = self.LOOT_AREA
        area = frame[y0:y0 + h0, x0:x0 + w0]
        if area.size == 0:
            return None
        hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
        lo, hi = self.LOOT_ICONS[key]
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        best = None
        for c in cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(c) < 150:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if not (0.6 <= w / h <= 1.6):
                continue
            # 战利品栏的图标贴着左边，靠右的都是背景装饰
            if best is None or x < best[0]:
                best = (x, y, w, h)
        if best is None:
            return None
        x, y, w, h = best
        return (x0 + x + w + 2, y0 + y - 2, 80, h + 4)

    def read_text(self, frame: np.ndarray,
                  roi: tuple[int, int, int, int],
                  lang: str = "eng", scale: int = 3, psm: int = 7) -> str:
        roi = self._clip(frame, roi)
        if roi is None:
            return ""
        x, y, w, h = roi
        crop = frame[y:y + h, x:x + w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
        return pytesseract.image_to_string(
            gray, lang=lang, config=f"--psm {psm}").strip()

    def read_ratio(self, frame: np.ndarray,
                   roi: tuple[int, int, int, int]) -> tuple[int, int] | None:
        """识别“剩余/总数”这类 x/y 小胶囊（工人数、可升级项数）。

        ROI 直接框住胶囊里的数字部分；上方的图标是亮色，不能参与二值化，
        所以这里不做亮字包围盒，只按 OTSU 反相取深色胶囊里的白字。
        """
        roi = self._clip(frame, roi)
        if roi is None:
            return None
        x, y, w, h = roi
        crop = frame[y:y + h, x:x + w]
        for scale in (6, 8):
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_LANCZOS4)
            # 胶囊所在背景并不统一，OTSU 偶尔会把底板和文字并到一起，
            # 所以固定阈值和 OTSU 都试一遍，谁能读出 x/y 就用谁
            for thr in (140, 170, 0):
                if thr:
                    bw = cv2.threshold(gray, thr, 255,
                                       cv2.THRESH_BINARY_INV)[1]
                else:
                    bw = cv2.threshold(
                        gray, 0, 255,
                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                bw = cv2.copyMakeBorder(bw, 15, 15, 15, 15,
                                        cv2.BORDER_CONSTANT, value=255)
                text = pytesseract.image_to_string(bw, config=self.RATIO_CONFIG)
                m = re.search(r"(\d)\s*[/|]\s*(\d)", "".join(text.split()))
                if m:
                    return int(m.group(1)), int(m.group(2))
        return None

    def read_duration(self, frame: np.ndarray,
                      roi: tuple[int, int, int, int]) -> int | None:
        """识别“X小时Y分钟”形式的剩余时间，返回秒数。需要中文语言包。

        ROI 只需覆盖文字可能出现的大致范围：内部会先找出亮色文字的紧包围盒，
        再做多组变体识别投票，避免把时长文字裁断或把描边当笔画。
        """
        if "chi_sim" not in self.languages:
            if not self._warned_zh:
                self._warned_zh = True
                logger.warning("未安装 tesseract 中文语言包 chi_sim，"
                               "建筑剩余时间将无法识别")
            return None
        box = self._text_bbox(frame, roi)
        if box is None:
            return None
        x, y, w, h = box
        crop = frame[max(0, y - 2):y + h + 2, max(0, x - 2):x + w + 2]

        unit_votes: Counter[str] = Counter()
        first_votes: Counter[int] = Counter()
        second_votes: Counter[int] = Counter()
        texts: list[str] = []
        for scale, thr, ksize in _DURATION_VARIANTS:
            text = self._ocr_variant(crop, scale, thr, ksize)
            if not text:
                continue
            texts.append(text)
            unit = _largest_unit(text)
            if unit:
                unit_votes[unit] += 1
            nums = [int(n) for n in _DIGITS_RE.findall(text)]
            if nums:
                first_votes[nums[0]] += 1
            if len(nums) >= 2:
                second_votes[nums[-1]] += 1

        seconds = self._combine(unit_votes, first_votes, second_votes)
        if seconds is None:
            logger.info("剩余时间识别失败：%r", texts)
        return seconds

    def _combine(self, unit_votes: Counter, first_votes: Counter,
                 second_votes: Counter) -> int | None:
        """按“最大单位 + 各数字的多数票”组合秒数。

        单位字（时/分/秒）比数字更容易被误识，所以只认“多数票”的数字；
        第二位数字没有共识时就按下一位的一半偏上估算，宁可略晚也不早。
        """
        if not first_votes:
            return None
        unit = unit_votes.most_common(1)[0][0] if unit_votes else None
        if unit is None:
            return None
        big, big_cnt = first_votes.most_common(1)[0]
        if big_cnt < 2:
            return None

        unit_sec, next_sec = _UNITS[unit]
        if unit == "second":
            return big if big < 60 else None

        limit = 24 if unit == "day" else 60
        if second_votes:
            small, small_cnt = second_votes.most_common(1)[0]
            if small < 10:
                # 十位数字常被单位字吃掉（28 → 8，23 → 3），优先补回两位数：
                # 先找同个位的两位数（证据最强），再找票数够的两位数
                wider = [(v, c) for v, c in second_votes.items() if v >= 10]
                same_tail = [(v, c) for v, c in wider if v % 10 == small]
                if same_tail:
                    small, small_cnt = max(same_tail, key=lambda x: x[1])
                elif wider:
                    best_v, best_c = max(wider, key=lambda x: x[1])
                    if best_c >= 2:
                        small, small_cnt = best_v, best_c
            if small_cnt >= 2:
                # 单位字常跟数字粘在一起（32秒 → 328），逐次去掉末位回到合法区间
                while small >= limit and small >= 10:
                    small //= 10
                if small < limit:
                    return big * unit_sec + small * next_sec
        return big * unit_sec + int(limit * 0.6) * next_sec

    def _ocr_variant(self, crop: np.ndarray, scale: int, thr: int,
                     ksize: int) -> str:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_LANCZOS4)
        mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)[1]
        if ksize:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (ksize, ksize))
            mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.copyMakeBorder(mask, 25, 25, 25, 25,
                                  cv2.BORDER_CONSTANT, value=255)
        text = pytesseract.image_to_string(mask, lang="chi_sim",
                                           config="--psm 7")
        return " ".join(text.split())

    def _text_bbox(self, frame: np.ndarray,
                   roi: tuple[int, int, int, int]
                   ) -> tuple[int, int, int, int] | None:
        """在 ROI 内找出亮色文字的紧包围盒（去掉行背景与装饰）。"""
        roi = self._clip(frame, roi)
        if roi is None:
            return None
        x, y, w, h = roi
        gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        mask = (gray > self.TEXT_THRESHOLD).astype(np.uint8)
        cols, rows = mask.sum(axis=0), mask.sum(axis=1)
        xs = np.where(cols >= 1)[0]
        ys = np.where(rows >= 2)[0]
        if len(xs) == 0 or len(ys) == 0:
            return None
        return (x + int(xs.min()), y + int(ys.min()),
                int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1)

    @staticmethod
    def _clip(frame: np.ndarray,
              roi: tuple[int, int, int, int]
              ) -> tuple[int, int, int, int] | None:
        """把 ROI 裁剪到画面内；完全越界时返回 None。"""
        x, y, w, h = roi
        fh, fw = frame.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(fw, x + w), min(fh, y + h)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1 - x0, y1 - y0
