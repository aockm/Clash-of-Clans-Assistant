import logging
import re

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)

# 例如：1天2小时 / 1小时52分钟 / 4分钟0秒 / 5秒 / 2小时0分
_DURATION_RE = re.compile(
    r"(?:(\d+)\s*天)?\s*(?:(\d+)\s*小?\s*时)?\s*(?:(\d+)\s*分\s*钟?)?\s*(?:(\d+)\s*秒)?")


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


class OcrEngine:
    """数字 OCR：灰度 → 上采样 → 二值化，白名单限定数字。"""

    CONFIG = '--psm 7 -c tessedit_char_whitelist=0123456789,'

    def __init__(self) -> None:
        try:
            self.languages = set(pytesseract.get_languages())
        except Exception:
            self.languages = set()
        self._warned_zh = False

    def read_number(self, frame: np.ndarray,
                    roi: tuple[int, int, int, int]) -> int | None:
        roi = self._clip(frame, roi)
        if roi is None:
            return None
        x, y, w, h = roi
        crop = frame[y:y + h, x:x + w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(bw, config=self.CONFIG).strip()
        digits = "".join(ch for ch in text if ch.isdigit())
        return int(digits) if digits else None

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

    def read_duration(self, frame: np.ndarray,
                      roi: tuple[int, int, int, int]) -> int | None:
        """识别“X小时Y分钟”形式的剩余时间，返回秒数。需要中文语言包。"""
        if "chi_sim" not in self.languages:
            if not self._warned_zh:
                self._warned_zh = True
                logger.warning("未安装 tesseract 中文语言包 chi_sim，"
                               "建筑剩余时间将无法识别")
            return None
        text = self.read_text(frame, roi, lang="chi_sim+eng",
                              scale=4, psm=7)
        seconds = parse_duration(text)
        if seconds is None and text:
            logger.info("剩余时间识别失败：%r", text)
        return seconds

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
