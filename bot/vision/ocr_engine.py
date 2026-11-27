import cv2
import numpy as np
import pytesseract


class OcrEngine:
    """数字 OCR：灰度 → 上采样 → 二值化，白名单限定数字。"""

    CONFIG = '--psm 7 -c tessedit_char_whitelist=0123456789,'

    def read_number(self, frame: np.ndarray,
                    roi: tuple[int, int, int, int]) -> int | None:
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
        x, y, w, h = roi
        crop = frame[y:y + h, x:x + w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
        return pytesseract.image_to_string(
            gray, lang=lang, config=f"--psm {psm}").strip()
