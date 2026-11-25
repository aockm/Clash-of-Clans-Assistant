import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    name: str
    score: float
    center: tuple[int, int]
    rect: tuple[int, int, int, int]


class TemplateMatcher:
    """模板匹配引擎：启动时预加载全部模板，支持 ROI 加速。"""

    def __init__(self, templates_dir: Path) -> None:
        self._templates: dict[str, np.ndarray] = {}
        self._load_all(templates_dir)

    def _load_all(self, root: Path) -> None:
        for p in root.rglob("*.png"):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                self._templates[p.stem] = img
        logger.info("Loaded %d templates", len(self._templates))

    def match(self, frame: np.ndarray, name: str,
              roi: tuple[int, int, int, int] | None = None,
              threshold: float = 0.85) -> MatchResult | None:
        tpl = self._templates.get(name)
        if tpl is None:
            logger.warning("Template not found: %s", name)
            return None
        x0, y0 = 0, 0
        search = frame
        if roi is not None:
            x0, y0, w, h = roi
            search = frame[y0:y0 + h, x0:x0 + w]
        res = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score < threshold:
            return None
        th, tw = tpl.shape[:2]
        rect = (x0 + loc[0], y0 + loc[1], tw, th)
        center = (rect[0] + tw // 2, rect[1] + th // 2)
        return MatchResult(name=name, score=float(score), center=center, rect=rect)
