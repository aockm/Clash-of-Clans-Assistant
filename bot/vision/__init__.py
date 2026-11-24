"""
Vision & Recognition Layer - 视觉与识别层模块
包含模板匹配、OCR、边缘检测等视觉能力
"""

from .template_matcher import TemplateMatcher, MatchResult
from .ocr_engine import OCREngine, OCRResult
from .edge_detector import EdgeDetector, DeployZone

__all__ = [
    "TemplateMatcher",
    "MatchResult",
    "OCREngine",
    "OCRResult",
    "EdgeDetector",
    "DeployZone",
]
