"""
Template Matcher - 模板匹配引擎

使用 OpenCV matchTemplate 检测 UI 元素（进攻按钮、捐兵图标、升级按钮等）。
模板需按分辨率组织多尺度版本。预加载所有模板到内存，避免循环中重复读盘。
"""

import os
import logging
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """模板匹配结果"""
    found: bool
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    confidence: float = 0.0
    template_name: str = ""
    
    @property
    def center(self) -> Tuple[int, int]:
        """匹配区域的中心点"""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """边界框 (x, y, width, height)"""
        return (self.x, self.y, self.width, self.height)
    
    def __repr__(self) -> str:
        if self.found:
            return f"MatchResult(found=True, center={self.center}, conf={self.confidence:.3f})"
        return "MatchResult(found=False)"


class TemplateMatcher:
    """
    模板匹配引擎
    
    使用 OpenCV TM_CCOEFF_NORMED 方法进行模板匹配。
    支持多尺度模板、ROI 区域限制、批量匹配。
    """
    
    def __init__(self, templates_dir: str = "assets/templates"):
        """
        初始化模板匹配器
        
        Args:
            templates_dir: 模板图片目录
        """
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV 不可用，无法使用模板匹配")
        
        self.templates_dir = templates_dir
        self._templates: Dict[str, np.ndarray] = {}
        self._template_scales: Dict[str, float] = {}
        
        # 默认匹配阈值
        self.default_threshold = 0.8
        
        logger.info(f"TemplateMatcher 初始化，模板目录: {templates_dir}")
    
    def load_template(self, name: str, path: Optional[str] = None, scale: float = 1.0) -> bool:
        """
        加载单个模板
        
        Args:
            name: 模板名称（用于后续匹配）
            path: 模板图片路径，None 则使用默认路径
            scale: 模板缩放因子（用于多分辨率支持）
            
        Returns:
            加载成功返回 True，否则返回 False
        """
        if path is None:
            path = os.path.join(self.templates_dir, f"{name}.png")
        
        if not os.path.exists(path):
            logger.error(f"模板文件不存在: {path}")
            return False
        
        try:
            template = cv2.imread(path, cv2.IMREAD_COLOR)
            if template is None:
                logger.error(f"无法读取模板图片: {path}")
                return False
            
            # 应用缩放
            if scale != 1.0:
                h, w = template.shape[:2]
                new_w, new_h = int(w * scale), int(h * scale)
                template = cv2.resize(template, (new_w, new_h))
                logger.debug(f"模板 {name} 缩放至 {new_w}x{new_h}")
            
            self._templates[name] = template
            self._template_scales[name] = scale
            
            logger.info(f"加载模板: {name} ({template.shape[1]}x{template.shape[0]})")
            return True
            
        except Exception as e:
            logger.error(f"加载模板失败 {name}: {e}")
            return False
    
    def load_templates_from_dir(self, subdir: Optional[str] = None) -> int:
        """
        从目录批量加载模板
        
        Args:
            subdir: 子目录名称，None 表示根目录
            
        Returns:
            成功加载的模板数量
        """
        if subdir:
            load_dir = os.path.join(self.templates_dir, subdir)
        else:
            load_dir = self.templates_dir
        
        if not os.path.exists(load_dir):
            logger.warning(f"模板目录不存在: {load_dir}")
            return 0
        
        count = 0
        for filename in os.listdir(load_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                name = os.path.splitext(filename)[0]
                path = os.path.join(load_dir, filename)
                if self.load_template(name, path):
                    count += 1
        
        logger.info(f"从 {load_dir} 加载了 {count} 个模板")
        return count
    
    def match(
        self,
        frame: np.ndarray,
        template_name: str,
        threshold: Optional[float] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        grayscale: bool = True
    ) -> MatchResult:
        """
        在帧中匹配单个模板
        
        Args:
            frame: 输入帧（BGR 格式）
            template_name: 模板名称
            threshold: 匹配阈值，None 使用默认值
            roi: 感兴趣区域 (x, y, width, height)，None 表示全图搜索
            grayscale: 是否转换为灰度图匹配（更快）
            
        Returns:
            MatchResult 匹配结果
        """
        if template_name not in self._templates:
            logger.warning(f"模板未加载: {template_name}")
            return MatchResult(found=False)
        
        template = self._templates[template_name]
        threshold = threshold if threshold is not None else self.default_threshold
        
        # 准备搜索图像
        if roi:
            x, y, w, h = roi
            search_img = frame[y:y+h, x:x+w]
        else:
            search_img = frame
            x, y = 0, 0
        
        if search_img.size == 0:
            return MatchResult(found=False)
        
        # 转换颜色空间
        if grayscale:
            search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # 模板尺寸
        th, tw = template.shape[:2]
        sh, sw = search_img.shape[:2]
        
        # 检查模板是否大于搜索图像
        if tw > sw or th > sh:
            logger.warning(f"模板尺寸 ({tw}x{th}) 大于搜索区域 ({sw}x{sh})")
            return MatchResult(found=False)
        
        # 执行模板匹配
        result = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # 判断是否匹配
        if max_val >= threshold:
            match_x = x + max_loc[0]
            match_y = y + max_loc[1]
            
            return MatchResult(
                found=True,
                x=match_x,
                y=match_y,
                width=tw,
                height=th,
                confidence=max_val,
                template_name=template_name
            )
        else:
            return MatchResult(found=False, confidence=max_val, template_name=template_name)
    
    def match_all(
        self,
        frame: np.ndarray,
        template_name: str,
        threshold: Optional[float] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        max_count: int = 10
    ) -> List[MatchResult]:
        """
        在帧中匹配模板的所有出现位置
        
        Args:
            frame: 输入帧
            template_name: 模板名称
            threshold: 匹配阈值
            roi: 感兴趣区域
            max_count: 最大返回数量
            
        Returns:
            匹配结果列表
        """
        if template_name not in self._templates:
            return []
        
        template = self._templates[template_name]
        threshold = threshold if threshold is not None else self.default_threshold
        
        # 准备搜索图像
        if roi:
            x, y, w, h = roi
            search_img = frame[y:y+h, x:x+w]
        else:
            search_img = frame
            x, y = 0, 0
        
        # 转灰度
        gray_search = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        th, tw = gray_template.shape[:2]
        
        # 执行匹配
        result = cv2.matchTemplate(gray_search, gray_template, cv2.TM_CCOEFF_NORMED)
        
        # 找到所有超过阈值的位置
        locations = np.where(result >= threshold)
        
        matches = []
        for pt in zip(*locations[::-1]):  # (x, y) 格式
            if len(matches) >= max_count:
                break
            
            match_x = x + pt[0]
            match_y = y + pt[1]
            
            matches.append(MatchResult(
                found=True,
                x=match_x,
                y=match_y,
                width=tw,
                height=th,
                confidence=result[pt[1], pt[0]],
                template_name=template_name
            ))
        
        # 去除重叠的匹配（非极大值抑制）
        matches = self._nms(matches, overlap_thresh=0.5)
        
        return matches
    
    def match_multi(
        self,
        frame: np.ndarray,
        template_names: List[str],
        threshold: Optional[float] = None,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict[str, MatchResult]:
        """
        同时匹配多个模板
        
        Args:
            frame: 输入帧
            template_names: 模板名称列表
            threshold: 匹配阈值
            roi: 感兴趣区域
            
        Returns:
            {模板名: MatchResult} 字典
        """
        results = {}
        for name in template_names:
            results[name] = self.match(frame, name, threshold, roi)
        return results
    
    def _nms(self, matches: List[MatchResult], overlap_thresh: float = 0.5) -> List[MatchResult]:
        """
        非极大值抑制，去除重叠的匹配
        
        Args:
            matches: 匹配结果列表
            overlap_thresh: 重叠阈值
            
        Returns:
            过滤后的匹配列表
        """
        if not matches:
            return []
        
        # 按置信度降序排序
        matches.sort(key=lambda m: m.confidence, reverse=True)
        
        keep = []
        for m in matches:
            # 检查与已保留匹配的重叠
            suppress = False
            for k in keep:
                if self._iou(m, k) > overlap_thresh:
                    suppress = True
                    break
            
            if not suppress:
                keep.append(m)
        
        return keep
    
    def _iou(self, a: MatchResult, b: MatchResult) -> float:
        """计算两个匹配的 IoU（交并比）"""
        x1 = max(a.x, b.x)
        y1 = max(a.y, b.y)
        x2 = min(a.x + a.width, b.x + b.width)
        y2 = min(a.y + a.height, b.y + b.height)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area_a = a.width * a.height
        area_b = b.width * b.height
        union = area_a + area_b - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def set_template(self, name: str, template: np.ndarray) -> None:
        """
        直接设置模板（用于动态生成的模板）
        
        Args:
            name: 模板名称
            template: 模板图像（BGR 格式）
        """
        self._templates[name] = template.copy()
        self._template_scales[name] = 1.0
        logger.info(f"设置模板: {name} ({template.shape[1]}x{template.shape[0]})")
    
    def get_template(self, name: str) -> Optional[np.ndarray]:
        """获取已加载的模板"""
        return self._templates.get(name)
    
    def list_templates(self) -> List[str]:
        """列出所有已加载的模板"""
        return list(self._templates.keys())
    
    def __repr__(self) -> str:
        return f"TemplateMatcher(templates={len(self._templates)}, dir={self.templates_dir})"
