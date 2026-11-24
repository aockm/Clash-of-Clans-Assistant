"""
Edge Detector - 边缘/区域检测器

战斗中识别红色不可部署区边界。
参考 HeyFang/coc-bot 的红区边缘扫描方案，动态确定可部署位置。
"""

import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DeployZone:
    """可部署区域"""
    x: int
    y: int
    width: int
    height: int
    is_deployable: bool = True
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


@dataclass
class EdgeResult:
    """边缘检测结果"""
    success: bool
    edges: Optional[np.ndarray] = None
    red_mask: Optional[np.ndarray] = None
    deployable_zone: Optional[DeployZone] = None
    red_regions: List[Tuple[int, int, int, int]] = None
    
    def __repr__(self) -> str:
        if self.success:
            zone = self.deployable_zone
            return f"EdgeResult(success=True, zone={zone})"
        return "EdgeResult(success=False)"


class EdgeDetector:
    """
    边缘/区域检测器
    
    主要用于战斗中：
    1. 检测红色不可部署区边界
    2. 计算可部署区域
    3. 生成部署坐标（沿边缘随机分布）
    """
    
    # 红色 HSV 范围（COC 中的红色边界）
    RED_LOWER_1 = np.array([0, 100, 100])
    RED_UPPER_1 = np.array([10, 255, 255])
    RED_LOWER_2 = np.array([160, 100, 100])
    RED_UPPER_2 = np.array([180, 255, 255])
    
    def __init__(
        self,
        red_sensitivity: int = 50,
        min_red_area: int = 100,
        edge_padding: int = 20
    ):
        """
        初始化边缘检测器
        
        Args:
            red_sensitivity: 红色检测灵敏度（调整饱和度阈值）
            min_red_area: 最小红色区域面积（过滤噪点）
            edge_padding: 部署区域距离边缘的内缩距离（像素）
        """
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV 不可用")
        
        self.red_sensitivity = red_sensitivity
        self.min_red_area = min_red_area
        self.edge_padding = edge_padding
        
        # 调整红色范围
        self.red_lower_1 = np.array([0, max(50, 100 - red_sensitivity), 100])
        self.red_upper_1 = np.array([10, 255, 255])
        self.red_lower_2 = np.array([160, max(50, 100 - red_sensitivity), 100])
        self.red_upper_2 = np.array([180, 255, 255])
        
        logger.info(
            f"EdgeDetector 初始化: sensitivity={red_sensitivity}, "
            f"min_area={min_red_area}, padding={edge_padding}"
        )
    
    def detect_red_regions(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        检测图像中的红色区域
        
        Args:
            frame: 输入帧（BGR 格式）
            
        Returns:
            (red_mask, regions) 红色掩码和区域列表 [(x,y,w,h), ...]
        """
        # 转 HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 创建红色掩码（红色在 HSV 中有两个范围）
        mask1 = cv2.inRange(hsv, self.red_lower_1, self.red_upper_1)
        mask2 = cv2.inRange(hsv, self.red_lower_2, self.red_upper_2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # 形态学操作清理噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 过滤小区域
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_red_area:
                x, y, w, h = cv2.boundingRect(contour)
                regions.append((x, y, w, h))
        
        return red_mask, regions
    
    def detect_deploy_zone(self, frame: np.ndarray) -> EdgeResult:
        """
        检测可部署区域（红色边界内的区域）
        
        Args:
            frame: 输入帧（BGR 格式）
            
        Returns:
            EdgeResult 包含可部署区域信息
        """
        h, w = frame.shape[:2]
        
        # 检测红色区域
        red_mask, red_regions = self.detect_red_regions(frame)
        
        if not red_regions:
            # 未检测到红色区域，可能不在战斗中
            logger.debug("未检测到红色边界区域")
            return EdgeResult(
                success=False,
                red_mask=red_mask,
                red_regions=[]
            )
        
        # 计算所有红色区域的包围盒
        min_x = min(r[0] for r in red_regions)
        min_y = min(r[1] for r in red_regions)
        max_x = max(r[0] + r[2] for r in red_regions)
        max_y = max(r[1] + r[3] for r in red_regions)
        
        # 可部署区域 = 红色边界内缩
        deploy_x = min_x + self.edge_padding
        deploy_y = min_y + self.edge_padding
        deploy_w = max(0, (max_x - min_x) - 2 * self.edge_padding)
        deploy_h = max(0, (max_y - min_y) - 2 * self.edge_padding)
        
        if deploy_w <= 0 or deploy_h <= 0:
            logger.warning("计算的可部署区域无效")
            return EdgeResult(
                success=False,
                red_mask=red_mask,
                red_regions=red_regions
            )
        
        zone = DeployZone(
            x=deploy_x,
            y=deploy_y,
            width=deploy_w,
            height=deploy_h,
            is_deployable=True
        )
        
        logger.debug(
            f"检测到可部署区域: ({deploy_x}, {deploy_y}) "
            f"{deploy_w}x{deploy_h}"
        )
        
        return EdgeResult(
            success=True,
            red_mask=red_mask,
            red_regions=red_regions,
            deployable_zone=zone
        )
    
    def generate_deploy_points(
        self,
        zone: DeployZone,
        count: int = 5,
        random_offset: int = 10
    ) -> List[Tuple[int, int]]:
        """
        在可部署区域内生成部署坐标点
        
        坐标沿边缘随机分布，模拟人工操作。
        
        Args:
            zone: 可部署区域
            count: 生成点的数量
            random_offset: 随机偏移量（像素）
            
        Returns:
            部署坐标列表 [(x, y), ...]
        """
        if not zone.is_deployable:
            return []
        
        points = []
        
        # 沿四条边分布
        edges = [
            # 上边
            lambda t: (zone.x + int(t * zone.width), zone.y),
            # 下边
            lambda t: (zone.x + int(t * zone.width), zone.y + zone.height),
            # 左边
            lambda t: (zone.x, zone.y + int(t * zone.height)),
            # 右边
            lambda t: (zone.x + zone.width, zone.y + int(t * zone.height)),
        ]
        
        for i in range(count):
            # 随机选择一条边
            edge_func = edges[i % len(edges)]
            t = np.random.random()
            
            base_x, base_y = edge_func(t)
            
            # 添加随机偏移
            offset_x = np.random.randint(-random_offset, random_offset + 1)
            offset_y = np.random.randint(-random_offset, random_offset + 1)
            
            final_x = base_x + offset_x
            final_y = base_y + offset_y
            
            points.append((final_x, final_y))
        
        return points
    
    def detect_edges(self, frame: np.ndarray, threshold1: int = 50, threshold2: int = 150) -> np.ndarray:
        """
        Canny 边缘检测（通用）
        
        Args:
            frame: 输入帧
            threshold1: Canny 低阈值
            threshold2: Canny 高阈值
            
        Returns:
            边缘图像
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2)
        return edges
    
    def detect_contours(
        self,
        frame: np.ndarray,
        min_area: int = 100,
        max_area: Optional[int] = None
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        检测轮廓
        
        Args:
            frame: 输入帧
            min_area: 最小轮廓面积
            max_area: 最大轮廓面积
            
        Returns:
            轮廓列表 [(x, y, w, h, area), ...]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            if max_area and area > max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            results.append((x, y, w, h, area))
        
        return results
    
    def set_red_sensitivity(self, sensitivity: int) -> None:
        """调整红色检测灵敏度"""
        self.red_sensitivity = sensitivity
        self.red_lower_1 = np.array([0, max(50, 100 - sensitivity), 100])
        self.red_lower_2 = np.array([160, max(50, 100 - sensitivity), 100])
        logger.info(f"红色检测灵敏度调整为: {sensitivity}")
    
    def __repr__(self) -> str:
        return (
            f"EdgeDetector(sensitivity={self.red_sensitivity}, "
            f"min_area={self.min_red_area}, padding={self.edge_padding})"
        )
