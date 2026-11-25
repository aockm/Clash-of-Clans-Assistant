import cv2
import numpy as np


class RedZoneEdgeDetector:
    """战斗中红区（不可部署区域）边界检测，输出边界外沿的可部署点集。"""

    # 部署边界线是橙红色（H≈13~25），不是纯红；纯红范围保留以兼容偏红的版本
    LOWER_RED1 = np.array([0, 130, 110])
    UPPER_RED1 = np.array([12, 255, 255])
    LOWER_RED2 = np.array([10, 140, 140])
    UPPER_RED2 = np.array([28, 255, 255])
    LOWER_RED3 = np.array([165, 130, 110])
    UPPER_RED3 = np.array([180, 255, 255])

    def _red_mask(self, frame: np.ndarray, map_bottom: int) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LOWER_RED1, self.UPPER_RED1) | \
               cv2.inRange(hsv, self.LOWER_RED2, self.UPPER_RED2) | \
               cv2.inRange(hsv, self.LOWER_RED3, self.UPPER_RED3)
        # 底部卡牌栏和右下按钮也会命中，剔除
        mask[map_bottom:, :] = 0
        return mask

    def _outer_component(self, mask: np.ndarray) -> np.ndarray:
        """取面积最大的连通域，排除远处零散的地形/装饰误命中。"""
        kernel = np.ones((9, 9), np.uint8)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
        if count <= 1:
            return np.zeros_like(mask)
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return np.where(labels == best, 255, 0).astype(np.uint8)

    @staticmethod
    def _resample(points: np.ndarray, spacing: int) -> list[tuple[int, int]]:
        """沿多边形按固定间距重采样。"""
        out: list[tuple[int, int]] = []
        carry = 0.0
        n = len(points)
        for i in range(n):
            p = points[i].astype(float)
            q = points[(i + 1) % n].astype(float)
            seg = np.linalg.norm(q - p)
            if seg == 0:
                continue
            pos = spacing - carry
            while pos <= seg:
                pt = p + (q - p) * (pos / seg)
                out.append((int(pt[0]), int(pt[1])))
                pos += spacing
            carry = (carry + seg) % spacing
        return out

    def boundary_points(self, frame: np.ndarray, margin: int = 40,
                        map_bottom: int = 365, spacing: int = 45) -> list[tuple[int, int]]:
        """红区外扩后取外边界（凸包），按周边顺序返回可部署点。"""
        mask = self._red_mask(frame, map_bottom)
        if cv2.countNonZero(mask) < 500:
            return []

        mask = self._outer_component(mask)
        kernel = np.ones((margin, margin), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)

        nonzero = cv2.findNonZero(dilated)
        if nonzero is None:
            return []
        hull = cv2.convexHull(nonzero)

        h, w = frame.shape[:2]
        points = [p for p in self._resample(hull.reshape(-1, 2), spacing)
                  if 10 < p[0] < w - 10 and 10 < p[1] < map_bottom - 5]
        return points

    def red_centroid(self, frame: np.ndarray,
                     map_bottom: int = 365) -> tuple[int, int] | None:
        """红区质心，约等于基地中心，可作为"点建筑"的默认落点。"""
        mask = self._outer_component(self._red_mask(frame, map_bottom))
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        return int(xs.mean()), int(ys.mean())
