"""
Battle Deployer - 战斗部署功能

在敌方基地部署兵种。
流程：EnemyBase → 部署兵种 → 战斗结束 → Results。
使用红区边缘检测确定可部署位置，英雄技能延迟激活。

兵种部署采用动态策略：通过检测底部兵种栏槽位是否有卡牌
（绿色草地占比低即为有兵），逐个槽位部署直到槽位清空，
从而完整部署军队中的全部兵种，无需为每个兵种配置模板。
"""

import time
import random
import logging
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np

from ..states.state_machine import GameState, StateMachine
from ..vision.template_matcher import TemplateMatcher
from ..vision.edge_detector import EdgeDetector, DeployZone

logger = logging.getLogger(__name__)


class DeployPattern(Enum):
    """部署模式"""
    EDGE = auto()       # 沿边缘部署
    CORNER = auto()     # 角落部署
    SPREAD = auto()     # 分散部署
    FOCUSED = auto()    # 集中部署


@dataclass
class HeroConfig:
    """英雄配置"""
    name: str
    template: str
    ability_template: str = "hero_ability"
    activate_delay: float = 5.0  # 英雄部署后多久激活技能


class BattleDeployer:
    """
    战斗部署器
    
    在战斗中自动部署兵种和激活英雄技能。
    使用边缘检测确定可部署区域，沿边缘随机分布部署。
    兵种栏槽位动态检测，逐个槽位清空实现全部部队部署。
    """
    
    # 兵种栏槽位几何（视频坐标 800x360，实测校准）
    SLOT_X0 = 155      # 第一个槽位中心 x
    SLOT_DX = 53       # 槽位间距
    SLOT_Y = 328       # 槽位中心 y
    SLOT_HALF = 24     # 槽位检测半宽
    SLOT_TOP = 300     # 槽位检测上边界
    SLOT_BOT = 356     # 槽位检测下边界
    MAX_SLOTS = 11     # 兵种栏槽位数
    BADGE_X_OFF = -5   # 数量角标区域相对槽位中心的 x 偏移
    BADGE_Y = 296      # 角标区域 y（视频坐标）
    BADGE_W = 34       # 角标区域宽
    BADGE_H = 16       # 角标区域高
    
    # 有效部署区（避开顶部 HUD 与底部兵种栏）
    FIELD_TOP = 30
    FIELD_BOT = 288
    FIELD_LEFT = 25
    FIELD_RIGHT = 775
    
    def __init__(
        self,
        state_machine: StateMachine,
        template_matcher: TemplateMatcher,
        edge_detector: EdgeDetector,
        click_callback=None,
        frame_provider: Optional[Callable[[], Optional[np.ndarray]]] = None,
        ocr=None
    ):
        """
        初始化战斗部署器
        
        Args:
            state_machine: 状态机
            template_matcher: 模板匹配器
            edge_detector: 边缘检测器
            click_callback: 点击回调 (x, y) -> bool
            frame_provider: 实时帧获取回调（部署过程中刷新画面用）
            ocr: OCR 引擎（识别兵种卡数量角标用）
        """
        self.sm = state_machine
        self.matcher = template_matcher
        self.edge_detector = edge_detector
        self.click = click_callback or self._default_click
        self._get_frame = frame_provider
        self.ocr = ocr
        
        # 英雄配置
        self.hero_configs: List[HeroConfig] = []
        
        # 部署区域缓存
        self._deploy_zone: Optional[DeployZone] = None
        
        # 统计
        self._troops_deployed = 0
        self._abilities_activated = 0
        self._battles_count = 0
        self._battle_active = False  # 本场战斗是否已部署
        
        # 配置
        self.deploy_zone_padding = 20
        self.hero_ability_delay = (4.0, 6.0)
        
        logger.info("战斗部署器初始化完成")
    
    def add_hero(self, config: HeroConfig) -> None:
        """添加英雄配置"""
        self.hero_configs.append(config)
        logger.debug(f"添加英雄: {config.name}")
    
    def detect_deploy_zone(self, frame: np.ndarray) -> Optional[DeployZone]:
        """
        检测可部署区域
        
        Args:
            frame: 当前帧（敌方基地画面）
            
        Returns:
            DeployZone 或 None
        """
        result = self.edge_detector.detect_deploy_zone(frame)
        
        if result.success and result.deployable_zone:
            self._deploy_zone = result.deployable_zone
            logger.info(
                f"检测到部署区域: {self._deploy_zone.bbox} "
                f"({self._deploy_zone.width}x{self._deploy_zone.height})"
            )
            return self._deploy_zone
        
        logger.warning("未检测到可部署区域")
        return None
    
    def generate_deploy_points(
        self,
        zone: DeployZone,
        count: int,
        pattern: DeployPattern = DeployPattern.EDGE
    ) -> List[Tuple[int, int]]:
        """
        生成部署坐标点
        
        Args:
            zone: 可部署区域
            count: 部署数量
            pattern: 部署模式
            
        Returns:
            坐标点列表
        """
        if pattern == DeployPattern.EDGE:
            return self.edge_detector.generate_deploy_points(
                zone, count, random_offset=15
            )
        
        elif pattern == DeployPattern.CORNER:
            # 在四个角落部署
            corners = [
                (zone.x, zone.y),
                (zone.x + zone.width, zone.y),
                (zone.x, zone.y + zone.height),
                (zone.x + zone.width, zone.y + zone.height),
            ]
            points = []
            per_corner = count // 4
            for corner in corners:
                for _ in range(per_corner):
                    offset_x = random.randint(-20, 20)
                    offset_y = random.randint(-20, 20)
                    points.append((corner[0] + offset_x, corner[1] + offset_y))
            return points
        
        elif pattern == DeployPattern.SPREAD:
            # 在区域内随机分散
            points = []
            for _ in range(count):
                x = random.randint(zone.x, zone.x + zone.width)
                y = random.randint(zone.y, zone.y + zone.height)
                points.append((x, y))
            return points
        
        elif pattern == DeployPattern.FOCUSED:
            # 在中心区域集中
            center_x = zone.x + zone.width // 2
            center_y = zone.y + zone.height // 2
            points = []
            for _ in range(count):
                offset_x = random.randint(-50, 50)
                offset_y = random.randint(-50, 50)
                points.append((center_x + offset_x, center_y + offset_y))
            return points
        
        return []
    
    def _fresh_frame(self, fallback: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """获取实时帧（未配置帧提供器或获取失败时返回 fallback）"""
        if self._get_frame:
            try:
                frame = self._get_frame()
                if frame is not None:
                    return frame
            except Exception as e:
                logger.debug(f"获取实时帧失败: {e}")
        return fallback
    
    def _slot_stats(self, frame: np.ndarray, slot: int) -> Tuple[float, float, float]:
        """
        获取兵种栏槽位区域的绿色占比、亮度均值与饱和度均值
        
        Returns:
            (green_ratio, val_mean, sat_mean)
        """
        cx = self.SLOT_X0 + slot * self.SLOT_DX
        x0 = max(0, cx - self.SLOT_HALF)
        x1 = min(frame.shape[1], cx + self.SLOT_HALF)
        seg = frame[self.SLOT_TOP:self.SLOT_BOT, x0:x1]
        if seg.size == 0:
            return 1.0, 0.0, 0.0
        hsv = cv2.cvtColor(seg, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (25, 40, 40), (95, 255, 255))
        return (
            green.mean() / 255.0,
            float(hsv[:, :, 2].mean()),
            float(hsv[:, :, 1].mean()),
        )
    
    def _slot_has_card(self, frame: np.ndarray, slot: int) -> bool:
        """槽位是否有可部署的兵种/英雄卡（非空槽、非灰卡）"""
        ratio, _, sat = self._slot_stats(frame, slot)
        return ratio < 0.6 and sat >= 60
    
    def _scan_slots(self, frame: np.ndarray) -> List[int]:
        """扫描兵种栏，返回有可部署卡牌的槽位索引"""
        return [i for i in range(self.MAX_SLOTS) if self._slot_has_card(frame, i)]
    
    def deploy_troops(self, frame: np.ndarray) -> int:
        """
        动态部署兵种栏全部兵种
        
        从左到右逐个选中槽位，持续点击战场直到该槽位卡牌清空
        （兵种全部部署）或变灰（英雄已下场），无需按兵种配置模板。
        
        Args:
            frame: 当前帧
            
        Returns:
            部署点击总数
        """
        if self.sm.current_state != GameState.ENEMY_BASE:
            logger.warning(f"当前状态不是战斗画面: {self.sm.current_state.name}")
            return 0
        
        # 部署点池：在战场矩形区（避开顶部 HUD 与底部兵种栏）随机分布。
        # 不依赖边缘检测——实测红区检测易把底部兵种栏误判为可部署区。
        points = [
            (random.randint(self.FIELD_LEFT, self.FIELD_RIGHT),
             random.randint(self.FIELD_TOP, self.FIELD_BOT))
            for _ in range(240)
        ]
        
        f = self._fresh_frame(frame)
        slots = self._scan_slots(f)
        if not slots:
            logger.warning("兵种栏未检测到兵种卡")
            return 0
        logger.info(f"兵种栏检测到 {len(slots)} 个兵种槽位: {[i + 1 for i in slots]}")
        try:
            cv2.imwrite("/tmp/deploy_start.png", f)
            for i in range(self.MAX_SLOTS):
                r, v, s = self._slot_stats(f, i)
                logger.info(f"槽位{i+1} 绿={r:.2f} 亮={v:.0f} 饱和={s:.0f}")
        except Exception:
            pass
        
        total = 0
        for slot in slots:
            total += self._deploy_slot(slot, points, f)
        
        return total
    
    def _read_count(self, frame: np.ndarray, slot: int) -> Optional[int]:
        """读取槽位兵种卡上的数量角标（x30 中的 30），无角标/无 OCR 返回 None"""
        if self.ocr is None:
            return None
        cx = self.SLOT_X0 + slot * self.SLOT_DX
        x0 = max(0, cx + self.BADGE_X_OFF)
        return self.ocr.recognize_badge_count(
            frame, (x0, self.BADGE_Y, self.BADGE_W, self.BADGE_H)
        )
    
    def _deploy_slot(self, slot: int, points: List[Tuple[int, int]], frame: np.ndarray) -> int:
        """
        按角标数量部署：读取剩余数量 → 点击相应次数 → 复查剩余数量，
        循环直到角标消失（全部部署完成）；数量不再减少时跳出防止无效点击。
        """
        cx = self.SLOT_X0 + slot * self.SLOT_DX
        deployed = 0
        prev = None
        for _ in range(8):  # 最多 8 轮复查
            f = self._fresh_frame(frame)
            count = self._read_count(f, slot)
            
            if count is None:
                # 角标不可见：若槽位仍疑似有卡（可能只剩 1 个角标消失），补点一次
                if self._slot_has_card(f, slot):
                    self.click(cx, self.SLOT_Y)
                    time.sleep(0.15)
                    x, y = random.choice(points)
                    self.click(x, y)
                    deployed += 1
                break
            
            if count <= 0:
                break
            if prev is not None and count >= prev:
                logger.warning(f"槽位 {slot + 1} 数量未减少（{count}），停止部署")
                break
            prev = count
            
            # 选中卡牌，按剩余数量点击战场部署
            self.click(cx, self.SLOT_Y)
            time.sleep(0.2)
            idx = random.randrange(len(points))
            for i in range(count):
                x, y = points[(idx + i) % len(points)]
                self.click(x, y)
                deployed += 1
                time.sleep(random.uniform(0.12, 0.28))
        
        self._troops_deployed += deployed
        logger.info(f"槽位 {slot + 1} 部署 {deployed} 次")
        return deployed
    
    def deploy_heroes(self, frame: np.ndarray) -> int:
        """
        部署英雄
        
        Args:
            frame: 当前帧
            
        Returns:
            部署的英雄数量
        """
        if self.sm.current_state != GameState.ENEMY_BASE:
            return 0
        
        zone = self._deploy_zone
        if zone is None:
            h, w = frame.shape[:2]
            zone = DeployZone(
                x=w // 4, y=h // 4,
                width=w // 2, height=h // 2,
                is_deployable=True
            )
        
        deployed = 0
        
        for hero in self.hero_configs:
            hero_card = self.matcher.match(frame, hero.template, threshold=0.7)
            if not hero_card.found:
                continue
            
            # 部署英雄
            points = self.generate_deploy_points(zone, 1, DeployPattern.EDGE)
            if points:
                self.click(*hero_card.center)
                time.sleep(0.1)
                self.click(*points[0])
                deployed += 1
                logger.info(f"部署英雄: {hero.name}")
        
        return deployed
    
    def activate_abilities(self, frame: np.ndarray) -> int:
        """
        激活英雄技能
        
        Args:
            frame: 当前帧
            
        Returns:
            激活的技能数量
        """
        activated = 0
        
        for hero in self.hero_configs:
            # 查找技能按钮
            ability_btn = self.matcher.match(frame, hero.ability_template, threshold=0.7)
            if ability_btn.found:
                if self.click(*ability_btn.center):
                    activated += 1
                    self._abilities_activated += 1
                    logger.info(f"激活英雄技能: {hero.name}")
        
        return activated
    
    def run_battle(self, frame: np.ndarray) -> None:
        """
        执行完整战斗流程
        
        Args:
            frame: 当前帧
        """
        if self.sm.current_state != GameState.ENEMY_BASE:
            return
        
        if self._battle_active:
            return  # 本场已部署，等待战斗结束
        
        self._battle_active = True
        self._battles_count += 1
        logger.info(f"开始战斗 #{self._battles_count}")
        
        # 1. 部署英雄（配置模板时；未配置时英雄随兵种栏动态部署一并下场）
        if self.hero_configs:
            self.deploy_heroes(frame)
            time.sleep(0.5)
        
        # 2. 动态部署兵种栏全部兵种
        self.deploy_troops(frame)
        
        # 3. 等待一段时间后激活技能
        time.sleep(random.uniform(*self.hero_ability_delay))
        self.activate_abilities(self._fresh_frame(frame))
        
        logger.info(
            f"战斗部署完成: 兵种={self._troops_deployed}, "
            f"技能={self._abilities_activated}"
        )
    
    def finish_battle(self) -> None:
        """战斗结束（进入结算界面）时调用，重置部署标记"""
        self._battle_active = False
    
    def _default_click(self, x: int, y: int) -> bool:
        logger.info(f"[模拟点击] ({x}, {y})")
        return True
    
    @property
    def troops_deployed(self) -> int:
        return self._troops_deployed
    
    @property
    def abilities_activated(self) -> int:
        return self._abilities_activated
    
    @property
    def battles_count(self) -> int:
        return self._battles_count
    
    def __repr__(self) -> str:
        return (
            f"BattleDeployer(battles={self._battles_count}, "
            f"troops={self._troops_deployed}, abilities={self._abilities_activated})"
        )
