"""
Battle Deployer - 战斗部署功能

在敌方基地部署兵种。
流程：EnemyBase → 部署兵种 → 战斗结束 → Results。
使用红区边缘检测确定可部署位置，英雄技能延迟激活。
"""

import time
import random
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum, auto

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
class DeployConfig:
    """部署配置"""
    troop_name: str
    troop_template: str
    count: int = 10
    pattern: DeployPattern = DeployPattern.EDGE
    delay_range: Tuple[float, float] = (0.1, 0.3)  # 部署间隔
    random_offset: int = 15  # 随机偏移像素


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
    """
    
    def __init__(
        self,
        state_machine: StateMachine,
        template_matcher: TemplateMatcher,
        edge_detector: EdgeDetector,
        click_callback=None
    ):
        """
        初始化战斗部署器
        
        Args:
            state_machine: 状态机
            template_matcher: 模板匹配器
            edge_detector: 边缘检测器
            click_callback: 点击回调 (x, y) -> bool
        """
        self.sm = state_machine
        self.matcher = template_matcher
        self.edge_detector = edge_detector
        self.click = click_callback or self._default_click
        
        # 部署配置
        self.deploy_configs: List[DeployConfig] = []
        self.hero_configs: List[HeroConfig] = []
        
        # 部署区域缓存
        self._deploy_zone: Optional[DeployZone] = None
        
        # 统计
        self._troops_deployed = 0
        self._abilities_activated = 0
        self._battles_count = 0
        
        # 配置
        self.deploy_zone_padding = 20
        self.hero_ability_delay = (4.0, 6.0)
        
        logger.info("战斗部署器初始化完成")
    
    def add_troop(self, config: DeployConfig) -> None:
        """添加兵种部署配置"""
        self.deploy_configs.append(config)
        logger.debug(f"添加兵种: {config.troop_name} x{config.count}")
    
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
    
    def deploy_troops(self, frame: np.ndarray) -> int:
        """
        部署所有配置的兵种
        
        Args:
            frame: 当前帧
            
        Returns:
            部署的兵种总数
        """
        if self.sm.current_state != GameState.ENEMY_BASE:
            logger.warning(f"当前状态不是战斗画面: {self.sm.current_state.name}")
            return 0
        
        # 检测部署区域
        zone = self.detect_deploy_zone(frame)
        if zone is None:
            # 使用默认区域
            h, w = frame.shape[:2]
            zone = DeployZone(
                x=self.deploy_zone_padding,
                y=self.deploy_zone_padding,
                width=w - 2 * self.deploy_zone_padding,
                height=h - 2 * self.deploy_zone_padding,
                is_deployable=True
            )
        
        total_deployed = 0
        
        for config in self.deploy_configs:
            # 生成部署点
            points = self.generate_deploy_points(zone, config.count, config.pattern)
            
            # 查找兵种卡牌
            troop_card = self.matcher.match(frame, config.troop_template, threshold=0.7)
            if not troop_card.found:
                logger.warning(f"未找到兵种卡牌: {config.troop_name}")
                continue
            
            # 依次点击部署
            for point in points:
                # 先点击兵种卡牌选中
                card_pos = troop_card.center
                self.click(card_pos[0], card_pos[1])
                time.sleep(random.uniform(0.05, 0.15))
                
                # 再点击部署位置
                if self.click(point[0], point[1]):
                    total_deployed += 1
                    self._troops_deployed += 1
                
                # 部署间隔
                time.sleep(random.uniform(*config.delay_range))
            
            logger.info(f"部署 {config.troop_name}: {len(points)} 个")
        
        return total_deployed
    
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
        
        self._battles_count += 1
        logger.info(f"开始战斗 #{self._battles_count}")
        
        # 1. 部署英雄
        self.deploy_heroes(frame)
        time.sleep(0.5)
        
        # 2. 部署兵种
        self.deploy_troops(frame)
        
        # 3. 等待一段时间后激活技能
        time.sleep(random.uniform(*self.hero_ability_delay))
        self.activate_abilities(frame)
        
        logger.info(
            f"战斗部署完成: 兵种={self._troops_deployed}, "
            f"技能={self._abilities_activated}"
        )
    
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
