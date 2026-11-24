"""
Attack Searcher - 攻击搜索功能

搜索对手并选择目标。
流程：MainVillage → AttackMenu → FindMatch → 选择对手。
可结合 OCR 跳过资源过低的对手。
"""

import time
import random
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np

from ..states.state_machine import GameState, StateMachine
from ..vision.template_matcher import TemplateMatcher
from ..vision.ocr_engine import OCREngine

logger = logging.getLogger(__name__)


@dataclass
class EnemyBaseInfo:
    """敌方基地信息"""
    gold: Optional[int] = None
    elixir: Optional[int] = None
    dark_elixir: Optional[int] = None
    town_hall_level: Optional[int] = None
    percent_loot: Optional[int] = None  # 可掠夺资源百分比
    
    @property
    def total_resources(self) -> int:
        return (self.gold or 0) + (self.elixir or 0) + (self.dark_elixir or 0) * 10
    
    def __repr__(self) -> str:
        return (
            f"EnemyBase(gold={self.gold}, elixir={self.elixir}, "
            f"dark={self.dark_elixir}, total={self.total_resources})"
        )


class AttackSearcher:
    """
    攻击搜索器
    
    自动搜索对手，根据资源条件决定是否攻击。
    """
    
    def __init__(
        self,
        state_machine: StateMachine,
        template_matcher: TemplateMatcher,
        ocr_engine: OCREngine,
        click_callback=None
    ):
        """
        初始化攻击搜索器
        
        Args:
            state_machine: 状态机
            template_matcher: 模板匹配器
            ocr_engine: OCR 引擎
            click_callback: 点击回调 (x, y) -> bool
        """
        self.sm = state_machine
        self.matcher = template_matcher
        self.ocr = ocr_engine
        self.click = click_callback or self._default_click
        
        # 搜索策略
        self.min_gold = 0           # 最低金币要求
        self.min_elixir = 0         # 最低圣水要求
        self.min_dark = 0           # 最低暗黑重油要求
        self.max_skip_count = 5     # 最大跳过次数后强制攻击
        
        # 统计
        self._search_count = 0
        self._skip_count = 0
        self._attack_count = 0
        self._last_search_ts = 0.0  # 上次发起搜索的时间戳
        self._transition_until = 0.0  # 界面切换宽限期截止时间
        self._army_troops: list = []  # 自动配兵列表 [{name, template}]
        self._army_step = 0  # 配兵流程步骤（0=未进行）
        
        # 配置
        self.action_delay = (0.5, 1.2)
        self.search_timeout = 30.0  # 搜索超时（秒）
        
        logger.info("攻击搜索器初始化完成")
    
    def set_strategy(
        self,
        min_gold: int = 0,
        min_elixir: int = 0,
        min_dark: int = 0,
        max_skip: int = 5
    ) -> None:
        """
        设置搜索策略
        
        Args:
            min_gold: 最低金币要求
            min_elixir: 最低圣水要求
            min_dark: 最低暗黑重油要求
            max_skip: 最大跳过次数
        """
        self.min_gold = min_gold
        self.min_elixir = min_elixir
        self.min_dark = min_dark
        self.max_skip_count = max_skip
        logger.info(
            f"搜索策略: gold>={min_gold}, elixir>={min_elixir}, "
            f"dark>={min_dark}, max_skip={max_skip}"
        )
    
    def start_search(self, frame: np.ndarray) -> bool:
        """
        开始搜索对手
        
        每帧只执行一步：
        - 已在攻击页面（找到"寻找对手"按钮）→ 点击开始搜索
        - 在主村庄 → 点击攻击按钮进入攻击页面
        
        Args:
            frame: 当前帧
            
        Returns:
            已发起对手搜索返回 True
        """
        # 攻击页面：点击"寻找对手"
        find_btn = self.matcher.match(frame, "find_match_button", threshold=0.7)
        if find_btn.found:
            # 搜索进行中/界面切换中，冷却期内不重复点击
            if time.time() - self._last_search_ts < 90:
                return False
            self._army_step = 0  # 新一轮进攻，重置上一轮配兵残留状态
            if self.click(*find_btn.center):
                logger.info("开始搜索对手")
                self._search_count += 1
                self._last_search_ts = time.time()
                self._transition_until = time.time() + 60  # 匹配+加载宽限期
                time.sleep(3.0)  # 等待匹配界面出现
                return True
            return False
        
        # 主村庄：点击攻击按钮进入攻击页面
        attack_btn = self.matcher.match(frame, "attack_button", threshold=0.8)
        if not attack_btn.found:
            logger.warning("未找到攻击按钮")
            return False
        
        if self.click(*attack_btn.center):
            logger.info("点击攻击按钮，进入攻击页面")
            time.sleep(2.0)  # 等待菜单打开动画结束
            return False
        
        return False
    
    def handle_army_page(self, frame: np.ndarray) -> bool:
        """
        处理"我的军队"页面（每帧一步）
        
        配置了 army.troops 时执行自动配兵：
        清空部队 → 打开兵种选择 → 逐个点击兵种卡片 → 关闭弹窗 → 进攻
        未配置时沿用游戏记忆直接点击"进攻!"。
        
        Args:
            frame: 当前帧
            
        Returns:
            已点击进攻返回 True
        """
        # 未配置配兵：沿用游戏记忆配好的兵直接进攻
        if not self._army_troops:
            if self.in_transition():
                return False
            return self._click_attack(frame)
        
        if self._army_step == 0:
            logger.info("自动配兵 [1/4]: 清空现有部队")
            self.click(705, 83)  # 军队营垃圾桶
            self._army_step = 1
            time.sleep(1.2)
            return False
        
        if self._army_step == 1:
            ok = self.matcher.match(frame, "army_confirm_ok", threshold=0.8)
            if ok.found:
                self.click(*ok.center)
                logger.info("自动配兵 [2/4]: 已确认移除，打开兵种选择")
                time.sleep(1.2)
            else:
                logger.info("自动配兵 [2/4]: 部队已空，打开兵种选择")
            self.click(500, 122)  # 空白军队栏 → 兵种选择弹窗
            self._army_step = 2
            time.sleep(1.5)
            return False
        
        if self._army_step == 2:
            added = []
            for troop in self._army_troops:
                r = self.matcher.match(frame, troop["template"], threshold=0.7)
                if r.found:
                    self.click(*r.center)
                    added.append(troop["name"])
                    time.sleep(0.4)
            logger.info(f"自动配兵 [3/4]: 已添加 {added or '无（兵种未训练）'}")
            self.click(400, 75)  # 点击上方区域关闭弹窗
            self._army_step = 3
            time.sleep(1.2)
            return False
        
        # step 3: 弹窗已关，点击进攻（成功后进入 step 4 等待界面切换）
        if self._army_step == 3:
            if self.in_transition():
                return False
            atk_btn = self.matcher.match(frame, "army_attack_button", threshold=0.7)
            if not atk_btn.found:
                # 兵种弹窗可能未关闭遮住进攻按钮，再点上方区域关闭
                self.click(400, 75)
                time.sleep(1.0)
                return False
            if self.click(*atk_btn.center):
                logger.info("点击进攻，进入敌方基地")
                self._attack_count += 1
                self._transition_until = time.time() + 15
                self._army_step = 4
                time.sleep(5.0)
                return True
            return False
        
        # step 4: 已点进攻，等待界面切换出军队页面
        if self._army_step == 4:
            if self.in_transition():
                return False
            atk_btn = self.matcher.match(frame, "army_attack_button", threshold=0.8)
            if atk_btn.found:
                # 宽限期后仍在军队页：进攻未生效（部队太少），关闭页面回村
                logger.warning("进攻未生效（可能部队过少），关闭军队页面返回村庄")
                self.click(705, 27)  # 军队页面右上角 X
                time.sleep(1.5)
            self._army_step = 0
            return False
        
        return False
    
    def _click_attack(self, frame: np.ndarray) -> bool:
        """点击"进攻!"进入敌方基地"""
        atk_btn = self.matcher.match(frame, "army_attack_button", threshold=0.7)
        if not atk_btn.found:
            return False
        
        if self.click(*atk_btn.center):
            logger.info("点击进攻，进入敌方基地")
            self._attack_count += 1
            self._transition_until = time.time() + 15  # 加载宽限期
            time.sleep(5.0)
            return True
        return False
    
    def in_transition(self) -> bool:
        """攻击流程界面切换宽限期内（加载中），应跳过异常恢复"""
        return time.time() < self._transition_until
    
    def set_army_config(self, troops: list) -> None:
        """设置自动配兵列表，每项需含 name 和 template"""
        self._army_troops = [t for t in troops if t.get("template")]
        if self._army_troops:
            names = [t.get("name", "?") for t in self._army_troops]
            logger.info(f"自动配兵已启用: {names}")
    
    def in_army_config(self) -> bool:
        """是否处于自动配兵流程中"""
        return self._army_step > 0
    
    def return_to_village(self, frame: np.ndarray) -> bool:
        """
        若当前处于攻击页面，单击屏幕右侧村庄区域返回主村庄
        
        Args:
            frame: 当前帧
            
        Returns:
            执行了返回操作返回 True
        """
        find_btn = self.matcher.match(frame, "find_match_button", threshold=0.7)
        if not find_btn.found:
            return False
        
        h, w = frame.shape[:2]
        if self.click(int(w * 0.875), int(h * 0.5)):
            logger.info("单击屏幕右侧，从攻击页面返回村庄")
            time.sleep(random.uniform(*self.action_delay))
            return True
        return False
    
    def analyze_enemy(self, frame: np.ndarray) -> EnemyBaseInfo:
        """
        分析敌方基地资源
        
        Args:
            frame: 当前帧（敌方基地画面）
            
        Returns:
            EnemyBaseInfo 敌方基地信息
        """
        info = EnemyBaseInfo()
        
        # OCR 读取资源数值
        # 金币区域（需要根据实际分辨率调整 ROI）
        gold_result = self.ocr.recognize_digits(
            frame, roi=(50, 10, 200, 40), allow_suffix=True
        )
        if gold_result.success:
            info.gold = gold_result.value
        
        # 圣水区域
        elixir_result = self.ocr.recognize_digits(
            frame, roi=(50, 50, 200, 40), allow_suffix=True
        )
        if elixir_result.success:
            info.elixir = elixir_result
        
        # 暗黑重油区域
        dark_result = self.ocr.recognize_digits(
            frame, roi=(50, 90, 200, 40), allow_suffix=True
        )
        if dark_result.success:
            info.dark_elixir = dark_result
        
        logger.debug(f"敌方基地分析: {info}")
        return info
    
    def should_attack(self, info: EnemyBaseInfo) -> bool:
        """
        判断是否应该攻击该对手
        
        Args:
            info: 敌方基地信息
            
        Returns:
            应该攻击返回 True
        """
        # 检查资源是否满足最低要求
        if self.min_gold > 0 and (info.gold or 0) < self.min_gold:
            return False
        
        if self.min_elixir > 0 and (info.elixir or 0) < self.min_elixir:
            return False
        
        if self.min_dark > 0 and (info.dark_elixir or 0) < self.min_dark:
            return False
        
        return True
    
    def skip_enemy(self, frame: np.ndarray) -> bool:
        """
        跳过当前对手，继续搜索
        
        Args:
            frame: 当前帧
            
        Returns:
            成功跳过返回 True
        """
        # 查找跳过按钮
        skip_btn = self.matcher.match(frame, "skip_button", threshold=0.7)
        if skip_btn.found:
            if self.click(*skip_btn.center):
                self._skip_count += 1
                logger.info(f"跳过对手 (累计跳过: {self._skip_count})")
                time.sleep(random.uniform(*self.action_delay))
                return True
        
        return False
    
    def run_search_cycle(self, frame: np.ndarray) -> Tuple[bool, Optional[EnemyBaseInfo]]:
        """
        执行一次搜索循环
        
        搜索对手 → 分析资源 → 决定是否攻击
        
        Args:
            frame: 当前帧
            
        Returns:
            (是否找到目标, 敌方基地信息)
        """
        # 分析敌方基地
        info = self.analyze_enemy(frame)
        
        # 判断是否攻击
        if self.should_attack(info):
            logger.info(f"找到满意目标: {info}")
            self._attack_count += 1
            self._skip_count = 0  # 重置跳过计数
            return (True, info)
        
        # 检查是否超过最大跳过次数
        if self._skip_count >= self.max_skip_count:
            logger.info(f"已达最大跳过次数 ({self.max_skip_count})，强制攻击")
            self._attack_count += 1
            self._skip_count = 0
            return (True, info)
        
        # 跳过当前对手
        self.skip_enemy(frame)
        return (False, info)
    
    def _default_click(self, x: int, y: int) -> bool:
        logger.info(f"[模拟点击] ({x}, {y})")
        return True
    
    @property
    def search_count(self) -> int:
        return self._search_count
    
    @property
    def attack_count(self) -> int:
        return self._attack_count
    
    @property
    def skip_count(self) -> int:
        return self._skip_count
    
    def __repr__(self) -> str:
        return (
            f"AttackSearcher(searches={self._search_count}, "
            f"attacks={self._attack_count}, skips={self._skip_count})"
        )
