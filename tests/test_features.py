"""
Test Features - 业务功能层模块测试

测试内容：
1. ResourceCollector 资源收集
2. BuildingUpgrader 建筑升级
3. TroopDonator 捐兵功能
4. AttackSearcher 攻击搜索
5. BattleDeployer 战斗部署
"""

import sys
import time
import logging
import unittest
from unittest.mock import Mock, MagicMock, patch

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(__file__).replace("\\tests\\test_features.py", ""))

from bot.states.state_machine import GameState, StateMachine
from bot.vision.template_matcher import TemplateMatcher, MatchResult
from bot.vision.ocr_engine import OCREngine, OCRResult
from bot.vision.edge_detector import EdgeDetector, DeployZone

from bot.features.collector import ResourceCollector, ResourceBuilding
from bot.features.upgrader import BuildingUpgrader, UpgradeTarget, UpgradePriority
from bot.features.donator import TroopDonator, DonationRequest
from bot.features.attacker import AttackSearcher, EnemyBaseInfo
from bot.features.deployer import BattleDeployer, DeployPattern, HeroConfig


class TestResourceCollector(unittest.TestCase):
    """测试资源收集器"""
    
    def setUp(self):
        self.sm = StateMachine(GameState.MAIN_VILLAGE)
        self.matcher = Mock(spec=TemplateMatcher)
        self.click_log = []
        
        def mock_click(x, y):
            self.click_log.append((x, y))
            return True
        
        self.collector = ResourceCollector(
            self.sm, self.matcher, click_callback=mock_click
        )
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.collector.sm, self.sm)
        self.assertEqual(self.collector.collect_count, 0)
    
    def test_add_building(self):
        """测试添加建筑"""
        building = ResourceBuilding("金矿", "gold_mine", "gold_bubble")
        self.collector.add_building(building)
        self.assertEqual(len(self.collector.buildings), 1)
    
    def test_setup_default_buildings(self):
        """测试设置默认建筑"""
        self.collector.setup_default_buildings()
        self.assertEqual(len(self.collector.buildings), 3)
    
    def test_collect_wrong_state(self):
        """测试错误状态下收集"""
        self.sm.force_transition(GameState.ATTACK_MENU)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        count = self.collector.collect(frame)
        self.assertEqual(count, 0)
    
    def test_collect_success(self):
        """测试成功收集"""
        self.collector.setup_default_buildings()
        self.collector.collect_interval = 0  # 禁用间隔
        
        # 模拟模板匹配返回结果
        self.matcher.match.return_value = MatchResult(
            found=True, x=100, y=100, width=50, height=50,
            confidence=0.9, template_name="gold_bubble"
        )
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        count = self.collector.collect(frame)
        
        self.assertGreater(count, 0)
        self.assertGreater(len(self.click_log), 0)
    
    def test_scan_buildings(self):
        """测试扫描建筑"""
        self.collector.setup_default_buildings()
        
        self.matcher.match.return_value = MatchResult(
            found=True, x=200, y=150, width=60, height=60,
            confidence=0.85, template_name="gold_mine"
        )
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        found = self.collector.scan_buildings(frame)
        
        self.assertGreater(len(found), 0)
        self.assertGreater(len(self.collector.known_buildings), 0)


class TestBuildingUpgrader(unittest.TestCase):
    """测试建筑升级器"""
    
    def setUp(self):
        self.sm = StateMachine(GameState.MAIN_VILLAGE)
        self.matcher = Mock(spec=TemplateMatcher)
        self.ocr = Mock(spec=OCREngine)
        self.click_log = []
        
        def mock_click(x, y):
            self.click_log.append((x, y))
            return True
        
        self.upgrader = BuildingUpgrader(
            self.sm, self.matcher, self.ocr, click_callback=mock_click
        )
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.upgrader.upgrade_count, 0)
    
    def test_add_target(self):
        """测试添加升级目标"""
        target = UpgradeTarget(
            name="金库",
            building_template="gold_storage",
            priority=UpgradePriority.HIGH
        )
        self.upgrader.add_target(target)
        self.assertEqual(len(self.upgrader.targets), 1)
    
    def test_get_builder_status(self):
        """测试获取工人状态"""
        self.matcher.match.return_value = MatchResult(
            found=True, x=50, y=50, width=40, height=40,
            confidence=0.9, template_name="builder_icon"
        )
        self.ocr.recognize_digits.return_value = OCRResult(
            success=True, text="5", value=5, confidence=0.95
        )
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        status = self.upgrader.get_builder_status(frame)
        
        self.assertIsNotNone(status)
        self.assertEqual(status.total, 5)
    
    def test_try_upgrade_wrong_state(self):
        """测试错误状态下升级"""
        self.sm.force_transition(GameState.ENEMY_BASE)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.upgrader.try_upgrade(frame)
        self.assertFalse(result)
    
    def test_try_upgrade_success(self):
        """测试成功升级"""
        target = UpgradeTarget(
            name="金库",
            building_template="gold_storage",
            upgrade_button_template="upgrade_btn"
        )
        self.upgrader.add_target(target)
        
        # 模拟匹配结果
        def mock_match(frame, template, threshold=0.7, roi=None):
            if template == "gold_storage":
                return MatchResult(
                    found=True, x=100, y=100, width=80, height=80,
                    confidence=0.9, template_name=template
                )
            elif template == "upgrade_btn":
                return MatchResult(
                    found=True, x=120, y=120, width=40, height=40,
                    confidence=0.85, template_name=template
                )
            return MatchResult(found=False)
        
        self.matcher.match.side_effect = mock_match
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.upgrader.try_upgrade(frame)
        
        self.assertTrue(result)
        self.assertEqual(self.upgrader.upgrade_count, 1)


class TestTroopDonator(unittest.TestCase):
    """测试捐兵器"""
    
    def setUp(self):
        self.sm = StateMachine(GameState.CLAN_CHAT)
        self.matcher = Mock(spec=TemplateMatcher)
        self.ocr = Mock(spec=OCREngine)
        self.click_log = []
        
        def mock_click(x, y):
            self.click_log.append((x, y))
            return True
        
        self.donator = TroopDonator(
            self.sm, self.matcher, self.ocr, click_callback=mock_click
        )
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.donator.donate_count, 0)
        self.assertEqual(self.donator.skip_count, 0)
    
    def test_detect_requests(self):
        """测试检测捐兵请求"""
        # 模拟检测到请求图标
        self.matcher.match_all.return_value = [
            MatchResult(
                found=True, x=200, y=150, width=50, height=50,
                confidence=0.9, template_name="donation_request_icon"
            )
        ]
        
        # 模拟匹配到兵种
        self.matcher.match.return_value = MatchResult(
            found=True, x=180, y=140, width=30, height=30,
            confidence=0.8, template_name="troop_barbarian"
        )
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        requests = self.donator.detect_donation_requests(frame)
        
        self.assertGreater(len(requests), 0)
        self.assertEqual(requests[0].troop_name, "野蛮人")
    
    def test_donate_success(self):
        """测试成功捐兵"""
        request = DonationRequest(
            troop_name="弓箭手",
            troop_template="troop_archer",
            count=5,
            position=(200, 150)
        )
        
        self.matcher.match.return_value = MatchResult(
            found=True, x=300, y=400, width=60, height=60,
            confidence=0.9, template_name="troop_archer"
        )
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.donator.donate(frame, request)
        
        self.assertTrue(result)
        self.assertEqual(self.donator.donate_count, 1)
        self.assertGreater(len(self.click_log), 0)


class TestAttackSearcher(unittest.TestCase):
    """测试攻击搜索器"""
    
    def setUp(self):
        self.sm = StateMachine(GameState.MAIN_VILLAGE)
        self.matcher = Mock(spec=TemplateMatcher)
        self.ocr = Mock(spec=OCREngine)
        self.click_log = []
        
        def mock_click(x, y):
            self.click_log.append((x, y))
            return True
        
        self.searcher = AttackSearcher(
            self.sm, self.matcher, self.ocr, click_callback=mock_click
        )
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.searcher.search_count, 0)
        self.assertEqual(self.searcher.attack_count, 0)
    
    def test_set_strategy(self):
        """测试设置策略"""
        self.searcher.set_strategy(
            min_gold=50000,
            min_elixir=50000,
            min_dark=1000,
            max_skip=3
        )
        
        self.assertEqual(self.searcher.min_gold, 50000)
        self.assertEqual(self.searcher.min_elixir, 50000)
        self.assertEqual(self.searcher.min_dark, 1000)
        self.assertEqual(self.searcher.max_skip_count, 3)
    
    def test_analyze_enemy(self):
        """测试分析敌方基地"""
        # 模拟 OCR 结果
        def mock_ocr(frame, roi=None, allow_suffix=False):
            if roi == (50, 10, 200, 40):
                return OCRResult(success=True, text="100K", value=100000, confidence=0.9)
            elif roi == (50, 50, 200, 40):
                return OCRResult(success=True, text="200K", value=200000, confidence=0.9)
            elif roi == (50, 90, 200, 40):
                return OCRResult(success=True, text="2K", value=2000, confidence=0.9)
            return OCRResult(success=False)
        
        self.ocr.recognize_digits.side_effect = mock_ocr
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        info = self.searcher.analyze_enemy(frame)
        
        self.assertEqual(info.gold, 100000)
        self.assertEqual(info.elixir, 200000)
        self.assertEqual(info.dark_elixir, 2000)
    
    def test_should_attack(self):
        """测试攻击判断"""
        self.searcher.set_strategy(min_gold=50000, min_elixir=50000)
        
        # 满足条件
        info1 = EnemyBaseInfo(gold=100000, elixir=100000)
        self.assertTrue(self.searcher.should_attack(info1))
        
        # 不满足条件
        info2 = EnemyBaseInfo(gold=30000, elixir=30000)
        self.assertFalse(self.searcher.should_attack(info2))
    
    def test_skip_enemy(self):
        """测试跳过对手"""
        self.matcher.match.return_value = MatchResult(
            found=True, x=500, y=400, width=80, height=40,
            confidence=0.9, template_name="skip_button"
        )
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.searcher.skip_enemy(frame)
        
        self.assertTrue(result)
        self.assertEqual(self.searcher.skip_count, 1)
    
    def test_run_search_cycle_attack(self):
        """测试搜索循环-攻击"""
        self.searcher.set_strategy(min_gold=50000)
        
        # 模拟分析结果
        with patch.object(self.searcher, 'analyze_enemy') as mock_analyze:
            mock_analyze.return_value = EnemyBaseInfo(gold=100000, elixir=100000)
            
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            found, info = self.searcher.run_search_cycle(frame)
            
            self.assertTrue(found)
            self.assertEqual(self.searcher.attack_count, 1)
    
    def test_run_search_cycle_skip(self):
        """测试搜索循环-跳过"""
        self.searcher.set_strategy(min_gold=500000)
        
        with patch.object(self.searcher, 'analyze_enemy') as mock_analyze:
            mock_analyze.return_value = EnemyBaseInfo(gold=100000, elixir=100000)
            
            with patch.object(self.searcher, 'skip_enemy', return_value=True):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                found, info = self.searcher.run_search_cycle(frame)
                
                self.assertFalse(found)
                self.assertEqual(self.searcher.skip_count, 1)


class TestBattleDeployer(unittest.TestCase):
    """测试战斗部署器"""
    
    def setUp(self):
        self.sm = StateMachine(GameState.ENEMY_BASE)
        self.matcher = Mock(spec=TemplateMatcher)
        self.edge_detector = Mock(spec=EdgeDetector)
        self.click_log = []
        
        def mock_click(x, y):
            self.click_log.append((x, y))
            return True
        
        self.deployer = BattleDeployer(
            self.sm, self.matcher, self.edge_detector, click_callback=mock_click
        )
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.deployer.troops_deployed, 0)
        self.assertEqual(self.deployer.battles_count, 0)
    
    def _make_battle_frame(self, filled_slots=(0,)):
        """构造 800x360 战斗画面：绿色背景 + 指定槽位放蓝色兵种卡"""
        frame = np.zeros((360, 800, 3), dtype=np.uint8)
        frame[:, :] = (40, 120, 40)  # 草地绿
        for i in filled_slots:
            cx = BattleDeployer.SLOT_X0 + i * BattleDeployer.SLOT_DX
            frame[BattleDeployer.SLOT_TOP:BattleDeployer.SLOT_BOT,
                  cx - BattleDeployer.SLOT_HALF:cx + BattleDeployer.SLOT_HALF] = (200, 120, 60)
        return frame
    
    def test_scan_slots(self):
        """测试兵种栏槽位检测"""
        frame = self._make_battle_frame(filled_slots=(0, 2))
        slots = self.deployer._scan_slots(frame)
        self.assertEqual(slots, [0, 2])
    
    def test_add_hero(self):
        """测试添加英雄"""
        config = HeroConfig(
            name="野蛮人之王",
            template="hero_barbarian_king"
        )
        self.deployer.add_hero(config)
        self.assertEqual(len(self.deployer.hero_configs), 1)
    
    def test_detect_deploy_zone(self):
        """测试检测部署区域"""
        from bot.vision.edge_detector import EdgeResult
        
        zone = DeployZone(x=50, y=50, width=540, height=380, is_deployable=True)
        self.edge_detector.detect_deploy_zone.return_value = EdgeResult(
            success=True,
            deployable_zone=zone
        )
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.deployer.detect_deploy_zone(frame)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.width, 540)
        self.assertEqual(result.height, 380)
    
    def test_generate_deploy_points_edge(self):
        """测试边缘部署点生成"""
        zone = DeployZone(x=100, y=100, width=400, height=300, is_deployable=True)
        
        # 模拟边缘检测器返回
        self.edge_detector.generate_deploy_points.return_value = [
            (100, 100), (500, 100), (100, 400), (500, 400)
        ]
        
        points = self.deployer.generate_deploy_points(zone, 4, DeployPattern.EDGE)
        
        self.assertEqual(len(points), 4)
    
    def test_generate_deploy_points_spread(self):
        """测试分散部署点生成"""
        zone = DeployZone(x=100, y=100, width=400, height=300, is_deployable=True)
        
        points = self.deployer.generate_deploy_points(zone, 10, DeployPattern.SPREAD)
        
        self.assertEqual(len(points), 10)
        
        # 验证所有点在区域内
        for x, y in points:
            self.assertGreaterEqual(x, 100)
            self.assertLessEqual(x, 500)
            self.assertGreaterEqual(y, 100)
            self.assertLessEqual(y, 400)
    
    def test_deploy_troops(self):
        """测试动态部署全部兵种"""
        from bot.vision.edge_detector import EdgeResult
        
        zone = DeployZone(x=100, y=100, width=400, height=200, is_deployable=True)
        self.edge_detector.detect_deploy_zone.return_value = EdgeResult(
            success=True, deployable_zone=zone
        )
        self.edge_detector.generate_deploy_points.return_value = [
            (150, 150), (400, 120), (600, 200), (250, 250)
        ]
        
        frame = self._make_battle_frame(filled_slots=(0,))
        empty = self._make_battle_frame(filled_slots=())
        
        # 首次扫描返回有卡画面，之后返回已清空画面
        state = {"first": True}
        def provider():
            if state["first"]:
                state["first"] = False
                return frame
            return empty
        
        # 角标：第一轮剩 3，第二轮无角标（已部署完）
        ocr = Mock()
        ocr.recognize_badge_count.side_effect = [3, None]
        
        deployer = BattleDeployer(
            self.sm, self.matcher, self.edge_detector,
            click_callback=lambda x, y: True,
            frame_provider=provider, ocr=ocr
        )
        count = deployer.deploy_troops(frame)
        
        # 按角标数量点击 3 次后复查无角标，停止部署
        self.assertEqual(count, 3)
        self.assertEqual(deployer.troops_deployed, 3)
    
    def test_run_battle(self):
        """测试完整战斗流程"""
        from bot.vision.edge_detector import EdgeResult
        
        zone = DeployZone(x=100, y=100, width=400, height=200, is_deployable=True)
        self.edge_detector.detect_deploy_zone.return_value = EdgeResult(
            success=True, deployable_zone=zone
        )
        self.edge_detector.generate_deploy_points.return_value = [
            (150, 150), (400, 120), (600, 200), (250, 250)
        ]
        
        self.matcher.match.return_value = MatchResult(
            found=False, x=0, y=0, width=0, height=0,
            confidence=0.0, template_name="none"
        )
        
        frame = self._make_battle_frame(filled_slots=(0, 1))
        empty = self._make_battle_frame(filled_slots=())
        
        state = {"first": True}
        def provider():
            if state["first"]:
                state["first"] = False
                return frame
            return empty
        
        ocr = Mock()
        ocr.recognize_badge_count.side_effect = [2, None, 2, None]
        
        deployer = BattleDeployer(
            self.sm, self.matcher, self.edge_detector,
            click_callback=lambda x, y: True,
            frame_provider=provider, ocr=ocr
        )
        deployer.hero_ability_delay = (0.01, 0.02)
        deployer.run_battle(frame)
        
        self.assertEqual(deployer.battles_count, 1)
        self.assertEqual(deployer.troops_deployed, 4)


def run_basic_tests():
    """运行基本测试"""
    print("\n" + "=" * 60)
    print("运行业务功能层模块基本测试")
    print("=" * 60)
    
    # 1. 测试资源收集器
    print("\n[1/5] 测试 ResourceCollector 资源收集器...")
    sm = StateMachine(GameState.MAIN_VILLAGE)
    matcher = Mock(spec=TemplateMatcher)
    click_log = []
    collector = ResourceCollector(sm, matcher, click_callback=lambda x, y: click_log.append((x, y)) or True)
    
    collector.setup_default_buildings()
    assert len(collector.buildings) == 3
    print("  ✓ 默认建筑设置正常")
    
    matcher.match.return_value = MatchResult(
        found=True, x=100, y=100, width=50, height=50,
        confidence=0.9, template_name="bubble"
    )
    collector.collect_interval = 0
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    count = collector.collect(frame)
    assert count > 0
    assert len(click_log) > 0
    print(f"  ✓ 资源收集成功: {count} 个建筑")
    
    # 2. 测试建筑升级器
    print("\n[2/5] 测试 BuildingUpgrader 建筑升级器...")
    ocr = Mock(spec=OCREngine)
    upgrader = BuildingUpgrader(sm, matcher, ocr, click_callback=lambda x, y: True)
    
    target = UpgradeTarget(
        name="金库",
        building_template="gold_storage",
        priority=UpgradePriority.HIGH
    )
    upgrader.add_target(target)
    assert len(upgrader.targets) == 1
    print("  ✓ 升级目标添加正常")
    
    # 3. 测试捐兵器
    print("\n[3/5] 测试 TroopDonator 捐兵器...")
    sm2 = StateMachine(GameState.CLAN_CHAT)
    donator = TroopDonator(sm2, matcher, ocr, click_callback=lambda x, y: True)
    
    request = DonationRequest(
        troop_name="野蛮人",
        troop_template="troop_barbarian",
        count=5,
        position=(200, 150)
    )
    matcher.match.return_value = MatchResult(
        found=True, x=300, y=400, width=60, height=60,
        confidence=0.9, template_name="troop_barbarian"
    )
    result = donator.donate(frame, request)
    assert result
    assert donator.donate_count == 1
    print(f"  ✓ 捐兵成功: {donator.donate_count} 次")
    
    # 4. 测试攻击搜索器
    print("\n[4/5] 测试 AttackSearcher 攻击搜索器...")
    searcher = AttackSearcher(sm, matcher, ocr, click_callback=lambda x, y: True)
    searcher.set_strategy(min_gold=50000, min_elixir=50000)
    
    info = EnemyBaseInfo(gold=100000, elixir=100000, dark_elixir=1000)
    assert searcher.should_attack(info)
    
    info2 = EnemyBaseInfo(gold=30000, elixir=30000)
    assert not searcher.should_attack(info2)
    print("  ✓ 攻击判断逻辑正常")
    
    # 5. 测试战斗部署器
    print("\n[5/5] 测试 BattleDeployer 战斗部署器...")
    sm3 = StateMachine(GameState.ENEMY_BASE)
    edge_detector = Mock(spec=EdgeDetector)
    deployer = BattleDeployer(sm3, matcher, edge_detector, click_callback=lambda x, y: True)
    
    battle_frame = np.zeros((360, 800, 3), dtype=np.uint8)
    battle_frame[:, :] = (40, 120, 40)
    cx = BattleDeployer.SLOT_X0
    battle_frame[BattleDeployer.SLOT_TOP:BattleDeployer.SLOT_BOT,
                 cx - BattleDeployer.SLOT_HALF:cx + BattleDeployer.SLOT_HALF] = (200, 120, 60)
    assert deployer._scan_slots(battle_frame) == [0]
    print("  ✓ 兵种栏槽位检测正常")
    
    zone = DeployZone(x=100, y=100, width=400, height=300, is_deployable=True)
    points = deployer.generate_deploy_points(zone, 10, DeployPattern.SPREAD)
    assert len(points) == 10
    print(f"  ✓ 部署点生成正常: {len(points)} 个点")
    
    print("\n" + "=" * 60)
    print("所有基本测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    run_basic_tests()
