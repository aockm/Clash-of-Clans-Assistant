"""
Test States - 状态机与决策层模块测试

测试内容：
1. StateMachine 状态机核心
2. StateClassifier 状态分类器
3. StateGraph & Navigator 状态图导航
4. FallbackHandler 异常恢复处理器
"""

import sys
import time
import logging
import unittest

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(__file__).replace("\\tests\\test_states.py", ""))

from bot.states.state_machine import GameState, StateMachine, StateContext
from bot.states.navigator import StateGraph, Navigator, NavigationStep, create_default_state_graph
from bot.states.fallback import (
    FallbackHandler, RecoveryAction, RecoveryStep, RecoveryPlan,
    create_default_fallback_handler
)


class TestStateMachine(unittest.TestCase):
    """测试 StateMachine 状态机核心"""
    
    def setUp(self):
        self.sm = StateMachine(initial_state=GameState.MAIN_VILLAGE)
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.sm.current_state, GameState.MAIN_VILLAGE)
        self.assertIsNone(self.sm.previous_state)
        self.assertEqual(self.sm.transition_count, 0)
    
    def test_transition(self):
        """测试状态转换"""
        result = self.sm.transition(GameState.ATTACK_MENU)
        self.assertTrue(result)
        self.assertEqual(self.sm.current_state, GameState.ATTACK_MENU)
        self.assertEqual(self.sm.previous_state, GameState.MAIN_VILLAGE)
        self.assertEqual(self.sm.transition_count, 1)
    
    def test_transition_same_state(self):
        """测试相同状态转换"""
        result = self.sm.transition(GameState.MAIN_VILLAGE)
        self.assertFalse(result)
        self.assertEqual(self.sm.transition_count, 0)
    
    def test_force_transition(self):
        """测试强制转换"""
        result = self.sm.force_transition(GameState.ENEMY_BASE)
        self.assertTrue(result)
        self.assertEqual(self.sm.current_state, GameState.ENEMY_BASE)
    
    def test_confirm_frames(self):
        """测试多帧确认"""
        self.sm.set_confirm_frames(GameState.ATTACK_MENU, 3)
        
        # 第一次转换尝试
        result = self.sm.transition(GameState.ATTACK_MENU)
        self.assertFalse(result)  # 还未达到确认帧数
        self.assertEqual(self.sm.current_state, GameState.MAIN_VILLAGE)
        
        # 第二次
        result = self.sm.transition(GameState.ATTACK_MENU)
        self.assertFalse(result)
        
        # 第三次
        result = self.sm.transition(GameState.ATTACK_MENU)
        self.assertTrue(result)
        self.assertEqual(self.sm.current_state, GameState.ATTACK_MENU)
    
    def test_callbacks(self):
        """测试回调函数"""
        entered = []
        exited = []
        transitions = []
        
        self.sm.add_enter_callback(
            GameState.ATTACK_MENU,
            lambda old, new: entered.append(new)
        )
        self.sm.add_exit_callback(
            GameState.MAIN_VILLAGE,
            lambda old, new: exited.append(old)
        )
        self.sm.add_transition_callback(
            lambda old, new, conf: transitions.append((old, new))
        )
        
        self.sm.transition(GameState.ATTACK_MENU)
        
        self.assertEqual(len(entered), 1)
        self.assertEqual(entered[0], GameState.ATTACK_MENU)
        self.assertEqual(len(exited), 1)
        self.assertEqual(exited[0], GameState.MAIN_VILLAGE)
        self.assertEqual(len(transitions), 1)
    
    def test_state_duration(self):
        """测试状态持续时间"""
        time.sleep(0.1)
        self.assertGreater(self.sm.state_duration, 0.09)
    
    def test_history(self):
        """测试状态历史"""
        self.sm.transition(GameState.ATTACK_MENU)
        self.sm.transition(GameState.ENEMY_BASE)
        
        history = self.sm.get_history(10)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].state, GameState.ATTACK_MENU)
        self.assertEqual(history[1].state, GameState.ENEMY_BASE)
    
    def test_reset(self):
        """测试重置"""
        self.sm.transition(GameState.ATTACK_MENU)
        self.sm.reset()
        
        self.assertEqual(self.sm.current_state, GameState.UNKNOWN)
        self.assertIsNone(self.sm.previous_state)


class TestStateGraph(unittest.TestCase):
    """测试 StateGraph 状态图"""
    
    def setUp(self):
        self.graph = StateGraph()
        self.graph.add_edge(GameState.MAIN_VILLAGE, GameState.ATTACK_MENU, "点击攻击")
        self.graph.add_edge(GameState.ATTACK_MENU, GameState.ATTACK_SEARCH, "搜索对手")
        self.graph.add_edge(GameState.ATTACK_SEARCH, GameState.ENEMY_BASE, "找到对手")
        self.graph.add_edge(GameState.ENEMY_BASE, GameState.BATTLE_RESULTS, "战斗结束")
        self.graph.add_edge(GameState.BATTLE_RESULTS, GameState.MAIN_VILLAGE, "返回村庄")
    
    def test_add_edge(self):
        """测试添加边"""
        self.assertTrue(self.graph.has_edge(GameState.MAIN_VILLAGE, GameState.ATTACK_MENU))
        self.assertFalse(self.graph.has_edge(GameState.ATTACK_MENU, GameState.MAIN_VILLAGE))
    
    def test_get_neighbors(self):
        """测试获取邻居"""
        neighbors = self.graph.get_neighbors(GameState.MAIN_VILLAGE)
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0][0], GameState.ATTACK_MENU)
    
    def test_get_action(self):
        """测试获取动作"""
        action = self.graph.get_action(GameState.MAIN_VILLAGE, GameState.ATTACK_MENU)
        self.assertEqual(action, "点击攻击")
        
        action = self.graph.get_action(GameState.ATTACK_MENU, GameState.MAIN_VILLAGE)
        self.assertIsNone(action)
    
    def test_get_predecessors(self):
        """测试获取前驱"""
        preds = self.graph.get_predecessors(GameState.ATTACK_MENU)
        self.assertEqual(len(preds), 1)
        self.assertIn(GameState.MAIN_VILLAGE, preds)


class TestNavigator(unittest.TestCase):
    """测试 Navigator 导航器"""
    
    def setUp(self):
        self.graph = create_default_state_graph()
        self.navigator = Navigator(self.graph)
    
    def test_find_path_adjacent(self):
        """测试相邻状态路径"""
        path = self.navigator.find_path(GameState.MAIN_VILLAGE, GameState.ATTACK_MENU)
        self.assertEqual(len(path), 1)
        self.assertEqual(path[0].from_state, GameState.MAIN_VILLAGE)
        self.assertEqual(path[0].to_state, GameState.ATTACK_MENU)
    
    def test_find_path_multi_step(self):
        """测试多步路径"""
        path = self.navigator.find_path(GameState.MAIN_VILLAGE, GameState.ENEMY_BASE)
        self.assertGreater(len(path), 1)
        
        # 验证路径连续性
        for i in range(len(path) - 1):
            self.assertEqual(path[i].to_state, path[i+1].from_state)
        
        # 验证起点和终点
        self.assertEqual(path[0].from_state, GameState.MAIN_VILLAGE)
        self.assertEqual(path[-1].to_state, GameState.ENEMY_BASE)
    
    def test_find_path_same_state(self):
        """测试相同状态"""
        path = self.navigator.find_path(GameState.MAIN_VILLAGE, GameState.MAIN_VILLAGE)
        self.assertEqual(len(path), 0)
    
    def test_is_reachable(self):
        """测试可达性"""
        self.assertTrue(
            self.navigator.is_reachable(GameState.MAIN_VILLAGE, GameState.ENEMY_BASE)
        )
    
    def test_get_next_step(self):
        """测试获取下一步"""
        step = self.navigator.get_next_step(GameState.MAIN_VILLAGE, GameState.ENEMY_BASE)
        self.assertIsNotNone(step)
        self.assertEqual(step.from_state, GameState.MAIN_VILLAGE)
    
    def test_get_distance(self):
        """测试距离计算"""
        dist = self.navigator.get_distance(GameState.MAIN_VILLAGE, GameState.ATTACK_MENU)
        self.assertEqual(dist, 1)
        
        dist = self.navigator.get_distance(GameState.MAIN_VILLAGE, GameState.ENEMY_BASE)
        self.assertGreater(dist, 1)
    
    def test_path_caching(self):
        """测试路径缓存"""
        # 第一次查找
        path1 = self.navigator.find_path(GameState.MAIN_VILLAGE, GameState.ENEMY_BASE)
        
        # 第二次查找应该使用缓存
        path2 = self.navigator.find_path(GameState.MAIN_VILLAGE, GameState.ENEMY_BASE)
        
        self.assertEqual(path1, path2)
    
    def test_full_cycle(self):
        """测试完整循环路径"""
        # 主村庄 -> 攻击 -> 搜索 -> 战斗 -> 结算 -> 主村庄
        path = self.navigator.find_path(
            GameState.MAIN_VILLAGE, GameState.MAIN_VILLAGE
        )
        # 相同状态返回空路径
        self.assertEqual(len(path), 0)
        
        # 但可以通过攻击菜单绕一圈
        path = self.navigator.find_path(
            GameState.BATTLE_RESULTS, GameState.MAIN_VILLAGE
        )
        self.assertGreater(len(path), 0)
        self.assertEqual(path[-1].to_state, GameState.MAIN_VILLAGE)


class TestFallbackHandler(unittest.TestCase):
    """测试 FallbackHandler 异常恢复处理器"""
    
    def setUp(self):
        self.handler = FallbackHandler()
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(len(self.handler._plans), 0)
        self.assertEqual(self.handler.stats["total"], 0)
    
    def test_register_plan(self):
        """测试注册恢复计划"""
        plan = RecoveryPlan(
            trigger_states=[GameState.POPUP],
            steps=[
                RecoveryStep(RecoveryAction.WAIT, "等待", {"duration": 1.0})
            ],
            target_state=GameState.MAIN_VILLAGE
        )
        self.handler.register_plan(plan)
        
        self.assertTrue(self.handler.should_handle(GameState.POPUP))
        self.assertFalse(self.handler.should_handle(GameState.UNKNOWN))
    
    def test_add_plan(self):
        """测试便捷添加方法"""
        self.handler.add_plan(
            trigger_states=[GameState.POPUP, GameState.UNKNOWN],
            target_state=GameState.MAIN_VILLAGE,
            steps=[
                RecoveryStep(RecoveryAction.TAP_BACK, "按返回键")
            ]
        )
        
        self.assertTrue(self.handler.should_handle(GameState.POPUP))
        self.assertTrue(self.handler.should_handle(GameState.UNKNOWN))
    
    def test_retry_mechanism(self):
        """测试重试机制"""
        self.handler.add_plan(
            trigger_states=[GameState.POPUP],
            target_state=GameState.MAIN_VILLAGE,
            steps=[
                RecoveryStep(RecoveryAction.WAIT, "等待", {"duration": 0.01})
            ],
            max_retries=2
        )
        
        self.assertTrue(self.handler.can_retry(GameState.POPUP))
        
        # 第一次重试
        self.handler.increment_retry(GameState.POPUP)
        self.assertTrue(self.handler.can_retry(GameState.POPUP))
        
        # 第二次重试
        self.handler.increment_retry(GameState.POPUP)
        self.assertFalse(self.handler.can_retry(GameState.POPUP))
        
        # 重置
        self.handler.reset_retry(GameState.POPUP)
        self.assertTrue(self.handler.can_retry(GameState.POPUP))
    
    def test_execute_recovery(self):
        """测试执行恢复"""
        self.handler.add_plan(
            trigger_states=[GameState.POPUP],
            target_state=GameState.MAIN_VILLAGE,
            steps=[
                RecoveryStep(RecoveryAction.WAIT, "等待", {"duration": 0.01}),
                RecoveryStep(RecoveryAction.TAP_BACK, "按返回键"),
            ]
        )
        
        result = self.handler.execute_recovery(GameState.POPUP)
        self.assertTrue(result)
        self.assertEqual(self.handler.stats["success"], 1)
    
    def test_callbacks(self):
        """测试回调"""
        started = []
        succeeded = []
        
        self.handler.add_plan(
            trigger_states=[GameState.POPUP],
            target_state=GameState.MAIN_VILLAGE,
            steps=[
                RecoveryStep(RecoveryAction.WAIT, "等待", {"duration": 0.01})
            ]
        )
        
        self.handler.on_recovery_start(lambda s, p: started.append(s))
        self.handler.on_recovery_success(lambda s, p: succeeded.append(s))
        
        self.handler.execute_recovery(GameState.POPUP)
        
        self.assertEqual(len(started), 1)
        self.assertEqual(len(succeeded), 1)
    
    def test_default_handler(self):
        """测试默认恢复处理器"""
        handler = create_default_fallback_handler()
        
        self.assertTrue(handler.should_handle(GameState.POPUP))
        self.assertTrue(handler.should_handle(GameState.MAINTENANCE))
        self.assertTrue(handler.should_handle(GameState.UNKNOWN))
        self.assertTrue(handler.should_handle(GameState.LOADING))


def run_basic_tests():
    """运行基本测试"""
    print("\n" + "=" * 60)
    print("运行状态机与决策层模块基本测试")
    print("=" * 60)
    
    # 1. 测试 StateMachine
    print("\n[1/4] 测试 StateMachine 状态机...")
    sm = StateMachine(initial_state=GameState.MAIN_VILLAGE)
    assert sm.current_state == GameState.MAIN_VILLAGE
    assert sm.transition_count == 0
    
    # 测试转换
    result = sm.transition(GameState.ATTACK_MENU)
    assert result, "状态转换失败"
    assert sm.current_state == GameState.ATTACK_MENU
    assert sm.previous_state == GameState.MAIN_VILLAGE
    print("  ✓ 状态转换正常")
    
    # 测试相同状态
    result = sm.transition(GameState.ATTACK_MENU)
    assert not result, "相同状态不应转换"
    print("  ✓ 相同状态不转换")
    
    # 测试多帧确认
    sm2 = StateMachine(initial_state=GameState.MAIN_VILLAGE)
    sm2.set_confirm_frames(GameState.ATTACK_MENU, 3)
    assert not sm2.transition(GameState.ATTACK_MENU)
    assert not sm2.transition(GameState.ATTACK_MENU)
    assert sm2.transition(GameState.ATTACK_MENU)
    assert sm2.current_state == GameState.ATTACK_MENU
    print("  ✓ 多帧确认正常")
    
    # 测试回调
    callback_log = []
    sm3 = StateMachine(initial_state=GameState.MAIN_VILLAGE)
    sm3.add_enter_callback(
        GameState.ATTACK_MENU,
        lambda old, new: callback_log.append(("enter", new))
    )
    sm3.add_transition_callback(
        lambda old, new, conf: callback_log.append(("transition", old, new))
    )
    sm3.transition(GameState.ATTACK_MENU)
    assert len(callback_log) == 2
    print("  ✓ 回调函数正常")
    
    # 测试历史
    history = sm3.get_history(10)
    assert len(history) == 1
    assert history[0].state == GameState.ATTACK_MENU
    print("  ✓ 状态历史正常")
    
    print("  ✓ StateMachine 测试通过")
    
    # 2. 测试 StateGraph & Navigator
    print("\n[2/4] 测试 StateGraph & Navigator 状态图导航...")
    graph = create_default_state_graph()
    navigator = Navigator(graph)
    
    # 测试相邻路径
    path = navigator.find_path(GameState.MAIN_VILLAGE, GameState.ATTACK_MENU)
    assert len(path) == 1
    assert path[0].action == "点击攻击按钮"
    print("  ✓ 相邻状态路径正常")
    
    # 测试多步路径
    path = navigator.find_path(GameState.MAIN_VILLAGE, GameState.ENEMY_BASE)
    assert len(path) > 1
    # 验证路径连续性
    for i in range(len(path) - 1):
        assert path[i].to_state == path[i+1].from_state, "路径不连续"
    assert path[0].from_state == GameState.MAIN_VILLAGE
    assert path[-1].to_state == GameState.ENEMY_BASE
    print(f"  ✓ 多步路径正常 ({len(path)} 步)")
    
    # 测试可达性
    assert navigator.is_reachable(GameState.MAIN_VILLAGE, GameState.BATTLE_RESULTS)
    print("  ✓ 可达性检测正常")
    
    # 测试距离
    dist = navigator.get_distance(GameState.MAIN_VILLAGE, GameState.ATTACK_MENU)
    assert dist == 1
    dist = navigator.get_distance(GameState.MAIN_VILLAGE, GameState.ENEMY_BASE)
    assert dist > 1
    print(f"  ✓ 距离计算正常 (MV->AB={navigator.get_distance(GameState.MAIN_VILLAGE, GameState.ATTACK_MENU)}, MV->EB={dist})")
    
    print("  ✓ Navigator 测试通过")
    
    # 3. 测试 FallbackHandler
    print("\n[3/4] 测试 FallbackHandler 异常恢复...")
    handler = create_default_fallback_handler()
    
    assert handler.should_handle(GameState.POPUP)
    assert handler.should_handle(GameState.MAINTENANCE)
    assert handler.should_handle(GameState.UNKNOWN)
    assert not handler.should_handle(GameState.MAIN_VILLAGE)
    print("  ✓ 恢复计划注册正常")
    
    # 测试恢复执行
    result = handler.execute_recovery(GameState.POPUP)
    assert result, "恢复执行失败"
    assert handler.stats["success"] >= 1
    print("  ✓ 恢复执行正常")
    
    # 测试重试机制
    handler2 = FallbackHandler()
    handler2.add_plan(
        trigger_states=[GameState.POPUP],
        target_state=GameState.MAIN_VILLAGE,
        steps=[RecoveryStep(RecoveryAction.WAIT, "等待", {"duration": 0.01})],
        max_retries=2
    )
    assert handler2.can_retry(GameState.POPUP)
    handler2.increment_retry(GameState.POPUP)
    handler2.increment_retry(GameState.POPUP)
    assert not handler2.can_retry(GameState.POPUP)
    handler2.reset_retry(GameState.POPUP)
    assert handler2.can_retry(GameState.POPUP)
    print("  ✓ 重试机制正常")
    
    print("  ✓ FallbackHandler 测试通过")
    
    # 4. 测试完整工作流
    print("\n[4/4] 测试完整工作流...")
    
    # 创建组件
    sm = StateMachine(initial_state=GameState.MAIN_VILLAGE)
    graph = create_default_state_graph()
    navigator = Navigator(graph)
    handler = create_default_fallback_handler()
    
    # 模拟：主村庄 -> 攻击菜单 -> 搜索对手 -> 战斗 -> 结算 -> 返回
    states = [
        GameState.ATTACK_MENU,
        GameState.ATTACK_SEARCH,
        GameState.ENEMY_BASE,
        GameState.BATTLE_RESULTS,
        GameState.MAIN_VILLAGE,
    ]
    
    for state in states:
        result = sm.transition(state)
        assert result, f"转换到 {state.name} 失败"
    
    assert sm.current_state == GameState.MAIN_VILLAGE
    assert sm.transition_count == 5
    print(f"  ✓ 完整循环完成: {sm.transition_count} 次转换")
    
    # 模拟异常恢复
    sm.force_transition(GameState.POPUP)
    assert sm.current_state == GameState.POPUP
    
    plan = handler.get_plan(GameState.POPUP)
    assert plan is not None
    assert plan.target_state == GameState.MAIN_VILLAGE
    
    handler.execute_recovery(GameState.POPUP)
    sm.force_transition(GameState.MAIN_VILLAGE)
    assert sm.current_state == GameState.MAIN_VILLAGE
    print("  ✓ 异常恢复流程正常")
    
    print("  ✓ 完整工作流测试通过")
    
    print("\n" + "=" * 60)
    print("所有基本测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    run_basic_tests()
