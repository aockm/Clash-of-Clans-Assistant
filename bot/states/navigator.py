"""
Navigator - 状态图导航器

维护状态可达关系，给定目标状态自动规划最短路径。
"""

import logging
from typing import Optional, List, Dict, Set, Tuple
from collections import deque
from dataclasses import dataclass

from .state_machine import GameState

logger = logging.getLogger(__name__)


@dataclass
class NavigationStep:
    """导航步骤"""
    from_state: GameState
    to_state: GameState
    action: str  # 执行的动作描述
    
    def __repr__(self) -> str:
        return f"Step({self.from_state.name} -> {self.to_state.name}: {self.action})"


class StateGraph:
    """
    状态图
    
    维护状态之间的可达关系和转换动作。
    """
    
    def __init__(self):
        """初始化状态图"""
        # 邻接表: {from_state: [(to_state, action), ...]}
        self._edges: Dict[GameState, List[Tuple[GameState, str]]] = {}
        
        # 反向邻接表（用于快速查找前驱）
        self._reverse_edges: Dict[GameState, List[GameState]] = {}
        
        logger.info("状态图初始化完成")
    
    def add_edge(self, from_state: GameState, to_state: GameState, action: str) -> None:
        """
        添加状态转换边
        
        Args:
            from_state: 起始状态
            to_state: 目标状态
            action: 转换动作描述
        """
        if from_state not in self._edges:
            self._edges[from_state] = []
        
        self._edges[from_state].append((to_state, action))
        
        # 更新反向边
        if to_state not in self._reverse_edges:
            self._reverse_edges[to_state] = []
        if from_state not in self._reverse_edges[to_state]:
            self._reverse_edges[to_state].append(from_state)
        
        logger.debug(f"添加边: {from_state.name} -> {to_state.name} ({action})")
    
    def get_neighbors(self, state: GameState) -> List[Tuple[GameState, str]]:
        """
        获取状态的所有邻居
        
        Args:
            state: 当前状态
            
        Returns:
            [(邻居状态, 动作), ...] 列表
        """
        return self._edges.get(state, [])
    
    def get_predecessors(self, state: GameState) -> List[GameState]:
        """
        获取状态的所有前驱
        
        Args:
            state: 当前状态
            
        Returns:
            前驱状态列表
        """
        return self._reverse_edges.get(state, [])
    
    def has_edge(self, from_state: GameState, to_state: GameState) -> bool:
        """检查是否存在边"""
        neighbors = self._edges.get(from_state, [])
        return any(to_state == n for n, _ in neighbors)
    
    def get_action(self, from_state: GameState, to_state: GameState) -> Optional[str]:
        """获取两个状态之间的动作"""
        neighbors = self._edges.get(from_state, [])
        for state, action in neighbors:
            if state == to_state:
                return action
        return None
    
    def get_all_states(self) -> Set[GameState]:
        """获取图中所有状态"""
        states = set(self._edges.keys())
        for neighbors in self._edges.values():
            for state, _ in neighbors:
                states.add(state)
        return states
    
    def __repr__(self) -> str:
        edge_count = sum(len(n) for n in self._edges.values())
        return f"StateGraph(states={len(self.get_all_states())}, edges={edge_count})"


class Navigator:
    """
    导航器
    
    使用 BFS 算法在状态图中寻找最短路径。
    """
    
    def __init__(self, state_graph: StateGraph):
        """
        初始化导航器
        
        Args:
            state_graph: 状态图
        """
        self.graph = state_graph
        self._path_cache: Dict[Tuple[GameState, GameState], List[NavigationStep]] = {}
        
        logger.info("导航器初始化完成")
    
    def find_path(
        self,
        from_state: GameState,
        to_state: GameState,
        max_depth: int = 10
    ) -> List[NavigationStep]:
        """
        使用 BFS 寻找最短路径
        
        Args:
            from_state: 起始状态
            to_state: 目标状态
            max_depth: 最大搜索深度
            
        Returns:
            导航步骤列表，无法到达时返回空列表
        """
        # 检查缓存
        cache_key = (from_state, to_state)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]
        
        # 起始状态等于目标状态
        if from_state == to_state:
            return []
        
        # BFS 搜索
        queue = deque([(from_state, [])])
        visited = {from_state}
        
        while queue:
            current_state, path = queue.popleft()
            
            # 检查深度限制
            if len(path) >= max_depth:
                continue
            
            # 遍历邻居
            for next_state, action in self.graph.get_neighbors(current_state):
                if next_state in visited:
                    continue
                
                new_path = path + [NavigationStep(current_state, next_state, action)]
                
                # 找到目标
                if next_state == to_state:
                    # 缓存路径
                    self._path_cache[cache_key] = new_path
                    logger.info(
                        f"找到路径: {from_state.name} -> {to_state.name}, "
                        f"步数: {len(new_path)}"
                    )
                    return new_path
                
                visited.add(next_state)
                queue.append((next_state, new_path))
        
        # 无法到达
        logger.warning(f"无法找到路径: {from_state.name} -> {to_state.name}")
        return []
    
    def get_next_step(
        self,
        from_state: GameState,
        to_state: GameState
    ) -> Optional[NavigationStep]:
        """
        获取到达目标状态的下一步
        
        Args:
            from_state: 当前状态
            to_state: 目标状态
            
        Returns:
            下一步导航步骤，无法到达时返回 None
        """
        path = self.find_path(from_state, to_state)
        if path:
            return path[0]
        return None
    
    def is_reachable(self, from_state: GameState, to_state: GameState) -> bool:
        """
        检查目标状态是否可达
        
        Args:
            from_state: 起始状态
            to_state: 目标状态
            
        Returns:
            可达返回 True，否则返回 False
        """
        path = self.find_path(from_state, to_state)
        return len(path) > 0 or from_state == to_state
    
    def get_distance(self, from_state: GameState, to_state: GameState) -> int:
        """
        获取两个状态之间的最短距离（步数）
        
        Args:
            from_state: 起始状态
            to_state: 目标状态
            
        Returns:
            最短距离，无法到达时返回 -1
        """
        path = self.find_path(from_state, to_state)
        return len(path) if path else -1
    
    def clear_cache(self) -> None:
        """清除路径缓存"""
        self._path_cache.clear()
        logger.debug("路径缓存已清除")
    
    def __repr__(self) -> str:
        return f"Navigator(graph={self.graph}, cached_paths={len(self._path_cache)})"


def create_default_state_graph() -> StateGraph:
    """
    创建默认的状态图
    
    定义 COC 游戏中常见的状态转换关系。
    
    Returns:
        配置好的状态图
    """
    graph = StateGraph()
    
    # 主村庄 -> 其他状态
    graph.add_edge(GameState.MAIN_VILLAGE, GameState.ATTACK_MENU, "点击攻击按钮")
    graph.add_edge(GameState.MAIN_VILLAGE, GameState.CLAN_CHAT, "点击部落聊天")
    graph.add_edge(GameState.MAIN_VILLAGE, GameState.BUILDING_INFO, "点击建筑")
    graph.add_edge(GameState.MAIN_VILLAGE, GameState.SHOP, "点击商店")
    graph.add_edge(GameState.MAIN_VILLAGE, GameState.SETTINGS, "点击设置")
    
    # 攻击菜单
    graph.add_edge(GameState.ATTACK_MENU, GameState.ATTACK_SEARCH, "点击寻找对手")
    graph.add_edge(GameState.ATTACK_MENU, GameState.MAIN_VILLAGE, "点击返回")
    graph.add_edge(GameState.ATTACK_SEARCH, GameState.MY_ARMY, "配兵界面出现")
    graph.add_edge(GameState.MY_ARMY, GameState.ENEMY_BASE, "点击进攻")
    graph.add_edge(GameState.MY_ARMY, GameState.ATTACK_MENU, "点击关闭")
    
    # 搜索对手
    graph.add_edge(GameState.ATTACK_SEARCH, GameState.ENEMY_BASE, "找到对手")
    graph.add_edge(GameState.ATTACK_SEARCH, GameState.ATTACK_SEARCH, "继续搜索")
    graph.add_edge(GameState.ATTACK_SEARCH, GameState.MAIN_VILLAGE, "取消搜索")
    
    # 战斗
    graph.add_edge(GameState.ENEMY_BASE, GameState.BATTLE_RESULTS, "战斗结束")
    
    # 战斗结算
    graph.add_edge(GameState.BATTLE_RESULTS, GameState.MAIN_VILLAGE, "点击返回")
    graph.add_edge(GameState.BATTLE_RESULTS, GameState.ATTACK_MENU, "再次攻击")
    
    # 部落聊天
    graph.add_edge(GameState.CLAN_CHAT, GameState.DONATE_TROOPS, "点击捐兵")
    graph.add_edge(GameState.CLAN_CHAT, GameState.MAIN_VILLAGE, "点击返回")
    
    # 捐兵
    graph.add_edge(GameState.DONATE_TROOPS, GameState.CLAN_CHAT, "完成捐兵")
    
    # 建筑信息
    graph.add_edge(GameState.BUILDING_INFO, GameState.BUILDER_MENU, "点击升级")
    graph.add_edge(GameState.BUILDING_INFO, GameState.MAIN_VILLAGE, "关闭面板")
    
    # 工人菜单
    graph.add_edge(GameState.BUILDER_MENU, GameState.UPGRADE_CONFIRM, "选择工人")
    graph.add_edge(GameState.BUILDER_MENU, GameState.BUILDING_INFO, "返回")
    
    # 升级确认
    graph.add_edge(GameState.UPGRADE_CONFIRM, GameState.MAIN_VILLAGE, "确认升级")
    graph.add_edge(GameState.UPGRADE_CONFIRM, GameState.BUILDING_INFO, "取消")
    
    # 商店
    graph.add_edge(GameState.SHOP, GameState.MAIN_VILLAGE, "关闭商店")
    
    # 设置
    graph.add_edge(GameState.SETTINGS, GameState.MAIN_VILLAGE, "关闭设置")
    
    # 弹窗
    graph.add_edge(GameState.POPUP, GameState.MAIN_VILLAGE, "关闭弹窗")
    
    # 维护公告
    graph.add_edge(GameState.MAINTENANCE, GameState.MAIN_VILLAGE, "等待维护结束")
    
    # 加载中
    graph.add_edge(GameState.LOADING, GameState.MAIN_VILLAGE, "加载完成")
    
    logger.info(f"默认状态图创建完成: {graph}")
    return graph
