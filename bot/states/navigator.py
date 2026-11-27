from collections import deque

from bot.states.base import GameState


class Navigator:
    """状态图最短路径导航。"""

    GRAPH: dict[GameState, list[GameState]] = {
        GameState.MAIN_VILLAGE: [GameState.ATTACK_MENU, GameState.CLAN_CHAT,
                                 GameState.BUILDER_MENU, GameState.BUILDING_PANEL,
                                 GameState.SETTINGS],
        GameState.ATTACK_MENU: [GameState.ARMY_SETUP, GameState.MAIN_VILLAGE],
        GameState.ARMY_SETUP: [GameState.FIND_MATCH, GameState.ATTACK_MENU],
        GameState.FIND_MATCH: [GameState.BATTLE, GameState.MAIN_VILLAGE],
        GameState.BATTLE: [GameState.RESULTS],
        GameState.RESULTS: [GameState.MAIN_VILLAGE],
        GameState.CLAN_CHAT: [GameState.MAIN_VILLAGE],
        GameState.BUILDER_MENU: [GameState.MAIN_VILLAGE, GameState.BUILDING_PANEL],
        GameState.BUILDING_PANEL: [GameState.BUILDER_MENU, GameState.MAIN_VILLAGE],
        GameState.SETTINGS: [GameState.SWITCH_ACCOUNT, GameState.MAIN_VILLAGE],
        GameState.SWITCH_ACCOUNT: [GameState.MAIN_VILLAGE],
        GameState.POPUP: [GameState.MAIN_VILLAGE],
    }

    def path(self, src: GameState, dst: GameState) -> list[GameState]:
        if src == dst:
            return [src]
        queue = deque([(src, [src])])
        visited = {src}
        while queue:
            node, route = queue.popleft()
            for nxt in self.GRAPH.get(node, []):
                if nxt in visited:
                    continue
                if nxt == dst:
                    return route + [nxt]
                visited.add(nxt)
                queue.append((nxt, route + [nxt]))
        return []
