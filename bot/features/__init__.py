"""
Feature Modules - 业务功能层模块
包含资源收集、建筑升级、捐兵、攻击搜索、战斗部署等功能
"""

from .collector import ResourceCollector
from .upgrader import BuildingUpgrader
from .donator import TroopDonator
from .attacker import AttackSearcher
from .deployer import BattleDeployer

__all__ = [
    "ResourceCollector",
    "BuildingUpgrader",
    "TroopDonator",
    "AttackSearcher",
    "BattleDeployer",
]
