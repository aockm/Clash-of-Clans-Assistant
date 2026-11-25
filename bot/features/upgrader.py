from bot.features.base import Feature


class Upgrader(Feature):
    """建筑升级（第三阶段实现）：OCR 读工人/费用 → 选择建筑 → 确认升级。"""

    name = "upgrader"

    def run(self) -> bool:
        raise NotImplementedError("第三阶段实现：建筑升级流程")
