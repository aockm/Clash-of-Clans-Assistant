import cv2
import numpy as np

CARD_BLUE_LOWER = np.array([95, 90, 90])
CARD_BLUE_UPPER = np.array([125, 255, 255])
CARD_SIZE = (64, 90)
OCCUPIED_RATIO = 0.10


def card_present(frame: np.ndarray, center: tuple[int, int],
                 size: tuple[int, int] = CARD_SIZE,
                 ratio: float = OCCUPIED_RATIO) -> bool:
    """兵种卡槽是否还有兵：卡牌底色为饱和蓝色，空槽是虚线灰框。"""
    w, h = size
    cx, cy = center
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = hsv[cy - h // 2:cy + h // 2, cx - w // 2:cx + w // 2]
    if roi.size == 0:
        return False
    blue = cv2.inRange(roi, CARD_BLUE_LOWER, CARD_BLUE_UPPER)
    return cv2.countNonZero(blue) / blue.size >= ratio


def occupied_cards(frame: np.ndarray, deploy_cfg) -> list[tuple[int, int]]:
    cards: list[tuple[int, int]] = []
    for i in range(deploy_cfg.card_slot_count):
        center = (deploy_cfg.card_slot_start[0] + i * deploy_cfg.card_slot_pitch,
                  deploy_cfg.card_slot_start[1])
        if card_present(frame, center):
            cards.append(center)
    return cards
