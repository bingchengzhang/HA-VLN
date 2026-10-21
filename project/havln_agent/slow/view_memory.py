"""视野记忆：给每个候选方向配一张"朝那边看过的画面"。

背景：单视角只有 90°，决策点上很多候选方向此刻看不到。
目的：大模型选方向时，每个候选尽量都有一张对应的图。
本质：记下最近的关键帧（RGB + 拍摄位姿）。视野内的候选直接在当前画面上画编号；
      视野外的候选，找一张拍摄位置离得近、拍摄朝向与候选方向接近的旧帧。
      开局转一圈（initial_spin）正好为四周各方向都存下一帧。
"""
import math
from collections import deque
from typing import List, Optional

import numpy as np

from ..geometry import wrap


class ViewMemory:
    def __init__(self, maxlen: int = 64):
        self.frames = deque(maxlen=maxlen)      # (step, pose, rgb)

    def add(self, step, pose, rgb, floor: int = 0):
        self.frames.append((step, pose, np.asarray(rgb)[..., :3].copy(), floor))

    def find(self, x, y, yaw, floor: int = 0, max_dist=2.0, max_ang_deg=35.0) -> Optional[int]:
        """返回最合适的旧帧下标：只在同一楼层里找；朝向误差为主、离得近与新近为辅。"""
        best, best_score = None, math.inf
        for k, (step, pose, _, fl) in enumerate(self.frames):
            if fl != floor:
                continue
            d = math.hypot(pose[0] - x, pose[1] - y)
            da = abs(math.degrees(wrap(pose[2] - yaw)))
            if d > max_dist or da > max_ang_deg:
                continue
            score = da + 10.0 * d - 0.01 * step
            if score < best_score:
                best, best_score = k, score
        return best

    def get(self, k):
        return self.frames[k][2]


def draw_candidates(rgb: np.ndarray, cands, hfov_deg: float) -> np.ndarray:
    """在当前画面上，为视野内的每个候选画一条竖线和编号。"""
    from PIL import Image, ImageDraw
    img = Image.fromarray(np.asarray(rgb)[..., :3].astype(np.uint8))
    w, h = img.size
    f = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    draw = ImageDraw.Draw(img)
    for c in cands:
        if not c.in_view:
            continue
        u = w / 2 - f * math.tan(math.radians(c.bearing_deg))   # 向左为正 → 图像左侧
        u = int(min(max(u, 4), w - 5))
        draw.line([(u, int(h * 0.45)), (u, h - 1)], fill=(255, 40, 40), width=3)
        draw.rectangle([u - 9, int(h * 0.45) - 20, u + 9, int(h * 0.45)], fill=(255, 40, 40))
        draw.text((u - 4, int(h * 0.45) - 18), str(c.cid), fill=(255, 255, 255))
    return np.asarray(img)


def build_images(rgb_now, pose, cands, memory: ViewMemory, hfov_deg: float, max_images: int = 4, floor: int = 0):
    """当前画面（带候选编号）+ 若干视野外候选的旧帧。返回 (images, captions)。"""
    images = [draw_candidates(rgb_now, cands, hfov_deg)]
    in_view = [c.cid for c in cands if c.in_view]
    captions = [f"Current view. Red numbered lines mark candidates {in_view} that are in view."]
    for c in cands:
        if len(images) >= max_images:
            break
        if c.in_view:
            continue
        k = memory.find(pose[0], pose[1], c.abs_yaw, floor)
        if k is None:
            continue
        c.view_ref = len(images)
        images.append(memory.get(k))
        captions.append(f"Earlier view looking toward candidate {c.cid} ({c.summary()['direction']}).")
    return images, captions
