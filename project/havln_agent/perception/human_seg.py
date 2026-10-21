"""人体实例分割。

背景：哪些像素是人是语义问题，深度上分不出来（人停下后和家具一样是静止表面）。
目的：给每个人一张逐像素的掩码（不是方框），人身后的墙不会被误当成人。
本质：现成的 YOLO-seg（COCO 里 person 是类别 0）。RGB 224×224 与深度 256×256 视场角相同，
      掩码按比例缩放到深度图尺寸即可对齐。
仿真里的人是 SMPL 网格渲染的，域差要实测：scripts/eval_segmenter.py 给出召回率，
不够再用 scripts/gen_seg_labels.py 生成的自动标签微调。
"""
from typing import Optional

import numpy as np


class HumanSegmenter:
    """接口：输入 RGB（H, W, 3, uint8），输出
         instance_ids：与深度图同尺寸的实例 id 图（0 为背景，k>0 为第 k 个人）
         labels：画面里检测到的其他物体类别（COCO 名称），供"子目标可能已完成"的判断用
    一次推理同时拿到两样东西，不额外花时间。"""

    def segment(self, rgb: np.ndarray, out_shape):
        raise NotImplementedError


class NullSegmenter(HumanSegmenter):
    """不检测人（调试建图用）。"""

    def segment(self, rgb, out_shape):
        return np.zeros(out_shape, dtype=np.int32), set()


class YoloSegSegmenter(HumanSegmenter):
    def __init__(self, weights: str = "yolov8m-seg.pt", conf: float = 0.35, device: Optional[str] = None,
                 min_pixels: int = 80, imgsz: int = 320):
        from ultralytics import YOLO          # 只在真正用到时导入，单元测试不依赖它
        self.model = YOLO(weights)
        self.conf, self.device, self.min_pixels, self.imgsz = conf, device, min_pixels, imgsz

    def segment(self, rgb, out_shape):
        res = self.model.predict(rgb[..., :3], imgsz=self.imgsz, conf=self.conf,
                                 device=self.device, verbose=False, retina_masks=True)[0]
        out = np.zeros(out_shape, dtype=np.int32)
        if res.boxes is None or len(res.boxes) == 0:
            return out, set()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        names = res.names
        labels = {names[c] for c in cls if c != 0}
        if res.masks is None:
            return out, labels
        masks = res.masks.data.cpu().numpy() > 0.5                 # (K, h, w)，retina_masks 下与原图同尺寸
        conf = res.boxes.conf.cpu().numpy()
        order = [k for k in np.argsort(conf) if cls[k] == 0]       # 只取人；低置信度先画，高置信度覆盖
        k_id = 0
        for k in order:
            m = _resize_nearest(masks[k], out_shape)
            if m.sum() < self.min_pixels:
                continue
            k_id += 1
            out[m] = k_id
        return out, labels


def _resize_nearest(mask: np.ndarray, shape) -> np.ndarray:
    h, w = mask.shape
    H, W = shape
    if (h, w) == (H, W):
        return mask
    ri = (np.arange(H) * h / H).astype(int)
    ci = (np.arange(W) * w / W).astype(int)
    return mask[ri][:, ci]


def build_segmenter(kind: str, weights: str = None, device=None, conf=0.35, min_pixels=80) -> HumanSegmenter:
    if kind == "none":
        return NullSegmenter()
    if kind == "yolo":
        return YoloSegSegmenter(weights or "yolov8m-seg.pt", conf=conf, device=device, min_pixels=min_pixels)
    raise ValueError(f"未知分割器 {kind}")
