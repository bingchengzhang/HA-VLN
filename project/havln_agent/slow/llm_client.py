"""OpenAI 兼容的对话接口：DeepSeek API 和本地 vLLM（Qwen2.5-VL）走同一套代码。

背景：官方镜像的 Python 环境被 Habitat 0.1.7 和 torch 2.0.1 锁死，把 7B 视觉模型装进去很容易冲突。
目的：大模型跑在独立的服务里（scripts/serve_vlm.sh 起一个 vLLM 容器），智能体只发 HTTP 请求。
本质：只依赖标准库发请求；所有调用按"模型 + 完整消息"做哈希缓存到磁盘。
      temperature 0 + 缓存，保证同一个 episode 重跑结果一致，获奖核验时也能复现。
"""
import base64
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
from typing import List, Optional

import numpy as np

from ..config import SlowCfg


class LLMError(RuntimeError):
    pass


def prepare_image(rgb: np.ndarray, side: int):
    """推理和造训练数据共用：缩放到 side×side 的 PIL 图。训练与推理的分辨率、
    以及 goal_box 的像素坐标系都以它为准（Qwen2.5-VL 输出的是输入图上的绝对像素坐标）。"""
    from PIL import Image
    img = Image.fromarray(np.asarray(rgb)[..., :3].astype(np.uint8))
    if img.size != (side, side):
        img = img.resize((side, side), Image.BILINEAR)
    return img


def image_to_data_url(rgb: np.ndarray, side: int) -> str:
    img = prepare_image(rgb, side)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class ChatClient:
    def __init__(self, cfg: SlowCfg):
        self.cfg = cfg
        self.api_key = os.environ.get(cfg.api_key_env, "EMPTY")
        os.makedirs(cfg.cache_dir, exist_ok=True)
        self.n_calls = 0
        self.n_cache_hits = 0
        self.last_cached = False

    def chat(self, system: str, text: str, images: Optional[List[np.ndarray]] = None,
             image_captions: Optional[List[str]] = None, max_tokens: Optional[int] = None,
             use_cache: bool = True) -> str:
        """发一轮对话，返回模型输出的文本。images 只在 supports_images=True 时发送。
        max_tokens：覆盖默认输出上限（会进入缓存键，改了上限就不会读到旧的截断回复）。
        use_cache=False：不读缓存、照常写缓存。上一次的回复解析失败、需要真正重试时用。"""
        content = [{"type": "text", "text": text}]
        if images and self.cfg.supports_images:
            for k, img in enumerate(images):
                cap = image_captions[k] if image_captions else f"Image {k}"
                content.append({"type": "text", "text": cap})
                content.append({"type": "image_url", "image_url": {"url": image_to_data_url(img, self.cfg.image_side)}})
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": content if len(content) > 1 else text}]
        body = {"model": self.cfg.model, "messages": messages, "temperature": self.cfg.temperature,
                "max_tokens": max_tokens or self.cfg.max_tokens}

        key = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        path = os.path.join(self.cfg.cache_dir, key[:2], key + ".json")
        self.last_cached = use_cache and os.path.exists(path)
        if self.last_cached:
            self.n_cache_hits += 1
            with open(path) as f:
                return json.load(f)["output"]

        out = self._post(body)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".partial"
        with open(tmp, "w") as f:
            json.dump({"model": self.cfg.model, "output": out, "time": time.time()}, f, ensure_ascii=False)
        os.replace(tmp, path)
        self.n_calls += 1
        return out

    def _post(self, body) -> str:
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        data = json.dumps(body).encode()
        # 本地 vLLM 不走代理：服务器上常设了 http_proxy，内网地址被转给代理就连不上
        handlers = [] if self.cfg.use_env_proxy else [urllib.request.ProxyHandler({})]
        opener = urllib.request.build_opener(*handlers)
        last = None
        for attempt in range(3):
            req = urllib.request.Request(url, data=data, method="POST", headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
            try:
                with opener.open(req, timeout=self.cfg.timeout_s) as r:
                    payload = json.loads(r.read().decode())
                return payload["choices"][0]["message"]["content"] or ""
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                last = e
                time.sleep(2 ** attempt)
        raise LLMError(f"调用 {url} 失败：{last}")


def extract_json(text: str) -> dict:
    """从模型输出里取出第一个完整的 JSON 对象（模型常在前后加解释或 ```json 包裹）。"""
    start = text.find("{")
    while start != -1:
        depth = 0
        for k in range(start, len(text)):
            ch = text[k]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:k + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError("输出里没有合法 JSON")
