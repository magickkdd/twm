import os
import re
import json
import time
import fcntl
import hashlib
import threading
import base64
from io import BytesIO

import numpy as np
from PIL import Image
from openai import OpenAI, RateLimitError, APITimeoutError

DEFAULT_RPM = 10.0
DEFAULT_TIMEOUT = 900.0
MAX_RETRIES = 15
INVALID_OBJECT_MAX_TRIES = 3

ACTION_RE = re.compile(r"(pick up|put down|insert|reorient|done)[ /\-*#`\[]*([a-z]+)?", re.I)

# 推理型模型会把结论放在最后，且常带 markdown 包裹；`parse_act_txt` 只认裸动作，
# 所以在这里把最后一次出现的动作对抽出来，交给下游的接口形状保持 "pick up red" 不变。
FORMAT_INSTRUCTION = ("\nOutput exactly one action as your final line, in the form "
                      "`<action> <color>`, e.g. `pick up red`. Put nothing after that line.")


def extract_action(text):
    matches = ACTION_RE.findall(text)
    if not matches:
        return text.strip()
    prim, obj = matches[-1]
    return f"{prim} {obj}".strip().lower()


def allowed_objects(prompt):
    m = re.search(r"following list: \[([^\]]*)\]", prompt)
    return set(m.group(1).replace(" ", "").split(",")) if m else set()

# 配额是所有 key、所有模型合并计算的，因此限速器必须全局共享而不是每个 agent 一个；
# 多进程分片时通过 DISCOVERY_API_LOCK 指向的文件锁共享同一个时间槽。
_LIMITER = None
_LIMITER_LOCK = threading.Lock()


def get_limiter():
    global _LIMITER
    with _LIMITER_LOCK:
        if _LIMITER is None:
            _LIMITER = RateLimiter(float(os.environ.get("DISCOVERY_API_RPM", DEFAULT_RPM)),
                                   lock_path=os.environ.get("DISCOVERY_API_LOCK") or None)
    return _LIMITER


def pil_to_base64(image_pil, size=(336, 336)):
    image_pil = image_pil.resize(size)
    buffered = BytesIO()
    image_pil.save(buffered, format="png")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class RateLimiter:
    """固定间隔限速器；给了 lock_path 就跨进程共享同一条时间槽队列。"""

    def __init__(self, rpm, lock_path=None):
        self.interval = 60.0 / rpm
        self.lock_path = lock_path
        self.lock = threading.Lock()
        self.next_slot = 0.0
        if lock_path:
            os.makedirs(os.path.dirname(lock_path), exist_ok=True)

    def _reserve_local(self):
        with self.lock:
            now = time.time()
            wait = self.next_slot - now
            self.next_slot = max(now, self.next_slot) + self.interval
        return wait

    def _reserve_shared(self):
        with open(self.lock_path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            txt = f.read().strip()
            next_slot = float(txt) if txt else 0.0
            now = time.time()
            wait = next_slot - now
            f.seek(0)
            f.truncate()
            f.write(repr(max(now, next_slot) + self.interval))
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)
        return wait

    def acquire(self):
        wait = self._reserve_shared() if self.lock_path else self._reserve_local()
        if wait > 0:
            time.sleep(wait)


class ResponseCache:
    """(model, prompt, 图像) -> 响应 的 jsonl 缓存，命中则不消耗 API 配额。"""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.lock = threading.Lock()
        self.store = {}
        self.hits = 0
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    self.store[rec["key"]] = rec["response"]

    @staticmethod
    def make_key(model, prompt, images_b64):
        h = hashlib.sha256()
        h.update(model.encode())
        h.update(prompt.encode())
        for img in images_b64:
            if img is not None:
                h.update(img.encode())
        return h.hexdigest()

    def get(self, key):
        with self.lock:
            if key in self.store:
                self.hits += 1
                return self.store[key]
        return None

    def put(self, key, response, usage, latency, raw=None):
        """只落盘、不进本次运行的查找表：同一状态重新问才能得到新答案。
        否则一旦轨迹卡在没有进展的状态上，请求逐字节重复，缓存会把偶发的绕圈锁成死循环。"""
        with self.lock:
            with open(self.path, "a") as f:
                f.write(json.dumps({"key": key, "response": response, "raw": raw,
                                    "usage": usage, "latency": latency}) + "\n")


class DiscoveryAgent(object):

    def __init__(self, api_key=None, base_url=None, model=None, cache_dir=None):
        api_key = os.environ.get("DISCOVERY_API_KEY", api_key)
        base_url = os.environ.get("DISCOVERY_API_BASE", base_url)
        if api_key is None or base_url is None:
            raise ValueError("set DISCOVERY_API_KEY and DISCOVERY_API_BASE")
        self.model = model or os.environ.get("DISCOVERY_MODEL", "Agents-A1")
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0,
                             timeout=float(os.environ.get("DISCOVERY_API_TIMEOUT", DEFAULT_TIMEOUT)))
        self.limiter = get_limiter()
        self.cache = ResponseCache(os.path.join(cache_dir, "api_cache.jsonl")) if cache_dir else None
        self.system_prompt = ("You are an intelligent robot equipped with cameras and robotic arms, "
                              "your primary task is to observe and interact with the objects on the desktop.")
        self.stats = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "total_latency": 0.0, "rate_limited": 0, "timeouts": 0, "empty_content": 0,
                      "invalid_object": 0}
        self.max_tokens = int(os.environ.get("DISCOVERY_MAX_TOKENS", "32768"))

    def parse_prompt(self, prompt, image_list):
        parts = []
        for i, chunk in enumerate(prompt.split("<image>")):
            parts.append({"type": "text", "text": chunk})
            if i < len(image_list) and image_list[i] is not None:
                parts.append({"type": "image_url",
                              "image_url": {"url": f"data:image/png;base64,{image_list[i]}"}})
        return [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": parts + [{"type": "text", "text": FORMAT_INSTRUCTION}]}]

    def act(self, image, goal_image, inp, next_image=None):
        b64 = []
        for img in (goal_image, image, next_image):
            if img is None:
                b64.append(None)
            else:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                b64.append(pil_to_base64(img))

        key = ResponseCache.make_key(self.model, inp + FORMAT_INSTRUCTION, b64)
        if self.cache:
            cached = self.cache.get(key)
            if cached is not None:
                self.stats["cache_hits"] += 1
                return extract_action(cached)

        messages = self.parse_prompt(inp, b64)
        allowed = allowed_objects(inp)
        retries = 0
        empty = 0
        invalid = 0
        budget = self.max_tokens
        while True:
            self.limiter.acquire()
            t0 = time.time()
            try:
                # 不传 temperature：SenseNova 一旦收到显式 temperature 就会让思维链
                # 烧满 max_tokens 并返回空正文，只有省略该参数才会正常给出 content
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=budget)
            except (RateLimitError, APITimeoutError) as e:
                # 429 来自共享 channel 限流，超时来自服务端长尾；两者都必须在这里重试，
                # 否则异常会冒到 run.py:335 的兜底 except，把整条轨迹直接判成失败
                kind = "429" if isinstance(e, RateLimitError) else "timeout"
                self.stats["rate_limited" if kind == "429" else "timeouts"] += 1
                retries += 1
                if retries >= MAX_RETRIES:
                    raise RuntimeError(f"DISCOVERY_API_GIVEUP {self.model}: 连续 {retries} 次 {kind}")
                time.sleep(min(60.0, 5.0 * 2 ** (retries - 1)))
                continue
            latency = time.time() - t0
            content = resp.choices[0].message.content
            if content:
                raw = content.strip()
                action = extract_action(raw)
                obj = action.split()[-1] if action else ""
                if (not allowed or obj in allowed or action.startswith("done")
                        or invalid >= INVALID_OBJECT_MAX_TRIES):
                    break
                # 模型会报出画面里不存在的颜色，这种动作会被 run.py:322 判为未知物体、
                # 整步作废，所以直接重问一次
                self.stats["invalid_object"] += 1
                invalid += 1
                continue
            # 推理 token 吃满预算导致正文为空；预算只对本次调用加倍，不污染后续调用
            self.stats["empty_content"] += 1
            empty += 1
            if empty >= 5:
                raise RuntimeError(f"DISCOVERY_API_GIVEUP {self.model}: 连续 {empty} 次空正文 "
                                   f"(max_tokens={budget})")
            budget = min(budget * 2, 65536)
        self.stats["calls"] += 1
        self.stats["total_latency"] += latency
        if resp.usage:
            self.stats["prompt_tokens"] += resp.usage.prompt_tokens
            self.stats["completion_tokens"] += resp.usage.completion_tokens
        if self.cache:
            usage = resp.usage.model_dump() if resp.usage else None
            self.cache.put(key, action, usage, latency, raw)
        return action
