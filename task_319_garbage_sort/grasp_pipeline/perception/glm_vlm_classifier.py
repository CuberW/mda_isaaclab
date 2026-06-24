"""GLM vision-language object naming and waste classification helpers."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from PIL import Image


DEFAULT_GLM_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_GLM_VLM_MODEL = "glm-5v-turbo"
WASTE_CATEGORIES = ("可回收物", "厨余垃圾", "其他垃圾", "有害垃圾")
API_KEY_ENV_NAMES = ("GLM_API_KEY", "ZHIPUAI_API_KEY", "BIGMODEL_API_KEY")


@dataclass(slots=True)
class VlmClassification:
    object_name: str
    waste_category: str
    confidence: float
    reason: str
    raw_text: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.waste_category in WASTE_CATEGORIES and self.confidence > 0.0


def get_glm_api_key() -> str:
    for name in API_KEY_ENV_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise RuntimeError(f"GLM API key is not set. Export one of: {', '.join(API_KEY_ENV_NAMES)}")


def image_to_data_url(image: Image.Image, *, max_size: int = 512) -> str:
    image = image.convert("RGB")
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def _normalize_category(value: str) -> str:
    value = str(value).strip()
    aliases = {
        "可回收垃圾": "可回收物",
        "可回收": "可回收物",
        "厨余": "厨余垃圾",
        "湿垃圾": "厨余垃圾",
        "干垃圾": "其他垃圾",
        "其它垃圾": "其他垃圾",
        "有害": "有害垃圾",
        "recyclable": "可回收物",
        "kitchen": "厨余垃圾",
        "food": "厨余垃圾",
        "other": "其他垃圾",
        "hazard": "有害垃圾",
        "hazardous": "有害垃圾",
    }
    lowered = value.lower()
    if value in WASTE_CATEGORIES:
        return value
    if lowered in aliases:
        return aliases[lowered]
    for key, category in aliases.items():
        if key in value or key in lowered:
            return category
    return value


def _classification_from_payload(payload: dict[str, Any], raw_text: str) -> VlmClassification:
    object_name = str(payload.get("object_name") or payload.get("name") or payload.get("物体名称") or "").strip()
    category = _normalize_category(payload.get("waste_category") or payload.get("category") or payload.get("垃圾类别") or "")
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence > 1.0:
        confidence = confidence / 100.0
    confidence = max(0.0, min(1.0, confidence))
    reason = str(payload.get("reason") or payload.get("理由") or "").strip()
    error = ""
    if not object_name:
        error = "VLM response did not include object_name."
    elif category not in WASTE_CATEGORIES:
        error = f"VLM response category is not one of {WASTE_CATEGORIES}: {category!r}"
    return VlmClassification(
        object_name=object_name or "unknown",
        waste_category=category,
        confidence=confidence,
        reason=reason,
        raw_text=raw_text,
        error=error,
    )


class GlmVlmClassifier:
    """Small REST client for GLM multimodal object naming and waste classification."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_GLM_VLM_MODEL,
        endpoint: str = DEFAULT_GLM_ENDPOINT,
        timeout_s: float = 30.0,
        max_retries: int = 1,
        max_image_size: int = 512,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.max_retries = max(0, int(max_retries))
        self.max_image_size = max_image_size

    def classify_crop(self, image: Image.Image, *, yolo_label: str = "", yolo_confidence: float = 0.0) -> VlmClassification:
        api_key = get_glm_api_key()
        prompt = (
            "你是垃圾分类机器人视觉识别模块。请只根据这张机器人头部相机的物体裁剪图，"
            "识别可见主体的物体名称，并归入且只能归入以下四类之一：可回收物、厨余垃圾、其他垃圾、有害垃圾。"
            "YOLO 的类别名可能不准确，只能作为很弱的参考。"
            "如果主体不是桌面上的待分类小物体，而是桌子、垃圾桶、机器人、地面、墙面或背景，"
            "请将 object_name 写成“非目标物体”，waste_category 写成“其他垃圾”，confidence 写成 0.0。"
            "请返回严格 JSON，不要 Markdown，不要额外解释，格式为："
            "{\"object_name\":\"中文物体名\",\"waste_category\":\"四类之一\",\"confidence\":0到1,\"reason\":\"一句话原因\"}。"
            f"YOLO弱参考标签：{yolo_label or '无'}，置信度：{yolo_confidence:.3f}。"
        )
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image, max_size=self.max_image_size)}},
                    ],
                }
            ],
            "temperature": 0.1,
            "stream": False,
        }
        body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
        response_payload = None
        last_error = ""
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                self.endpoint,
                data=body_bytes,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = f"GLM HTTP {exc.code}: {detail}"
                if exc.code in {408, 500, 502, 503, 504} and attempt < self.max_retries:
                    time.sleep(min(2.0 * (attempt + 1), 5.0))
                    continue
                return VlmClassification("unknown", "", 0.0, "", error=last_error)
            except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
                last_error = f"GLM request failed after attempt {attempt + 1}/{self.max_retries + 1}: {exc!r}"
                if attempt < self.max_retries:
                    time.sleep(min(2.0 * (attempt + 1), 5.0))
                    continue
                return VlmClassification("unknown", "", 0.0, "", error=last_error)
            except Exception as exc:
                return VlmClassification("unknown", "", 0.0, "", error=f"GLM request failed: {exc!r}")

        if response_payload is None:
            return VlmClassification("unknown", "", 0.0, "", error=last_error or "GLM request failed without a response.")

        try:
            raw_text = response_payload["choices"][0]["message"]["content"]
            if isinstance(raw_text, list):
                raw_text = "".join(str(item.get("text", item)) for item in raw_text)
            payload = _extract_json(str(raw_text))
            return _classification_from_payload(payload, str(raw_text))
        except Exception as exc:
            return VlmClassification(
                "unknown",
                "",
                0.0,
                "",
                raw_text=json.dumps(response_payload, ensure_ascii=False)[:1000],
                error=f"Could not parse GLM response: {exc!r}",
            )
