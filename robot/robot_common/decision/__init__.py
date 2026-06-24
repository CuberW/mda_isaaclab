"""
Decision Hub - Task parsing, planning, and VLA policy execution.

Components:
  - LLM Task Parser: Natural language → structured task description
  - Task Planner: DAG-based task decomposition
  - State Machine: Task lifecycle management
  - VLA Interface: Vision-Language-Action model wrapper (OpenVLA, etc.)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from enum import Enum

import numpy as np


# ── Task parsing result ──────────────────────────────────────
class TaskPhase(Enum):
    APPROACH = "approach"
    GRASP = "grasp"
    LIFT = "lift"
    CARRY = "carry"
    PLACE = "place"
    NAVIGATE = "navigate"
    RELEASE = "release"
    SCAN = "scan"
    VERIFY = "verify"


class ArmRole(Enum):
    MASTER = "master"      # Primary arm
    SLAVE = "slave"        # Secondary arm
    SYNC = "sync"          # Synchronized movement
    SINGLE = "single"      # Single arm operation


@dataclass
class SubTask:
    """A single sub-task in the DAG."""
    id: str = ""
    phase: TaskPhase = TaskPhase.APPROACH
    arm: str = "left"       # left | right | both
    arm_role: ArmRole = ArmRole.SINGLE
    target_object: str = ""
    target_position: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    grasp_pose: Optional[np.ndarray] = None
    constraints: dict = field(default_factory=dict)
    depends_on: list = field(default_factory=list)  # Sub-task IDs this depends on
    duration: float = 0.0


@dataclass
class TaskPlan:
    """Structured task plan from LLM or state machine."""
    instruction: str = ""
    task_type: str = ""          # carry | grasp | garbage_sort
    arm_role: ArmRole = ArmRole.SINGLE
    target_objects: list = field(default_factory=list)
    target_region: str = ""
    trash_category: str = ""     # For task 3.19
    grasp_constraints: dict = field(default_factory=dict)
    sub_tasks: list = field(default_factory=list)  # List[SubTask]
    error_recovery: list = field(default_factory=list)


# ── Task Router ──────────────────────────────────────────────
class TaskRouter:
    """Routes natural language instruction to appropriate task pipeline."""

    # Keyword mapping for task classification
    CARRY_KEYWORDS = ["搬运", "搬", "carry", "transport", "move together",
                      "抬起", "拿起", "双手", "一起", "协同", "抱"]
    GRASP_KEYWORDS = ["抓取", "抓", "递给我", "递", "拿", "grasp", "fetch",
                      "pick", "hand me", "看起来像", "像", "那个"]
    GARBAGE_KEYWORDS = ["垃圾", "分类", "回收", "trash", "garbage", "recycle",
                        "sort", "扔掉", "清理", "厨余", "可回收", "有害"]

    @classmethod
    def classify(cls, instruction: str) -> str:
        """
        Classify instruction into task type.

        Returns:
            "dual_arm_vla" | "garbage_sort"
        """
        text = instruction.lower()

        carry_score = sum(1 for kw in cls.CARRY_KEYWORDS if kw.lower() in text)
        grasp_score = sum(1 for kw in cls.GRASP_KEYWORDS if kw.lower() in text)
        garbage_score = sum(1 for kw in cls.GARBAGE_KEYWORDS if kw.lower() in text)

        if garbage_score > carry_score and garbage_score > grasp_score:
            return "garbage_sort"
        elif carry_score > grasp_score:
            return "dual_arm_vla"
        else:
            return "garbage_sort"  # Default; task 3.7/open-vocab grasp has been retired.


# ── LLM Task Parser ──────────────────────────────────────────
class LLMTaskParser:
    """Parse natural language instructions into structured task plans.

    Uses OpenAI-compatible API (DeepSeek, Qwen, OpenAI, etc.).
    Set DEEPSEEK_API_KEY or OPENAI_API_KEY env var, or configure endpoint in YAML.

    Supported API providers:
      - DeepSeek:    https://api.deepseek.com/v1    (recommended, cheap)
      - Qwen:        https://dashscope.aliyuncs.com/compatible-mode/v1
      - OpenAI:      https://api.openai.com/v1
      - Local:       http://localhost:8000/v1       (vLLM/Ollama)
    """

    # System prompt for task parsing
    SYSTEM_PROMPT = """你是一个机器人任务解析器。将用户的自然语言指令解析为JSON格式的任务计划。

输出格式:
{
  "task_type": "carry|garbage_sort",
  "arm_role": "sync|master_slave|single",
  "target_objects": ["物体名"],
  "target_region": "目标区域",
  "trash_category": "可回收|厨余垃圾|有害垃圾|其他垃圾",
  "constraints": {"max_tilt_deg": 5.0, "max_sync_error_m": 0.02}
}

规则:
- "搬运/抬起/抱起/一起" → task_type: "carry", arm_role: "sync"
- "左手扶/右手放" → task_type: "carry", arm_role: "master_slave"
- "垃圾分类/回收/扔掉" → task_type: "garbage_sort", arm_role: "single"
- 中文指令返回中文分类名称

只返回JSON，不要其他文字。"""

    # DeepSeek model tiers (actual API model names)
    DEEPSEEK_MODELS = {
        "flash": "deepseek-v4-flash",     # Fast, cheap
        "pro": "deepseek-v4-pro",         # Highest quality
    }

    def __init__(self, model_name: str = "deepseek-v4-flash", endpoint: str = ""):
        self.model_name = model_name
        self.endpoint = endpoint
        self._client = None
        self._api_available = False
        self._model_tier = "flash"
        self._init_client()

    def _init_client(self):
        """Initialize OpenAI-compatible API client."""
        import os

        # Auto-load .env file if present
        env_file = Path(__file__).resolve().parent.parent.parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        key, val = key.strip(), val.strip()
                        if key not in os.environ:
                            os.environ[key] = val

        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai package not installed. pip install openai")
            return

        # Resolve endpoint and API key
        api_key = None
        base_url = self.endpoint

        if not base_url or not api_key:
            # Try DeepSeek first (cheapest, Chinese-optimized)
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if api_key:
                base_url = base_url or "https://api.deepseek.com/v1"
                self.model_name = self.model_name or "deepseek-chat"
                logger.info("LLM: Using DeepSeek API")
            else:
                # Try OpenAI
                api_key = os.environ.get("OPENAI_API_KEY")
                if api_key:
                    base_url = base_url or "https://api.openai.com/v1"
                    self.model_name = self.model_name or "gpt-4o-mini"
                    logger.info("LLM: Using OpenAI API")
                else:
                    # Try Qwen (DashScope)
                    api_key = os.environ.get("DASHSCOPE_API_KEY")
                    if api_key:
                        base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
                        self.model_name = self.model_name or "qwen-plus"
                        logger.info("LLM: Using Qwen DashScope API")

        if not api_key:
            logger.warning("No API key found. Set DEEPSEEK_API_KEY, OPENAI_API_KEY, "
                          "or DASHSCOPE_API_KEY. Using heuristic parser as fallback.")
            return

        try:
            self._client = OpenAI(api_key=api_key, base_url=base_url)
            self.endpoint = base_url
            self._api_available = True
            logger.info(f"LLM API client ready: {base_url} (model={self.model_name})")
        except Exception as e:
            logger.warning(f"Failed to init API client: {e}")

    def _call_api(self, instruction: str) -> dict:
        """Call LLM API to parse instruction. Returns parsed dict or None."""
        if not self._client or not self._api_available:
            return None

        import json
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            text = response.choices[0].message.content.strip()
            # Extract JSON from response (handle markdown code blocks)
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"LLM returned invalid JSON: {e}")
            return None
        except Exception as e:
            logger.warning(f"LLM API call failed: {e}")
            return None

    def parse(self, instruction: str, task_type: str = "") -> TaskPlan:
        """Parse instruction into structured TaskPlan.

        Tries API first, falls back to heuristic parser.
        """
        task_type = task_type or TaskRouter.classify(instruction)

        # Try API
        api_result = self._call_api(instruction)
        if api_result:
            task_type = api_result.get("task_type", task_type)
            if task_type == "grasp":
                task_type = "garbage_sort"
            arm_role_str = api_result.get("arm_role", "single").upper()
            # Map API responses to enum
            role_map = {
                "SYNC": "SYNC", "SYNCHRONIZED": "SYNC",
                "MASTER_SLAVE": "MASTER", "MASTER": "MASTER",
                "SLAVE": "SLAVE", "SINGLE": "SINGLE",
            }
            arm_role_name = role_map.get(arm_role_str, "SINGLE")
            plan = TaskPlan(
                instruction=instruction,
                task_type=task_type,
                arm_role=ArmRole[arm_role_name],
                target_objects=api_result.get("target_objects", []),
                target_region=api_result.get("target_region", ""),
                trash_category=api_result.get("trash_category", ""),
                grasp_constraints=api_result.get("constraints", {}),
            )
            # Generate subtasks from API result
            plan.sub_tasks = self._generate_subtasks(plan)
            logger.info(f"LLM API parsed: task={task_type}, role={plan.arm_role.name}")
            return plan

        # Fallback to heuristic
        logger.info("Using heuristic parser (no API)")
        return self.parse_heuristic(instruction)

    def _generate_subtasks(self, plan: TaskPlan) -> list:
        """Generate subtask DAG from parsed plan."""
        if plan.task_type == "garbage_sort":
            return self._make_garbage_subtasks(plan)
        elif plan.task_type == "carry":
            return self._make_carry_subtasks(plan)
        return []

    def _make_garbage_subtasks(self, plan: TaskPlan) -> list:
        return [
            SubTask(id="scan", phase=TaskPhase.SCAN, arm="single", target_object="scene"),
            SubTask(id="detect", phase=TaskPhase.APPROACH, arm="right", target_object="all_garbage"),
            SubTask(id="classify", phase=TaskPhase.VERIFY, arm="single", target_object="detected_items"),
            SubTask(id="grasp", phase=TaskPhase.GRASP, arm="right", target_object="target_garbage"),
            SubTask(id="navigate_to_bin", phase=TaskPhase.NAVIGATE, arm="single", target_object="target_bin"),
            SubTask(id="release", phase=TaskPhase.RELEASE, arm="right", target_object="target_bin"),
            SubTask(id="verify_empty", phase=TaskPhase.VERIFY, arm="single", target_object="scene"),
        ]

    def _make_carry_subtasks(self, plan: TaskPlan) -> list:
        if plan.arm_role == ArmRole.SYNC:
            return [
                SubTask(id="parse_roles", phase=TaskPhase.VERIFY, arm="single", target_object="instruction"),
                SubTask(id="detect_object", phase=TaskPhase.APPROACH, arm="single", target_object="target_object"),
                SubTask(id="left_approach", phase=TaskPhase.APPROACH, arm="left", arm_role=ArmRole.MASTER, target_object="target_object"),
                SubTask(id="right_approach", phase=TaskPhase.APPROACH, arm="right", arm_role=ArmRole.SLAVE, target_object="target_object", depends_on=["left_approach"]),
                SubTask(id="dual_grasp", phase=TaskPhase.GRASP, arm="both", arm_role=ArmRole.SYNC, target_object="target_object", depends_on=["left_approach", "right_approach"]),
                SubTask(id="sync_lift", phase=TaskPhase.LIFT, arm="both", arm_role=ArmRole.SYNC, target_object="target_object", depends_on=["dual_grasp"]),
                SubTask(id="sync_carry", phase=TaskPhase.CARRY, arm="both", arm_role=ArmRole.SYNC, target_object="target_region", depends_on=["sync_lift"]),
                SubTask(id="sync_place", phase=TaskPhase.PLACE, arm="both", arm_role=ArmRole.SYNC, target_object="target_region", depends_on=["sync_carry"]),
            ]
        else:
            # Master-slave
            return [
                SubTask(id="left_hold", phase=TaskPhase.GRASP, arm="left", arm_role=ArmRole.MASTER, target_object="container"),
                SubTask(id="right_place", phase=TaskPhase.PLACE, arm="right", arm_role=ArmRole.SLAVE, target_object="item", depends_on=["left_hold"]),
            ]

    def parse_heuristic(self, instruction: str) -> TaskPlan:
        """Heuristic parser (no LLM). Covers all three task types."""
        task_type = TaskRouter.classify(instruction)
        plan = TaskPlan(instruction=instruction, task_type=task_type)

        if task_type == "garbage_sort":
            plan.sub_tasks = self._make_garbage_subtasks(plan)
        elif task_type == "dual_arm_vla":
            # Detect arm role from keywords
            if any(kw in instruction for kw in ["一起", "协同", "同步", "共同", "双手", "搬运"]):
                plan.arm_role = ArmRole.SYNC
                plan.grasp_constraints = {"max_tilt_deg": 5.0, "max_sync_error_m": 0.02}
            elif any(kw in instruction for kw in ["扶", "固定"]):
                plan.arm_role = ArmRole.MASTER
            plan.sub_tasks = self._make_carry_subtasks(plan)

        return plan


# Import logger locally to avoid circular
from robot_common.infra.logging import logger
