from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from app.schemas import ToolTrace


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = ""


@dataclass
class ModelTurn:
    """Normalized model output, independent of the backend."""

    assistant_message: dict[str, Any]
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


class ModelAdapter(Protocol):
    def generate(
        self, messages: list[dict[str, Any]], tool_schemas: list[dict[str, Any]]
    ) -> ModelTurn: ...


@dataclass(frozen=True)
class HarnessLimits:
    max_steps: int = 8
    max_tool_errors: int = 3
    max_repeated_call: int = 2
    max_wall_seconds: float = 180.0
    max_observation_chars: int = 20_000


@dataclass
class HarnessResult:
    answer: str
    trace: list[ToolTrace]
    steps: int
    termination_reason: str
    api_usage: dict[str, int]


class AgentHarness:
    """Explicit agent runtime owned by this project, not by the model backend."""

    def __init__(
        self,
        model: ModelAdapter,
        tool_schemas: list[dict[str, Any]],
        tool_registry: dict[str, Callable[..., dict[str, Any]]],
        limits: HarnessLimits | None = None,
    ):
        self.model = model
        self.tool_schemas = tool_schemas
        self.tool_registry = tool_registry
        self.limits = limits or HarnessLimits()

    @staticmethod
    def _call_signature(call: ToolCall) -> str:
        return json.dumps(
            {"name": call.name, "arguments": call.arguments},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    def _observation(self, call: ToolCall, result: dict[str, Any]) -> dict[str, Any]:
        text = json.dumps(result, ensure_ascii=False, default=str)
        if len(text) > self.limits.max_observation_chars:
            text = text[: self.limits.max_observation_chars] + "...<truncated>"
        observation = {"role": "tool", "content": text}
        if call.id:
            observation["tool_call_id"] = call.id
        return observation

    def run(self, question: str, system_prompt: str) -> HarnessResult:
        started = time.perf_counter()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        trace: list[ToolTrace] = []
        repeated_calls: dict[str, int] = {}
        tool_errors = 0
        api_usage: dict[str, int] = {}

        for step in range(1, self.limits.max_steps + 1):
            if time.perf_counter() - started > self.limits.max_wall_seconds:
                return HarnessResult(
                    "任务超过运行时间预算，已安全终止。",
                    trace,
                    step - 1,
                    "wall_time_budget",
                    api_usage,
                )

            turn = self.model.generate(messages, self.tool_schemas)
            for key, value in turn.usage.items():
                if isinstance(value, int):
                    api_usage[key] = api_usage.get(key, 0) + value
            messages.append(turn.assistant_message)
            if not turn.tool_calls:
                return HarnessResult(
                    turn.content or "模型没有给出最终答案。",
                    trace,
                    step,
                    "final_answer",
                    api_usage,
                )

            for call in turn.tool_calls:
                signature = self._call_signature(call)
                repeated_calls[signature] = repeated_calls.get(signature, 0) + 1
                if repeated_calls[signature] > self.limits.max_repeated_call:
                    return HarnessResult(
                        f"检测到重复工具调用，已终止：{call.name}",
                        trace,
                        step,
                        "repeated_action",
                        api_usage,
                    )

                tool_started = time.perf_counter()
                status = "ok"
                if call.name not in self.tool_registry:
                    result = {"error": f"Unknown tool: {call.name}"}
                    status = "error"
                elif not isinstance(call.arguments, dict):
                    result = {"error": "Tool arguments must be a JSON object"}
                    status = "error"
                elif "__parse_error__" in call.arguments:
                    result = {"error": "Model returned invalid JSON tool arguments"}
                    status = "error"
                else:
                    try:
                        result = self.tool_registry[call.name](**call.arguments)
                    except Exception as exc:
                        result = {"error": str(exc)}
                        status = "error"

                trace.append(
                    ToolTrace(
                        step=step,
                        name=call.name,
                        arguments=call.arguments,
                        result=result,
                        status=status,
                        elapsed_seconds=time.perf_counter() - tool_started,
                    )
                )
                messages.append(self._observation(call, result))

                if status == "error":
                    tool_errors += 1
                    if tool_errors >= self.limits.max_tool_errors:
                        return HarnessResult(
                            "工具失败次数达到预算，已安全终止。",
                            trace,
                            step,
                            "tool_error_budget",
                            api_usage,
                        )

        return HarnessResult(
            "达到最大决策步数，任务未能可靠完成。请缩小问题范围。",
            trace,
            self.limits.max_steps,
            "step_budget",
            api_usage,
        )
