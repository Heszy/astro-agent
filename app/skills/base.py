from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class SkillExecution:
    """Stable envelope for a deterministic multi-tool scientific workflow."""

    skill: str
    version: str
    status: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "skill": {"name": self.skill, "version": self.version},
            "status": self.status,
            "steps": self.steps,
            "result": self.result,
            "artifacts": self.artifacts,
            "warnings": self.warnings,
        }


class ScientificSkill(Protocol):
    name: str
    version: str

    def run(self, **arguments: Any) -> dict[str, Any]: ...

    def describe(self) -> dict[str, Any]: ...
