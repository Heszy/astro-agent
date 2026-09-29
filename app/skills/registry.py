from __future__ import annotations

from typing import Any

from app.skills.base import ScientificSkill


class SkillRegistry:
    """Registry for named workflows; names are the only model-facing choices."""

    def __init__(self, skills: list[ScientificSkill] | None = None):
        self._skills = {skill.name: skill for skill in skills or []}

    def register(self, skill: ScientificSkill) -> None:
        if skill.name in self._skills:
            raise ValueError(f"Skill already registered: {skill.name}")
        self._skills[skill.name] = skill

    def execute(self, skill_name: str, **arguments: Any) -> dict[str, Any]:
        skill = self._skills.get(skill_name)
        if skill is None:
            raise ValueError(f"Unknown skill: {skill_name}")
        return skill.run(**arguments)

    def describe(self) -> list[dict[str, Any]]:
        return [self._skills[name].describe() for name in sorted(self._skills)]
