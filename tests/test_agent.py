from pathlib import Path

from app.agent import SYSTEM_PROMPT
from app.catalog import CatalogAnalyzer
from app.harness import AgentHarness, HarnessLimits, ModelTurn, ToolCall
from app.tools import TOOL_SCHEMAS, build_tool_registry


class ScriptedModel:
    def __init__(self, turns):
        self.turns = iter(turns)

    def generate(self, messages, tool_schemas):
        return next(self.turns)


def test_harness_executes_tool_and_returns_audit_trace(tmp_path):
    analyzer = CatalogAnalyzer(Path("data/sample_catalog.csv"), tmp_path)
    model = ScriptedModel(
        [
            ModelTurn(
                assistant_message={"role": "assistant", "content": ""},
                tool_calls=[ToolCall("get_catalog_schema", {})],
            ),
            ModelTurn(
                assistant_message={"role": "assistant", "content": "目录共有720行。"},
                content="目录共有720行。",
            ),
        ]
    )
    harness = AgentHarness(model, TOOL_SCHEMAS, build_tool_registry(analyzer))
    result = harness.run("目录有多少行？", SYSTEM_PROMPT)
    assert result.answer == "目录共有720行。"
    assert result.termination_reason == "final_answer"
    assert result.steps == 2
    assert result.trace[0].name == "get_catalog_schema"
    assert result.trace[0].status == "ok"
    assert result.trace[0].result["rows"] == 720


def test_harness_stops_repeated_actions(tmp_path):
    analyzer = CatalogAnalyzer(Path("data/sample_catalog.csv"), tmp_path)
    repeated = ModelTurn(
        assistant_message={"role": "assistant", "content": ""},
        tool_calls=[ToolCall("get_catalog_schema", {})],
    )
    model = ScriptedModel([repeated, repeated, repeated])
    harness = AgentHarness(
        model,
        TOOL_SCHEMAS,
        build_tool_registry(analyzer),
        HarnessLimits(max_repeated_call=2),
    )
    result = harness.run("不断查看字段", SYSTEM_PROMPT)
    assert result.termination_reason == "repeated_action"
    assert len(result.trace) == 2


def test_harness_stops_after_tool_error_budget(tmp_path):
    analyzer = CatalogAnalyzer(Path("data/sample_catalog.csv"), tmp_path)
    turns = [
        ModelTurn(
            assistant_message={"role": "assistant", "content": ""},
            tool_calls=[ToolCall("missing_tool_1", {})],
        ),
        ModelTurn(
            assistant_message={"role": "assistant", "content": ""},
            tool_calls=[ToolCall("missing_tool_2", {})],
        ),
    ]
    harness = AgentHarness(
        ScriptedModel(turns),
        TOOL_SCHEMAS,
        build_tool_registry(analyzer),
        HarnessLimits(max_tool_errors=2),
    )
    result = harness.run("调用错误工具", SYSTEM_PROMPT)
    assert result.termination_reason == "tool_error_budget"
    assert all(item.status == "error" for item in result.trace)
