from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.agent import AstroAgent
from app.catalog import CatalogAnalyzer
from app.config import get_settings


CASES = [
    ("这个目录有哪些字段？", "get_catalog_schema"),
    ("三级结构一共有多少个？", "query_structures"),
    ("拟合半径与速度弥散的对数标度关系。", "fit_scaling_relation"),
    ("只使用质量大于1000太阳质量的结构拟合线宽尺度关系。", "fit_scaling_relation"),
    ("按柱密度1e22 cm^-2分成两组，比较线宽尺度关系。", "compare_scaling_relations"),
    ("画出半径与速度弥散的对数散点和拟合图。", "make_plot"),
]


def main() -> None:
    settings = get_settings()
    if settings.deepseek_api_key is None:
        raise SystemExit("DEEPSEEK_API_KEY is not configured in .env")
    analyzer = CatalogAnalyzer(settings.astro_data_path, settings.astro_results_dir)
    agent = AstroAgent(
        analyzer=analyzer,
        api_key=settings.deepseek_api_key.get_secret_value(),
        model=settings.deepseek_model,
        base_url=settings.deepseek_base_url,
        thinking=settings.deepseek_thinking,
    )
    records = []
    for question, expected_tool in CASES:
        response = agent.ask(question)
        used_tools = [item.name for item in response.trace]
        passed = expected_tool in used_tools
        records.append(
            {
                "question": question,
                "expected_tool": expected_tool,
                "used_tools": json.dumps(used_tools, ensure_ascii=False),
                "passed": passed,
                "elapsed_seconds": response.elapsed_seconds,
                "total_tokens": response.api_usage.get("total_tokens", 0),
                "answer": response.answer,
            }
        )
        print(f"{'PASS' if passed else 'FAIL'} | {expected_tool} | {used_tools}")

    output = Path("results/agent_benchmark.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output, index=False)
    print(f"Tool-selection accuracy: {frame['passed'].mean():.1%}")
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
