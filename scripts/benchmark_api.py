from __future__ import annotations

import argparse
import concurrent.futures
import statistics
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

from app.config import get_settings


PROMPT = "用三句话解释为什么科学数据分析中的数值计算应由确定性工具完成，而不是由语言模型直接猜测。"


def one_request(client: OpenAI, model: str) -> dict[str, float | int]:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=160,
        stream=False,
        extra_body={"thinking": {"type": "disabled"}},
    )
    wall_seconds = time.perf_counter() - started
    usage = response.usage.model_dump(exclude_none=True) if response.usage else {}
    completion_tokens = int(usage.get("completion_tokens", 0))
    return {
        "wall_seconds": wall_seconds,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": completion_tokens,
        "total_tokens": int(usage.get("total_tokens", 0)),
        "output_tokens_per_second": completion_tokens / wall_seconds if wall_seconds else 0,
        "prompt_cache_hit_tokens": int(usage.get("prompt_cache_hit_tokens", 0)),
        "prompt_cache_miss_tokens": int(usage.get("prompt_cache_miss_tokens", 0)),
    }


def main() -> None:
    settings = get_settings()
    if settings.deepseek_api_key is None:
        raise SystemExit("DEEPSEEK_API_KEY is not configured in .env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()
    client = OpenAI(
        api_key=settings.deepseek_api_key.get_secret_value(),
        base_url=settings.deepseek_base_url,
    )

    records = []
    for concurrency in args.concurrency:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(
                pool.map(
                    lambda _: one_request(client, settings.deepseek_model),
                    range(args.requests),
                )
            )
        latencies = [float(item["wall_seconds"]) for item in results]
        for item in results:
            records.append({"concurrency": concurrency, **item})
        p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 2 else latencies[0]
        print(
            f"concurrency={concurrency}: mean={statistics.mean(latencies):.2f}s, "
            f"p95={p95:.2f}s, total_tokens={sum(int(x['total_tokens']) for x in results)}"
        )

    output = Path("results/deepseek_api_benchmark.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    print(f"Saved raw results to {output}")
    print("Use the current official DeepSeek pricing page to calculate cost; prices can change.")


if __name__ == "__main__":
    main()
