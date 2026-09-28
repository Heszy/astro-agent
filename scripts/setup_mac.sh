#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v python3 >/dev/null 2>&1; then
  echo "未找到 python3。请先执行：brew install python@3.12"
  exit 1
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
python scripts/generate_sample_data.py
pytest -q

echo
echo "Python环境与数据层已就绪。下一步执行："
echo "  cp .env.example .env"
echo "  编辑 .env，填入 DEEPSEEK_API_KEY"
echo "  source .venv/bin/activate"
echo "  uvicorn app.api:app --reload"
