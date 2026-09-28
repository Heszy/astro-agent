from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.agent import AstroAgent
from app.catalog import CatalogAnalyzer, CatalogError
from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse, HealthResponse


settings = get_settings()
app = FastAPI(
    title="AstroAgent",
    version="0.1.0",
    description="A verifiable local LLM agent for molecular-cloud catalog analysis.",
)


def get_analyzer() -> CatalogAnalyzer:
    return CatalogAnalyzer(settings.astro_data_path, settings.astro_results_dir)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    rows = None
    status = "ok"
    try:
        rows = len(get_analyzer().df)
    except CatalogError:
        status = "data_missing"
    api_key_configured = settings.deepseek_api_key is not None
    return HealthResponse(
        status=status,
        provider="deepseek",
        model=settings.deepseek_model,
        data_path=str(settings.astro_data_path),
        rows=rows,
        api_key_configured=api_key_configured,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        analyzer = get_analyzer()
        if settings.deepseek_api_key is None:
            raise HTTPException(
                status_code=503,
                detail="DEEPSEEK_API_KEY is not configured. Copy .env.example to .env and set it.",
            )
        agent = AstroAgent(
            analyzer=analyzer,
            api_key=settings.deepseek_api_key.get_secret_value(),
            model=settings.deepseek_model,
            base_url=settings.deepseek_base_url,
            thinking=settings.deepseek_thinking,
        )
        return agent.ask(request.question)
    except CatalogError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Agent failed. Check the DeepSeek API key, balance and network: {exc}",
        ) from exc
