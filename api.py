import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from .model import Summarizer, SummaryOut
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="AI Document Summarizer", version="0.1.0")
app.mount(
    "/ui",
    StaticFiles(directory="src/nlp/summarizer/ui", html=True),
    name="ui",
)

# single global model instance
_sm = Summarizer(
    model_name=os.getenv("MODEL_NAME"),
    device=os.getenv("DEVICE", "auto"),
)

class SummarizeIn(BaseModel):
    text: str = Field(..., min_length=1, description="Raw document text")
    max_new_tokens: int = 128
    min_new_tokens: int = 32
    temperature: float = 1.0

class HealthOut(BaseModel):
    status: str

class SummarizeOut(BaseModel):
    summary: str
    tokens: int
    latency_ms: int

@app.get("/health", response_model=HealthOut)
def health():
    return HealthOut(status="ok")

@app.post("/summarize", response_model=SummarizeOut)
def summarize(payload: SummarizeIn):
    out: SummaryOut = _sm.summarize(
        text=payload.text,
        max_new_tokens=payload.max_new_tokens,
        min_new_tokens=payload.min_new_tokens,
        temperature=payload.temperature,
    )
    return SummarizeOut(summary=out.text, tokens=out.tokens, latency_ms=out.latency_ms)

