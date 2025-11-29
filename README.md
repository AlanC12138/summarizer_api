![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/python-3.11-blue.svg?style=for-the-badge&logo=python)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)
![Status](https://img.shields.io/badge/status-active-success?style=for-the-badge)

```markdown
# AI Document Summarizer

End-to-end NLP module: model inference (BART/T5), FastAPI service, Docker packaging, and basic evaluation.

## Features
- Pretrained abstractive summarization (Hugging Face `transformers`)
- FastAPI endpoints: `/health`, `/summarize`
- Batch + single text support
- Deterministic runs (`seed`), CPU/GPU compatible
- Dockerized deployment

## Project layout
```

src/nlp/summarizer/
├── api.py                # FastAPI app
├── model.py              # model wrapper (load/run)
├── requirements.txt      # module deps
├── Dockerfile            # service image
├── README.md
└── tests/
└── test_smoke.py

````

## Quickstart (local)
```bash
# from repo root
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r src/nlp/summarizer/requirements.txt

# run API
uvicorn src.nlp.summarizer.api:app --host 0.0.0.0 --port 8000 --reload
````

## API

### Health

```
GET /health  ->  {"status":"ok"}
```

### Summarize

```
POST /summarize
{
  "text": "long article text ...",
  "max_new_tokens": 128,
  "min_new_tokens": 32,
  "temperature": 0.7
}
```

Response:

```
{"summary": "concise summary...","tokens": 116,"latency_ms": 245}
```

cURL:

```bash
curl -s -X POST http://localhost:8000/summarize \
 -H "Content-Type: application/json" \
 -d '{"text":"<long text here>","max_new_tokens":128,"min_new_tokens":32}'
```

## Python usage

```python
from src.nlp.summarizer.model import Summarizer

sm = Summarizer(model_name="facebook/bart-large-cnn", device="auto", seed=42)
out = sm.summarize("your long text", max_new_tokens=128, min_new_tokens=32)
print(out.text, out.tokens, out.latency_ms)
```

## Configuration

Environment variables (optional):

```
MODEL_NAME=facebook/bart-large-cnn   # or t5-small, google/pegasus-xsum, etc.
DEVICE=auto                          # auto | cpu | cuda
SEED=42
MAX_INPUT_TOKENS=2048
```

## Docker

```bash
# build
docker build -t summarizer:latest -f src/nlp/summarizer/Dockerfile .

# run
docker run --rm -p 8000:8000 \
  -e MODEL_NAME=facebook/bart-large-cnn \
  -e DEVICE=auto \
  summarizer:latest
```

## Requirements

```
torch>=2.2
transformers>=4.41
accelerate>=0.30
fastapi>=0.111
uvicorn[standard]>=0.30
pydantic>=2.7
```

## Evaluation (optional)

Use CNN/DailyMail or a custom set. Log ROUGE and latency.

```
src/results/summarizer_metrics.md
| model                    | rouge1 | rougeL | avg_latency_ms | notes |
|--------------------------|--------|--------|----------------|-------|
| bart-large-cnn (cpu)     |  -     |   -    |        -       | seed=42
| bart-large-cnn (cuda)    |  -     |   -    |        -       | A100/T4/RTX3060
```

Run example:

```bash
python -m src.nlp.summarizer.model --eval_file data/sample_articles.jsonl --limit 100
```

`sample_articles.jsonl` format: one JSON per line with `{"text": "...", "summary": "..."}`.

## Tests

```bash
pytest -q src/nlp/summarizer/tests/test_smoke.py
```

## Roadmap

* Streaming endpoint (`/summarize/stream`)
* Batch endpoint (`/summarize/batch`)
* Quantization (bitsandbytes / torch.compile)
* Simple web UI (FastAPI + HTML)

