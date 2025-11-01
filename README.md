# AI Document Summarizer

End-to-end NLP module: model inference (BART/T5), FastAPI service, Docker packaging, and basic evaluation.

## Features
- Pretrained abstractive summarization (Hugging Face `transformers`)
- FastAPI endpoints: `/health`, `/summarize`
- Batch + single text support
- Deterministic runs (`seed`), CPU/GPU compatible
- Dockerized deployment

## Project layout

src/nlp/summarizer/
├── api.py # FastAPI app
├── model.py # model wrapper (load/run)
├── requirements.txt # module deps
├── Dockerfile # service image
├── README.md
└── tests/
└── test_smoke.py


## Quickstart (local)
```bash
# from repo root
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r src/nlp/summarizer/requirements.txt

# run API
uvicorn src.nlp.summarizer.api:app --host 0.0.0.0 --port 8000 --reload
