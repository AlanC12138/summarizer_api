
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


@dataclass
class SummaryOut:
    text: str
    tokens: int
    latency_ms: int


def _resolve_device(device: str) -> int | str:
    device = (device or "auto").lower()
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return 0 if torch.cuda.is_available() else "cpu"
    # auto
    return 0 if torch.cuda.is_available() else "cpu"


class Summarizer:
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        max_input_tokens: Optional[int] = None,
    ):
        self.model_name = model_name or os.getenv("MODEL_NAME", "facebook/bart-large-cnn")
        self.device = _resolve_device(device or os.getenv("DEVICE", "auto"))
        self.max_input_tokens = int(os.getenv("MAX_INPUT_TOKENS", max_input_tokens or 2048))

        if seed is None:
            seed = int(os.getenv("SEED", "42"))
        torch.manual_seed(seed)

        tok = AutoTokenizer.from_pretrained(self.model_name)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._tok = tok
        self._pipe = pipeline(
            "summarization",
            model=mdl,
            tokenizer=tok,
            device=0 if self.device == 0 else -1,  # -1 = CPU
        )

    def summarize(
        self,
        text: str,
        max_new_tokens: int = 128,
        min_new_tokens: int = 32,
        temperature: float = 1.0,
    ) -> SummaryOut:
        if not isinstance(text, str) or len(text.strip()) == 0:
            return SummaryOut("", 0, 0)

        # truncate inputs to avoid OOM
        inputs = self._tok(
            text,
            truncation=True,
            max_length=self.max_input_tokens,
            return_tensors="pt",
        )
        num_input_tokens = int(inputs["input_ids"].shape[-1])

        t0 = time.time()
        out = self._pipe(
            text,
            do_sample=temperature is not None and temperature > 0 and temperature != 1.0,
            temperature=max(0.1, float(temperature)),
            max_new_tokens=int(max_new_tokens),
            min_new_tokens=int(min_new_tokens),
            clean_up_tokenization_spaces=True,
        )[0]["summary_text"]
        latency_ms = int((time.time() - t0) * 1000)

        # rough token count = input + generated length
        gen_tokens = min_new_tokens if len(out) == 0 else max_new_tokens
        return SummaryOut(out, num_input_tokens + gen_tokens, latency_ms)


# Optional CLI: quick smoke usage
if __name__ == "__main__":
    import argparse, sys, json

    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, default=None, help="Text to summarize")
    ap.add_argument("--model_name", type=str, default=None)
    ap.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--min_new_tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    text = args.text or sys.stdin.read()
    sm = Summarizer(model_name=args.model_name, device=args.device)
    out = sm.summarize(
        text=text,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
    )
    print(json.dumps(out.__dict__, ensure_ascii=False))
