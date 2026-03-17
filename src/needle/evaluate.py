#!/usr/bin/env python3
"""
Needle-in-a-haystack evaluation for CARE-generated MLA checkpoints.

Example:
    python -m needle.evaluate \
        --model-path outputs/qwen2.5-1.5B-mla-care-rank256 \
        --context-lengths 4000 8000 16000 \
        --depth-percents 10 30 50 70 90
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_NEEDLE = (
    "\nThe best thing to do in San Francisco is eat a sandwich and sit in "
    "Dolores Park on a sunny day.\n"
)
DEFAULT_QUESTION = "What is the best thing to do in San Francisco?"


@dataclass
class NeedleResult:
    model: str
    context_length: int
    depth_percent: int
    exact_match: float
    token_recall: float
    response: str
    needle: str
    question: str
    elapsed_seconds: float


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def token_recall_score(prediction: str, answer: str) -> float:
    answer_tokens = normalize_text(answer).split()
    if not answer_tokens:
        return 0.0
    pred_tokens = set(normalize_text(prediction).split())
    hits = sum(token in pred_tokens for token in answer_tokens)
    return hits / len(answer_tokens)


def default_haystack_dir() -> Path | None:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "TMP/CARE/needle/PaulGrahamEssays",
        Path.cwd() / "needle/PaulGrahamEssays",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class CARENeedleTester:
    def __init__(
        self,
        model_path: str,
        needle: str = DEFAULT_NEEDLE,
        retrieval_question: str = DEFAULT_QUESTION,
        haystack_dir: str | None = None,
        output_dir: str = "outputs/needle",
        dtype: str = "bf16",
        device_map: str = "auto",
        attn_implementation: str = "sdpa",
        use_chat_template: bool = False,
        max_new_tokens: int = 64,
    ):
        self.model_path = model_path
        self.needle = needle
        self.retrieval_question = retrieval_question
        self.output_dir = Path(output_dir)
        self.use_chat_template = use_chat_template
        self.max_new_tokens = max_new_tokens

        haystack = Path(haystack_dir) if haystack_dir else default_haystack_dir()
        if haystack is None or not haystack.exists():
            raise FileNotFoundError(
                "No haystack directory found. Please pass --haystack-dir explicitly."
            )
        self.haystack_dir = haystack

        torch_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[dtype]

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            _attn_implementation=attn_implementation,
        ).eval()

    def load_haystack_text(self, min_context_tokens: int) -> str:
        text = ""
        txt_files = sorted(self.haystack_dir.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in {self.haystack_dir}")

        while len(self.tokenizer.encode(text)) < min_context_tokens:
            for file_path in txt_files:
                text += file_path.read_text(errors="ignore")
                if len(self.tokenizer.encode(text)) >= min_context_tokens:
                    break
        return text

    def insert_needle(self, context: str, context_length: int, depth_percent: int) -> str:
        needle_tokens = self.tokenizer.encode(self.needle, add_special_tokens=False)
        context_tokens = self.tokenizer.encode(context, add_special_tokens=False)

        if len(context_tokens) + len(needle_tokens) > context_length:
            context_tokens = context_tokens[: context_length - len(needle_tokens)]

        if depth_percent >= 100:
            merged = context_tokens + needle_tokens
        else:
            insertion_point = int(len(context_tokens) * (depth_percent / 100.0))
            merged = context_tokens[:insertion_point] + needle_tokens + context_tokens[insertion_point:]

        return self.tokenizer.decode(merged, skip_special_tokens=False)

    def build_prompt(self, context: str) -> str:
        user_prompt = (
            "Read the following context carefully and answer the retrieval question.\n\n"
            f"<context>\n{context}\n</context>\n\n"
            f"Question: {self.retrieval_question}\n"
            "Answer briefly and only with the information supported by the context."
        )
        if self.use_chat_template and getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": user_prompt}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return user_prompt

    @torch.no_grad()
    def generate_response(self, prompt: str) -> tuple[str, float]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        model_device = self.model.model.embed_tokens.weight.device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        start = time.time()
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        elapsed = time.time() - start
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return response, elapsed

    def evaluate_once(self, context_length: int, depth_percent: int) -> NeedleResult:
        base_context = self.load_haystack_text(context_length)
        context = self.insert_needle(base_context, context_length, depth_percent)
        prompt = self.build_prompt(context)
        response, elapsed = self.generate_response(prompt)
        exact_match = 1.0 if normalize_text(self.needle) in normalize_text(response) else 0.0
        token_recall = token_recall_score(response, self.needle)
        return NeedleResult(
            model=Path(self.model_path).name,
            context_length=context_length,
            depth_percent=depth_percent,
            exact_match=exact_match,
            token_recall=token_recall,
            response=response,
            needle=self.needle,
            question=self.retrieval_question,
            elapsed_seconds=elapsed,
        )

    def run(self, context_lengths: list[int], depth_percents: list[int]) -> list[NeedleResult]:
        results = []
        for context_length in context_lengths:
            for depth_percent in depth_percents:
                result = self.evaluate_once(context_length, depth_percent)
                print(
                    f"[needle] len={context_length} depth={depth_percent}% "
                    f"exact={result.exact_match:.1f} recall={result.token_recall:.3f} "
                    f"time={result.elapsed_seconds:.1f}s"
                )
                results.append(result)
        return results

    def save_results(self, results: list[NeedleResult]) -> Path:
        model_name = Path(self.model_path).name.replace("/", "_")
        out_dir = self.output_dir / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "needle_results.json"
        with path.open("w") as f:
            json.dump([result.__dict__ for result in results], f, indent=2)
        return path


def parse_args():
    parser = argparse.ArgumentParser(description="Needle-in-a-haystack evaluation for CARE MLA checkpoints")
    parser.add_argument("--model-path", required=True, help="Path to a generated MLA checkpoint under outputs/")
    parser.add_argument("--haystack-dir", default=None, help="Directory of haystack .txt files")
    parser.add_argument("--output-dir", default="outputs/needle", help="Directory to save needle results")
    parser.add_argument("--needle", default=DEFAULT_NEEDLE)
    parser.add_argument("--retrieval-question", default=DEFAULT_QUESTION)
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[4000, 8000, 16000, 32000])
    parser.add_argument("--depth-percents", nargs="+", type=int, default=[0, 25, 50, 75, 100])
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--use-chat-template", action="store_true", default=False)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    tester = CARENeedleTester(
        model_path=args.model_path,
        needle=args.needle,
        retrieval_question=args.retrieval_question,
        haystack_dir=args.haystack_dir,
        output_dir=args.output_dir,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        use_chat_template=args.use_chat_template,
        max_new_tokens=args.max_new_tokens,
    )
    results = tester.run(args.context_lengths, args.depth_percents)
    path = tester.save_results(results)
    print(f"Saved needle results to: {path}")


if __name__ == "__main__":
    main()
