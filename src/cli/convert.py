"""CLI entry point for the CARE MLA conversion pipeline."""

from __future__ import annotations

import argparse

from converter import run_conversion

DESCRIPTION = """\
Converts standard GQA/MHA checkpoints into DeepSeek-style MLA checkpoints.

Pipeline stages:
  1. Partial RoPE separation
  2. Low-rank QKV decomposition
  3. Checkpoint save + config rewrite

Examples:
  python -m cli.convert --model-path meta-llama/Llama-3.1-8B-Instruct \
      --save-path outputs/llama-mla --kv-lora-rank 512

  python -m cli.convert --model-path <MODEL> --save-path outputs/llama-mla \
      --kv-lora-rank 512 --kv-decomp-method transmla-care
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CARE MLA converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=DESCRIPTION,
    )

    parser.add_argument("--model-path", type=str, required=True, help="HuggingFace model to convert")
    parser.add_argument("--save-path", type=str, required=True, help="Output directory")
    parser.add_argument("--dtype", type=str, choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument(
        "--cal-dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4", "alpaca"],
        default="wikitext2",
    )
    parser.add_argument("--cal-nsamples", type=int, default=256, help="Number of calibration samples")
    parser.add_argument("--cal-batch-size", type=int, default=8)
    parser.add_argument("--cal-max-seqlen", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--ppl-dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4", "alpaca"],
        default="wikitext2",
        help="Dataset for PPL evaluation (independent of calibration dataset)",
    )
    parser.add_argument("--ppl-eval-batch-size", type=int, default=2, help="Batch size for PPL eval (0 to skip)")
    parser.add_argument(
        "--run-lm-eval",
        action="store_true",
        default=False,
        help="Run lm-eval benchmarks after conversion",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=["arc_easy", "arc_challenge", "hellaswag", "piqa", "MMLU", "winogrande"],
        help="lm-eval tasks to run when --run-lm-eval is enabled",
    )

    parser.add_argument("--freqfold", type=str, default="auto", help="Freqfold for RoPE separation (int or 'auto')")
    parser.add_argument("--collapse", type=str, default="auto", help="Collapse factor (int or 'auto')")
    parser.add_argument("--qk-mqa-dim", type=int, default=64, help="RoPE query/key dimension after folding")

    parser.add_argument("--kv-lora-rank", type=int, default=512, help="KV low-rank target dimension")
    parser.add_argument(
        "--dynamic-rank",
        action="store_true",
        default=False,
        help="Enable dynamic per-layer KV rank allocation under a fixed total budget",
    )
    parser.add_argument(
        "--min-rank",
        type=int,
        default=None,
        help="Minimum per-layer KV rank when --dynamic-rank is enabled (default: rank//2)",
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        default=None,
        help="Maximum per-layer KV rank when --dynamic-rank is enabled (default: rank*2)",
    )
    parser.add_argument(
        "--kv-decomp-method",
        type=str,
        choices=["transmla", "no-sqrt-care", "care", "transmla-care"],
        default="transmla",
        help="KV joint decomposition method",
    )
    parser.add_argument("--cwsvd-percdamp", type=float, default=0.01, help="Damping ratio for CWSVD-family methods")
    parser.add_argument(
        "--cal-mode",
        type=str,
        choices=["auto", "full", "layerwise"],
        default="auto",
        help="Calibration mode: 'full' = single forward pass (TransMLA-style), "
             "'layerwise' = layer-by-layer with propagated hidden states, "
             "'auto' = full for transmla/transmla-care, layerwise for no-sqrt-care/care",
    )
    parser.add_argument("--q-lora-rank", type=int, default=None, help="Q low-rank dim (None = no Q compression)")
    parser.add_argument("--balance-kv-ratio", type=float, default=1, help="KV balance normalization ratio")
    parser.add_argument("--use-qkv-norm", action="store_true", default=False)

    parser.add_argument(
        "--deepseek-style",
        action="store_true",
        default=False,
        help="Use DeepSeek-v3 style modeling/config files",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run_conversion(parse_args(argv))


if __name__ == "__main__":
    main()
