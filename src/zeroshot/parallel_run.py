#!/usr/bin/env python3
"""Parallel GPU dispatcher for zero-shot K/V decomposition experiments.

Generates all (method × rank × dynamic) tasks and dispatches them across
available GPUs. When a GPU finishes one task, the next pending task is
launched on it immediately.

Usage:
    python -m zeroshot.parallel_run \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --ranks 64 128 256 512

    python -m zeroshot.parallel_run \
        --model-path Qwen/Qwen3-4B-Instruct-2507 \
        --ranks 64 128 256 512 --gpu-list 0 1 2 3
"""

import argparse
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import gpustat
except ImportError:
    gpustat = None

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("parallel_run")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_METHODS = ["palu", "asvd", "mha2mla", "no-sqrt-care", "care", "svdllm"]
SUPPORTED_METHODS = ["palu", "asvd", "mha2mla", "no-sqrt-care", "care", "svdllm"]
DEFAULT_BENCHMARKS = [
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "piqa",
    "MMLU",
    "openbookqa",
    "race",
    "winogrande",
]


def normalize_dataset_name(name: str) -> str:
    """Normalize dataset aliases for zero-shot script entrypoints."""
    aliases = {"wiki": "wikitext2", "wikitext": "wikitext2"}
    return aliases.get(name.strip().lower(), name.strip().lower())


class GPUPool:
    """Thread-safe GPU allocator that tracks per-GPU task slots."""

    def __init__(
        self,
        gpu_list: list[int],
        min_free_mem_mb: int = 20480,
        max_tasks_per_gpu: int = 3,
        gpu_reuse_cooldown_seconds: float = 45.0,
    ):
        self.gpu_list = set(gpu_list)
        self.min_free_mem_mb = min_free_mem_mb
        self.max_tasks_per_gpu = max_tasks_per_gpu
        self.gpu_reuse_cooldown_seconds = gpu_reuse_cooldown_seconds
        self._in_use_counts: dict[int, int] = {}
        self._last_launch_time: dict[int, float] = {}
        self._lock = threading.Lock()

    def acquire(self, poll_interval: float = 10.0) -> int:
        while True:
            with self._lock:
                try:
                    stats = gpustat.GPUStatCollection.new_query()
                except Exception:
                    time.sleep(poll_interval)
                    continue
                for gpu in stats.gpus:
                    idx = gpu["index"]
                    free_mem_mb = gpu["memory.total"] - gpu["memory.used"]
                    active_tasks = self._in_use_counts.get(idx, 0)
                    since_last_launch = time.time() - self._last_launch_time.get(idx, 0.0)
                    if (
                        idx in self.gpu_list
                        and active_tasks < self.max_tasks_per_gpu
                        and free_mem_mb >= self.min_free_mem_mb
                        and (active_tasks == 0 or since_last_launch >= self.gpu_reuse_cooldown_seconds)
                    ):
                        self._in_use_counts[idx] = active_tasks + 1
                        self._last_launch_time[idx] = time.time()
                        return idx
            time.sleep(poll_interval)

    def release(self, gpu_id: int):
        with self._lock:
            active_tasks = self._in_use_counts.get(gpu_id, 0)
            if active_tasks <= 1:
                self._in_use_counts.pop(gpu_id, None)
            else:
                self._in_use_counts[gpu_id] = active_tasks - 1


def build_tasks(args) -> list[dict]:
    """Generate all (cal_dataset, method, rank, dynamic) task configs."""
    tasks = []
    cal_datasets = args.cal_datasets or [args.cal_dataset]
    cal_datasets = list(dict.fromkeys(normalize_dataset_name(name) for name in cal_datasets))
    requested_methods = args.methods or DEFAULT_METHODS
    for cal_dataset in cal_datasets:
        for rank in args.ranks:
            for method in requested_methods:
                tasks.append(
                    {
                        "cal_dataset": cal_dataset,
                        "method": method,
                        "rank": rank,
                        "dynamic": False,
                    }
                )
                if method in ["no-sqrt-care", "care", "transmla-care"]:
                    tasks.append(
                        {
                            "cal_dataset": cal_dataset,
                            "method": method,
                            "rank": rank,
                            "dynamic": True,
                        }
                    )
    return tasks


def run_task(task: dict, gpu_id: int, args, pool: GPUPool):
    cal_dataset = task["cal_dataset"]
    method = task["method"]
    rank = task["rank"]
    dynamic = task["dynamic"]
    model_short = args.model_path.replace("/", "_")
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    log_dir = output_root / "logs" / model_short
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir = str(output_root / cal_dataset)

    tag = (
        f"{cal_dataset}-{method}-dynamic-rank{rank}"
        if dynamic
        else f"{cal_dataset}-{method}-rank{rank}"
    )
    log_file = log_dir / f"{model_short}-{tag}.log"
    label = f"{method} rank={rank} cal={cal_dataset}" + (" dynamic" if dynamic else "")

    cmd = [
        sys.executable, "-m", "zeroshot.convert",
        "--model-path", args.model_path,
        "--method", method,
        "--rank", str(rank),
        "--cal-dataset", cal_dataset,
        "--cal-max-seqlen", str(args.cal_max_seqlen),
        "--ppl-dataset", normalize_dataset_name(args.ppl_dataset),
        "--output-dir", output_dir,
    ]
    if args.benchmarks:
        cmd.extend(["--benchmarks", *args.benchmarks])
    if getattr(args, "ppl_eval_batch_size", None) is not None:
        cmd.extend(["--ppl-eval-batch-size", str(args.ppl_eval_batch_size)])
    if dynamic:
        cmd.append("--dynamic-rank")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logger.info(f"[GPU {gpu_id}] START  {label}  ->  {log_file.name}")
    t0 = time.time()
    try:
        with open(log_file, "w") as lf:
            proc = subprocess.run(
                cmd, stdout=lf, stderr=subprocess.STDOUT,
                env=env, cwd=str(PROJECT_ROOT),
            )
        elapsed = time.time() - t0
        status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
        logger.info(f"[GPU {gpu_id}] {status}  {label}  ({elapsed:.0f}s)")
    except Exception as e:
        logger.error(f"[GPU {gpu_id}] ERROR  {label}: {e}")
    finally:
        time.sleep(5)
        pool.release(gpu_id)


def main():
    parser = argparse.ArgumentParser(description="Parallel zero-shot K/V decomposition runner")
    parser.add_argument("--model-path", required=True, help="HuggingFace model path")
    parser.add_argument("--ranks", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--gpu-list", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument(
        "--gpu-free-mem-threshold",
        type=int,
        default=20480,
        help="Minimum free GPU memory (MB) required to launch a new task",
    )
    parser.add_argument(
        "--gpu-max-tasks-per-gpu",
        type=int,
        default=3,
        help="Maximum number of concurrent tasks allowed on one GPU",
    )
    parser.add_argument(
        "--gpu-reuse-cooldown-seconds",
        type=float,
        default=45.0,
        help="Wait this many seconds before launching another task on the same GPU",
    )
    parser.add_argument("--cal-dataset", default="alpaca")
    parser.add_argument(
        "--cal-datasets",
        nargs="+",
        default=None,
        help="Run all listed calibration datasets in one sweep (e.g. wiki c4 alpaca ptb)",
    )
    parser.add_argument("--ppl-dataset", default="wikitext2")
    parser.add_argument("--cal-max-seqlen", type=int, default=32)
    parser.add_argument("--output-root", default="outputs/zero-shot")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=SUPPORTED_METHODS,
        help="Methods to sweep (default: palu no-sqrt-care care svdllm)",
    )
    parser.add_argument("--benchmarks", nargs="*", default=DEFAULT_BENCHMARKS,
                        help="lm-eval tasks (omit or pass none to skip benchmarks)")
    parser.add_argument("--ppl-eval-batch-size", type=int, default=None,
                        help="PPL eval batch size (0 to skip PPL eval entirely)")
    args = parser.parse_args()

    if gpustat is None:
        raise RuntimeError(
            "parallel_run.py requires gpustat and its dependencies. "
            "Please install gpustat in the active environment."
        )

    pool = GPUPool(
        args.gpu_list,
        min_free_mem_mb=args.gpu_free_mem_threshold,
        max_tasks_per_gpu=args.gpu_max_tasks_per_gpu,
        gpu_reuse_cooldown_seconds=args.gpu_reuse_cooldown_seconds,
    )
    tasks = build_tasks(args)
    cal_datasets = args.cal_datasets or [args.cal_dataset]
    cal_datasets = list(dict.fromkeys(normalize_dataset_name(name) for name in cal_datasets))

    logger.info(f"Model:  {args.model_path}")
    logger.info(f"Ranks:  {args.ranks}")
    logger.info(f"Cal datasets: {cal_datasets}")
    logger.info(f"PPL dataset: {normalize_dataset_name(args.ppl_dataset)}")
    logger.info(f"Methods: {args.methods or DEFAULT_METHODS}")
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    logger.info(f"Output root: {output_root}")
    logger.info(f"GPUs:   {sorted(args.gpu_list)}")
    logger.info(f"Min free GPU memory: {args.gpu_free_mem_threshold} MB")
    logger.info(f"Max tasks per GPU: {args.gpu_max_tasks_per_gpu}")
    logger.info(f"GPU reuse cooldown: {args.gpu_reuse_cooldown_seconds:.0f}s")
    logger.info(f"Tasks:  {len(tasks)} total")
    logger.info("=" * 60)

    threads = []
    for task in tasks:
        gpu_id = pool.acquire()
        t = threading.Thread(target=run_task, args=(task, gpu_id, args, pool))
        t.start()
        threads.append(t)
        time.sleep(2)

    for t in threads:
        t.join()

    logger.info("=" * 60)
    logger.info("All tasks complete.")


if __name__ == "__main__":
    main()
