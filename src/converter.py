import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from lora_qkv import low_rank_qkv
from modify_config import modify_config
from partial_rope import partial_rope
from utils import get_dataset, prepare_dataloader, prepare_test_dataloader, evaluate_ppl


def run_lm_eval_benchmarks(model, tokenizer, benchmark_names):
    """Run lm-eval-harness benchmarks using the same flow as zero-shot convert."""
    import numpy as np
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    mmlu_subjects = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
        "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
        "college_medicine", "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
        "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology", "high_school_statistics",
        "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
        "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
        "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes",
        "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions",
    ]

    available_tasks = set(TaskManager().all_tasks)
    eval_tasks = []
    has_mmlu = False
    mmlu_mode = None
    non_mmlu_benchmarks = []
    for name in benchmark_names:
        if name == "MMLU":
            has_mmlu = True
            if "mmlu" in available_tasks:
                eval_tasks.append("mmlu")
                mmlu_mode = "unified"
            else:
                legacy_tasks = [f"hendrycksTest-{s}" for s in mmlu_subjects]
                if all(task in available_tasks for task in legacy_tasks):
                    eval_tasks.extend(legacy_tasks)
                    mmlu_mode = "legacy"
                else:
                    print("Skipping MMLU: no compatible task names found in installed lm-eval.")
        else:
            eval_tasks.append(name)
            non_mmlu_benchmarks.append(name)

    if not eval_tasks:
        print("No valid lm-eval tasks to run.")
        return {}

    print(f"Running lm-eval benchmarks: {benchmark_names} ({len(eval_tasks)} tasks total)")

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=1,
        max_batch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=eval_tasks,
        num_fewshot=0,
        batch_size=1,
        max_batch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_cache=None,
    )

    def extract_acc(metric_dict):
        for key in ("acc_norm,none", "acc_norm", "acc,none", "acc"):
            if key in metric_dict:
                return metric_dict[key]
        return 0.0

    benchmark_results = {}
    for name in non_mmlu_benchmarks:
        if name in results["results"]:
            metric_dict = results["results"][name]
            benchmark_results[name] = {
                "acc": extract_acc(metric_dict),
                "acc_stderr": metric_dict.get("acc_stderr", 0.0),
            }

    if has_mmlu and mmlu_mode == "unified":
        mmlu_metrics = results["results"].get("mmlu", {})
        benchmark_results["MMLU"] = {
            "acc": float(extract_acc(mmlu_metrics)),
            "num_subtasks": None,
        }
    elif has_mmlu and mmlu_mode == "legacy":
        legacy_tasks = [f"hendrycksTest-{s}" for s in mmlu_subjects]
        accs = []
        for task in legacy_tasks:
            if task in results["results"]:
                accs.append(float(extract_acc(results["results"][task])))
        benchmark_results["MMLU"] = {
            "acc": float(np.mean(accs)) if accs else 0.0,
            "num_subtasks": len(accs),
        }

    return benchmark_results


def load_model_and_tokenizer(args):
    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map=args.device,
        _attn_implementation="sdpa",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    assert model.config.model_type in ["llama", "qwen2", "qwen3", "mistral", "mimo"] or not args.deepseek_style

    return model, tokenizer


def get_dataset_loader(tokenizer: AutoTokenizer, **kwargs):
    cal_dataset = get_dataset(kwargs["cal_dataset"])
    train_loader = prepare_dataloader(
        dataset=cal_dataset["train"],
        tokenizer=tokenizer,
        max_seqlen=kwargs["cal_max_seqlen"],
        batch_size=kwargs["cal_batch_size"],
        nsamples=kwargs["cal_nsamples"],
        seed=kwargs["seed"],
    )
    if kwargs["ppl_eval_batch_size"] > 0:
        ppl_dataset_name = kwargs.get("ppl_dataset", "wikitext2")
        ppl_dataset = get_dataset(ppl_dataset_name)
        test_loader = prepare_test_dataloader(
            dataset=ppl_dataset["test"], tokenizer=tokenizer, batch_size=kwargs["ppl_eval_batch_size"]
        )
    else:
        test_loader = None

    return train_loader, test_loader


def run_conversion(args):
    print("\nOriginal Model")
    model, tokenizer = load_model_and_tokenizer(args)
    train_loader, test_loader = get_dataset_loader(tokenizer, **vars(args))

    if test_loader:
        message = "Evaluating original model's ppl"
        dataset_ppl = evaluate_ppl(model, tokenizer.pad_token_id, test_loader, message)
        print(f'Original ppl: {dataset_ppl:.4f}')

    print("\nPartial RoPE Model")
    if args.collapse == "auto":
        head_dim = model.config.head_dim if hasattr(model.config, "head_dim") and model.config.head_dim is not None else model.config.hidden_size // model.config.num_attention_heads
        model.config.head_dim = head_dim
        args.collapse = head_dim // args.qk_mqa_dim
        print(f"Auto collapse: {args.collapse} (head_dim={head_dim} / qk_mqa_dim={args.qk_mqa_dim})")
    else:
        args.collapse = int(args.collapse)

    model = partial_rope(model, tokenizer, train_loader, test_loader, **vars(args))
    if args.freqfold == "auto":
        args.freqfold = model[1]
        model = model[0]

    print("\nLoraQKV Model")
    model = low_rank_qkv(model, tokenizer, train_loader, test_loader, **vars(args))

    if getattr(args, "run_lm_eval", False):
        print("\nLM-Eval Benchmarks")
        benchmark_results = run_lm_eval_benchmarks(model, tokenizer, args.benchmarks)
        if benchmark_results:
            for name, res in benchmark_results.items():
                acc = res.get("acc", 0.0)
                stderr = res.get("acc_stderr", "")
                stderr_str = f" +/- {stderr:.4f}" if isinstance(stderr, float) and stderr else ""
                print(f"  {name}: {acc:.4f}{stderr_str}")
            print()

    if getattr(args, "dynamic_rank", False):
        print(
            "\nSkipping save: dynamic per-layer KV ranks are currently evaluation-only "
            "because the exported MLA config still assumes one global kv_lora_rank."
        )
        return

    print(f"\nSaving model and tokenizer to {args.save_path}...")
    model.save_pretrained(os.path.join(args.save_path))
    tokenizer.save_pretrained(os.path.join(args.save_path))
    modify_config(model, os.path.join(args.save_path, "config.json"), args)
