"""
Zero-shot K/V weight decomposition evaluation.

Decomposes K and V projection weights via SVD-family methods without
Partial RoPE or MLA conversion stages. Measures quality impact of pure
low-rank approximation at various ranks.

Usage:
    python -m zeroshot.convert \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --method svd --rank 256
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    get_dataset,
    prepare_dataloader,
    prepare_test_dataloader,
    evaluate_ppl,
    map_tensors,
)
from lora_qkv import cwsvd_decompose

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SUPPORTED_METHODS = ("palu", "asvd", "mha2mla", "no-sqrt-care", "care", "svdllm")
COVARIANCE_METHODS = ("no-sqrt-care", "care", "svdllm")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot K/V weight decomposition evaluation"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="HuggingFace model path"
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=SUPPORTED_METHODS,
        help="Decomposition method",
    )
    parser.add_argument(
        "--rank", type=int, required=True, help="Target rank for K/V decomposition"
    )
    parser.add_argument(
        "--dynamic-rank",
        action="store_true",
        default=False,
        help="Enable dynamic per-layer rank allocation under a fixed total rank budget",
    )
    parser.add_argument(
        "--min-rank",
        type=int,
        default=None,
        help="Minimum rank per layer when --dynamic-rank is enabled (default: rank//2)",
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        default=None,
        help="Maximum rank per layer when --dynamic-rank is enabled (default: rank*2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/zero-shot",
        help="Output directory (default: outputs/zero-shot)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["bf16", "fp16"],
        help="Model dtype (default: fp16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="CUDA device (default: auto)",
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Damping for covariance-weighted methods (default: 0.01)",
    )
    parser.add_argument(
        "--asvd-alpha",
        type=float,
        default=0.5,
        help="Activation-aware scaling exponent for ASVD (default: 0.5)",
    )
    parser.add_argument(
        "--asvd-scaling-method",
        type=str,
        default="abs_mean",
        choices=["abs_mean", "abs_max"],
        help="Activation statistic used to scale ASVD inputs (default: abs_mean)",
    )
    parser.add_argument(
        "--cal-dataset",
        type=str,
        default="alpaca",
        help="Calibration dataset (default: alpaca)",
    )
    parser.add_argument(
        "--cal-nsamples",
        type=int,
        default=256,
        help="Number of calibration samples (default: 256)",
    )
    parser.add_argument(
        "--cal-max-seqlen",
        type=int,
        default=32,
        help="Max calibration sequence length (default: 32)",
    )
    parser.add_argument(
        "--cal-batch-size",
        type=int,
        default=8,
        help="Calibration batch size (default: 8)",
    )
    parser.add_argument(
        "--ppl-dataset",
        type=str,
        default="wikitext2",
        help="PPL evaluation dataset (default: wikitext2)",
    )
    parser.add_argument(
        "--ppl-eval-batch-size",
        type=int,
        default=2,
        help="PPL eval batch size (default: 2, 0=skip)",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="*",
        default=None,
        help='Space-separated lm-eval task names (e.g. "arc_easy hellaswag piqa MMLU")',
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    return parser.parse_args()


def load_model_and_tokenizer(args):
    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    logger.info(f"Loading model: {args.model_path} (dtype={args.dtype}, device={args.device})")
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
    return model, tokenizer


class _HessianAccumulator:
    """Accumulates input second-moment (Hessian) for a single nn.Linear,
    using the 2/N running-average normalization from the original pipeline."""

    def __init__(self, layer: torch.nn.Linear):
        self.columns = layer.weight.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=layer.weight.device)
        self.nsamples = 0

    def add_batch(self, inp):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = inp.float()
        self.H += 2 / self.nsamples * inp.matmul(inp.t())


def _get_calibration_loader(tokenizer, dataset_name, nsamples, seqlen, seed):
    """Load calibration data as random fixed-length token slices (old-style)."""
    import random
    from datasets import load_dataset as hf_load_dataset
    random.seed(seed)
    if dataset_name == "alpaca":
        ds = hf_load_dataset("tatsu-lab/alpaca", split="train[5%:]")
        enc = tokenizer("\n\n".join(ds["text"]), return_tensors="pt")
    elif dataset_name == "wikitext2":
        ds = hf_load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        enc = tokenizer("\n\n".join(ds["text"]), return_tensors="pt")
    elif dataset_name == "c4":
        ds = hf_load_dataset(
            "allenai/c4", "en",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
            verification_mode="no_checks",
        )
        enc = tokenizer("\n\n".join(ds["text"][:2000]), return_tensors="pt")
    elif dataset_name == "ptb":
        from huggingface_hub import hf_hub_url
        ds = hf_load_dataset(
            "parquet",
            data_files={"train": hf_hub_url(
                repo_id="ptb-text-only/ptb_text_only",
                filename="penn_treebank/train/0000.parquet",
                repo_type="dataset",
                revision="refs/convert/parquet",
            )},
            split="train",
        )
        enc = tokenizer("\n\n".join(ds["sentence"]), return_tensors="pt")
    else:
        raise ValueError(f"Unsupported cal dataset: {dataset_name}")
    loader = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        loader.append(enc.input_ids[:, i : i + seqlen])
    return loader


@torch.no_grad()
def collect_calibration_covariances(model, tokenizer, args):
    """
    Layer-by-layer Hessian collection with propagated hidden states.

    Each layer's Hessian is computed from hidden states that have already
    passed through all preceding (already-decomposed, in the decompose path)
    layers.  This matches the original tmp/ pipeline precision.
    """
    logger.info("Collecting layer-by-layer calibration Hessians...")
    model.eval()
    model.config.use_cache = False
    layers = model.model.layers
    device = model.model.embed_tokens.weight.device

    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)

    cal_loader = _get_calibration_loader(
        tokenizer, args.cal_dataset, args.cal_nsamples, args.cal_max_seqlen, args.seed,
    )

    dtype = next(iter(model.parameters())).dtype
    seqlen = cal_loader[0].shape[-1]
    nsamples = len(cal_loader)
    hidden_size = model.config.hidden_size
    inps = torch.zeros((nsamples, seqlen, hidden_size), dtype=dtype, device=device)
    cache = {"i": 0, "attention_mask": None, "position_embeddings": None}
    original_layer_devices = [next(layer.parameters()).device for layer in layers]
    has_meta = any(d.type == "meta" for d in original_layer_devices)
    keep_layers_on_device = all(d.type == "cuda" for d in original_layer_devices)

    if has_meta:
        raise RuntimeError(
            "Some model layers are on the meta device (CPU/disk offloaded). "
            "The model is too large for the available GPU memory with device_map='auto'. "
            "Try using more GPUs (e.g., CUDA_VISIBLE_DEVICES=0,1,2,3) so the full model "
            "fits in GPU memory, or reduce the model size."
        )

    if keep_layers_on_device:
        logger.info(
            "Calibration Hessian collection: keeping layers on GPU with per-sample forwards",
        )
    else:
        logger.info(
            "Calibration Hessian collection: falling back to CPU offload path with per-sample forwards",
        )

    class _Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            if name == "module":
                return super().__getattr__(name)
            return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            if cache["i"] < nsamples:
                inps[cache["i"]] = inp
                cache["i"] += 1
                cache["attention_mask"] = kwargs.get("attention_mask")
                cache["position_embeddings"] = kwargs.get("position_embeddings")
            raise ValueError

    layers[0] = _Catcher(layers[0])
    for batch in cal_loader:
        try:
            model(batch.to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    attention_mask = cache["attention_mask"]
    position_embeddings = cache["position_embeddings"]

    covariances = {}
    outs = None
    for i in range(len(layers)):
        layer = layers[i] if keep_layers_on_device else layers[i].to(device)
        layer_device = next(layer.parameters()).device

        if inps.device != layer_device:
            inps = inps.to(layer_device, non_blocking=True)
        if outs is None or outs.device != layer_device:
            outs = torch.empty_like(inps, device=layer_device)

        layer_attention_mask = map_tensors(attention_mask, device=layer_device)
        layer_position_embeddings = map_tensors(position_embeddings, device=layer_device)

        k_acc = _HessianAccumulator(layer.self_attn.k_proj)
        v_acc = _HessianAccumulator(layer.self_attn.v_proj)

        def _make_hook(acc):
            def fn(module, inp, out):
                acc.add_batch(inp[0].detach())
            return fn

        hk = layer.self_attn.k_proj.register_forward_hook(_make_hook(k_acc))
        hv = layer.self_attn.v_proj.register_forward_hook(_make_hook(v_acc))

        for j in range(nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=layer_attention_mask,
                position_embeddings=layer_position_embeddings,
                use_cache=False,
            )[0]

        hk.remove()
        hv.remove()

        covariances[i] = k_acc.H if keep_layers_on_device else k_acc.H.cpu()
        logger.info(f"Layer {i}: Hessian collected (nsamples={k_acc.nsamples})")

        if not keep_layers_on_device:
            layers[i] = layer.cpu()
            torch.cuda.empty_cache()

        del layer, k_acc, v_acc, layer_attention_mask, layer_position_embeddings
        inps, outs = outs, inps

    if not keep_layers_on_device:
        for layer in layers:
            layer.to(device)

    return covariances


def collect_calibration_asvd_scales(model, tokenizer, args):
    logger.info("Collecting calibration data for ASVD scales...")
    cal_dataset = get_dataset(args.cal_dataset)
    cal_loader = prepare_dataloader(
        dataset=cal_dataset["train"],
        tokenizer=tokenizer,
        max_seqlen=args.cal_max_seqlen,
        batch_size=args.cal_batch_size,
        nsamples=args.cal_nsamples,
        seed=args.seed,
    )

    scale_stats = {}
    scale_counts = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            x = input[0].detach().float().cpu().reshape(-1, input[0].shape[-1])
            if args.asvd_scaling_method == "abs_mean":
                batch_stats = x.abs().mean(dim=0)
                scale_stats[layer_idx] = (
                    batch_stats if layer_idx not in scale_stats else scale_stats[layer_idx] + batch_stats
                )
                scale_counts[layer_idx] = scale_counts.get(layer_idx, 0) + 1
            else:
                batch_stats = x.abs().amax(dim=0)
                scale_stats[layer_idx] = (
                    batch_stats if layer_idx not in scale_stats else torch.maximum(scale_stats[layer_idx], batch_stats)
                )
        return hook_fn

    for idx, layer in enumerate(model.model.layers):
        h = layer.self_attn.k_proj.register_forward_hook(make_hook(idx))
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(cal_loader, desc="Calibration forward passes"):
            batch = map_tensors(batch, model.model.embed_tokens.weight.device)
            batch.pop("labels", None)
            model(**batch, use_cache=False)

    for h in hooks:
        h.remove()

    scales = {}
    for layer_idx in sorted(scale_stats.keys()):
        stats = scale_stats[layer_idx]
        if args.asvd_scaling_method == "abs_mean":
            stats = stats / max(scale_counts.get(layer_idx, 1), 1)
        scales[layer_idx] = stats.clamp_min(1e-6)
    torch.cuda.empty_cache()
    return scales


def decompose_svd(W, rank):
    """
    Plain SVD decomposition: W_approx = U[:,:rank] @ diag(S[:rank]) @ V[:rank,:].

    Args:
        W: [d_out, d_in] weight matrix
        rank: target rank

    Returns:
        W_approx: [d_out, d_in] low-rank approximation
    """
    original_dtype = W.dtype
    W_f64 = W.double()
    U, S, Vh = torch.linalg.svd(W_f64, full_matrices=False)
    W_approx = (U[:, :rank] * S[:rank].unsqueeze(0)) @ Vh[:rank, :]
    return W_approx.to(original_dtype)


def decompose_mha2mla(W_k, W_v, rank):
    """
    Joint SVD over concatenated K/V weights with a shared right factor.

    This is the zero-shot analogue of sharing one KV latent space:
    stack K and V on the output dimension, compute one rank-r SVD,
    reconstruct, then split the approximation back into K and V.
    """
    original_dtype = W_k.dtype
    W_kv = torch.cat([W_k.double(), W_v.double()], dim=0)
    U, S, Vh = torch.linalg.svd(W_kv, full_matrices=False)
    W_kv_approx = (U[:, :rank] * S[:rank].unsqueeze(0)) @ Vh[:rank, :]
    W_k_approx, W_v_approx = torch.split(W_kv_approx, [W_k.shape[0], W_v.shape[0]], dim=0)
    return W_k_approx.to(original_dtype), W_v_approx.to(original_dtype)


def decompose_cwsvd(W, H, rank, percdamp, decomp_method):
    """
    no-sqrt-care or care decomposition using cwsvd_decompose.

    Args:
        W: [d_out, d_in] weight matrix
        H: [d_in, d_in] covariance matrix
        rank: target rank
        percdamp: damping factor
        decomp_method: one of no-sqrt-care / care

    Returns:
        W_approx: [d_out, d_in] low-rank approximation
    """
    original_dtype = W.dtype
    H = H.to(W.device)
    up, down = cwsvd_decompose(
        W,
        H.clone(),
        rank,
        percdamp=percdamp,
        decomp_method=decomp_method,
    )
    W_approx = up @ down
    return W_approx.to(original_dtype)


def asvd_decompose(W, scaling_diag_matrix, rank, alpha=0.5):
    """Activation-aware SVD decomposition used only by zero-shot evaluation."""
    original_dtype = W.dtype
    W_f64 = W.double()
    scale = scaling_diag_matrix.to(device=W.device, dtype=torch.float64).clamp_min(1e-6)
    scale = scale.pow(alpha)

    W_scale = W_f64 * scale.unsqueeze(0)
    U, S, Vh = torch.linalg.svd(W_scale, full_matrices=False)
    up = (U[:, :rank] * S[:rank].unsqueeze(0)).to(original_dtype).contiguous()
    down = (Vh[:rank, :] / scale.unsqueeze(0)).to(original_dtype).contiguous()

    logger.info(
        "[ASVD] alpha=%.4f, up: absmax=%.4e, down: absmax=%.4e",
        alpha,
        torch.max(torch.abs(up)).item(),
        torch.max(torch.abs(down)).item(),
    )
    return up, down


def decompose_asvd(W, scale, rank, alpha):
    """Activation-aware SVD decomposition using calibration-derived input scales."""
    original_dtype = W.dtype
    up, down = asvd_decompose(W, scale, rank, alpha=alpha)
    W_approx = up @ down
    return W_approx.to(original_dtype)


def _damped_covariance(H, percdamp, device, dtype):
    """Apply the same diagonal damping convention used elsewhere."""
    H = H.to(device=device, dtype=dtype).clone()
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1], device=device)
    H[diag, diag] += damp
    return H


def _stable_cholesky(H):
    """Cholesky with a small jitter fallback for sample covariances."""
    chol, info = torch.linalg.cholesky_ex(H)
    if int(info.max()) == 0:
        return chol

    mean_diag = torch.mean(torch.diag(H)).clamp_min(torch.tensor(1e-8, device=H.device, dtype=H.dtype))
    eye = torch.eye(H.shape[-1], device=H.device, dtype=H.dtype)
    jitter = mean_diag * 1e-6
    for _ in range(6):
        chol, info = torch.linalg.cholesky_ex(H + jitter * eye)
        if int(info.max()) == 0:
            return chol
        jitter *= 10

    evals = torch.linalg.eigvalsh(H)
    shift = (-torch.min(evals)).clamp_min(0) + mean_diag * 1e-6
    return torch.linalg.cholesky(H + shift * eye)


def decompose_svdllm(W, H, rank, percdamp):
    """
    Minimal SVD-LLM-style whitening decomposition.

    This follows the original repo's core idea for one weight matrix:
    build a Cholesky whitening factor L from the input covariance, run SVD
    on W @ L, then map the right factor back with L^{-1}.
    """
    original_dtype = W.dtype
    W_f64 = W.double()
    H_f64 = _damped_covariance(H, percdamp, W.device, torch.float64)
    chol = _stable_cholesky(H_f64)
    W_scale = W_f64 @ chol

    U, S, Vh = torch.linalg.svd(W_scale, full_matrices=False)
    try:
        chol_inv = torch.linalg.inv(chol)
    except RuntimeError:
        chol_inv = torch.linalg.pinv(chol)

    right = Vh[:rank, :] @ chol_inv
    W_approx = (U[:, :rank] * S[:rank].unsqueeze(0)) @ right
    return W_approx.to(original_dtype)


def _get_hw_matrix_for_ranking(W, H, percdamp, use_sqrt):
    """Build matrix whose singular values drive dynamic rank allocation."""
    H = H.to(W.device, dtype=torch.float32).clone()
    damp_multiplier = 10.0 if use_sqrt else 1.0
    damp = percdamp * torch.mean(torch.diag(H)) * damp_multiplier
    diag = torch.arange(H.shape[-1], device=H.device)
    H[diag, diag] += damp

    if use_sqrt:
        evals, evecs = torch.linalg.eigh(H)
        evals = torch.clamp(evals, min=1e-8)
        sqrt_h = (evecs * torch.sqrt(evals).unsqueeze(0)) @ evecs.mT
        return sqrt_h @ W.float().mT
    return H @ W.float().mT


def _get_svdllm_matrix_for_ranking(W, H, percdamp):
    """Singular values of W @ chol(H) drive SVD-LLM rank allocation."""
    H = _damped_covariance(H, percdamp, W.device, torch.float32)
    chol = _stable_cholesky(H)
    return chol.mT @ W.float().mT


def _get_asvd_matrix_for_ranking(W, scale, alpha):
    """Singular values of the act-aware scaled weight drive ASVD rank allocation."""
    scale = scale.to(W.device, dtype=torch.float32).clamp_min(1e-6).pow(alpha)
    return (W.float() * scale.unsqueeze(0)).mT


def _collect_layer_singular_values(model, args, covariances=None, asvd_scales=None):
    """Collect per-layer singular values for K/V rank allocation."""
    k_svals = []
    v_svals = []

    for idx, layer in enumerate(model.model.layers):
        W_k = layer.self_attn.k_proj.weight.data
        W_v = layer.self_attn.v_proj.weight.data

        if args.method == "palu":
            k_s = torch.linalg.svdvals(W_k.float()).cpu()
            v_s = torch.linalg.svdvals(W_v.float()).cpu()
        elif args.method == "asvd":
            scale = asvd_scales[idx]
            k_s = torch.linalg.svdvals(_get_asvd_matrix_for_ranking(W_k, scale, args.asvd_alpha)).cpu()
            v_s = torch.linalg.svdvals(_get_asvd_matrix_for_ranking(W_v, scale, args.asvd_alpha)).cpu()
        elif args.method == "mha2mla":
            kv_s = torch.linalg.svdvals(torch.cat([W_k.float(), W_v.float()], dim=0)).cpu()
            k_s = kv_s
            v_s = kv_s
        elif args.method == "svdllm":
            H = covariances[idx]
            k_s = torch.linalg.svdvals(_get_svdllm_matrix_for_ranking(W_k, H, args.percdamp)).cpu()
            v_s = torch.linalg.svdvals(_get_svdllm_matrix_for_ranking(W_v, H, args.percdamp)).cpu()
        else:
            H = covariances[idx]
            Y_k = _get_hw_matrix_for_ranking(
                W_k, H, args.percdamp, use_sqrt=(args.method == "care")
            )
            Y_v = _get_hw_matrix_for_ranking(
                W_v, H, args.percdamp, use_sqrt=(args.method == "care")
            )
            k_s = torch.linalg.svdvals(Y_k).cpu()
            v_s = torch.linalg.svdvals(Y_v).cpu()
        k_svals.append(k_s)
        v_svals.append(v_s)

    return k_svals, v_svals


def _prepare_spectrum_stats(singular_values_list):
    """Precompute squared-spectrum statistics for dynamic rank scoring."""
    stats = []
    for svals in singular_values_list:
        svals_np = svals.detach().cpu().numpy().astype(np.float64, copy=False)
        sq_svals = np.square(svals_np)
        prefix_energy = np.concatenate(([0.0], np.cumsum(sq_svals)))
        total_energy = float(prefix_energy[-1]) if len(prefix_energy) > 0 else 0.0
        stats.append(
            {
                "sq_svals": sq_svals,
                "prefix_energy": prefix_energy,
                "total_energy": total_energy,
            }
        )
    return stats


def _normalized_propagation_horizon(layer_idx, num_layers):
    """Return a normalized estimate of how far layer errors propagate downstream."""
    remaining_layers = num_layers - layer_idx
    mean_remaining_layers = (num_layers + 1) / 2.0
    return float(np.sqrt(remaining_layers / mean_remaining_layers))


def _propagated_residual_gain(layer_stats, current_rank, min_rank, layer_idx, num_layers):
    """Score one more rank by cost-normalized propagated residual reduction."""
    sq_svals = layer_stats["sq_svals"]
    total_energy = layer_stats["total_energy"]
    if current_rank >= len(sq_svals) or total_energy <= 0.0:
        return 0.0
    delta_ratio = float(sq_svals[current_rank] / total_energy)
    explained_ratio = float(layer_stats["prefix_energy"][current_rank] / total_energy)
    residual_ratio = max(1.0 - explained_ratio, 1e-12)
    local_residual_gain = delta_ratio / residual_ratio
    propagation_horizon = _normalized_propagation_horizon(layer_idx, num_layers)
    rank_cost = np.sqrt(max(float(current_rank), 1.0) / max(float(min_rank), 1.0))
    return (local_residual_gain * propagation_horizon) / rank_cost


def _allocate_dynamic_ranks(singular_values_list, total_budget, min_rank, max_rank):
    """Greedy budgeted rank allocation using propagated residual reduction."""
    num_layers = len(singular_values_list)
    ranks = [min_rank] * num_layers
    spectrum_stats = _prepare_spectrum_stats(singular_values_list)
    remaining = total_budget - num_layers * min_rank
    if remaining < 0:
        raise ValueError(
            f"Invalid rank bounds: min_rank={min_rank} uses more budget than total={total_budget}"
        )

    for _ in range(remaining):
        best_layer = -1
        best_score = -1.0
        for layer_idx, svals in enumerate(singular_values_list):
            if ranks[layer_idx] >= max_rank:
                continue
            current_rank = ranks[layer_idx]
            score = _propagated_residual_gain(
                spectrum_stats[layer_idx], current_rank, min_rank, layer_idx, num_layers
            )
            if score > best_score:
                best_score = score
                best_layer = layer_idx
        if best_layer == -1:
            break
        ranks[best_layer] += 1
    return ranks


def build_kv_rank_lists(model, args, covariances=None, asvd_scales=None):
    """Return per-layer K/V rank lists (uniform or dynamic)."""
    num_layers = len(model.model.layers)
    if not args.dynamic_rank:
        uniform = [args.rank] * num_layers
        return uniform, uniform

    min_rank = args.min_rank if args.min_rank is not None else max(1, args.rank // 2)
    max_rank = args.max_rank if args.max_rank is not None else args.rank * 2
    if min_rank > max_rank:
        raise ValueError(f"min_rank ({min_rank}) cannot be greater than max_rank ({max_rank})")
    if max_rank * num_layers < args.rank * num_layers:
        raise ValueError("max_rank is too small to satisfy total rank budget")

    if args.method == "asvd" and asvd_scales is None:
        raise ValueError("ASVD scales are required for dynamic rank allocation with ASVD")
    if args.method not in ("palu", "asvd", "mha2mla") and covariances is None:
        raise ValueError("Covariances are required for dynamic rank allocation with CWSVD methods")

    logger.info(
        "Building dynamic ranks with budget=%d, min_rank=%d, max_rank=%d",
        args.rank * num_layers,
        min_rank,
        max_rank,
    )
    k_svals, v_svals = _collect_layer_singular_values(model, args, covariances, asvd_scales)
    total_budget = args.rank * num_layers
    k_ranks = _allocate_dynamic_ranks(k_svals, total_budget, min_rank, max_rank)
    v_ranks = _allocate_dynamic_ranks(v_svals, total_budget, min_rank, max_rank)
    logger.info(
        "Dynamic K ranks: min=%d max=%d mean=%.2f",
        min(k_ranks),
        max(k_ranks),
        float(np.mean(k_ranks)),
    )
    logger.info(
        "Dynamic V ranks: min=%d max=%d mean=%.2f",
        min(v_ranks),
        max(v_ranks),
        float(np.mean(v_ranks)),
    )
    print(f"\n--- Dynamic Rank Distribution (budget={total_budget}, "
          f"min={min_rank}, max={max_rank}) ---")
    print("  Gain metric: ((delta_residual_ratio / current_residual_ratio) * normalized_propagation_horizon) / rank_cost")
    print(f"  K ranks per layer: {k_ranks}")
    print(f"  V ranks per layer: {v_ranks}")
    print()
    return k_ranks, v_ranks


def decompose_kv_weights(model, args, covariances=None, asvd_scales=None, k_rank_list=None, v_rank_list=None):
    """
    Decompose K and V projection weights in-place for all layers.

    Args:
        model: the HuggingFace model
        args: parsed arguments
        covariances: dict of per-layer covariance matrices for covariance-weighted methods
    """
    method = args.method
    rank = args.rank
    num_layers = len(model.model.layers)

    logger.info(f"Decomposing K/V weights: method={method}, rank={rank}, layers={num_layers}")
    if k_rank_list is None or v_rank_list is None:
        k_rank_list, v_rank_list = [rank] * num_layers, [rank] * num_layers

    for idx, layer in enumerate(tqdm(model.model.layers, desc=f"Decomposing ({method}, rank={rank})")):
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj
        W_k = k_proj.weight.data
        W_v = v_proj.weight.data
        k_rank = int(k_rank_list[idx])
        v_rank = int(v_rank_list[idx])

        if method == "palu":
            W_k_approx = decompose_svd(W_k, k_rank)
            W_v_approx = decompose_svd(W_v, v_rank)
        elif method == "asvd":
            scale = asvd_scales[idx]
            W_k_approx = decompose_asvd(W_k, scale, k_rank, args.asvd_alpha)
            W_v_approx = decompose_asvd(W_v, scale, v_rank, args.asvd_alpha)
        elif method == "mha2mla":
            W_k_approx, W_v_approx = decompose_mha2mla(W_k, W_v, k_rank)
        elif method == "no-sqrt-care":
            H = covariances[idx]
            W_k_approx = decompose_cwsvd(W_k, H, k_rank, args.percdamp, "no-sqrt-care")
            W_v_approx = decompose_cwsvd(W_v, H, v_rank, args.percdamp, "no-sqrt-care")
        elif method == "care":
            H = covariances[idx]
            W_k_approx = decompose_cwsvd(
                W_k, H, k_rank, args.percdamp, "care"
            )
            W_v_approx = decompose_cwsvd(
                W_v, H, v_rank, args.percdamp, "care"
            )
        elif method == "svdllm":
            H = covariances[idx]
            W_k_approx = decompose_svdllm(W_k, H, k_rank, args.percdamp)
            W_v_approx = decompose_svdllm(W_v, H, v_rank, args.percdamp)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Replace weights in-place
        k_proj.weight.data = W_k_approx.to(W_k.dtype)
        v_proj.weight.data = W_v_approx.to(W_v.dtype)

        logger.debug(
            f"Layer {idx}: K err={torch.norm(W_k - W_k_approx).item():.4e}, "
            f"V err={torch.norm(W_v - W_v_approx).item():.4e}"
        )

    logger.info("K/V weight decomposition complete.")
    return k_rank_list, v_rank_list


def run_lm_eval_benchmarks(model, tokenizer, benchmark_names):
    """
    Run lm-eval-harness benchmarks using installed lm-eval.

    Returns:
        dict mapping task_name -> {acc, acc_stderr, ...}
    """
    import numpy as np
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    MMLU_SUBJECTS = [
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
        "security_studies", "sociology", "us_foreign_policy", "virology",         "world_religions",
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
                legacy_tasks = [f"hendrycksTest-{s}" for s in MMLU_SUBJECTS]
                if all(task in available_tasks for task in legacy_tasks):
                    eval_tasks.extend(legacy_tasks)
                    mmlu_mode = "legacy"
                else:
                    logger.warning("Skipping MMLU: no compatible task names found in installed lm-eval.")
        else:
            eval_tasks.append(name)
            non_mmlu_benchmarks.append(name)

    if not eval_tasks:
        logger.warning("No valid lm-eval tasks to run.")
        return {}

    logger.info(f"Running lm-eval benchmarks: {benchmark_names} ({len(eval_tasks)} tasks total)")

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
        batch_size=1,  # explicit for reproducibility
        max_batch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_cache=None,
    )

    def _extract_acc(metric_dict):
        for key in ("acc_norm,none", "acc_norm", "acc,none", "acc"):
            if key in metric_dict:
                return metric_dict[key]
        return 0.0

    benchmark_results = {}
    for name in non_mmlu_benchmarks:
        if name in results["results"]:
            metric_dict = results["results"][name]
            benchmark_results[name] = {
                "acc": _extract_acc(metric_dict),
                "acc_stderr": metric_dict.get("acc_stderr", 0.0),
            }

    if has_mmlu and mmlu_mode == "unified":
        mmlu_metrics = results["results"].get("mmlu", {})
        benchmark_results["MMLU"] = {
            "acc": float(_extract_acc(mmlu_metrics)),
            "num_subtasks": None,
        }
    elif has_mmlu and mmlu_mode == "legacy":
        mmlu_tasks = [f"hendrycksTest-{s}" for s in MMLU_SUBJECTS]
        accs = []
        for task in mmlu_tasks:
            if task in results["results"]:
                accs.append(float(_extract_acc(results["results"][task])))
        benchmark_results["MMLU"] = {
            "acc": float(np.mean(accs)) if accs else 0.0,
            "num_subtasks": len(accs),
        }

    return benchmark_results


def save_results(results, args):
    model_short = args.model_path.replace("/", "_")
    results_dir = os.path.join(args.output_dir, "results", model_short)
    os.makedirs(results_dir, exist_ok=True)

    filename = f"{args.method}-rank{args.rank}.json"
    filepath = os.path.join(results_dir, filename)

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {filepath}")
    return filepath


def main():
    args = parse_args()
    start_time = time.time()
    torch.manual_seed(args.seed)

    print("\n" + "=" * 60)
    print("Zero-Shot K/V Decomposition Evaluation".center(60))
    print("=" * 60)
    print(f"  Model:  {args.model_path}")
    print(f"  Method: {args.method}")
    print(f"  Rank:   {args.rank}")
    print("=" * 60 + "\n")

    model, tokenizer = load_model_and_tokenizer(args)

    test_loader = None
    if args.ppl_eval_batch_size > 0:
        ppl_dataset = get_dataset(args.ppl_dataset)
        test_loader = prepare_test_dataloader(
            dataset=ppl_dataset["test"],
            tokenizer=tokenizer,
            batch_size=args.ppl_eval_batch_size,
        )

    original_ppl = None
    if test_loader is not None:
        print("\n--- Original Model PPL ---")
        original_ppl = evaluate_ppl(
            model, tokenizer.pad_token_id, test_loader, "Evaluating original PPL"
        )
        print(f"Original PPL: {original_ppl:.4f}\n")

    covariances = None
    asvd_scales = None
    if args.method == "asvd":
        asvd_scales = collect_calibration_asvd_scales(model, tokenizer, args)
    if args.method in COVARIANCE_METHODS:
        covariances = collect_calibration_covariances(model, tokenizer, args)

    k_rank_list, v_rank_list = build_kv_rank_lists(model, args, covariances=covariances, asvd_scales=asvd_scales)
    k_rank_list, v_rank_list = decompose_kv_weights(
        model, args, covariances=covariances, asvd_scales=asvd_scales, k_rank_list=k_rank_list, v_rank_list=v_rank_list
    )

    if covariances is not None:
        del covariances
        torch.cuda.empty_cache()

    decomposed_ppl = None
    if test_loader is not None:
        print("\n--- Decomposed Model PPL ---")
        decomposed_ppl = evaluate_ppl(
            model, tokenizer.pad_token_id, test_loader, "Evaluating decomposed PPL"
        )
        print(f"Decomposed PPL: {decomposed_ppl:.4f}")
        if original_ppl is not None:
            print(f"PPL increase: {decomposed_ppl - original_ppl:.4f} ({(decomposed_ppl / original_ppl - 1) * 100:.2f}%)\n")

    benchmark_results = None
    if args.benchmarks:
        print("\n--- LM-Eval Benchmarks ---")
        benchmark_results = run_lm_eval_benchmarks(model, tokenizer, args.benchmarks)
        for name, res in benchmark_results.items():
            acc = res.get("acc", 0.0)
            stderr = res.get("acc_stderr", "")
            stderr_str = f" +/- {stderr:.4f}" if isinstance(stderr, float) and stderr else ""
            print(f"  {name}: {acc:.4f}{stderr_str}")
        print()

    elapsed = time.time() - start_time
    results = {
        "model": args.model_path,
        "method": args.method,
        "rank": args.rank,
        "dynamic_rank": args.dynamic_rank,
        "min_rank": args.min_rank,
        "max_rank": args.max_rank,
        "dynamic_rank_policy": "propagated_residual_greedy" if args.dynamic_rank else None,
        "dynamic_rank_gain_metric": (
            "delta_residual_ratio_over_current_residual_ratio_times_normalized_propagation_horizon_over_rank_cost"
            if args.dynamic_rank else None
        ),
        "k_rank_list": k_rank_list if args.dynamic_rank else None,
        "v_rank_list": v_rank_list if args.dynamic_rank else None,
        "dtype": args.dtype,
        "percdamp": args.percdamp,
        "asvd_alpha": args.asvd_alpha if args.method == "asvd" else None,
        "asvd_scaling_method": args.asvd_scaling_method if args.method == "asvd" else None,
        "original_ppl": original_ppl,
        "decomposed_ppl": decomposed_ppl,
        "ppl_dataset": args.ppl_dataset,
        "benchmarks": benchmark_results,
        "cal_dataset": args.cal_dataset,
        "cal_nsamples": args.cal_nsamples,
        "cal_max_seqlen": args.cal_max_seqlen,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }

    filepath = save_results(results, args)

    print("=" * 60)
    print("Summary".center(60))
    print("=" * 60)
    print(f"  Model:          {args.model_path}")
    print(f"  Method:         {args.method}")
    print(f"  Rank:           {args.rank}")
    if original_ppl is not None:
        print(f"  Original PPL:   {original_ppl:.4f}")
    if decomposed_ppl is not None:
        print(f"  Decomposed PPL: {decomposed_ppl:.4f}")
    if benchmark_results:
        for name, res in benchmark_results.items():
            print(f"  {name}: {res.get('acc', 0.0):.4f}")
    print(f"  Elapsed:        {elapsed:.1f}s")
    print(f"  Results:        {filepath}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
