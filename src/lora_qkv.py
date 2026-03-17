import logging
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave

from utils import pca_calc, get_qkv_calibrate_outputs, evaluate_ppl, statistics_qkv_rmsnorm, sqrtm


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@torch.no_grad()
def second_moment_calc(X: list[torch.Tensor], device: str, add_bias: bool = False) -> torch.Tensor:
    H = None
    for X_batch in X:
        X_batch = X_batch.double().to(device)
        if add_bias:
            ones = torch.ones(*X_batch.shape[:-1], 1, dtype=X_batch.dtype, device=X_batch.device)
            X_batch = torch.cat([X_batch, ones], dim=-1)
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)
        H = H_batch if H is None else H + H_batch
    return H


@torch.no_grad()
def build_separate_kv_ranking_inputs(
    self_attn,
    key_outputs: list[torch.Tensor],
    value_outputs: list[torch.Tensor],
    qk_mqa_dim: int,
    balance_kv_ratio: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build separate K/V ranking operators and moments for dynamic-rank scoring."""
    latent_dim = self_attn.latent_dim
    num_attention_heads = self_attn.num_attention_heads
    head_dim = self_attn.head_dim

    if balance_kv_ratio is not None:
        k_outputs_norm = torch.cat(
            [key.reshape(-1, latent_dim)[:, qk_mqa_dim:] for key in key_outputs]
        ).norm(p=2, dim=0).mean()
        v_outputs_norm = torch.cat(
            [value.reshape(-1, latent_dim) for value in value_outputs]
        ).norm(p=2, dim=0).mean()
        ratio = k_outputs_norm / (v_outputs_norm * balance_kv_ratio)
    else:
        ratio = 1.0

    key_nope_outputs = [key[:, :, qk_mqa_dim:] / ratio for key in key_outputs]
    value_latent_outputs = list(value_outputs)
    H_k = second_moment_calc(key_nope_outputs, self_attn.k_proj.weight.device, add_bias=False)
    H_v = second_moment_calc(value_latent_outputs, self_attn.k_proj.weight.device, add_bias=False)

    k_b_nope_weight = self_attn.k_up_proj.weight.data[:, qk_mqa_dim:].clone() * ratio
    k_b_nope_weight = k_b_nope_weight.view(
        num_attention_heads * head_dim,
        latent_dim - qk_mqa_dim,
    ).to(torch.float64)
    v_b_nope_weight = self_attn.v_up_proj.weight.data.clone().view(
        num_attention_heads * head_dim,
        latent_dim,
    ).to(torch.float64)
    return k_b_nope_weight, H_k, v_b_nope_weight, H_v


@torch.no_grad()
def cwsvd_decompose(
    W: torch.Tensor,
    H: torch.Tensor,
    rank: int,
    percdamp: float = 0.01,
    decomp_method: str = "no-sqrt-care",
) -> tuple[torch.Tensor, torch.Tensor]:
    # W: [d_out, d_in], H: [d_in, d_in]
    dev = W.device
    assert decomp_method in ("no-sqrt-care", "care"), (
        f"Unsupported decomp_method: {decomp_method}"
    )

    if decomp_method == "care":
        # Preserve the old TMP/CARE sqrt-HWSVD path for care:
        # - float32 math
        # - stronger damping (x10)
        # - explicit sqrt(H)
        # - Y = sqrt(H) @ W^T factorization
        W = W.float()
        H = H.float()
        damp = percdamp * torch.mean(torch.diag(H)) * 10
        diag = torch.arange(H.shape[-1], device=dev)
        H[diag, diag] += damp
        use_scipy_sqrt = _env_flag("SQRTM_USE_SCIPY", default=False)
        sqrt_h = sqrtm(H, use_scipy=use_scipy_sqrt)
        hinv = torch.linalg.inv(sqrt_h)
        y = sqrt_h @ W.t()
        U, S, Vh = torch.linalg.svd(y, full_matrices=False)
        U = U[:, :rank]
        Vh = Vh[:rank, :]
        down = hinv @ U @ torch.diag(S[:rank])  # [d_in, rank]
        up = Vh  # [rank, d_out]
    else:
        W = W.float()
        H = H.float()
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[-1], device=dev)
        H[diag, diag] += damp

        try:
            hinv = torch.linalg.inv(H)
        except RuntimeError:
            hinv = torch.linalg.pinv(H)

        y = H @ W.t()  # [d_in, d_out]
        U, S, V = torch.linalg.svd(y, full_matrices=False)
        U = U[:, :rank]
        V = V[:rank, :]
        down = hinv @ U @ torch.diag(S[:rank])  # [d_in, rank]
        up = V.t().contiguous()      # [d_out, rank]
        down = down.t().contiguous()  # [rank, d_in]

    if decomp_method == "care":
        up = up.t().contiguous()      # [d_out, rank]
        down = down.t().contiguous()  # [rank, d_in]

    logging.info(
        f"[{decomp_method}] up: absmax={torch.max(torch.abs(up)).item():.4e}, "
        f"down: absmax={torch.max(torch.abs(down)).item():.4e}"
    )
    if decomp_method == "care":
        logging.info(
            "[care] sqrtm backend: %s",
            "scipy_cpu" if use_scipy_sqrt else "torch_psd_gpu",
        )
    return up, down  # up: [d_out, rank], down: [rank, d_in]


@torch.no_grad()
def _damped_psd_eigh(
    H: torch.Tensor,
    percdamp: float = 0.01,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigendecompose a damped PSD matrix once so SVD-style bases can reuse it."""
    dev = H.device
    H = H.double().clone()
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1], device=dev)
    H[diag, diag] += damp
    evals, evecs = torch.linalg.eigh(H)
    evals = torch.clamp(evals, min=eps)
    return evals, evecs


@torch.no_grad()
def _matrix_power_from_eigh(
    evals: torch.Tensor,
    evecs: torch.Tensor,
    power: float,
) -> torch.Tensor:
    scaled = evecs * evals.pow(power).unsqueeze(0)
    return scaled @ evecs.transpose(-2, -1)


@torch.no_grad()
def svd_project_basis(
    W: torch.Tensor,
    H: torch.Tensor,
    rank: int,
    percdamp: float = 0.01,
    power: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an orthonormal input basis from the left singular space of H^power W^T."""
    evals, evecs = _damped_psd_eigh(H, percdamp)
    H_power = _matrix_power_from_eigh(evals, evecs, power)
    y = H_power @ W.double().t()
    U, S, _ = torch.linalg.svd(y, full_matrices=False)
    basis = U[:, :rank].contiguous()
    svals = S[:rank].contiguous()
    return basis, svals


@torch.no_grad()
def svd_project_singular_values(
    W: torch.Tensor,
    H: torch.Tensor,
    max_rank: int,
    percdamp: float = 0.01,
    power: float = 1.0,
) -> torch.Tensor:
    """Top singular values of the joint SVD ranking operator."""
    evals, evecs = _damped_psd_eigh(H, percdamp)
    H_power = _matrix_power_from_eigh(evals, evecs, power)
    svals = torch.linalg.svdvals(H_power @ W.double().t())
    return svals[:max_rank].detach().cpu()


def _prepare_spectrum_stats(singular_values_list: list[torch.Tensor]):
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


def _normalized_propagation_horizon(layer_idx: int, num_layers: int) -> float:
    remaining_layers = num_layers - layer_idx
    mean_remaining_layers = (num_layers + 1) / 2.0
    return float(np.sqrt(remaining_layers / mean_remaining_layers))


def _propagated_residual_gain(layer_stats, current_rank: int, min_rank: int, layer_idx: int, num_layers: int) -> float:
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


def allocate_joint_dynamic_ranks(
    singular_values_list: list[torch.Tensor],
    total_budget: int,
    min_rank: int,
    max_rank: int,
) -> list[int]:
    """Greedy budgeted rank allocation using propagated residual reduction."""
    num_layers = len(singular_values_list)
    spectrum_stats = _prepare_spectrum_stats(singular_values_list)
    remaining = total_budget - num_layers * min_rank
    if remaining < 0:
        raise ValueError(
            f"Invalid rank bounds: min_rank={min_rank} uses more budget than total={total_budget}"
        )
    ranks = [min_rank] * num_layers

    for _ in range(remaining):
        best_layer = -1
        best_score = -1.0
        for layer_idx, _ in enumerate(singular_values_list):
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


def allocate_separate_branch_ranks(
    k_singular_values_list: list[torch.Tensor],
    v_singular_values_list: list[torch.Tensor],
    total_budget: int,
    branch_min_rank: int,
    branch_max_rank: int,
) -> tuple[list[int], list[int]]:
    """Allocate separate K/V budgets under one shared total latent budget."""
    k_budget = total_budget // 2
    v_budget = total_budget - k_budget
    k_ranks = allocate_joint_dynamic_ranks(
        k_singular_values_list, k_budget, branch_min_rank, branch_max_rank
    )
    v_ranks = allocate_joint_dynamic_ranks(
        v_singular_values_list, v_budget, branch_min_rank, branch_max_rank
    )
    return k_ranks, v_ranks


class LoraQKV(nn.Module):
    def __init__(
        self,
        self_attn,
        query_outputs,
        key_outputs,
        value_outputs,
        q_lora_rank=None,
        qk_mqa_dim=64,
        collapse=1,
        kv_lora_rank=896,
        kv_decomp_method="transmla",
        cwsvd_percdamp=0.01,
        use_qkv_norm=False,
        balance_kv_ratio=None,
        rms_norm_eps=1e-6,
    ):
        super().__init__()
        assert qk_mqa_dim * collapse == self_attn.head_dim

        self.config = self_attn.config
        self.dtype = self_attn.q_proj.weight.dtype
        self.layer_idx = self_attn.layer_idx
        self.num_attention_heads = self_attn.num_attention_heads
        self.head_dim = self_attn.head_dim
        self.qk_mqa_dim = qk_mqa_dim
        self.collapse = collapse
        self.latent_dim = self_attn.latent_dim
        self.attention_dropout = self_attn.attention_dropout
        self.hidden_size = self_attn.hidden_size
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.kv_decomp_method = kv_decomp_method
        self.cwsvd_percdamp = cwsvd_percdamp
        self.new_sqrt_basis_power = float(os.environ.get("NEW_SQRT_BASIS_POWER", "4.0"))
        assert self.kv_lora_rank <= 2 * self.latent_dim - self.qk_mqa_dim, f"kv_lora_rank ({self.kv_lora_rank}) must be less than 2 * latent_dim ({self.latent_dim}) - qk_mqa_dim ({self.qk_mqa_dim})"
        assert self.kv_decomp_method in ["transmla", "no-sqrt-care", "care", "transmla-care"], f"Unknown kv_decomp_method: {self.kv_decomp_method}"

        self.attention_function = ALL_ATTENTION_FUNCTIONS["sdpa"]
        self.scaling = (self.head_dim + self.qk_mqa_dim)**(-0.5)

        q_bias = self_attn.q_proj.bias is not None
        k_bias = self_attn.k_proj.bias is not None
        v_bias = self_attn.v_proj.bias is not None
        assert q_bias == k_bias == v_bias, f"q_bias ({q_bias}), k_bias ({k_bias}), v_bias ({v_bias}) must be the same"
        self.attention_bias = q_bias

        if q_lora_rank is not None:
            self.q_a_proj = nn.Linear(
                self.hidden_size,
                q_lora_rank,
                bias=False,
                device=self_attn.q_proj.weight.device,
                dtype=self.dtype,
            )
            if use_qkv_norm:
                self.q_a_layernorm = nn.RMSNorm(q_lora_rank, device=self_attn.q_proj.weight.device, dtype=self.dtype, eps=rms_norm_eps)
            self.q_b_proj = nn.Linear(
                q_lora_rank,
                self.num_attention_heads * (self.qk_mqa_dim + self.head_dim),
                bias=self.attention_bias,
                device=self_attn.q_proj.weight.device,
                dtype=self.dtype,
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_attention_heads * (self.qk_mqa_dim + self.head_dim),
                bias=self.attention_bias,
                device=self_attn.q_proj.weight.device,
                dtype=self.dtype,
            )
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            kv_lora_rank + qk_mqa_dim,
            bias=self.attention_bias,
            device=self_attn.k_proj.weight.device,
            dtype=self.dtype,
        )
        if use_qkv_norm:
            self.kv_a_layernorm = nn.RMSNorm(kv_lora_rank, device=self_attn.k_proj.weight.device, dtype=self.dtype, eps=rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            self.num_attention_heads * self.head_dim * 2,
            bias=False,
            device=self_attn.k_proj.weight.device,
            dtype=self.dtype,
        )
        self.o_proj = self_attn.o_proj

        if balance_kv_ratio is not None:
            k_outputs_norm = torch.cat([key.reshape(-1, self.latent_dim)[:,self.qk_mqa_dim:] for key in key_outputs]).norm(p=2,dim=0).mean()
            v_outputs_norm = torch.cat([value.reshape(-1, self.latent_dim)[:,self.qk_mqa_dim:] for value in value_outputs]).norm(p=2,dim=0).mean()
            ratio = k_outputs_norm / (v_outputs_norm * balance_kv_ratio)
            self_attn.k_proj.weight.data[self.qk_mqa_dim:] /= ratio
            if self.attention_bias:
                self_attn.k_proj.bias.data[self.qk_mqa_dim:] /= ratio
            self_attn.k_up_proj.weight.data[:, self.qk_mqa_dim:] *= ratio
        else:
            ratio = 1
        key_nope_outputs = [key_outputs[i][:,:,qk_mqa_dim:] / ratio for i in range(len(key_outputs))]
        value_latent_outputs = list(value_outputs)
        kv_outputs = [torch.cat([key_nope_outputs[i], value_latent_outputs[i]], dim=-1) for i in range(len(key_nope_outputs))]

        if self.q_lora_rank is not None:
            R_q = pca_calc(query_outputs, self_attn.q_proj.weight.device)
        else:
            R_q = None
        if self.kv_decomp_method == "transmla":
            R_kv = pca_calc(kv_outputs, self_attn.k_proj.weight.device)
            H_kv = None
        else:
            R_kv = None
            H_kv = second_moment_calc(kv_outputs, self_attn.k_proj.weight.device, add_bias=False)
        self._init_weights(self_attn, R_q, R_kv, H_kv)

    def _init_weights(self, self_attn, R_q, R_kv, H_kv):
        k_a_rope_weight, k_a_nope_weight = self_attn.k_proj.weight.data.split([self.qk_mqa_dim, self.latent_dim - self.qk_mqa_dim], dim=0)
        k_b_rope_weight, k_b_nope_weight = self_attn.k_up_proj.weight.data.split([self.qk_mqa_dim, self.latent_dim - self.qk_mqa_dim], dim=1)
        k_b_rope_weight = k_b_rope_weight.view(self.num_attention_heads, self.head_dim, self.qk_mqa_dim)
        k_b_nope_weight = k_b_nope_weight.view(self.num_attention_heads, self.head_dim, self.latent_dim-self.qk_mqa_dim)

        v_a_nope_weight  = self_attn.v_proj.weight.data
        v_b_nope_weight = self_attn.v_up_proj.weight.data
        v_b_nope_weight = v_b_nope_weight.view(self.num_attention_heads, self.head_dim, self.latent_dim)

        if self.attention_bias:
            q_bias = self_attn.q_proj.bias.data
            v_bias = self_attn.v_proj.bias.data
            k_bias_rope, k_bias_nope = self_attn.k_proj.bias.data.split([self.qk_mqa_dim, self.latent_dim - self.qk_mqa_dim], dim=0)

        original_scaling = getattr(self.config, "query_pre_attn_scalar", self.head_dim)**-0.5
        scaling = original_scaling / self.scaling
        if self.q_lora_rank is not None:
            q_weight = self_attn.q_proj.weight.data.to(torch.float64)

            q_a_weight = (R_q.T @ q_weight)[: self.q_lora_rank].to(self.dtype)
            self.q_a_proj.weight.data = q_a_weight.contiguous()

            q_b_weight = R_q[:, :self.q_lora_rank].to(self.dtype)
            q_b_weight = q_b_weight.view(self.num_attention_heads, self.head_dim, self.q_lora_rank)
            q_b_rope_weight = torch.einsum("hdq,hdk->hkq", q_b_weight, k_b_rope_weight)
            q_b_with_mqa_weight = torch.cat([q_b_weight, q_b_rope_weight], dim=1).reshape(
                self.num_attention_heads * (self.head_dim + self.qk_mqa_dim), self.q_lora_rank
            )
            self.q_b_proj.weight.data = q_b_with_mqa_weight.contiguous() * scaling

        else:
            q_weight = self_attn.q_proj.weight.data.view(self.num_attention_heads, self.head_dim, self.hidden_size)
            q_rope_weight = torch.einsum("hdD,hdk->hkD", q_weight, k_b_rope_weight)
            q_with_mqa_weight = torch.cat([q_weight, q_rope_weight], dim=1).reshape(
                self.num_attention_heads * (self.head_dim + self.qk_mqa_dim), self.hidden_size
            )
            self.q_proj.weight.data = q_with_mqa_weight.contiguous() * scaling

        if self.attention_bias:
            q_bias = q_bias.reshape(self.num_attention_heads, self.head_dim)
            q_rope_bias = torch.einsum("hd,hdk->hk", q_bias.to(torch.float64), k_b_rope_weight.to(torch.float64)).to(self.dtype)
            q_bias = torch.cat([q_bias, q_rope_bias], dim=1).flatten().contiguous() * scaling
            if self.q_lora_rank is not None:
                self.q_b_proj.bias.data = q_bias
            else:
                self.q_proj.bias.data = q_bias

        kv_a_nope_weight = torch.cat([k_a_nope_weight, v_a_nope_weight], dim=0).to(torch.float64)
        if self.attention_bias:
            kv_a_nope_bias = torch.cat([k_bias_nope, v_bias]).unsqueeze(-1).to(torch.float64)
            kv_a_nope_weight = torch.cat([kv_a_nope_weight, kv_a_nope_bias], dim=-1)
        kv_b_nope_weight = torch.cat(
            [
                torch.cat([k_b_nope_weight, torch.zeros_like(v_b_nope_weight)], dim=-1),
                torch.cat([torch.zeros_like(k_b_nope_weight), v_b_nope_weight], dim=-1)
            ],
            dim=1
        ).reshape(2 * self.num_attention_heads * self.head_dim, 2 * self.latent_dim - self.qk_mqa_dim).to(torch.float64)

        if R_kv is not None:
            kv_a_nope_weight = (R_kv.T @ kv_a_nope_weight)[: self.kv_lora_rank].to(self.dtype)
            if self.attention_bias:
                kv_a_nope_weight, kv_a_nope_bias = torch.split(kv_a_nope_weight, [self.hidden_size, 1], dim=-1)
                kv_a_nope_bias = kv_a_nope_bias.flatten().to(self.dtype)
            kv_b_nope_weight = (kv_b_nope_weight @ R_kv)[:, :self.kv_lora_rank].to(self.dtype)
        elif self.kv_decomp_method == "transmla-care":
            assert H_kv is not None, "transmla-care requires joint KV moments"
            R_kv, svals = svd_project_basis(
                kv_b_nope_weight,
                H_kv,
                self.kv_lora_rank,
                self.cwsvd_percdamp,
                power=self.new_sqrt_basis_power,
            )
            print(
                "[transmla-care] joint_svd_projection "
                f"power={self.new_sqrt_basis_power:.2f}, sigma0={svals[0].item():.4e}, "
                f"sigma_last={svals[-1].item():.4e}"
            )
            kv_a_nope_weight = (R_kv.T @ kv_a_nope_weight)[: self.kv_lora_rank].to(self.dtype)
            if self.attention_bias:
                kv_a_nope_weight, kv_a_nope_bias = torch.split(kv_a_nope_weight, [self.hidden_size, 1], dim=-1)
                kv_a_nope_bias = kv_a_nope_bias.flatten().to(self.dtype)
            kv_b_nope_weight = (kv_b_nope_weight @ R_kv)[:, :self.kv_lora_rank].to(self.dtype)
        else:
            assert H_kv is not None, "H_kv is required for CWSVD-family methods"
            kv_b_up, kv_b_down = cwsvd_decompose(
                kv_b_nope_weight,
                H_kv,
                self.kv_lora_rank,
                self.cwsvd_percdamp,
                decomp_method=self.kv_decomp_method,
            )
            kv_b_nope_weight = kv_b_up.to(self.dtype)
            kv_a_nope_weight = (kv_b_down.double() @ kv_a_nope_weight).to(self.dtype)
            if self.attention_bias:
                kv_a_nope_weight, kv_a_nope_bias = torch.split(kv_a_nope_weight, [self.hidden_size, 1], dim=-1)
                kv_a_nope_bias = kv_a_nope_bias.flatten().to(self.dtype)
        self.kv_b_proj.weight.data = kv_b_nope_weight.contiguous()
        kv_a_proj_with_mqa_weight = torch.cat([kv_a_nope_weight, k_a_rope_weight], dim=0)
        self.kv_a_proj_with_mqa.weight.data = kv_a_proj_with_mqa_weight.contiguous()
        if self.attention_bias:
            kv_a_proj_with_mqa_bias = torch.cat([kv_a_nope_bias, k_bias_rope])
            self.kv_a_proj_with_mqa.bias.data = kv_a_proj_with_mqa_bias.contiguous()


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        past_key_values = None,  # compat with transformers >= 4.50
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # compat: merge plural form
        if past_key_value is None and past_key_values is not None:
            past_key_value = past_key_values
        bsz, q_len, _ = hidden_states.size()

        # query
        if self.q_lora_rank is not None:
            query_states = self.q_a_proj(hidden_states)
            if hasattr(self, "q_a_layernorm"):
                query_states = self.q_a_layernorm(query_states)
            query_states = self.q_b_proj(query_states)
        else:
            query_states = self.q_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_attention_heads, -1).transpose(1, 2)
        q_nope, q_rope = query_states.split([self.head_dim, self.qk_mqa_dim], dim=-1)

        # key and value
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_nope, k_rope = compressed_kv.split([self.kv_lora_rank, self.qk_mqa_dim], dim=-1)
        kv_nope = kv_nope.view(bsz, 1, q_len, self.kv_lora_rank)
        k_rope = k_rope.view(bsz, 1, q_len, self.qk_mqa_dim)

        cos, sin = position_embeddings
        q_rope, k_rope = apply_rotary_pos_emb_interleave(q_rope, k_rope, cos[ :, :, : : self.collapse], sin[ :, :, : : self.collapse])
        query_states = torch.cat([q_nope, q_rope], dim=-1)

        if hasattr(self, "kv_a_layernorm"):
            kv_nope = self.kv_a_layernorm(kv_nope)
        kv_nope = self.kv_b_proj(kv_nope).view(bsz, q_len, self.num_attention_heads, self.head_dim * 2).transpose(1, 2)
        k_nope, value_states = kv_nope.split([self.head_dim, self.head_dim], dim=-1)
        key_states = torch.cat([k_nope, repeat_kv(k_rope, self.num_attention_heads)], dim=-1)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output, attn_weights = self.attention_function(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            softcap=getattr(self.config, "attn_logit_softcapping", None)
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


@torch.no_grad()
def _capture_layer_qkv(layer, inps, nsamples, attention_mask, position_embeddings):
    """Forward all samples through one layer, capturing q/k/v proj outputs."""
    query_out, key_out, value_out = [], [], []
    outs = torch.zeros_like(inps)

    def _hook(store):
        def fn(module, inp, out):
            store.append(out.detach().cpu())
        return fn

    hq = layer.self_attn.q_proj.register_forward_hook(_hook(query_out))
    hk = layer.self_attn.k_proj.register_forward_hook(_hook(key_out))
    hv = layer.self_attn.v_proj.register_forward_hook(_hook(value_out))

    seqlen = inps.shape[1]
    if position_embeddings is not None:
        pe = tuple(p[:, :seqlen] if p.shape[1] >= seqlen else p for p in position_embeddings)
    else:
        pe = None

    for j in range(nsamples):
        outs[j] = layer(
            inps[j].unsqueeze(0),
            attention_mask=None,
            position_embeddings=pe,
            use_cache=False,
        )[0]

    hq.remove()
    hk.remove()
    hv.remove()
    return outs, query_out, key_out, value_out


def _capture_first_layer_inputs(model, train_loader):
    """Run calibration data through the embedding layer to capture inputs
    to the first transformer layer, along with attention_mask and
    position_embeddings.  Works like the Catcher pattern in the old pipeline."""
    import torch.nn as nn

    layers = model.model.layers
    device = model.model.embed_tokens.weight.device

    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)

    dtype = next(iter(model.parameters())).dtype
    hidden_size = model.config.hidden_size
    all_inps = []
    cache = {"attention_mask": None, "position_embeddings": None}

    class _Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            if name == "module":
                return super().__getattr__(name)
            return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            all_inps.append(inp.detach())
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_embeddings"] = kwargs.get("position_embeddings")
            raise ValueError

    layers[0] = _Catcher(layers[0])
    model.config.use_cache = False
    for batch in train_loader:
        try:
            if isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
                batch.pop("labels", None)
                model(**batch, use_cache=False)
            else:
                model(batch.to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    if not all_inps:
        raise RuntimeError("No calibration inputs captured")
    seqlen = all_inps[0].shape[1]
    all_inps = [x for x in all_inps if x.shape[1] == seqlen]
    inps = torch.cat(all_inps, dim=0)
    nsamples = inps.shape[0]
    return inps, nsamples, cache["attention_mask"], cache["position_embeddings"]


def low_rank_qkv(model, tokenizer, train_loader, test_loader, **kwargs):
    num_layers = len(model.model.layers)
    dynamic_rank = kwargs.get("dynamic_rank", False)
    layers = model.model.layers
    device = model.model.embed_tokens.weight.device
    kv_decomp_method = kwargs.get("kv_decomp_method", "transmla")
    keep_on_device = all(
        next(layer.parameters()).device.type == device.type for layer in layers
    )

    cal_mode = kwargs.get("cal_mode", "auto")
    if cal_mode == "auto":
        use_layerwise = kv_decomp_method in ("no-sqrt-care", "care") or dynamic_rank
    elif cal_mode == "layerwise":
        use_layerwise = True
    else:
        use_layerwise = dynamic_rank

    if not use_layerwise:
        # ---- TransMLA-style: full forward pass, original-model QKV outputs ----
        message = "Calibrating rope-removed model's qkv outputs"
        rm_rope_qkv_outputs = get_qkv_calibrate_outputs(model, train_loader, message)

        for layer_idx, layer in enumerate(layers):
            print(f"[LoraQKV] Layer {layer_idx}/{num_layers}: {kv_decomp_method} decomposing ...")
            setattr(layer, "self_attn", LoraQKV(
                layer.self_attn,
                rm_rope_qkv_outputs["query"][layer_idx],
                rm_rope_qkv_outputs["key"][layer_idx],
                rm_rope_qkv_outputs["value"][layer_idx],
                q_lora_rank=kwargs["q_lora_rank"],
                qk_mqa_dim=kwargs["qk_mqa_dim"],
                collapse=kwargs["collapse"],
                kv_lora_rank=kwargs["kv_lora_rank"],
                kv_decomp_method=kv_decomp_method,
                cwsvd_percdamp=kwargs["cwsvd_percdamp"],
                use_qkv_norm=kwargs["use_qkv_norm"],
                balance_kv_ratio=kwargs["balance_kv_ratio"],
                rms_norm_eps=model.config.rms_norm_eps,
            ))
        del rm_rope_qkv_outputs
    else:
        # ---- Layer-by-layer: needed for CWSVD-family and dynamic rank ----
        print("Capturing layer inputs for layer-by-layer calibration...")
        inps, nsamples, attention_mask, position_embeddings = _capture_first_layer_inputs(
            model, train_loader,
        )

        if dynamic_rank:
            base_rank = int(kwargs["kv_lora_rank"])
            joint_min_rank = kwargs.get("min_rank")
            joint_max_rank = kwargs.get("max_rank")
            joint_min_rank = int(joint_min_rank) if joint_min_rank is not None else max(1, base_rank // 2)
            joint_max_rank = int(joint_max_rank) if joint_max_rank is not None else base_rank * 2
            if joint_min_rank > joint_max_rank:
                raise ValueError(
                    f"min_rank ({joint_min_rank}) cannot be greater than max_rank ({joint_max_rank})"
                )
            branch_min_rank = max(1, joint_min_rank // 2)
            branch_max_rank = max(branch_min_rank, joint_max_rank // 2)
            power = float(os.environ.get("NEW_SQRT_BASIS_POWER", "4.0"))
            total_budget = base_rank * num_layers

            print("Collecting singular values for dynamic rank allocation...")
            k_svals, v_svals = [], []
            tmp_inps = inps.clone()
            for layer_idx in range(num_layers):
                layer = layers[layer_idx] if keep_on_device else layers[layer_idx].to(device)
                tmp_outs, _, key_out, value_out = _capture_layer_qkv(
                    layer, tmp_inps, nsamples, attention_mask, position_embeddings,
                )
                k_weight, H_k, v_weight, H_v = build_separate_kv_ranking_inputs(
                    layer.self_attn, key_out, value_out,
                    kwargs["qk_mqa_dim"], kwargs["balance_kv_ratio"],
                )
                k_svals.append(svd_project_singular_values(
                    k_weight, H_k, max_rank=branch_max_rank,
                    percdamp=kwargs["cwsvd_percdamp"], power=power,
                ))
                v_svals.append(svd_project_singular_values(
                    v_weight, H_v, max_rank=branch_max_rank,
                    percdamp=kwargs["cwsvd_percdamp"], power=power,
                ))
                if not keep_on_device:
                    layers[layer_idx] = layer.cpu()
                del key_out, value_out
                torch.cuda.empty_cache()
                tmp_inps = tmp_outs
            del tmp_inps, tmp_outs

            k_rank_list, v_rank_list = allocate_separate_branch_ranks(
                k_svals, v_svals, total_budget, branch_min_rank, branch_max_rank
            )
            kv_rank_list = [k + v for k, v in zip(k_rank_list, v_rank_list)]
            print(
                f"\n--- Joint Dynamic Rank Distribution "
                f"(budget={total_budget}, joint_min={joint_min_rank}, joint_max={joint_max_rank}, "
                f"branch_min={branch_min_rank}, branch_max={branch_max_rank}, power={power:.2f}) ---"
            )
            print(f"  k ranks per layer: {k_rank_list}")
            print(f"  v ranks per layer: {v_rank_list}")
            print(f"  kv ranks per layer: {kv_rank_list}")
            print(
                f"  stats: min={min(kv_rank_list)}, max={max(kv_rank_list)}, "
                f"mean={sum(kv_rank_list) / len(kv_rank_list):.2f}"
            )
            print()
        else:
            kv_rank_list = [kwargs["kv_lora_rank"]] * num_layers

        for layer_idx in range(num_layers):
            layer = layers[layer_idx] if keep_on_device else layers[layer_idx].to(device)
            print(f"[LoraQKV] Layer {layer_idx}/{num_layers}: calibrating + decomposing ...")

            layer_input = inps
            outs, query_out, key_out, value_out = _capture_layer_qkv(
                layer, layer_input, nsamples, attention_mask, position_embeddings,
            )

            setattr(layer, "self_attn", LoraQKV(
                layer.self_attn,
                query_out,
                key_out,
                value_out,
                q_lora_rank=kwargs["q_lora_rank"],
                qk_mqa_dim=kwargs["qk_mqa_dim"],
                collapse=kwargs["collapse"],
                kv_lora_rank=kv_rank_list[layer_idx],
                kv_decomp_method=kwargs["kv_decomp_method"],
                cwsvd_percdamp=kwargs["cwsvd_percdamp"],
                use_qkv_norm=kwargs["use_qkv_norm"],
                balance_kv_ratio=kwargs["balance_kv_ratio"],
                rms_norm_eps=model.config.rms_norm_eps,
            ))

            del query_out, key_out, value_out, outs
            torch.cuda.empty_cache()

            seqlen = layer_input.shape[1]
            if position_embeddings is not None:
                pe = tuple(p[:, :seqlen] if p.shape[1] >= seqlen else p for p in position_embeddings)
            else:
                pe = None
            new_outs = torch.zeros_like(layer_input)
            with torch.no_grad():
                for j in range(nsamples):
                    new_outs[j] = layer(
                        layer_input[j].unsqueeze(0),
                        attention_mask=None,
                        position_embeddings=pe,
                        use_cache=False,
                    )[0]
            inps = new_outs
            del layer_input

            if not keep_on_device:
                layers[layer_idx] = layer.cpu()
                del layer
                torch.cuda.empty_cache()

        if not keep_on_device:
            for layer in layers:
                layer.to(device)

    if kwargs["use_qkv_norm"]:
        lora_qkv_outputs = get_qkv_calibrate_outputs(model, train_loader)
        for layer_idx, layer in enumerate(model.model.layers):
            statistics_qkv_rmsnorm(
                layer.self_attn,
                lora_qkv_outputs["q_a_proj"][layer_idx] if len(lora_qkv_outputs["q_a_proj"]) > layer_idx else None,
                lora_qkv_outputs["kv_a_proj"][layer_idx]
            )

    if test_loader:
        message = "Evaluating lora-qkv model's ppl"
        dataset_ppl = evaluate_ppl(model, tokenizer.pad_token_id, test_loader, message)
        print(f'Low rank approximate QKV ppl: {dataset_ppl:.4f}')

    return model