import dataclasses

import safetensors.torch
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, list_repo_files
from torch import Tensor
from torch.nn.attention.bias import causal_lower_right


# TFLOPS is from specsheet, membw is measured memcpy bw.
def get_sol():
    gpu_name = torch.cuda.get_device_name()
    if "5090" in gpu_name:
        sol = 209.5, 1500
    elif "A100" in gpu_name:
        sol = 312, 1700
    elif "H200" in gpu_name:
        sol = 1979, 4000
    else:
        sol = 1e9, 1e9
    return sol


@dataclasses.dataclass(frozen=True, slots=True)
class ModelParams:
    input_embeds: Tensor  # [vocab_size, dim]
    l_attn_norm: Tensor  # [depth, dim]
    l_wqkv: Tensor  # [depth, qkv_dim, dim]
    l_q_norm: Tensor  # [depth, head_dim]
    l_k_norm: Tensor  # [depth, head_dim]
    l_wo: Tensor  # [depth, dim, q_dim]
    l_mlp_norm: Tensor  # [depth, dim]
    l_w13: Tensor  # [depth, mlp_dim * 2, dim]
    l_w2: Tensor  # [depth, dim, mlp_dim]
    norm: Tensor  # [dim]
    lm_head: Tensor | None
    num_layers: int
    num_heads: int
    num_kv_heads: int

    @staticmethod
    def from_pretrained(repo_id: str):
        return ModelParams.from_state_dict(load_hf_state_dict(repo_id))

    @staticmethod
    def from_state_dict(state_dict: dict[str, Tensor]):
        input_embeds = state_dict["model.embed_tokens.weight"]
        norm = state_dict["model.norm.weight"]
        lm_head = state_dict["lm_head.weight"]
        if lm_head is input_embeds:
            lm_head = None

        num_heads = 0
        num_kv_heads = 0

        l_attn_norm = []
        l_wqkv = []
        l_q_norm = []
        l_k_norm = []
        l_wo = []
        l_mlp_norm = []
        l_w13 = []
        l_w2 = []

        for i in range(1000):
            prefix = f"model.layers.{i}."
            layer = {k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}
            if len(layer) == 0:
                break

            wq, wk, wv, wo = [layer[f"self_attn.{x}_proj.weight"] for x in "qkvo"]
            w1, w2, w3 = [layer[f"mlp.{x}_proj.weight"] for x in ("gate", "down", "up")]

            num_heads = wq.shape[0] // 128
            num_kv_heads = wk.shape[0] // 128

            l_attn_norm.append(layer["input_layernorm.weight"])
            l_wqkv.append(torch.cat([wq, wk, wv], dim=0))
            l_q_norm.append(layer["self_attn.q_norm.weight"])
            l_k_norm.append(layer["self_attn.k_norm.weight"])
            l_wo.append(wo)
            l_mlp_norm.append(layer["post_attention_layernorm.weight"])
            l_w13.append(torch.cat([w1, w3], dim=0))
            l_w2.append(w2)

        return ModelParams(
            input_embeds,
            torch.stack(l_attn_norm, dim=0),
            torch.stack(l_wqkv, dim=0),
            torch.stack(l_q_norm, dim=0),
            torch.stack(l_k_norm, dim=0),
            torch.stack(l_wo, dim=0),
            torch.stack(l_mlp_norm, dim=0),
            torch.stack(l_w13, dim=0),
            torch.stack(l_w2, dim=0),
            norm,
            lm_head,
            len(l_attn_norm),
            num_heads,
            num_kv_heads,
        )

    def to(self, *args, **kwargs):
        return ModelParams(*[x.to(*args, **kwargs) if x is not None else None for x in self])


@dataclasses.dataclass(slots=True)
class ModelBuffers:
    rope: Tensor
    kv_cache: Tensor
    position: int

    @staticmethod
    def create(
        num_kv_heads: int,
        num_layers: int,
        max_length: int = 40960,
        head_dim: int = 128,
        device: torch.types.Device = None,
        kv_dtype: torch.dtype | None = None,
    ):
        rope = _precompute_rope(max_length, head_dim, theta=1e6).to(device)
        kv_cache = torch.empty(num_layers, 2, max_length, num_kv_heads, head_dim, dtype=kv_dtype, device=device)
        position = 0
        return ModelBuffers(rope, kv_cache, position)


def _rms_norm(x: Tensor, weight: Tensor):
    return F.rms_norm(x, x.shape[-1:], weight, eps=1e-6)


# we use combined w13 to align with vLLM/SGLang impl.
# this helps with the baseline significantly for small shapes.
def mlp_ref(x: Tensor, norm: Tensor, w13: Tensor, w2: Tensor):
    gate, up = (_rms_norm(x, norm) @ w13.T).chunk(2, dim=1)
    return x + (F.silu(gate) * up) @ w2.T


def _precompute_rope(max_length: int, dim: int, theta: float):
    omega = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    timestep = torch.arange(max_length, dtype=torch.float32)
    freqs = timestep[:, None] * omega  # [max_length, dim/2]
    freqs = torch.cat([freqs, freqs], dim=-1)  # [max_length, dim]
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)  # [max_length, dim * 2]


def _apply_rope(x: Tensor, rope: Tensor):
    # x: (..., L, num_heads, head_dim)
    # rope: (L, head_dim * 2)
    x_f32 = x.float()
    x1, x2 = x_f32.chunk(2, dim=-1)
    rotate_half = torch.cat([-x2, x1], dim=-1)

    cos, sin = rope.unsqueeze(-2).chunk(2, dim=-1)  # (L, 1, head_dim) each
    return (x_f32 * cos + rotate_half * sin).to(x.dtype)


def attn_ref(
    x: Tensor,  # (num_tokens, dim)
    norm: Tensor,  # (dim)
    kv_cache: Tensor,  # (2, max_context, num_kv_heads, head_dim)
    wqkv: Tensor,  # (qkv_dim, dim)
    q_norm: Tensor,
    k_norm: Tensor,
    rope: Tensor,  # (num_tokens, head_dim * 2)
    wo: Tensor,  # (dim, q_dim)
    position: int,
):
    num_tokens, _ = x.shape
    _, _, num_kv_heads, head_dim = kv_cache.shape
    num_heads = wqkv.shape[0] // head_dim - num_kv_heads * 2

    # input projection
    qkv = (_rms_norm(x, norm) @ wqkv.T).unflatten(-1, (-1, head_dim))  # (num_tokens, -1, head_dim)
    q, k, v = qkv.split([num_heads, num_kv_heads, num_kv_heads], dim=-2)

    # apply QK norm and rope
    q = _apply_rope(_rms_norm(q, q_norm), rope)
    k = _apply_rope(_rms_norm(k, k_norm), rope)

    # update KV cache
    kv_cache[0, position : position + num_tokens] = k
    kv_cache[1, position : position + num_tokens] = v

    # attention
    attn_mask = causal_lower_right(num_tokens, position + num_tokens) if num_tokens > 1 else None
    o = F.scaled_dot_product_attention(
        q.transpose(0, 1),
        kv_cache[0, : position + num_tokens].transpose(0, 1),
        kv_cache[1, : position + num_tokens].transpose(0, 1),
        attn_mask=attn_mask,
        enable_gqa=True,
    ).transpose(0, 1)

    # output projection
    return x + o.flatten(-2) @ wo.T


def model_ref(input_ids: Tensor, params: ModelParams, buffers: ModelBuffers):
    num_tokens = input_ids.shape[0]
    position = buffers.position

    x = params.input_embeds[input_ids]
    rope = buffers.rope[position : position + num_tokens]

    for i in range(params.num_layers):
        x = attn_ref(
            x,
            params.l_attn_norm[i],
            buffers.kv_cache[i],
            params.l_wqkv[i],
            params.l_q_norm[i],
            params.l_k_norm[i],
            rope,
            params.l_wo[i],
            position,
        )
        x = mlp_ref(x, params.l_mlp_norm[i], params.l_w13[i], params.l_w2[i])

    # increment position
    buffers.position += num_tokens

    # LM head for the last token only
    x = _rms_norm(x[-1], params.norm)

    lm_head = params.lm_head
    if lm_head is None:
        lm_head = params.input_embeds

    logits = x @ lm_head.T
    return logits.argmax(-1)


def load_hf_state_dict(repo_id: str):
    state_dict = dict()

    for filename in list_repo_files(repo_id):
        if not filename.endswith(".safetensors"):
            continue

        local_path = hf_hub_download(repo_id, filename)
        state_dict.update(safetensors.torch.load_file(local_path))

    return state_dict


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # compare in FP32
    model_id = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto").eval().float()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    messages = [dict(role="user", content="hi, how are you?")]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")["input_ids"]

    # HF reference
    max_new_tokens = 10
    tokens = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    expected = tokens[0, input_ids.shape[1] :]

    # ours
    # buffers will be modified in-place
    params = ModelParams.from_state_dict(model.state_dict())
    buffers = ModelBuffers.create(params.num_kv_heads, params.num_layers, kv_dtype=params.input_embeds.dtype)

    # prefill
    outputs = [model_ref(input_ids.squeeze(0), params, buffers)]

    # decode
    for _ in range(max_new_tokens - 1):
        outputs.append(model_ref(outputs[-1].unsqueeze(0), params, buffers))

    actual = torch.stack(outputs, dim=0)
    torch.testing.assert_close(expected, actual)
