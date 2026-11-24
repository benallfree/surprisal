"""A leaner single-token forward pass for Qwen2 models in MLX, used by the codecs instead of model(...).

- q/k/v projections fused into one matmul, gate/up fused into one matmul (fewer kernels per layer).
- Everything with fixed shapes (norms, projections, rope, MLP, residuals) runs in mx.compile'd blocks; only the
  KV-cache write and attention run uncompiled, so attention sees the exact context length (no padded buffer).
- Optional pruned output layer: logits only for a given sorted list of token ids.

It computes the same function as the stock model, but with different kernels, so its probabilities are not
bit-identical to model(...). Encoder and decoder must both use it.
"""
import mlx.core as mx
import mlx.nn as nn


def _cat_linear(layers):
    """Concatenate Linear or QuantizedLinear layers along the output dimension into one layer."""
    first = layers[0]
    bias = [l.bias for l in layers] if "bias" in first else None
    if hasattr(first, "scales"):
        out = nn.QuantizedLinear(first.group_size, 1, bias=bias is not None, group_size=first.group_size, bits=first.bits)
        out.weight = mx.concatenate([l.weight for l in layers], axis=0)
        out.scales = mx.concatenate([l.scales for l in layers], axis=0)
        out.biases = mx.concatenate([l.biases for l in layers], axis=0)
    else:
        out = nn.Linear(1, 1, bias=bias is not None)
        out.weight = mx.concatenate([l.weight for l in layers], axis=0)
    if bias is not None:
        out.bias = mx.concatenate(bias, axis=0)
    mx.eval(out.parameters())
    return out


class FastQwen:
    def __init__(self, model, keep=None):
        self.model = model
        inner = model.model
        self.blocks = []
        for layer in inner.layers:
            a = layer.self_attn
            qkv = _cat_linear([a.q_proj, a.k_proj, a.v_proj])
            gu = _cat_linear([layer.mlp.gate_proj, layer.mlp.up_proj])
            self.blocks.append(self._make(layer, a, qkv, gu))
        self.keep = keep
        if keep is not None:
            emb = inner.embed_tokens
            idx = mx.array(keep, dtype=mx.int32)
            if hasattr(emb, "scales"):
                w, s, b = emb.weight[idx], emb.scales[idx], emb.biases[idx]
                mx.eval(w, s, b)
                gs, bits = emb.group_size, emb.bits
                self.head = lambda h: mx.quantized_matmul(h, w, s, b, transpose=True, group_size=gs, bits=bits)
            elif hasattr(model, "lm_head"):
                w = model.lm_head.weight[idx]
                mx.eval(w)
                self.head = lambda h: h @ w.T
            else:
                w = emb.weight[idx]
                mx.eval(w)
                self.head = lambda h: h @ w.T
        else:
            self.head = model.lm_head if hasattr(model, "lm_head") else inner.embed_tokens.as_linear
        norm = inner.norm
        head = self.head
        self.final = mx.compile(lambda h: head(norm(h))[0, 0])

    @staticmethod
    def _make(layer, a, qkv, gu):
        nq, nkv = a.n_heads, a.n_kv_heads

        def pre(h, pos):
            x = qkv(layer.input_layernorm(h))  # [1, 1, (nq + 2 nkv) * hd]
            hd = x.shape[-1] // (nq + 2 * nkv)
            q = x[..., : nq * hd].reshape(1, 1, nq, hd).transpose(0, 2, 1, 3)
            k = x[..., nq * hd : (nq + nkv) * hd].reshape(1, 1, nkv, hd).transpose(0, 2, 1, 3)
            v = x[..., (nq + nkv) * hd :].reshape(1, 1, nkv, hd).transpose(0, 2, 1, 3)
            return a.rope(q, offset=pos), a.rope(k, offset=pos), v

        def post(h, o):
            h = h + a.o_proj(o.transpose(0, 2, 1, 3).reshape(1, 1, -1))
            g, u = mx.split(gu(layer.post_attention_layernorm(h)), 2, axis=-1)
            return h + layer.mlp.down_proj(nn.silu(g) * u)

        return mx.compile(pre), mx.compile(post), a.scale

    def __call__(self, tok, cache):
        """tok: int32 [1, 1]; cache: mlx_lm prompt cache. Returns float logits [V] (or [len(keep)])."""
        h = self.model.model.embed_tokens(tok)
        pos = mx.array(cache[0].offset, dtype=mx.int32)
        for (pre, post, scale), c in zip(self.blocks, cache):
            q, k, v = pre(h, pos)
            K, V = c.update_and_fetch(k, v)
            o = mx.fast.scaled_dot_product_attention(q, K, V, scale=scale)
            h = post(h, o)
        return self.final(h)
