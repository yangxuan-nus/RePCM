"""Region-aware cardiac mesh motion synthesis with a shape-conditioned prior."""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================================================
# Basic blocks (SDPA MHA + FFN)
# ============================================================
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class Attention(nn.Module):
    """SDPA self-attn. Bool attn_mask True=ALLOW."""

    def __init__(self, dim, heads=4, dim_head=64):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.Linear(inner, dim)

        nn.init.zeros_(self.to_out.weight)
        if self.to_out.bias is not None:
            nn.init.zeros_(self.to_out.bias)

    def forward(self, x, attn_mask=None):
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        if attn_mask is not None:
            if attn_mask.ndim == 3:
                attn_mask = attn_mask[:, None, :, :]
            attn_mask = attn_mask.to(torch.bool)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

    def forward_pair(self, shape, motion, attn_mask=None):
        h = self.heads
        q = rearrange(self.to_q(shape), "b n (h d) -> b h n d", h=h)
        k = rearrange(self.to_k(shape), "b n (h d) -> b h n d", h=h)
        v0 = rearrange(self.to_v(shape), "b n (h d) -> b h n d", h=h)
        vt = rearrange(self.to_v(motion), "b n (h d) -> b h n d", h=h)
        if attn_mask is not None and attn_mask.ndim == 3:
            attn_mask = attn_mask[:, None]
        y0 = F.scaled_dot_product_attention(q, k, v0, attn_mask=attn_mask)
        yt = F.scaled_dot_product_attention(q, k, vt, attn_mask=attn_mask)
        y0 = rearrange(y0, "b h n d -> b n (h d)")
        yt = rearrange(yt, "b h n d -> b n (h d)")
        return self.to_out(y0), self.to_out(yt)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class TokenMPBlock(nn.Module):
    """PreNorm self-attn + FFN."""

    def __init__(self, dim, heads=4, dim_head=64, ff_mult=4):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, dim_head=dim_head)
        self.n2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=ff_mult)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.n1(x), attn_mask=attn_mask)
        x = x + self.ff(self.n2(x))
        return x

    def forward_pair(self, shape, motion, attn_mask=None):
        d0, dt = self.attn.forward_pair(self.n1(shape), self.n1(motion), attn_mask)
        shape = shape + d0
        motion = motion + dt
        shape = shape + self.ff(self.n2(shape))
        motion = motion + self.ff(self.n2(motion))
        return shape, motion


# Embeddings
# ============================================================
class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=256):
        super().__init__()
        assert hidden_dim % 6 == 0
        e = torch.pow(2, torch.arange(hidden_dim // 6)).float() * math.pi
        bx = torch.cat([e, torch.zeros_like(e), torch.zeros_like(e)])
        by = torch.cat([torch.zeros_like(e), e, torch.zeros_like(e)])
        bz = torch.cat([torch.zeros_like(e), torch.zeros_like(e), e])
        basis = torch.stack([bx, by, bz], dim=0)
        self.register_buffer("basis", basis)
        self.mlp = nn.Linear(hidden_dim + 3, dim)

    @staticmethod
    def embed(x, basis):
        proj = torch.einsum("bnd,de->bne", x, basis)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)

    def forward(self, x):
        emb = self.embed(x, self.basis)
        return self.mlp(torch.cat([emb, x], dim=-1))


class TrajEncoding(nn.Module):
    def __init__(self, out_dim, num_freqs=10, include_input=True, input_dims=48):
        super().__init__()
        self.include_input = include_input
        self.input_dims = input_dims
        freq = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        self.register_buffer("freq", freq.view(-1, 1))
        latent_dim = input_dims * (2 * num_freqs + int(include_input))
        self.mlp = nn.Linear(latent_dim, out_dim)

    def forward(self, x):
        assert x.shape[-1] == self.input_dims
        x_exp = x.unsqueeze(2)
        xw = x_exp * (self.freq.view(1, 1, -1, 1) * math.pi)
        sinf = torch.sin(xw).flatten(2, 3)
        cosf = torch.cos(xw).flatten(2, 3)
        feat = torch.cat([x, sinf, cosf], dim=-1) if self.include_input else torch.cat([sinf, cosf], dim=-1)
        return self.mlp(feat)


# ============================================================
# Distributions
# ============================================================
class DiagonalGaussianDistribution:
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        if deterministic:
            self.std = torch.zeros_like(self.mean)

    def sample(self, temp: float = 1.0):
        eps = torch.randn_like(self.std)
        return self.mean + (self.std * float(temp)) * eps

    def kl(self, other=None):
        if other is None:
            return 0.5 * torch.mean(self.mean.pow(2) + self.std.pow(2) - 1.0 - self.logvar, dim=(-1, -2))
        mq, lvq = self.mean, self.logvar
        mp, lvp = other.mean, other.logvar
        kl = 0.5 * ((lvq - lvp).exp() + (mq - mp).pow(2) * torch.exp(-lvp) - 1.0 + (lvp - lvq))
        return kl.mean(dim=(-1, -2))


# ============================================================
# Token SyncAttention (shared routing, Q/K from t0 only)
# ============================================================
class TokenNeighborSync(nn.Module):
    """Share shape-stream attention weights between anatomy and motion."""

    def __init__(self, dim):
        super().__init__()
        self.nq = nn.LayerNorm(dim)
        self.nk = nn.LayerNorm(dim)
        self.nv0 = nn.LayerNorm(dim)
        self.nvt = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v0 = nn.Linear(dim, dim, bias=False)
        self.to_vt = nn.Linear(dim, dim, bias=False)
        self.out0 = nn.Linear(dim, dim)
        self.outt = nn.Linear(dim, dim)
        nn.init.zeros_(self.out0.weight)
        nn.init.zeros_(self.out0.bias)
        nn.init.zeros_(self.outt.weight)
        nn.init.zeros_(self.outt.bias)

    def forward(self, t0, tt, attn_mask=None):
        q = self.to_q(self.nq(t0))
        k = self.to_k(self.nk(t0))
        v0 = self.to_v0(self.nv0(t0))
        vt = self.to_vt(self.nvt(tt))

        Q = q[:, None]
        K = k[:, None]
        V0 = v0[:, None]
        VT = vt[:, None]
        if attn_mask is not None and attn_mask.ndim == 3:
            attn_mask = attn_mask[:, None]

        o0 = F.scaled_dot_product_attention(Q, K, V0, attn_mask=attn_mask)[:, 0]
        ot = F.scaled_dot_product_attention(Q, K, VT, attn_mask=attn_mask)[:, 0]
        return self.out0(o0), self.outt(ot)


class DualFFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.n0 = nn.LayerNorm(dim)
        self.nt = nn.LayerNorm(dim)
        self.f0 = nn.Sequential(nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Linear(dim * mult, dim))
        self.ft = nn.Sequential(nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Linear(dim * mult, dim))
        nn.init.zeros_(self.f0[-1].weight)
        nn.init.zeros_(self.f0[-1].bias)
        nn.init.zeros_(self.ft[-1].weight)
        nn.init.zeros_(self.ft[-1].bias)

    def forward(self, t0, tt):
        return self.f0(self.n0(t0)), self.ft(self.nt(tt))


# ============================================================

# ============================================================
class DyMeshVAE_FiLM_MoE_Unified_Final(nn.Module):
    """Conditional mesh VAE with functional regions and motion experts.

    Anchor indices, vertex-region assignments and adjacency masks are
    supplied by the motion-based partitioning stage.
    """

    def __init__(
        self,
        enc_depth=8,
        dec_depth=8,
        dim=256,
        heads=4,
        dim_head=64,
        ff_mult=4,
        T=25,
        latent_dim=16,
        output_dim=None,
        num_traj=512,
        num_tokens=16,
        token_mp_layers=1,
        decode_chunk=20000,
        bind_mode="neighbor",
        use_xt_cond_prior=True,
        region_every=2,
        anchor_neighbor_layers=1,
        anchor_nei_every=2,
        anchor_nei_phase=0,
        use_moe_prior=True,
        num_experts=4,
        gate_temp=0.2,
        prior_sample_temp: float = 1.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.T = int(T)
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim if output_dim is not None else 3 * self.T)

        self.num_traj = int(num_traj)
        self.num_tokens = int(num_tokens)

        self.decode_chunk = int(decode_chunk) if decode_chunk is not None else None
        self.bind_mode = str(bind_mode)
        self.use_xt_cond_prior = bool(use_xt_cond_prior)

        self.region_every = int(region_every)
        self.anchor_neighbor_layers = int(anchor_neighbor_layers)
        self.anchor_nei_every = int(anchor_nei_every)
        self.anchor_nei_phase = int(anchor_nei_phase)

        self.use_moe_prior = bool(use_moe_prior)
        self.num_experts = int(num_experts)
        self.gate_temp = float(gate_temp)
        if self.gate_temp <= 0:
            raise ValueError("gate_temp must be positive")
        self.prior_sample_temp = float(prior_sample_temp)

        self.register_buffer("idx_fixed", None, persistent=False)
        self.register_buffer("A_allow", None, persistent=False)
        self.register_buffer("vertex_token_id", None, persistent=False)
        self.register_buffer("anchor_token_id", None, persistent=False)
        self.register_buffer("AK_allow", None, persistent=False)

        self.point_embed = PointEmbed(dim=self.dim)
        self.traj_embed = TrajEncoding(out_dim=self.dim, input_dims=3 * self.T)

        self.token_sync = nn.ModuleList([TokenNeighborSync(self.dim) for _ in range(int(token_mp_layers))])
        self.token_ffn = nn.ModuleList([DualFFN(self.dim) for _ in range(int(token_mp_layers))])

        self.to_gamma_x0 = nn.Linear(self.dim, self.dim)
        self.to_beta_x0 = nn.Linear(self.dim, self.dim)
        self.to_gamma_xt = nn.Linear(self.dim, self.dim)
        self.to_beta_xt = nn.Linear(self.dim, self.dim)
        for m in [self.to_gamma_x0, self.to_beta_x0, self.to_gamma_xt, self.to_beta_xt]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        self.enc_self = nn.ModuleList([
            TokenMPBlock(self.dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult)
            for _ in range(int(enc_depth))
        ])
        self.dec_self = nn.ModuleList([
            TokenMPBlock(self.dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult)
            for _ in range(int(dec_depth))
        ])

        if self.anchor_neighbor_layers > 0:
            self.anchor_nei = nn.ModuleList([
                TokenMPBlock(self.dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult)
                for _ in range(self.anchor_neighbor_layers)
            ])
        else:
            self.anchor_nei = nn.ModuleList([])

        self.mean_fc_x0 = nn.Linear(self.dim, self.latent_dim)
        self.mean_fc_xt = nn.Linear(self.dim, self.latent_dim)
        self.logvar_fc_xt = nn.Linear(self.dim, self.latent_dim)

        if self.use_moe_prior:
            self.prior_mean_xt_experts = nn.ModuleList([
                nn.Linear(self.dim, self.latent_dim) for _ in range(self.num_experts)
            ])
            self.prior_logvar_xt_experts = nn.ModuleList([
                nn.Linear(self.dim, self.latent_dim) for _ in range(self.num_experts)
            ])
            self.register_buffer("shape_protos", torch.zeros(self.num_experts, self.latent_dim), persistent=True)
            self.register_buffer("shape_protos_valid", torch.tensor(0, dtype=torch.long), persistent=True)
        elif self.use_xt_cond_prior:
            self.prior_mean_xt = nn.Linear(self.dim, self.latent_dim)
            self.prior_logvar_xt = nn.Linear(self.dim, self.latent_dim)

        self.proj_x0 = nn.Linear(self.latent_dim, self.dim)
        self.proj_xt = nn.Linear(self.latent_dim, self.dim)

        self.fc_query = nn.Linear(self.dim, self.dim * 2)
        self.to_q = nn.Linear(self.dim * 2, self.dim * 2, bias=False)
        self.to_k = nn.Linear(self.dim, self.dim * 2, bias=False)
        self.to_v = nn.Linear(self.dim, self.dim * 2, bias=False)
        self.to_outputs = nn.Linear(self.dim * 2, self.output_dim)
        nn.init.zeros_(self.to_outputs.weight)
        nn.init.zeros_(self.to_outputs.bias)

    def set_fixed_idx(self, idx_1d: torch.Tensor):
        assert idx_1d.ndim == 1 and idx_1d.numel() == self.num_traj
        self.idx_fixed = idx_1d.long()

    def set_vertex_token_id(self, vertex_token_id: torch.Tensor):
        assert vertex_token_id.ndim == 1
        self.vertex_token_id = vertex_token_id.long()

    def build_anchor_token_id(self):
        if self.vertex_token_id is None or self.idx_fixed is None:
            raise RuntimeError("Need vertex_token_id and idx_fixed before build_anchor_token_id().")
        vtid = self.vertex_token_id.to(self.idx_fixed.device)
        self.anchor_token_id = vtid[self.idx_fixed.long()]
        mx = int(self.anchor_token_id.max().item())
        if mx + 1 > self.num_tokens:
            self.num_tokens = mx + 1

    def set_token_graph(self, A_allow: torch.Tensor):
        assert A_allow.ndim == 2 and A_allow.shape[0] == A_allow.shape[1]
        A = A_allow.bool().clone()
        A.fill_diagonal_(True)
        self.A_allow = A
        self.num_tokens = int(A.shape[0])

    def set_anchor_graph(self, AK_allow: torch.Tensor):
        assert AK_allow.ndim == 2 and AK_allow.shape[0] == AK_allow.shape[1]
        assert int(AK_allow.shape[0]) == int(self.num_traj), f"AK_allow must be (K,K) with K={self.num_traj}"
        A = AK_allow.bool().clone()
        A.fill_diagonal_(True)
        self.AK_allow = A

    @torch.no_grad()
    def set_shape_prototypes(self, protos: torch.Tensor):
        if not self.use_moe_prior:
            return
        assert protos.ndim == 2
        assert protos.size(0) == self.num_experts and protos.size(1) == self.latent_dim
        self.shape_protos.copy_(protos.to(self.shape_protos.device))
        self.shape_protos_valid.fill_(1)

    @torch.no_grad()
    def update_shape_prototypes(self, embeddings: torch.Tensor, momentum: float = 0.95):
        if not self.use_moe_prior:
            return
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        embeddings = embeddings.detach().to(self.shape_protos).reshape(-1, self.latent_dim)
        if embeddings.shape[0] == 0:
            return
        if not bool(self.shape_protos_valid.item()):
            directions = F.normalize(embeddings, dim=-1)
            chosen = [int(directions.norm(dim=-1).argmax())]
            for _ in range(1, self.num_experts):
                similarities = directions @ directions[chosen].T
                scores = similarities.max(dim=1).values
                scores[chosen] = float("inf")
                chosen.append(int(scores.argmin()))
            self.set_shape_prototypes(embeddings[chosen])
            return

        directions = F.normalize(embeddings, dim=-1)
        prototypes = F.normalize(self.shape_protos, dim=-1)
        assignment = (directions @ prototypes.T).argmax(dim=-1)
        for expert_idx in range(self.num_experts):
            selected = embeddings[assignment == expert_idx]
            if selected.numel():
                self.shape_protos[expert_idx].lerp_(selected.mean(dim=0), 1 - momentum)

    def _gather(self, feat_full, idx_BK):
        B, N, D = feat_full.shape
        return torch.gather(feat_full, 1, idx_BK.unsqueeze(-1).expand(-1, -1, D))

    def _token_mask(self, B, device):
        if self.A_allow is None:
            return None
        return self.A_allow.to(device)[None].expand(B, -1, -1).contiguous()

    def _anchor_mask(self, B, device):
        if self.AK_allow is None:
            return None
        return self.AK_allow.to(device)[None].expand(B, -1, -1).contiguous()

    def _anchor_to_token_pool(self, a):
        if self.anchor_token_id is None:
            raise RuntimeError("anchor_token_id is None. Call build_anchor_token_id().")
        B, K, D = a.shape
        G = int(self.num_tokens)
        atid = self.anchor_token_id.to(a.device)

        t = torch.zeros(B, G, D, device=a.device, dtype=a.dtype)
        t.scatter_add_(1, atid[None, :, None].expand(B, K, D), a)

        cnt = torch.zeros(B, G, device=a.device, dtype=a.dtype)
        cnt.scatter_add_(1, atid[None, :].expand(B, K), torch.ones(B, K, device=a.device, dtype=a.dtype))
        t = t / cnt.unsqueeze(-1).clamp_min(1.0)
        return t

    def _token_to_anchor_gather(self, t):
        atid = self.anchor_token_id.to(t.device)
        return t.index_select(1, atid)

    def _film_writeback(self, a, t, gamma_fc, beta_fc):
        tb = self._token_to_anchor_gather(t)
        gamma = gamma_fc(tb)
        beta = beta_fc(tb)
        return a + gamma * F.layer_norm(a, (a.shape[-1],)) + beta

    def _vertex_anchor_mask(self, s, e, B, device):
        if self.vertex_token_id is None or self.anchor_token_id is None:
            return None
        vt = self.vertex_token_id[s:e].to(device)
        at = self.anchor_token_id.to(device)[None, :]

        if self.bind_mode == "hard":
            allow = (at == vt[:, None])
        elif self.bind_mode == "neighbor":
            if self.A_allow is None:
                allow = (at == vt[:, None])
            else:
                A = self.A_allow.to(device)
                allowed_tok = A[vt]
                at_clamped = at.expand(vt.numel(), -1).clamp(0, self.num_tokens - 1)
                allow = allowed_tok.gather(1, at_clamped)
        else:
            raise ValueError(f"Unknown bind_mode={self.bind_mode}")

        return allow[None, None].expand(B, 1, -1, -1).contiguous()

    def _token_exchange_return_tokens(self, x0, xt, B, device):
        t0 = self._anchor_to_token_pool(x0)
        tt = self._anchor_to_token_pool(xt)
        tmask = self._token_mask(B, device)

        for attn, ffn in zip(self.token_sync, self.token_ffn):
            dt0, dtt = attn(t0, tt, attn_mask=tmask)
            t0 = t0 + dt0
            tt = tt + dtt
            ft0, ftt = ffn(t0, tt)
            t0 = t0 + ft0
            tt = tt + ftt
        return t0, tt

    def _anchor_neighbor_refine_all(self, x0, xt, B, device):
        if len(self.anchor_nei) == 0:
            return x0, xt
        if self.AK_allow is None:
            raise RuntimeError("anchor_neighbor_layers>0 but AK_allow is None. Call set_anchor_graph().")
        amask = self._anchor_mask(B, device)
        for blk in self.anchor_nei:
            x0, xt = blk.forward_pair(x0, xt, attn_mask=amask)
        return x0, xt

    def _anchor_neighbor_step(self, x0, xt, B, device, layer_idx: int):
        if len(self.anchor_nei) == 0:
            return x0, xt
        if self.AK_allow is None:
            raise RuntimeError("anchor_nei enabled but AK_allow is None. Call set_anchor_graph().")
        amask = self._anchor_mask(B, device)
        blk = self.anchor_nei[layer_idx % len(self.anchor_nei)]
        x0, xt = blk.forward_pair(x0, xt, attn_mask=amask)
        return x0, xt

    def _gate_weights(self, e_s: torch.Tensor) -> torch.Tensor:
        if not self.use_moe_prior or not bool(self.shape_protos_valid.item()):
            return e_s.new_full((e_s.shape[0], self.num_experts), 1.0 / self.num_experts)
        e = F.normalize(e_s, dim=-1)
        p = F.normalize(self.shape_protos.to(device=e_s.device, dtype=e_s.dtype), dim=-1)
        return F.softmax((e @ p.T) / self.gate_temp, dim=-1)

    def _moe_prior_params(self, x0_feat: torch.Tensor, e_s: torch.Tensor):
        w = self._gate_weights(e_s.detach())
        mu = torch.stack([expert(x0_feat) for expert in self.prior_mean_xt_experts], dim=2)
        logvar = torch.stack([expert(x0_feat) for expert in self.prior_logvar_xt_experts], dim=2)
        weights = w[:, None, :, None]
        mu_prior = (mu * weights).sum(dim=2)
        var_prior = (logvar.exp() * weights).sum(dim=2)
        return mu_prior, var_prior.clamp_min(1e-8).log(), w

    def encode(self, pc, single_frame=False, prior_sample_temp: float | None = None):
        B, T, N, _ = pc.shape
        device = pc.device

        pc0 = pc[:, 0]
        if single_frame:
            pct_rel = torch.zeros(B, N, 3 * self.T, device=device, dtype=pc.dtype)
        else:
            pct_rel = (pc - pc[:, :1]).permute(0, 2, 1, 3).flatten(2, 3)

        pc0_full = self.point_embed(pc0)
        pct_full = self.traj_embed(pct_rel)

        if self.idx_fixed is None:
            raise RuntimeError("Need idx_fixed loaded.")
        idx = self.idx_fixed.to(device).view(1, -1).expand(B, -1).contiguous()

        x0 = self._gather(pc0_full, idx)
        xt = self._gather(pct_full, idx)

        if self.anchor_neighbor_layers > 0:
            x0, xt = self._anchor_neighbor_refine_all(x0, xt, B, device)

        for i, blk in enumerate(self.enc_self):
            if self.anchor_nei_every > 0 and ((i + self.anchor_nei_phase) % self.anchor_nei_every == 0):
                if self.anchor_neighbor_layers > 0:
                    x0, xt = self._anchor_neighbor_step(x0, xt, B, device, layer_idx=i)

            x0, xt = blk.forward_pair(x0, xt)

            if self.region_every > 0 and ((i + 1) % self.region_every == 0):
                if self.anchor_token_id is not None and (self.A_allow is not None):
                    t0, tt = self._token_exchange_return_tokens(x0, xt, B, device)
                    x0 = self._film_writeback(x0, t0, self.to_gamma_x0, self.to_beta_x0)
                    xt = self._film_writeback(xt, tt, self.to_gamma_xt, self.to_beta_xt)

        x0_lat = self.mean_fc_x0(x0)
        e_s = x0_lat.mean(dim=1)
        temp = float(self.prior_sample_temp if prior_sample_temp is None else prior_sample_temp)

        if single_frame:
            if self.use_moe_prior:
                pmu, plv, gate_w = self._moe_prior_params(x0, e_s)
            elif self.use_xt_cond_prior:
                pmu = self.prior_mean_xt(x0)
                plv = self.prior_logvar_xt(x0)
                gate_w = None
            else:
                pmu = torch.zeros_like(x0_lat)
                plv = torch.zeros_like(x0_lat)
                gate_w = None
            prior = DiagonalGaussianDistribution(pmu, plv)
            xt_lat = prior.sample(temp=temp)
            kl = torch.zeros(B, device=device)
        else:
            qmu = self.mean_fc_xt(xt)
            qlv = self.logvar_fc_xt(xt)
            post = DiagonalGaussianDistribution(qmu, qlv)

            if self.use_moe_prior:
                pmu, plv, gate_w = self._moe_prior_params(x0, e_s)
            elif self.use_xt_cond_prior:
                pmu = self.prior_mean_xt(x0)
                plv = self.prior_logvar_xt(x0)
                gate_w = None
            else:
                pmu = torch.zeros_like(qmu)
                plv = torch.zeros_like(qlv)
                gate_w = None
            prior = DiagonalGaussianDistribution(pmu, plv)

            kl = post.kl(other=prior)
            xt_lat = post.sample()

        lat = torch.cat([x0_lat, xt_lat], dim=-1)
        extra = {
            "e_s": e_s,
            "gate_w": gate_w,
            "shape_protos_valid": int(self.shape_protos_valid.item()) if self.use_moe_prior else -1,
        }
        return kl, lat, pc0_full, extra

    def decode(self, lat, queries_full, pc0_full):
        B, N, _ = pc0_full.shape
        device = pc0_full.device

        x0_lat, xt_lat = lat.chunk(2, dim=-1)
        x0 = self.proj_x0(x0_lat)
        xt = self.proj_xt(xt_lat)

        if self.anchor_neighbor_layers > 0:
            x0, xt = self._anchor_neighbor_refine_all(x0, xt, B, device)

        for i, blk in enumerate(self.dec_self):
            if self.anchor_nei_every > 0 and ((i + self.anchor_nei_phase) % self.anchor_nei_every == 0):
                if self.anchor_neighbor_layers > 0:
                    x0, xt = self._anchor_neighbor_step(x0, xt, B, device, layer_idx=i)

            x0, xt = blk.forward_pair(x0, xt)

            if self.region_every > 0 and ((i + 1) % self.region_every == 0):
                if self.anchor_token_id is not None and (self.A_allow is not None):
                    t0, tt = self._token_exchange_return_tokens(x0, xt, B, device)
                    x0 = self._film_writeback(x0, t0, self.to_gamma_x0, self.to_beta_x0)
                    xt = self._film_writeback(xt, tt, self.to_gamma_xt, self.to_beta_xt)

        K = self.to_k(x0)[:, None]
        V = self.to_v(xt)[:, None]

        outs = []
        chunk = self.decode_chunk if self.decode_chunk is not None else N
        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            qv = self.fc_query(pc0_full[:, s:e, :])
            Q = self.to_q(qv)[:, None]

            vmask = self._vertex_anchor_mask(s, e, B, device)

            out = F.scaled_dot_product_attention(Q, K, V, attn_mask=vmask)[:, 0]

            out = self.to_outputs(out).view(B, e - s, self.T, 3).permute(0, 2, 1, 3)
            outs.append(out)

        recon = torch.cat(outs, dim=2) + queries_full[:, None]
        return recon

    def forward(self, pc, queries, single_frame=False, return_extra=True, prior_sample_temp: float | None = None):
        kl, lat, pc0_full, extra = self.encode(pc, single_frame=single_frame, prior_sample_temp=prior_sample_temp)
        recon = self.decode(lat, queries, pc0_full)
        out = {"recon_pc": recon, "pc": pc, "kl_temp": kl, "latent": lat}
        if return_extra:
            out.update(extra)
        return out

RePCM = DyMeshVAE_FiLM_MoE_Unified_Final
