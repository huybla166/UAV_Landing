"""Time-series forecasting baselines from the Time-Series-Library (THUML, https://github.com/thuml/Time-Series-Library,
MIT License): PatchTST (Nie et al., ICLR 2023), iTransformer (Liu et al., ICLR 2024) and TimeXer (Wang et al., NeurIPS
2024). Long-term-forecasting path only, features 'M' (heave, roll, pitch in, the same three out).

Same code as TSLib models/PatchTST.py, models/iTransformer.py, models/TimeXer.py and the layers they use
(layers/Transformer_EncDec.py, layers/SelfAttention_Family.py, layers/Embed.py); the attention and encoder layers are the
ones already vendored in experiments/k2vae (TSLib code via K2VAE, MIT), copied here so this file needs torch only.
Adaptations:
  * x_mark = None: a simulation run has no calendar time, so there are no time-stamp features. TSLib's data loaders
    always supply them (hour, weekday, ...), and iTransformer and TimeXer then add them as extra variate tokens; here
    the tokens are the three motion channels only (DataEmbedding_inverted's x_mark=None branch).
  * PatchTST head dropout 0 as in the official PatchTST Weather script (TSLib reuses `dropout` for the head).
The rest does not change a model's computation:
  * forward(x) takes the past window (B, seq_len, C) and returns (B, pred_len, C); the unused decoder inputs are dropped.
  * `x_enc /= stdev` written out of place (same values).
  * FullAttention runs through torch's fused scaled_dot_product_attention (USE_SDPA; same softmax(QK^T/sqrt(E)),
    dropout on the attention weights, times V), 1.2x faster on the RTX 3060; USE_SDPA = False gives TSLib's einsum.
Instance normalisation (the 'Non-stationary Transformer' mean/std of each input window, use_norm = 1) is part of
every model, as in TSLib.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

USE_SDPA = True     # FullAttention through F.scaled_dot_product_attention (same computation as TSLib's einsum path)


# ---------------------------------------------------------------- layers (TSLib)
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        scale = self.scale or 1. / math.sqrt(E)
        if self.mask_flag:
            raise NotImplementedError("causal mask not used by these encoders")
        if USE_SDPA and not self.output_attention:
            # the same softmax(scale Q K^T) with dropout on the weights, then V, by PyTorch's fused kernel (faster)
            V = F.scaled_dot_product_attention(queries.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2),
                                               dropout_p=self.dropout.p if self.training else 0.0, scale=scale)
            return V.transpose(1, 2).contiguous(), None
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous(), (A if self.output_attention else None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau=tau, delta=delta)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                                   # x (B, C, L)
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):                      # x (B, L, C) -> (B, C, d_model): one token per variate
        x = x.permute(0, 2, 1)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):                                   # x (B, n_vars, d_model, patch_num)
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        return x.transpose(*self.dims).contiguous() if self.contiguous else x.transpose(*self.dims)


def instance_norm(x_enc):
    """Normalisation from the Non-stationary Transformer, as every TSLib forecast(): per window and channel."""
    means = x_enc.mean(1, keepdim=True).detach()
    x_enc = x_enc - means
    stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    return x_enc / stdev, means, stdev


def instance_denorm(dec_out, means, stdev, pred_len):
    dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, pred_len, 1))
    return dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, pred_len, 1))


def full_attention_layer(c):
    return AttentionLayer(FullAttention(False, c["factor"], attention_dropout=c["dropout"], output_attention=False),
                          c["d_model"], c["n_heads"])


# ---------------------------------------------------------------- PatchTST (TSLib models/PatchTST.py)
class PatchTST(nn.Module):
    """Paper: https://arxiv.org/pdf/2211.14730.pdf. Channel-independent patches, padding = stride at the end."""

    def __init__(self, c):
        super().__init__()
        self.seq_len, self.pred_len = c["seq_len"], c["pred_len"]
        patch_len, stride = c["patch_len"], c["stride"]
        padding = stride
        self.patch_embedding = PatchEmbedding(c["d_model"], patch_len, stride, padding, c["dropout"])
        self.encoder = Encoder(
            [EncoderLayer(full_attention_layer(c), c["d_model"], c["d_ff"], dropout=c["dropout"],
                          activation=c["activation"]) for _ in range(c["e_layers"])],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(c["d_model"]), Transpose(1, 2)))
        self.head_nf = c["d_model"] * int((c["seq_len"] - patch_len) / stride + 2)
        # TSLib passes configs.dropout to the head; the official PatchTST sets head_dropout separately (0 for Weather)
        self.head = FlattenHead(c["enc_in"], self.head_nf, c["pred_len"],
                                head_dropout=c.get("head_dropout", c["dropout"]))

    def forward(self, x_enc):
        x_enc, means, stdev = instance_norm(x_enc)
        x_enc = x_enc.permute(0, 2, 1)                                       # (B, C, L)
        enc_out, n_vars = self.patch_embedding(x_enc)                        # (B*C, patch_num, d_model)
        enc_out, _ = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)                                # (B, C, d_model, patch_num)
        dec_out = self.head(enc_out).permute(0, 2, 1)                        # (B, pred_len, C)
        return instance_denorm(dec_out, means, stdev, self.pred_len)


# ---------------------------------------------------------------- iTransformer (TSLib models/iTransformer.py)
class iTransformer(nn.Module):
    """Paper: https://arxiv.org/abs/2310.06625. Each variate's whole window is one token."""

    def __init__(self, c):
        super().__init__()
        self.seq_len, self.pred_len = c["seq_len"], c["pred_len"]
        self.enc_embedding = DataEmbedding_inverted(c["seq_len"], c["d_model"], c["dropout"])
        self.encoder = Encoder(
            [EncoderLayer(full_attention_layer(c), c["d_model"], c["d_ff"], dropout=c["dropout"],
                          activation=c["activation"]) for _ in range(c["e_layers"])],
            norm_layer=torch.nn.LayerNorm(c["d_model"]))
        self.projection = nn.Linear(c["d_model"], c["pred_len"], bias=True)

    def forward(self, x_enc):
        x_enc, means, stdev = instance_norm(x_enc)
        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        return instance_denorm(dec_out, means, stdev, self.pred_len)


# ---------------------------------------------------------------- TimeXer (TSLib models/TimeXer.py, features 'M')
class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                                   # x (B, C, L)
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class XEncoderLayer(nn.Module):
    """TimeXer's EncoderLayer: self-attention over the patch tokens + the global token, then cross-attention of the
    global token to the variate (exogenous) tokens."""

    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
        x = self.norm1(x)
        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(x_glb, cross, cross, attn_mask=cross_mask, tau=tau,
                                                       delta=delta)[0])
        x_glb_attn = torch.reshape(x_glb_attn, (x_glb_attn.shape[0] * x_glb_attn.shape[1],
                                                x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)
        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class XEncoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x


class TimeXer(nn.Module):
    """Paper: https://arxiv.org/abs/2402.19072. features 'M': every channel is an endogenous series (patch tokens +
    a global token) and, through DataEmbedding_inverted, also an exogenous variate token. seq_len must be a multiple
    of patch_len: TSLib unfolds from the start without padding, so a remainder would drop the latest samples."""

    def __init__(self, c):
        super().__init__()
        if c["seq_len"] % c["patch_len"]:
            raise ValueError(f"TimeXer: seq_len {c['seq_len']} is not a multiple of patch_len {c['patch_len']}")
        self.seq_len, self.pred_len = c["seq_len"], c["pred_len"]
        self.patch_len = c["patch_len"]
        self.patch_num = int(c["seq_len"] // c["patch_len"])
        self.n_vars = c["enc_in"]
        self.en_embedding = EnEmbedding(self.n_vars, c["d_model"], self.patch_len, c["dropout"])
        self.ex_embedding = DataEmbedding_inverted(c["seq_len"], c["d_model"], c["dropout"])
        self.encoder = XEncoder(
            [XEncoderLayer(full_attention_layer(c), full_attention_layer(c), c["d_model"], c["d_ff"],
                           dropout=c["dropout"], activation=c["activation"]) for _ in range(c["e_layers"])],
            norm_layer=torch.nn.LayerNorm(c["d_model"]))
        self.head_nf = c["d_model"] * (self.patch_num + 1)
        self.head = FlattenHead(c["enc_in"], self.head_nf, c["pred_len"], head_dropout=c["dropout"])

    def forward(self, x_enc):                               # forecast_multi with use_norm = 1
        x_enc, means, stdev = instance_norm(x_enc)
        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc, None)
        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out).permute(0, 2, 1)
        dec_out = instance_denorm(dec_out, means, stdev, self.pred_len)
        return dec_out[:, -self.pred_len:, :]


MODELS = {"patchtst": PatchTST, "itransformer": iTransformer, "timexer": TimeXer}

# Configurations of the official long-term-forecasting scripts for the Weather data set (21 physical channels, long
# series), the closest of the benchmark sets to this task, checked against the fetched scripts on 08/11/2026:
#   PatchTST     yuqinie98/PatchTST PatchTST_supervised/scripts/PatchTST/weather.sh: d_model 128, n_heads 16, e_layers 3,
#                d_ff 256, dropout 0.2, head_dropout 0, patch_len 16, stride 8 (fc_dropout 0.2 only feeds the pretraining head)
#   iTransformer thuml/iTransformer scripts/multivariate_forecasting/Weather/iTransformer.sh: d_model 512, d_ff 512,
#                e_layers 3; n_heads 8, dropout 0.1 from run.py
#   TimeXer      thuml/TimeXer scripts/multivariate/Weather/TimeXer.sh, pred_len 96 (the closest to 100): d_model 256,
#                e_layers 1, d_ff 512; n_heads 8, dropout 0.1, patch_len 16 from run.py -> 15 here so that it divides 300
# activation gelu and factor (unused by FullAttention) from run.py. Their seq_len is 96 (PatchTST 336); here every model
# gets the same 300 samples (60 s). Learning rate, batch size and epochs: see tsf_baselines.py.
CONFIGS = {
    "patchtst": dict(d_model=128, n_heads=16, e_layers=3, d_ff=256, dropout=0.2, head_dropout=0.0, patch_len=16, stride=8),
    "itransformer": dict(d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.1),
    "timexer": dict(d_model=256, n_heads=8, e_layers=1, d_ff=512, dropout=0.1, patch_len=15),
}
COMMON = dict(factor=1, activation="gelu")


def config(name, seq_len, pred_len, enc_in, **over):
    c = dict(COMMON, **CONFIGS[name], seq_len=seq_len, pred_len=pred_len, enc_in=enc_in)
    c.update(over)
    return c


def build(name, seq_len, pred_len, enc_in, **over):
    c = config(name, seq_len, pred_len, enc_in, **over)
    return MODELS[name](c), c


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
