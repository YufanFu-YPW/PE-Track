import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.triton.layernorm import RMSNorm
# from tools import _init_weights
# from functools import partial

class MultiHeadAttentionWithThreshold(nn.Module):
    def __init__(self, d_model, h=8, dropout=0.0):
        super(MultiHeadAttentionWithThreshold, self).__init__()
        self.d_k = d_model // h
        self.h = h

        # 线性变换层保持不变
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.raw_thresholds = nn.Parameter(torch.Tensor(h, 1, 1))
        nn.init.constant_(self.raw_thresholds, 0.1)

        self.dropout = nn.Dropout(dropout)
        self.softplus = nn.Softplus(beta=1, threshold=20)  # Softplus

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        thresholds = self.softplus(self.raw_thresholds).unsqueeze(0)

        # Apply threshold filtering (retain only scores larger than the threshold)
        scores = scores.masked_fill(scores <= thresholds, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        x = torch.matmul(attention, V)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.W_o(x)


class SpaceAttBlock(nn.Module):
    def __init__(self, d_model, heads=8, norm_epsilon=1e-5, rms_norm=False, dropout=0.0):
        super(SpaceAttBlock, self).__init__()
        self.v_proj = nn.Linear(2 * d_model, d_model, bias=False)
        self.self_attn = MultiHeadAttentionWithThreshold(d_model, heads, dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model)
        )

        self.norm1 = (nn.LayerNorm if not rms_norm else RMSNorm)(d_model, eps=norm_epsilon)
        self.norm2 = (nn.LayerNorm if not rms_norm else RMSNorm)(d_model, eps=norm_epsilon)
        self.dropout = nn.Dropout(dropout)
        # self.apply(partial(_init_weights,n_layer=1))

    def forward(self, query, key, mask=None):
        v = query.repeat(1, key.shape[1], 1)
        v = torch.cat((key, v), 2)
        v = self.v_proj(v)

        attn_output = self.self_attn(query, key, v, mask)
        x = query + self.dropout(attn_output)
        x = self.norm1(x)

        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x
    






if __name__ == '__main__':
    model = SpaceAttBlock(d_model=256)
    a = torch.randn(3, 1, 256)
    b = torch.randn(3, 5, 256)
    out = model(a, b)
    print(out.shape)
