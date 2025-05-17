import math
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from einops import repeat, einsum, rearrange
from mamba_ssm.ops.triton.layernorm import RMSNorm
from models.HSMP.tools import _init_weights_residual


class FusionMamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=1, norm_epsilon=1e-5, rms_norm=False):
        super(FusionMamba, self).__init__()
        self.d_model = d_model  # Enter feature dimensions
        self.d_state = d_state  # Dimensions of state space model
        self.d_conv = d_conv  # Convolution kernel size
        self.expand = expand  # Feature expansion multiple
        self.dt_rank = math.ceil(d_model / 16)
        self.d_inner = int(self.expand * self.d_model)  # Extended internal dimensions

        # Linear projection layer, expanding input features to twice the internal dimension
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, bias=True,
                                kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1)

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)

        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))  # e * d_model, e * d_model

        self.hidden_proj = nn.Linear(self.d_model, self.d_state * self.d_inner, bias=False)
        self.history_cond_proj = nn.Linear(self.d_state * self.d_inner, self.d_model, bias=False)

        self.silu = nn.SiLU()

        # Linear projection of the output layer, projecting the internal dimension back to the original feature dimension
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.norm = (nn.LayerNorm if not rms_norm else RMSNorm)(d_model, eps=norm_epsilon)
        self.apply(partial(_init_weights_residual,n_layer=1))


    def ssm(self, x, space_cond):
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)

        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        h = self.hidden_proj(space_cond)  # (b, d_in * n)
        h = rearrange(h, 'b (d_in n) -> b d_in n', d_in=self.d_inner, n=self.d_state)
        # h = repeat(h, 'n -> d n', d=self.d_inner)

        y, hidden_state = self.selective_scan(x, delta, A, B, C, D, h)

        return y, hidden_state


    def selective_scan(self, u, delta, A, B, C, D, hidden_state):
        (b, l, d_in) = u.shape  # (B,L,D)
        n = A.shape[1]  # N

        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))  # (B,L,D) * (D,N) -> (B,L,D,N)

        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')  # (B,L,D)*(B,L,N)*(B,L,D)->(B,L,D,N)

        ys = []
        for i in range(l):
            hidden_state = deltaA[:, i] * hidden_state + deltaB_u[:, i]  # x(t + 1) = Ax(t) + Bu(t)
            y = einsum(hidden_state, C[:, i, :], 'b d_in n, b n -> b d_in')  # y(t) = Cx(t)  (B,D,N)*(B,N)->(B,D)
            ys.append(y)
        y = torch.stack(ys, dim=1)  # (b, l, d_in)  (B,L,D)
        y = y + u * D  # y(t) = Cx(t)+Du(t)
        return y, hidden_state # (B,L,D)


    def forward(self, x, space_cond):
        space_cond = space_cond.squeeze(dim=1)
        (b, l, d) = x.shape  # shape (b,l,d)
        res_x = x

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)  # x: (b,l,d_in), res: (b,l,d_in)

        x = rearrange(x, 'b l d_in -> b d_in l')  # shape (b,l,d_in)->(b,d_in,l)
        x = self.conv1d(x)[:, :, :l]  # (b,d_in,l)
        x = rearrange(x, 'b d_in l -> b l d_in')  # (b,d_in,l)->(b,l,d_in)
        # x = F.silu(x)  # (b,l,d_in)
        x = self.silu(x)
        y, hidden_state = self.ssm(x, space_cond)  # (b,l,d_in) (b,d_in,n)
        # y = y * F.silu(res)  # (b,l,d_in)
        y = y * self.silu(res)
        output = self.out_proj(y)  # (b,l,d_in)-> (b,l,d)

        output = self.norm(output + res_x)


        history_cond = rearrange(hidden_state, 'b d_in n -> b (d_in n)', d_in=self.d_inner, n=self.d_state)
        history_cond = self.history_cond_proj(history_cond)
        history_cond = history_cond.unsqueeze(1)

        return output, history_cond  


if __name__ == '__main__':
    model = FusionMamba(d_model=256)
    a = torch.randn(3, 5, 256)
    b = torch.randn(3, 1, 256)
    out, h = model(a, b)
    print(out.shape)
    print(h.shape)