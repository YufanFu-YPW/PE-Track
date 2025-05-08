import torch
import torch.nn as nn
import math
from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from models.HSMP.tools import _init_weights_residual


class GeneralMambaBlock(nn.Module):
    def __init__(self, d_model, norm_epsilon=1e-5, rms_norm=False, residual_in_fp32=False, fused_add_norm=True):
        super(GeneralMambaBlock, self).__init__()

        norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
        self.mamba_block = Block(d_model, mixer_cls=Mamba, norm_cls=norm_cls, 
                                 fused_add_norm=fused_add_norm, residual_in_fp32=residual_in_fp32)

    def forward(self, x, residual, inference_params=None):
        return self.mamba_block(x, residual, inference_params)


class GeneralHistoryEncoder(nn.Module):
    def __init__(self, d_model, n_layers, norm_epsilon=1e-5, rms_norm=False, residual_in_fp32=False, fused_add_norm=True):
        super(GeneralHistoryEncoder, self).__init__()
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.layers = nn.ModuleList([
            GeneralMambaBlock(d_model, norm_epsilon, rms_norm, residual_in_fp32, fused_add_norm)
            for _ in range(n_layers)
        ])
        self.norm = (nn.LayerNorm if not rms_norm else RMSNorm)(d_model, eps=norm_epsilon)

        self.apply(partial(_init_weights_residual,n_layer=n_layers))

    def forward(self, x, inference_params=None):
        # x: (batch_size, seq_len, d_model)
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual, inference_params=inference_params)

        if not self.fused_add_norm:
            residual = (x + residual) if residual is not None else x
            x = self.norm(residual.to(dtype=self.norm.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            x = fused_add_norm_fn(
                x,
                self.norm.weight,
                self.norm.bias,
                eps=self.norm.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return x


class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super(BiMambaBlock, self).__init__()
        self.forward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.proj = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)

        forward_out = self.forward_mamba(x)

        reversed_x = torch.flip(x, dims=[1])
        backward_out = self.backward_mamba(reversed_x)
        backward_out = torch.flip(backward_out, dims=[1])

        combined = torch.cat([forward_out, backward_out], dim=-1)
        projected = self.proj(combined)

        return projected + residual


class BiHistoryEncoder(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, n_layers=2):
        super(BiHistoryEncoder, self).__init__()
        self.layers = nn.ModuleList([
            BiMambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    






if __name__ == '__main__':
    model = GeneralHistoryEncoder(d_model=256, n_layers=2, fused_add_norm=True, rms_norm=True).cuda()
    # model = BiHistoryEncoder(d_model=256).cuda()
    a = torch.randn(3, 5, 256)
    out = model(a.cuda())
    print(out.shape)
