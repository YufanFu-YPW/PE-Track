import torch
import torch.nn as nn
from models.HSMP.history_mamba import GeneralHistoryEncoder, BiHistoryEncoder
from models.HSMP.space_attention import SpaceAttBlock
from models.HSMP.fusion_block import FusionMamba
from mamba_ssm.ops.triton.layernorm import RMSNorm
from models.HSMP.tools import _init_weights
from functools import partial

class HSBlock(nn.Module):
    def __init__(self, d_model=256, d_state=16, d_conv=4, expand=2, fusion_expand=1, mamba_layers=2,
                 bi_mamba=False, heads=8, norm_epsilon=1e-5, rms_norm=False, dropout=0.0):
        super(HSBlock, self).__init__()
        if bi_mamba:
            self.mamba_encoder = BiHistoryEncoder(d_model, d_state, d_conv, expand, mamba_layers)
        else:
            self.mamba_encoder = GeneralHistoryEncoder(d_model, mamba_layers, norm_epsilon, rms_norm)

        self.att_encoder = SpaceAttBlock(d_model, heads, norm_epsilon, rms_norm, dropout)

        self.fusion = FusionMamba(d_model, d_state, d_conv, fusion_expand, norm_epsilon, rms_norm)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model)
        )
        self.norm = (nn.LayerNorm if not rms_norm else RMSNorm)(d_model, eps=norm_epsilon)

    def forward(self, s, q, k):
        mamba_output =self.mamba_encoder(s)

        att_output = self.att_encoder(q, k)

        fusion_s, history_cond = self.fusion(mamba_output, att_output)

        new_q = self.feed_forward(history_cond)
        new_q = self.norm(att_output + new_q)

        return fusion_s, new_q
        # return mamba_output, att_output


class PreBlock(nn.Module):
    def __init__(self, d_model=256, d_state=16, d_conv=4, expand=2, mamba_layers=2, bi_mamba=False, heads=8, 
                 norm_epsilon=1e-5, rms_norm=False, dropout=0.0):
        super(PreBlock, self).__init__()
        self.d_model = d_model
        if bi_mamba:
            self.mamba_encoder = BiHistoryEncoder(d_model, d_state, d_conv, expand, mamba_layers)
        else:
            self.mamba_encoder = GeneralHistoryEncoder(d_model, mamba_layers, norm_epsilon, rms_norm)

        self.att_encoder = SpaceAttBlock(d_model, heads, norm_epsilon, rms_norm, dropout)
        self.pre_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, s, q, k):
        b = s.shape[0]
        mamba_output = self.mamba_encoder(s)
        mamba_output = mamba_output[:, -1].view(b, 1, self.d_model).squeeze(dim=1)
        att_output = self.att_encoder(q, k).squeeze(dim=1)
        mix = torch.cat((mamba_output, att_output), -1)
        output = self.pre_head(mix)

        return output


class HSModel(nn.Module):
    def __init__(self, d_model=256, v_size=8, block_layers=1, d_state=16, d_conv=4, expand=2,
                 fusion_expand=2, mamba_layers=2, pre_mamba_layers=2, bi_mamba=False, heads=8,  
                 norm_epsilon=1e-5, rms_norm=True, dropout=0.0):
        super(HSModel, self).__init__()
        self.embedding = nn.Embedding(v_size, d_model)
        self.q_proj = nn.Linear(v_size, d_model, bias=False)
        self.k_proj = nn.Linear(v_size, d_model, bias=False)

        self.layers = nn.ModuleList([
            HSBlock(d_model, d_state, d_conv, expand, fusion_expand, mamba_layers, bi_mamba, heads,  norm_epsilon, rms_norm, dropout)
            for _ in range(block_layers)
        ])

        self.pre_layer = PreBlock(d_model, d_state, d_conv, expand, pre_mamba_layers, bi_mamba, heads, norm_epsilon, rms_norm, dropout)
        self.apply(partial(_init_weights))
        


    def forward(self, history, space):
        b = history.shape[0]
        s = torch.bmm(history, self.embedding.weight.clone().unsqueeze(0).repeat(b, 1, 1))# b l d_model
        box = history[:, -1, :]  # [batch_size,8]
        q = box.unsqueeze(1)  # [batch_size,1,8]
        q = self.q_proj(q)
        k = self.k_proj(space)  # [batch_size,space_len,8]

        for layer in self.layers:
            s, q = layer(s, q, k)

        offsets = self.pre_layer(s, q, k)
        output = box[:, :4] + offsets

        return output






if __name__ == '__main__':
    model = HSModel(d_model=256, rms_norm=True).cuda()
    print(model)
    a = torch.randn(3, 5, 8)
    b = torch.randn(3, 10, 8)
    out = model(a.cuda(), b.cuda())
    print(out.shape)









