import torch
from torch.nn import Module
from models.other_models.TrackSSM.motion_decoder import Time_info_decoder
from models.other_models.TrackSSM.mamba_encoder import Time_info_aggregation
import numpy as np



class TrackSSM(Module):
    def __init__(self):
        super().__init__()
        self.encoder = Time_info_aggregation()
        self.ssm_decoder = Time_info_decoder()

    def forward(self, conds):
        cond_flow = self.encoder(conds)
        track_pred = self.ssm_decoder(conds[:, -1, :4], cond_flow)

        return track_pred






