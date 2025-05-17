import torch
from torch.nn import Module
from models.other_models.DiffMOT import diffusion
from models.other_models.DiffMOT.diffusion import VarianceSchedule, D2MP_OB
from models.other_models.DiffMOT.condition_embedding import History_motion_embedding
import numpy as np

class DiffMOT(Module):
    def __init__(self):
        super().__init__()
        # self.config = config
        # self.device = device
        self.encoder = History_motion_embedding()
        self.diffnet = getattr(diffusion, "HMINet")

        self.diffusion = D2MP_OB(
            # net = self.diffnet(point_dim=2, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False),
            net=self.diffnet(point_dim=4, context_dim=256, tf_layer=3, residual=False),
            var_sched = VarianceSchedule(
                num_steps=100,
                beta_T=5e-2,
                mode='linear'
            ),
            # config=self.config
            config = None
        )


    def forward(self, conds):
        cond_encodeds = self.encoder(conds)
        box = conds[:,-1,:]
        track_pred = self.diffusion.sample(cond_encodeds, sample=1, bestof=True, flexibility=0.0, ret_traj=False)
        track_pred = track_pred.mean(0)
        track_pred = track_pred + box[:,:4]

        return track_pred


