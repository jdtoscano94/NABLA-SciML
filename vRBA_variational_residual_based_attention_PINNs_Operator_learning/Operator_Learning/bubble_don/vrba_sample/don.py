import torch
import torch.nn as nn
import numpy as np
import sys

class DeepONet(nn.Module):
    def __init__(self,par):
        super(DeepONet,self).__init__()
        torch.manual_seed(23)

        self.par = par

        bn_res = self.par['bn_res']
        tn_res = self.par['tn_res']
        ld     = self.par['ld']

        self.branch_net = nn.Sequential(
                     nn.Linear(bn_res, 100),
                     nn.GELU(),
                     nn.Linear(100,100),
                     nn.GELU(),
                     nn.Linear(100,100),
                     nn.GELU(),
                     nn.Linear(100,100),
                     nn.GELU(),
                     nn.Linear(100,ld),
                     )
        
        self.trunk_net = nn.Sequential(
                     nn.Linear(tn_res, 100),
                     nn.GELU(),
                     nn.Linear(100,100),
                     nn.GELU(),
                     nn.Linear(100,100),
                     nn.GELU(),
                     nn.Linear(100,100),
                     nn.GELU(),
                     nn.Linear(100,ld),
                     )

    def forward(self, X_func, X_loc):

        # X_func = (X_func - self.par['X_func_shift'])/self.par['X_func_scale']
        # X_loc  = (X_loc  - self.par['X_loc_shift'] )/self.par['X_loc_scale'] 

        bn = self.branch_net(X_func)
        tn = self.trunk_net(X_loc)

        out = torch.einsum('bj,nj->bn', bn, tn)

        # out = out*self.par['out_scale'] + self.par['out_shift']
        
        return out
