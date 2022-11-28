from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class MIXSTDLoss(nn.Module):
    def __init__(self,opt):
        super(MIXSTDLoss, self).__init__()
        self.opt = opt
        self.cross_ent = nn.CrossEntropyLoss()
        self.KL = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, logit_s, logit_t, target):

        stdt = torch.std(logit_t, dim=-1,keepdim=True)
        stds = torch.std(logit_s, dim=-1, keepdim=True)

        ## CLS ##        
        loss = -F.log_softmax(logit_s/stds,-1) * target
        loss_cls = self.opt.gamma * (torch.sum(loss))/logit_s.shape[0]        
        ## STD KD ## 
        p_s = F.log_softmax(logit_s/stds, dim=1)
        p_t = F.softmax(logit_t/stdt, dim=1)
        std_KD = self.KL(p_s, p_t) 
        loss_div = self.opt.alpha * std_KD

        return loss_cls, loss_div
