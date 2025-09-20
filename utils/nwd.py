import torch
import torch.nn as nn
import torch.nn.functional as F
from .grl import WarmStartGradientReverseLayer


class NuclearWassersteinDiscrepancy(nn.Module):
    def __init__(self, classifier: nn.Module):
        super(NuclearWassersteinDiscrepancy, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.classifier = classifier

    @staticmethod
    def n_discrepancy(y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        pre_s, pre_t = F.softmax(y_s, dim=1), F.softmax(y_t, dim=1)
        loss = (-torch.norm(pre_t, 'nuc') + torch.norm(pre_s, 'nuc')) / y_t.shape[0]
        return loss

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        f_grl = self.grl(f)
        y = self.classifier(f_grl, not_skip=True)
        y_s, y_t = y.chunk(2, dim=0)

        loss = self.n_discrepancy(y_s, y_t)
        return loss

class NWD_Ours(nn.Module):
    def __init__(self, classifier: nn.Module):
        super(NWD_Ours, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.classifier = classifier

    @staticmethod
    def n_discrepancy(y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        pre_s, pre_t = F.softmax(y_s, dim=1), F.softmax(y_t, dim=1)
        loss = -torch.norm(pre_t, 'nuc') /  y_t.shape[0] + torch.norm(pre_s, 'nuc') / y_s.shape[0]
        return loss

    def forward(self, f_s, f_t):
        f_grl_s = self.grl(f_s)
        f_grl_t = self.grl(f_t)
        y_s = self.classifier(f_grl_s)
        y_t = self.classifier(f_grl_t)
        loss = self.n_discrepancy(y_s, y_t)
        return loss

class NWD_New(nn.Module):
    def __init__(self, classifier: nn.Module):
        super(NWD_New, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.classifier = classifier

    @staticmethod
    def n_discrepancy(y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        pre_s, pre_t = F.softmax(y_s, dim=1), F.softmax(y_t, dim=1)
        loss = -torch.norm(pre_t, 'nuc') /  y_t.shape[0] + torch.norm(pre_s, 'nuc') / y_s.shape[0]
        return loss

    def forward(self, f_s, f_t, f_s_c, f_t_c):
        f_grl_s = self.grl(f_s)
        f_grl_t = self.grl(f_t)
        f_grl_s_c = self.grl(f_s_c)
        f_grl_t_c = self.grl(f_t_c)
        y_s = self.classifier(f_grl_s)
        y_t = self.classifier(f_grl_t)
        y_s_c = self.classifier(f_grl_s_c)
        y_t_c = self.classifier(f_grl_t_c)
        s_comb = torch.cat([y_s,y_s_c], dim=0)
        t_comb = torch.cat([y_t,y_t_c], dim=0)
        loss = self.n_discrepancy(s_comb, t_comb) + 0.2*self.n_discrepancy(y_t, y_t_c)
        return loss

class NWD_Seg(nn.Module):
    def __init__(self, classifier: nn.Module):
        super(NWD_Seg, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.classifier = classifier

    @staticmethod
    def n_discrepancy(y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        pre_s, pre_t = F.softmax(y_s, dim=1), F.softmax(y_t, dim=1)
        loss = -torch.norm(pre_t, 'nuc') /  y_t.shape[0] + torch.norm(pre_s, 'nuc') / y_s.shape[0]
        return loss

    def forward(self, f_s, f_t, f_s_c, f_t_c, target_weight=0.2):
        f_grl_s = self.grl(f_s)
        f_grl_t = self.grl(f_t)
        f_grl_s_c = self.grl(f_s_c)
        f_grl_t_c = self.grl(f_t_c)
        y_s = self.classifier(f_grl_s)
        y_s = y_s.reshape(-1, y_s.size(-1))
        y_t = self.classifier(f_grl_t)
        y_t = y_t.reshape(-1, y_t.size(-1))
        y_s_c = self.classifier(f_grl_s_c)
        y_s_c = y_s_c.reshape(-1, y_s_c.size(-1))
        y_t_c = self.classifier(f_grl_t_c)
        y_t_c = y_t_c.reshape(-1, y_t_c.size(-1))
        s_comb = torch.cat([y_s,y_s_c], dim=0)
        t_comb = torch.cat([y_t,y_t_c], dim=0)
        loss = self.n_discrepancy(s_comb, t_comb) + target_weight*self.n_discrepancy(y_t, y_t_c)
        return loss