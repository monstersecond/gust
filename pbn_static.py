"""
Descript the network structure
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


thresh1 = 1
thresh2 = -1
lens1 = 0.5 
lens2 = 1
decay = 0
tau_rfr = 9

device = utils.PARAM["device"]


class ActFun1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh1).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh1) < lens1
        return grad_input * temp.float()

class ActFun2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input): 
        ctx.save_for_backward(input)
        return input.gt(thresh2).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = input == 0
        return grad_input * temp.float()


act_fun1 = ActFun1.apply
act_fun2 = ActFun2.apply


def mem_update(x, ctx, rfr, mmax, device=utils.PARAM["device"]):
    p = x * ctx
    noise = torch.rand(p.size(), device=device)
    mem = p + noise
    spike = act_fun1(mem)* act_fun2(-rfr)
    rfr = rfr - 1 + tau_rfr * spike
    rfr = mmax(rfr)
    return spike, rfr


class PBN2(nn.Module):
    def __init__(self, input_size, hidden_size, T):
        super(PBN2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T

        self.mmax = nn.ReLU()
        self.T = T
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_size),
            nn.Sigmoid()
        )

        self.encoder_core = nn.Linear(input_size, hidden_size)
        self.RNN_activation=nn.Sigmoid()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, input_size),
            nn.Sigmoid()
        )

        self.sample = mem_update

    def forward(self, x, hidden, rfr, IT, p=0.5, device=utils.PARAM["device"]):
        T = self.T
        batch_size = x.shape[0]
        spike = torch.zeros(batch_size, self.T, self.input_size, device=device)
        spike_pre = torch.zeros(batch_size, self.T, self.input_size, device=device)
        spk_all = torch.zeros(IT,self.T, self.input_size)
        ctx_some = torch.zeros(batch_size, self.T, self.input_size, device=device)
        for it in range(IT):
            spike_preit = spike
            for t in range(T):
                ctx = self.decoder(hidden[:, t, :].clone())
                ctx_some[:, t, :] = ctx
                decay_ratio = torch.rand((batch_size, 784), device=device)
                decay_mask = (decay_ratio < p).float()
                spike_pre = spike[:, t - 1, :] * decay_mask
                spike[:, t, :], rfr = self.sample(x, ctx.clone(), rfr.clone(), mmax=self.mmax, device=device)
                hidden[:, t, :] = self.encoder(spike[:, t, :].clone() + spike_pre.clone())
            spk_all[it, :, :] = spike[0, :, :].detach().cpu()
        return spike, hidden, rfr, ctx_some, spk_all, spike_preit
