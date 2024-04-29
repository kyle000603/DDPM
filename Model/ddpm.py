import torch
import torch.nn as nn


class DDPM(nn.Module):
    def __init__(self, network:nn.Module, device:torch.device, steps=200,
                  min_beta=0.0001, max_beta=0.02, image_chw=(1, 28, 28)) -> None:
        super(DDPM, self).__init__()
        self.network = network.to(device)
        self.steps = steps
        self.device = device
        self.betas = torch.linspace(min_beta, max_beta, steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.Tensor([torch.prod(self.alphas[:i+1]) \
                                        for i in range(len(self.alphas))]).to(device)
        self.img_chw = image_chw

    def forward(self, x0:torch.Tensor, t, eta=None):
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]
        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)
        # 
        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy
    
    def backward(self, x, t):
        return self.network(x, t)