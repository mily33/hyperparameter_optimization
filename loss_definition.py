from torch.distributions import Normal
import torch.nn as nn
import torch


class PolicyLoss(nn.Module):
    def __init__(self):
        super(PolicyLoss, self).__init__()

    def forward(self, data, mu, sigma):
        distribution = Normal(mu, sigma)
        loss_value = torch.mean(distribution.log_prob(data[:, 0]) * data[:, 1])
        return loss_value
