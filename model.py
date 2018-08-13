import torch
import torch.nn as nn
from torch.distributions import Normal
from external_memory import Memory


def job(q, mu, sigma, sf):
    distribution = Normal(mu, sigma)
    x = distribution.sample()
    y = sf.predict(x)
    q.put([x, y])


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, time_step):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.time_step = time_step
        self.rnn = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)
        self.linear = nn.Sequential(nn.Linear(self.hidden_size, 2 * (self.input_size - 1)), nn.Softmax())
        self.memory = None

    def forward(self, in_put):
        output = []
        h = torch.autograd.Variable(torch.zeros(in_put.size(0), self.hidden_size))
        c = torch.autograd.Variable(torch.zeros(in_put.size(0), self.hidden_size))
        mu, sigma = None, None
        mus = []
        for i in range(self.time_step):
            h, c = self.rnn(in_put, (h, c))
            mu_sigma = self.linear(h)
            mu = mu_sigma[:, 0:self.input_size - 1]
            sigma = mu_sigma[:, 0:self.input_size - 1]
            next_input, data = self.memory.process(mu, sigma)
            mus.append(10 * mu)
            output.append(data[0])
            in_put = torch.FloatTensor(next_input)
        output = torch.Tensor(output)
        mus = torch.Tensor(mus)
        return output, mu, sigma, mus

