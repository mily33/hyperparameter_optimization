import torch
from torch.distributions import Normal
import numpy as np
from scipy.stats import norm
from gaussian_process import SampleFunctions


class Memory(object):
    def __init__(self, size):
        self.size = size
        self.sf = SampleFunctions(5)
        self.sf.sample(self.size)
        self.x = np.atleast_2d(10 * np.random.rand(self.size)).T
        self.y = self.sf.predict(self.x)
        self.data = np.concatenate((self.x, self.y), 1)
        self.data = self.data[:, :, np.newaxis]
        self.w = np.ones_like(self.x) / self.x.shape[1]
        self.t = np.zeros_like(self.x)

    def process(self, mu, sigma):
        m = Normal(mu, sigma)
        x = m.sample()
        y = np.float32(self.sf.predict(x))
        self.add_sample(x, y)
        out = np.concatenate((x, y), 1)
        self.update_weight(mu, sigma)
        sample_output = self.sample_data()
        return torch.from_numpy(out), sample_output

    def add_sample(self, x, y):
        data = np.concatenate((x, y), 1)
        self.data = np.dstack((self.data, data))
        self.t = np.concatenate((self.t + 1, np.ones(shape=(self.x.shape[0], 1), dtype=np.int32)), 1)
        self.w = np.concatenate((self.w, np.zeros(shape=(self.w.shape[0], 1), dtype=np.float32)), 1)

    def sample_data(self):
        out = []
        for i in range(self.w.shape[0]):
            idx = np.random.choice(len(self.w[i]), 1, p=self.w[i])
            out.append(self.data[i, :, idx][0])
        return out

    def update_weight(self, mu, sigma):
        x = self.data[:, 0, :]
        for i in range(x.shape[0]):
            p = norm.pdf(x[i, :], loc=mu[i].data.numpy(), scale=sigma[i].data.numpy())
            exp_p = np.exp(p)
            self.w[i, :] = exp_p / np.sum(exp_p)
