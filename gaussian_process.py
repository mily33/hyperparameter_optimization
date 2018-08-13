import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
from sklearn import gaussian_process


class SampleFunctions(object):
    def __init__(self, num_prior):
        self.function = lambda x: x * np.sin(x) * np.cos(x)
        self.prior_x = np.atleast_2d(10 * np.random.rand(num_prior)).T
        self.prior_y = self.function(self.prior_x).ravel()
        self.kernels = [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
                   1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
                   1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
                                        length_scale_bounds=(0.1, 10.0),
                                        periodicity_bounds=(1.0, 10.0)),
                   ConstantKernel(0.1, (0.01, 10.0))
                   * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.0, 10.0)) ** 2),
                   1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
                                nu=1.5)]
        self.gp = gaussian_process.GaussianProcessRegressor(self.kernels[4])
        self.gp.fit(self.prior_x, self.prior_y)
        self.function_list = None
        self.query_x = None
        self.query_y = None

    def sample(self, num=1):
        self.query_x = np.atleast_2d(10 * np.random.rand(50)).T
        self.query_y = self.gp.sample_y(self.query_x, num)
        self.function_list = []

        for i in range(num):
            self.function_list.append(gaussian_process.GaussianProcessRegressor(self.kernels[4]))
            self.function_list[i].fit(self.query_x, self.query_y[:, i])
            # y.append(self.function_list[i].predict(x))
        #     plt.plot(x, y[i])
        # plt.show()

    def predict(self, x):
        y = []
        for i in range(len(self.function_list)):
            result = self.function_list[i].predict(np.atleast_2d(x[i]))
            y.append(result)
        return np.atleast_2d(y)
