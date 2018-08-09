# # pytorch代码示例
# ''' 对于最简单的 RNN，我们可以使用下面两种方式去调用，分别是 torch.nn.RNNCell() 和 torch.nn.RNN()， 这两种方式的区别在于 RNNCell() 只能接受序列中单步的输入，且必须传入隐藏状态， 而 RNN() 可以接受一个序列的输入，默认会传入全 0 的隐藏状态，也可以自己申明隐藏状态传入。 '''
# # rnn_single.weight_ih [torch.FloatTensor of size 200x100]
# # rnn_single.weight_hh [torch.FloatTensor of size 200x200]
# rnn_single = nn.RNNCell(input_size=100, hidden_size=200)
# # 构造一个序列，长为 6，batch 是 5， 特征是 100
# x = Variable(torch.randn(6, 5, 100)) # 这是 rnn 的输入格式
# # 定义初始的记忆状态
# h_t = Variable(torch.zeros(5, 200))
# # 传入 rnn
# out = []
# for i in range(6): # 通过循环 6 次作用在整个序列上
#     h_t = rnn_single(x[i], h_t)
#     out.append(h_t)
# # out.shape: torch.Size([6, 5, 200])
# # ------------------------------
# rnn_seq = nn.RNN(100, 200)
# out, h_t = rnn_seq(x) # 使用默认的全0隐藏状态
# # 自己定义初始的隐藏状态 [torch.FloatTensor of size 1x5x200]
# h_0 = Variable(torch.randn(1, 5, 200))
# out, h_t = rnn_seq(x, h_0)
# # out.shape: torch.Size([6, 5, 200])
import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

np.random.seed(1)


def f(x):
    """The function to predict."""
    return 2 * x * np.sin(x) * np.cos(x)


# ----------------------------------------------------------------------
#  First the noiseless case
# X = np.atleast_2d([1., 2., 3., 4., 5., 6., 7., 8.]).T
X = 10 * np.random.rand(4)
X = np.atleast_2d(X).T
# Observations
y = f(X).ravel()

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(0, 10, 1000)).T

# Instanciate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
fig = plt.figure()
plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
plt.plot(x, y_pred, 'b-', label=u'Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')

# ----------------------------------------------------------------------
# now the noisy case
X = np.linspace(0.1, 9.9, 20)
X = np.atleast_2d(X).T

# Observations and noise
y = f(X).ravel()
dy = 0.5 + 1.0 * np.random.random(y.shape)
noise = np.random.normal(0, dy)
y += noise

# Instanciate a Gaussian Process model
gp = GaussianProcessRegressor(kernel=kernel, alpha=(dy / y) ** 2,
                              n_restarts_optimizer=10)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
fig = plt.figure()
plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label=u'Observations')
plt.plot(x, y_pred, 'b-', label=u'Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')

plt.show()