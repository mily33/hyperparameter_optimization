from opt import Flag
import torch
from gaussian_process import SampleFunctions


def main():
    sf = SampleFunctions(5)
    sf.sample(16)
    x = torch.Tensor([[0.1], [0.8]])
    y = sf.predict(x)
    print(y)


if __name__ == '__main__':
    main()
