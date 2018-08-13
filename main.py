from opt import Flag
import time
from gaussian_process import SampleFunctions
from model import Model
from RNNTrainer import RNNTrainer


def main():
    model = Model(2, 64, 100)
    pathModel = './m-11082018-150904.pth.tar'
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    RNNTrainer.train(model=model, checkpoint=None, trMaxEpoch=100, trBatchSize=16, launchTimestamp=timestampLaunch)
    # RNNTrainer.test(model=model, pathModel=pathModel)


if __name__ == '__main__':
    main()
