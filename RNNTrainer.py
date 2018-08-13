import torch
import time
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loss_definition import PolicyLoss
import numpy as np
from matplotlib import pyplot as plt
from external_memory import Memory
import re


class RNNTrainer(object):

    def train(model, checkpoint, trMaxEpoch, trBatchSize, launchTimestamp):

        # -------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min')

        loss = PolicyLoss()
        # ---- Load checkpoint
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])

        # ---- TRAIN THE NETWORK

        lossMIN = 100000

        for epochID in range(0, trMaxEpoch):
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime

            mean_loss = RNNTrainer.epoch_train(model, optimizer, trBatchSize, trMaxEpoch, loss)

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime

            scheduler.step(mean_loss)

            torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN,
                        'optimizer': optimizer.state_dict()}, 'm-' + launchTimestamp + '.pth.tar')
            print('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(mean_loss))

            # if mean_loss < lossMIN:
            #     lossMIN = mean_loss
            #     torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN,
            #                 'optimizer': optimizer.state_dict()}, 'm-' + launchTimestamp + '.pth.tar')
            #     print('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(mean_loss))
            # else:
            #     print('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(mean_loss))

            # --------------------------------------------------------------------------------

    def epoch_train(model, optimizer, batchSize, maxIter, loss):
        model.train()
        loss_mean = 0
        for i in range(maxIter):
            model.memory = Memory(batchSize)
            input = model.memory.sample_data()
            varInput = torch.autograd.Variable(torch.FloatTensor(input))
            data, mu, sigma, _ = model(varInput)
            lossvalue = loss(data, mu, sigma)
            loss_mean += lossvalue.item()
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()

        return loss_mean / maxIter

    def test(model, pathModel):

        checkpoint = torch.load(pathModel)

        model.load_state_dict(checkpoint['state_dict'])

        model.eval()
        print('Start eval......')
        model.memory = Memory(1)
        model.memory.sf.sample(1)
        input = model.memory.sample_data()
        varInput = torch.autograd.Variable(torch.Tensor(input))
        data, mu, _, mus = model(varInput)
        x = np.atleast_2d(np.linspace(0, 10, 1000)).T
        y = model.memory.sf.function_list[0].predict(x)
        dots = model.memory.sf.function_list[0].predict(np.atleast_2d(mus).T)
        plt.plot(x, y)
        plt.scatter(mus.data.numpy(), dots)
        plt.scatter(mus[-1].data.numpy(), dots[-1], c='r')
        plt.show()
