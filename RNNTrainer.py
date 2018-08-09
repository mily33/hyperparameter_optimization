import torch
import time
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loss_definition import PolicyLoss


class RNNTrainer(object):

    def train(self, model, checkpoint, trMaxEpoch,
              nnClassCount, trBatchSize,
              launchTimestamp, loss_form):

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

            self.epoch_train(model, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
            lossVal, losstensor = self.epoch_val(model, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime

            scheduler.step(losstensor)

            if lossVal < lossMIN:
                lossMIN = lossVal
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN,
                            'optimizer': optimizer.state_dict()}, 'm-' + launchTimestamp + '.pth.tar')
                print('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
            else:
                print('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))

                # --------------------------------------------------------------------------------

    def epoch_train(self, model, optimizer, scheduler, trMaxEpoch, nnClassCount, loss):
        model.train()
        for batchID, (input, target) in enumerate(dataLoader):

            varInput = torch.autograd.Variable(input)
            data, mu, sigma = model(varInput)

            lossvalue = loss(data, mu, sigma)
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()

    def epoch_val(self, model, optimizer, scheduler, trMaxEpoch, nnClassCount, loss):
        lossVal, losstensor = 0, 0
        return lossVal, losstensor