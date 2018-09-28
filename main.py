import torch
from torch import nn
from torch.autograd import Variable
from torch import optim

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools

import utility as util
from modules import Encoder, Decoder

logger = util.setup_log()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info("Using computation device %s.", device)


class DaRnn:

    def __init__(self, file_data, logger, encoder_hidden_size=64, decoder_hidden_size=64, T=10,
                 learning_rate=0.01, batch_size=128, parallel=True, debug=False):
        self.T = T
        dat = pd.read_csv(file_data, nrows=100 if debug else None)
        self.logger = logger
        self.logger.info("Shape of data: %s.\nMissing in data: %s.", dat.shape, dat.isnull().sum().sum())

        self.X = dat.loc[:, [x for x in dat.columns.tolist() if x != 'NDX']].as_matrix()
        self.y = np.array(dat.NDX)
        self.batch_size = batch_size

        self.encoder = Encoder(input_size=self.X.shape[1], hidden_size=encoder_hidden_size, T=T,
                               logger=logger).to(device)
        self.decoder = Decoder(encoder_hidden_size=encoder_hidden_size,
                               decoder_hidden_size=decoder_hidden_size,
                               T=T, logger=logger).to(device)

        if parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder_optimizer = optim.Adam(
            params=itertools.filterfalse(lambda p: not p.requires_grad, self.encoder.parameters()),
            lr=learning_rate)
        self.decoder_optimizer = optim.Adam(
            params=itertools.filterfalse(lambda p: not p.requires_grad, self.decoder.parameters()),
            lr=learning_rate)
        # self.learning_rate = learning_rate

        self.train_size = int(self.X.shape[0] * 0.7)
        self.y = self.y - np.mean(self.y[:self.train_size])  # Question: why Adam requires data to be normalized?
        self.logger.info("Training size: %d.", self.train_size)

    def train(self, n_epochs=10):
        iter_per_epoch = int(np.ceil(self.train_size * 1. / self.batch_size))
        logger.info("Iterations per epoch: %3.3f ~ %d.", self.train_size * 1. / self.batch_size, iter_per_epoch)
        self.iter_losses = np.zeros(n_epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(n_epochs)

        self.loss_func = nn.MSELoss()

        n_iter = 0

        for i in range(n_epochs):
            perm_idx = np.random.permutation(self.train_size - self.T)
            j = 0
            while j < self.train_size:
                batch_idx = perm_idx[j:(j + self.batch_size)]
                X = np.zeros((len(batch_idx), self.T - 1, self.X.shape[1]))
                y_history = np.zeros((len(batch_idx), self.T - 1))
                y_target = self.y[batch_idx + self.T]

                for k in range(len(batch_idx)):
                    X[k, :, :] = self.X[batch_idx[k]: (batch_idx[k] + self.T - 1), :]
                    y_history[k, :] = self.y[batch_idx[k]: (batch_idx[k] + self.T - 1)]

                loss = self.train_iteration(X, y_history, y_target)
                self.iter_losses[i * iter_per_epoch + j // self.batch_size] = loss
                # if (j / self.batch_size) % 50 == 0:
                #    self.logger.info("Epoch %d, Batch %d: loss = %3.3f.", i, j / self.batch_size, loss)
                j += self.batch_size
                n_iter += 1

                if n_iter % 10000 == 0 and n_iter > 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                '''
                if learning_rate > self.learning_rate:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * .9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * .9
                    learning_rate *= .9
                '''

            self.epoch_losses[i] = np.mean(self.iter_losses[range(i * iter_per_epoch, (i + 1) * iter_per_epoch)])
            if i % 10 == 0:
                self.logger.info("Epoch %d, loss: %3.3f.", i, self.epoch_losses[i])

            if i % 10 == 0:
                y_train_pred = self.predict(on_train=True)
                y_test_pred = self.predict(on_train=False)
                y_pred = np.concatenate((y_train_pred, y_test_pred))
                plt.figure()
                plt.plot(range(1, 1 + len(self.y)), self.y, label="True")
                plt.plot(range(self.T, len(y_train_pred) + self.T), y_train_pred, label='Predicted - Train')
                plt.plot(range(self.T + len(y_train_pred), len(self.y) + 1), y_test_pred, label='Predicted - Test')
                plt.legend(loc='upper left')
                plt.show()

    def train_iteration(self, X, y_history, y_target):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_weighted, input_encoded = self.encoder(
            Variable(torch.from_numpy(X).type(torch.FloatTensor).to(device)))
        y_pred = self.decoder(input_encoded,
                              Variable(torch.from_numpy(y_history).type(torch.FloatTensor).to(device)))

        y_true = Variable(torch.from_numpy(y_target).type(torch.FloatTensor)).to(device)
        loss = self.loss_func(y_pred.squeeze(), y_true)
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        # if loss.data[0] < 10:
        #     self.logger.info("MSE: %s, loss: %s.", loss.data, (y_pred[:, 0] - y_true).pow(2).mean())

        return loss.data[0]

    def predict(self, on_train=False):
        if on_train:
            y_pred = np.zeros(self.train_size - self.T + 1)
        else:
            y_pred = np.zeros(self.X.shape[0] - self.train_size)

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i: (i + self.batch_size)]
            X = np.zeros((len(batch_idx), self.T - 1, self.X.shape[1]))
            y_history = np.zeros((len(batch_idx), self.T - 1))

            for j in range(len(batch_idx)):
                if on_train:
                    X[j, :, :] = self.X[range(batch_idx[j], batch_idx[j] + self.T - 1), :]
                    y_history[j, :] = self.y[range(batch_idx[j], batch_idx[j] + self.T - 1)]

                else:
                    X[j, :, :] = self.X[
                                 range(batch_idx[j] + self.train_size - self.T, batch_idx[j] + self.train_size - 1), :]
                    y_history[j, :] = self.y[
                        range(batch_idx[j] + self.train_size - self.T, batch_idx[j] + self.train_size - 1)]

            y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor)).to(device)
            _, input_encoded = self.encoder(
                Variable(torch.from_numpy(X).type(torch.FloatTensor)).to(device))
            y_pred[i:(i + self.batch_size)] = self.decoder(input_encoded, y_history).cpu().data.numpy()[:, 0]
            i += self.batch_size

        return y_pred


model = DaRnn(file_data='data/nasdaq100_padding.csv', logger=logger, parallel=False,
              learning_rate=.001)
model.train(n_epochs=500)
y_pred = model.predict()

plt.figure()
plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
plt.show()

plt.figure()
plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
plt.show()

plt.figure()
plt.plot(y_pred, label='Predicted')
plt.plot(model.y[model.train_size:], label="True")
plt.legend(loc='upper left')
plt.show()
