import torch
from torch import nn
from torch.autograd import Variable
from torch import optim

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import utility as util
from modules import Encoder, Decoder

from types import TrainConfig, TrainData, DaRnnNet

logger = util.setup_log()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info("Using computation device %s.", device)


def da_rnn(file_nm: str, encoder_hidden_size=64, decoder_hidden_size=64,
           T=10, learning_rate=0.01, batch_size=128, parallel=True, debug=False):
    dat = pd.read_csv(file_nm, nrows=100 if debug else None)
    logger.info(f"Shape of data: {dat.shape}.\nMissing in data: {dat.isnull().sum().sum()}.")

    # TODO: Normalize the data?
    # TODO: there's probably a more elegant way to index this
    train_data = TrainData(dat.loc[:, [x for x in dat.columns.tolist() if x != 'NDX']].as_matrix(),
                           np.array(dat.NDX))
    train_cfg = TrainConfig(T, int(train_data.feats.shape[0] * 0.7), batch_size, nn.MSELoss)
    logger.info(f"Training size: {train_cfg.train_size:d}.")

    encoder = Encoder(input_size=train_data.feats.shape[1],
                      hidden_size=encoder_hidden_size,
                      T=T, logger=logger).to(device)
    decoder = Decoder(encoder_hidden_size=encoder_hidden_size,
                      decoder_hidden_size=decoder_hidden_size,
                      T=T, logger=logger).to(device)
    if parallel:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)

    encoder_optimizer = optim.Adam(
        params=[p for p in encoder.parameters() if p.requires_grad],
        lr=learning_rate)
    decoder_optimizer = optim.Adam(
        params=[p for p in decoder.parameters() if p.requires_grad],
        lr=learning_rate)
    da_rnn_net = DaRnnNet(encoder, decoder, encoder_optimizer, decoder_optimizer)

    return train_cfg, train_data, da_rnn_net


def train(net: DaRnnNet, train_data: TrainData, t_cfg: TrainConfig, n_epochs=10):
    iter_per_epoch = int(np.ceil(t_cfg.train_size * 1. / t_cfg.batch_size))
    iter_losses = np.zeros(n_epochs * iter_per_epoch)
    epoch_losses = np.zeros(n_epochs)
    logger.info(f"Iterations per epoch: {t_cfg.train_size * 1. / t_cfg.batch_size:3.3f} ~ {iter_per_epoch:d}.")
    n_iter = 0

    for i in range(n_epochs):
        perm_idx = np.random.permutation(t_cfg.train_size - t_cfg.T)
        j = 0

        while j < t_cfg.train_size:
            batch_idx = perm_idx[j:(j + t_cfg.batch_size)]
            X = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.feats.shape[1]))
            y_history = np.zeros((len(batch_idx), t_cfg.T - 1))
            y_target = train_data.targs[batch_idx + t_cfg.T]

            for k in range(len(batch_idx)):
                X[k, :, :] = train_data.feats[batch_idx[k]: (batch_idx[k] + t_cfg.T - 1), :]
                y_history[k, :] = train_data.targs[batch_idx[k]: (batch_idx[k] + t_cfg.T - 1)]

            loss = train_iteration(net, t_cfg.loss_func, X, y_history, y_target)
            iter_losses[i * iter_per_epoch + j // t_cfg.batch_size] = loss
            # if (j / t_cfg.batch_size) % 50 == 0:
            #    self.logger.info("Epoch %d, Batch %d: loss = %3.3f.", i, j / t_cfg.batch_size, loss)
            j += t_cfg.batch_size
            n_iter += 1

            # TODO: Where did this Learning Rate adjusment schedule come from? Why not just use the Cosine?
            if n_iter % 10000 == 0 and n_iter > 0:
                for param_group in net.enc_opt.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9
                for param_group in net.dec_opt.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9

        epoch_losses[i] = np.mean(iter_losses[range(i * iter_per_epoch, (i + 1) * iter_per_epoch)])
        if i % 10 == 0:
            logger.info("Epoch %d, loss: %3.3f.", i, epoch_losses[i])

        if i % 10 == 0:
            y_train_pred = predict(net, train_data,
                                   t_cfg.train_size, t_cfg.batch_size, t_cfg.T,
                                   on_train=True)
            y_test_pred = predict(net, train_data,
                                  t_cfg.train_size, t_cfg.batch_size, t_cfg.T,
                                  on_train=False)
            plt.figure()
            plt.plot(range(1, 1 + len(train_data.targs)), train_data.targs,
                     label="True")
            plt.plot(range(t_cfg.T, len(y_train_pred) + t_cfg.T), y_train_pred,
                     label='Predicted - Train')
            plt.plot(range(t_cfg.T + len(y_train_pred), len(train_data.targs) + 1), y_test_pred,
                     label='Predicted - Test')
            plt.legend(loc='upper left')
            plt.show()

        return iter_losses, epoch_losses


def train_iteration(t_net: DaRnnNet, loss_func, X, y_history, y_target):
    t_net.enc_opt.zero_grad()
    t_net.dec_opt.zero_grad()

    input_weighted, input_encoded = t_net.encoder(
        Variable(torch.from_numpy(X).type(torch.FloatTensor).to(device)))
    y_pred = t_net.decoder(input_encoded,
                           Variable(torch.from_numpy(y_history).type(torch.FloatTensor).to(device)))

    y_true = Variable(torch.from_numpy(y_target).type(torch.FloatTensor)).to(device)
    loss = loss_func(y_pred.squeeze(), y_true)
    loss.backward()

    t_net.enc_opt.step()
    t_net.dec_opt.step()

    # if loss.data[0] < 10:
    #     self.logger.info("MSE: %s, loss: %s.", loss.data, (y_pred[:, 0] - y_true).pow(2).mean())

    return loss.data[0]


def predict(t_net: DaRnnNet, t_dat: TrainData, train_size: int, batch_size: int, T: int, on_train=False):
    if on_train:
        y_pred = np.zeros(train_size - T + 1)
    else:
        y_pred = np.zeros(t_dat.feats.shape[0] - train_size)

    i = 0
    while i < len(y_pred):
        batch_idx = np.array(range(len(y_pred)))[i: (i + batch_size)]
        X = np.zeros((len(batch_idx), T - 1, t_dat.feats.shape[1]))
        y_history = np.zeros((len(batch_idx), T - 1))

        for j in range(len(batch_idx)):
            if on_train:
                idx = range(batch_idx[j], batch_idx[j] + T - 1)
            else:
                idx = range(batch_idx[j] + train_size - T, batch_idx[j] + train_size - 1)

            X[j, :, :] = t_dat.feats[idx, :]
            y_history[j, :] = t_dat.targs[idx]

        y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor)).to(device)
        _, input_encoded = t_net.encoder(
            Variable(torch.from_numpy(X).type(torch.FloatTensor)).to(device))
        y_pred[i:(i + batch_size)] = t_net.decoder(input_encoded, y_history).cpu().data.numpy()[:, 0]
        i += batch_size

    return y_pred


config, data, model = da_rnn(file_nm='data/nasdaq100_padding.csv', parallel=False, learning_rate=.001)
iter_loss, epoch_loss = train(model, data, config, n_epochs=500)
final_y_pred = predict(model, data, config.train_size, config.batch_size, config.T)

plt.figure()
plt.semilogy(range(len(iter_loss)), iter_loss)
plt.show()

plt.figure()
plt.semilogy(range(len(epoch_loss)), epoch_loss)
plt.show()

plt.figure()
plt.plot(final_y_pred, label='Predicted')
plt.plot(data.targs[config.train_size:], label="True")
plt.legend(loc='upper left')
plt.show()
