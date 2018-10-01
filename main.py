import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import utility as util
from modules import Encoder, Decoder

import typing
import collections

logger = util.setup_log()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info("Using computation device %s.", device)


class TrainConfig(typing.NamedTuple):
    T: int
    train_size: int
    batch_size: int
    loss_func: typing.Callable


class TrainData(typing.NamedTuple):
    feats: np.ndarray
    targs: np.ndarray


DaRnnNet = collections.namedtuple("DaRnnNet", ["encoder", "decoder", "enc_opt", "dec_opt"])


def da_rnn(file_nm: str, encoder_hidden_size=64, decoder_hidden_size=64,
           T=10, learning_rate=0.01, batch_size=128, debug=False):
    dat = pd.read_csv(file_nm, nrows=100 if debug else None)
    logger.info(f"Shape of data: {dat.shape}.\nMissing in data: {dat.isnull().sum().sum()}.")

    scaler = StandardScaler().fit(dat)
    proc_dat = scaler.transform(dat)

    col_idx = list(dat.columns).index("NDX")
    mask = np.ones(proc_dat.shape[1], dtype=bool)
    mask[col_idx] = False
    feats = proc_dat[:, mask]
    targs = proc_dat[:, ~mask]
    train_data = TrainData(feats, targs.squeeze())

    train_cfg = TrainConfig(T, int(train_data.feats.shape[0] * 0.7), batch_size, nn.MSELoss())
    logger.info(f"Training size: {train_cfg.train_size:d}.")

    encoder = Encoder(input_size=train_data.feats.shape[1],
                      hidden_size=encoder_hidden_size,
                      T=T, logger=logger).to(device)
    decoder = Decoder(encoder_hidden_size=encoder_hidden_size,
                      decoder_hidden_size=decoder_hidden_size,
                      T=T, logger=logger).to(device)

    encoder_optimizer = optim.Adam(
        params=[p for p in encoder.parameters() if p.requires_grad],
        lr=learning_rate)
    decoder_optimizer = optim.Adam(
        params=[p for p in decoder.parameters() if p.requires_grad],
        lr=learning_rate)
    da_rnn_net = DaRnnNet(encoder, decoder, encoder_optimizer, decoder_optimizer)

    return train_cfg, train_data, da_rnn_net


def train(net: DaRnnNet, train_data: TrainData, t_cfg: TrainConfig, n_epochs=10, save_plots=False):
    iter_per_epoch = int(np.ceil(t_cfg.train_size * 1. / t_cfg.batch_size))
    iter_losses = np.zeros(n_epochs * iter_per_epoch)
    epoch_losses = np.zeros(n_epochs)
    logger.info(f"Iterations per epoch: {t_cfg.train_size * 1. / t_cfg.batch_size:3.3f} ~ {iter_per_epoch:d}.")

    n_iter = 0

    for i in range(n_epochs):
        perm_idx = np.random.permutation(t_cfg.train_size - t_cfg.T)

        for j in range(0, t_cfg.train_size, t_cfg.batch_size):
            feats, y_history, y_target = prep_train_data(j, perm_idx, t_cfg, train_data)

            loss = train_iteration(net, t_cfg.loss_func, feats, y_history, y_target)
            iter_losses[i * iter_per_epoch + j // t_cfg.batch_size] = loss
            # if (j / t_cfg.batch_size) % 50 == 0:
            #    self.logger.info("Epoch %d, Batch %d: loss = %3.3f.", i, j / t_cfg.batch_size, loss)
            n_iter += 1

            adjust_learning_rate(net, n_iter)

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
            util.save_or_show_plot(f"pred_{i}.png", save_plots)

    return iter_losses, epoch_losses


def prep_train_data(j: int, perm_idx, t_cfg: TrainConfig, train_data: TrainData):
    batch_idx = perm_idx[j:(j + t_cfg.batch_size)]
    feats = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.feats.shape[1]))
    y_history = np.zeros((len(batch_idx), t_cfg.T - 1))
    y_target = train_data.targs[batch_idx + t_cfg.T]

    for b_i, b_idx in enumerate(batch_idx):
        b_slc = slice(b_idx, (b_idx + t_cfg.T - 1))
        feats[b_i, :, :] = train_data.feats[b_slc, :]
        y_history[b_i, :] = train_data.targs[b_slc]

    return feats, y_history, y_target


def adjust_learning_rate(net: DaRnnNet, n_iter: int):
    # TODO: Where did this Learning Rate adjustment schedule come from? Why not just use the Cosine?
    if n_iter % 10000 == 0 and n_iter > 0:
        for enc_params, dec_params in zip(net.enc_opt.param_groups, net.dec_opt.param_groups):
            enc_params['lr'] = enc_params['lr'] * 0.9
            dec_params['lr'] = dec_params['lr'] * 0.9


def train_iteration(t_net: DaRnnNet, loss_func: typing.Callable, X, y_history, y_target):
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

    return loss.item()


def predict(t_net: DaRnnNet, t_dat: TrainData, train_size: int, batch_size: int, T: int, on_train=False):
    if on_train:
        y_pred = np.zeros(train_size - T + 1)
    else:
        y_pred = np.zeros(t_dat.feats.shape[0] - train_size)

    for y_i in range(0, len(y_pred), batch_size):
        batch_idx = np.array(range(len(y_pred)))[y_i: (y_i + batch_size)]
        X = np.zeros((len(batch_idx), T - 1, t_dat.feats.shape[1]))
        y_history = np.zeros((len(batch_idx), T - 1))

        for b_i, b_idx in enumerate(batch_idx):
            if on_train:
                idx = range(b_idx, b_idx + T - 1)
            else:
                idx = range(b_idx + train_size - T, b_idx + train_size - 1)

            X[b_i, :, :] = t_dat.feats[idx, :]
            y_history[b_i, :] = t_dat.targs[idx]

        y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor)).to(device)
        _, input_encoded = t_net.encoder(
            Variable(torch.from_numpy(X).type(torch.FloatTensor)).to(device))
        y_pred[y_i:(y_i + batch_size)] = t_net.decoder(input_encoded, y_history).cpu().data.numpy()[:, 0]

    return y_pred


save_plots = True

config, data, model = da_rnn(file_nm='data/nasdaq100_padding.csv', learning_rate=.001)
iter_loss, epoch_loss = train(model, data, config, n_epochs=500, save_plots=save_plots)
final_y_pred = predict(model, data, config.train_size, config.batch_size, config.T)

plt.figure()
plt.semilogy(range(len(iter_loss)), iter_loss)
util.save_or_show_plot("iter_loss.png", save_plots)

plt.figure()
plt.semilogy(range(len(epoch_loss)), epoch_loss)
util.save_or_show_plot("epoch_loss.png", save_plots)

plt.figure()
plt.plot(final_y_pred, label='Predicted')
plt.plot(data.targs[config.train_size:], label="True")
plt.legend(loc='upper left')
util.save_or_show_plot("final_predicted.png", save_plots)
