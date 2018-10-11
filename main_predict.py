import json
import os

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules import Encoder, Decoder
from utils import numpy_to_tvar
import utils


def predict(encoder, decoder, t_dat, batch_size: int, T: int):
    y_pred = np.zeros(t_dat.feats.shape[0] - T + 1)

    for y_i in range(0, len(y_pred), batch_size):
        y_slc = slice(y_i, y_i + batch_size)
        batch_idx = range(len(y_pred))[y_slc]
        b_len = len(batch_idx)
        X = np.zeros((b_len, T - 1, t_dat.feats.shape[1]))
        y_history = np.zeros((b_len, T - 1))

        for b_i, b_idx in enumerate(batch_idx):
            idx = range(b_idx, b_idx + T - 1)

            X[b_i, :, :] = t_dat.feats[idx, :]
            y_history[b_i, :] = t_dat.targs[idx]

        y_history = numpy_to_tvar(y_history)
        _, input_encoded = encoder(numpy_to_tvar(X))
        y_pred[y_slc] = decoder(input_encoded, y_history).cpu().data.numpy()[:, 0]

    return y_pred


debug = True
save_plots = False

with open(os.path.join("data", "enc_kwargs.json"), "r") as fi:
    enc_kwargs = json.load(fi)
enc = Encoder(**enc_kwargs)
enc.load_state_dict(torch.load(os.path.join("data", "encoder.torch")))

with open(os.path.join("data", "dec_kwargs.json"), "r") as fi:
    dec_kwargs = json.load(fi)
dec = Decoder(**dec_kwargs)
dec.load_state_dict(torch.load(os.path.join("data", "decoder.torch")))

data = pd.read_csv(os.path.join("data", "nasdaq100_padding.csv"), nrows=100 if debug else None)

final_y_pred = predict(enc, dec, data, 128, 10)

plt.figure()
plt.plot(final_y_pred, label='Predicted')
plt.plot(data.targs, label="True")
plt.legend(loc='upper left')
utils.save_or_show_plot("final_predicted.png", save_plots)
