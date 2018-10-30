import json
import os
from time import process_time
from typing import List, Tuple

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib

from modules import Encoder, Decoder
from utils import numpy_to_tvar
import utils
from constants import device
from main_predict import preprocess_data


def predict(encoder, decoder, t_dat, batch_size: int, T: int) -> Tuple[np.ndarray, List[float]]:
    y_pred = np.zeros((t_dat.feats.shape[0] - T + 1, t_dat.targs.shape[1]))
    run_times = []

    for y_i in range(0, len(y_pred), batch_size):
        y_slc = slice(y_i, y_i + batch_size)
        batch_idx = range(len(y_pred))[y_slc]
        b_len = len(batch_idx)
        X = np.zeros((b_len, T - 1, t_dat.feats.shape[1]))
        y_history = np.zeros((b_len, T - 1, t_dat.targs.shape[1]))

        for b_i, b_idx in enumerate(batch_idx):
            idx = range(b_idx, b_idx + T - 1)

            X[b_i, :, :] = t_dat.feats[idx, :]
            y_history[b_i, :] = t_dat.targs[idx]

        t_val_start = process_time()
        _, input_encoded = encoder(numpy_to_tvar(X))
        y_pred[y_slc] = decoder(input_encoded, numpy_to_tvar(y_history)).cpu().data.numpy()
        t_val_end = process_time()
        run_times.append(t_val_start - t_val_end)

    return y_pred, run_times


debug = False
save_plots = False

with open(os.path.join("data", "enc_kwargs.json"), "r") as fi:
    enc_kwargs = json.load(fi)
enc = Encoder(**enc_kwargs)
enc.load_state_dict(torch.load(os.path.join("data", "encoder.torch"), map_location=device))

with open(os.path.join("data", "dec_kwargs.json"), "r") as fi:
    dec_kwargs = json.load(fi)
dec = Decoder(**dec_kwargs)
dec.load_state_dict(torch.load(os.path.join("data", "decoder.torch"), map_location=device))

scaler = joblib.load(os.path.join("data", "scaler.pkl"))
raw_data = pd.read_csv(os.path.join("data", "nasdaq100_padding.csv"), nrows=100 if debug else None)
targ_cols = ("NDX",)
data = preprocess_data(raw_data, targ_cols, scaler)

with open(os.path.join("data", "da_rnn_kwargs.json"), "r") as fi:
    da_rnn_kwargs = json.load(fi)
final_y_pred, run_times = predict(enc, dec, data, **da_rnn_kwargs)

print(len(run_times))
np.save(os.path.join("data", "profiling", "base.npy"), run_times)

plt.figure()
plt.plot(final_y_pred, label='Predicted')
plt.plot(data.targs[(da_rnn_kwargs["T"]-1):], label="True")
plt.legend(loc='upper left')
utils.save_or_show_plot("base_predicted_reloaded.png", save_plots)
