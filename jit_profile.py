import json
import os

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib

from modules import Encoder, Decoder
import utils
from constants import device
from main_predict import preprocess_data
from profile import predict

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

jit_enc = torch.jit.trace(enc, torch.Tensor(da_rnn_kwargs["batch_size"],
                                            da_rnn_kwargs["T"] - 1,
                                            data.feats.shape[1]))
jit_dec = torch.jit.trace(dec,
                          (torch.Tensor(da_rnn_kwargs["batch_size"],
                                        da_rnn_kwargs["T"] - 1,
                                        64),
                           torch.Tensor(da_rnn_kwargs["batch_size"],
                                        da_rnn_kwargs["T"] - 1,
                                        data.targs.shape[1])))
final_y_pred, run_times = predict(jit_enc, jit_dec, data, **da_rnn_kwargs)

print(len(run_times))
np.save(os.path.join("data", "profiling", "base_jit.npy"), run_times)

plt.figure()
plt.plot(final_y_pred, label='Predicted')
plt.plot(data.targs[(da_rnn_kwargs["T"]-1):], label="True")
plt.legend(loc='upper left')
utils.save_or_show_plot("jit_predicted_reloaded.png", save_plots)
