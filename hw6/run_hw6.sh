#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /dropbox/20-21/575k/env

# Vanilla RNN, default parameters.
python run.py

# Vanilla RNN, with L2 regularization at 1e-4 and dropout at 0.5.
python run.py --l2 1e-4 --dropout 0.5

# LSTM, default parameters.
python run.py --lstm

# LSTM, with L2 regularization at 1e-4 and dropout at 0.5.
python run.py --lstm --l2 1e-4 --dropout 0.5
