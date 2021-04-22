#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /dropbox/20-21/575k/env

python run.py

# python run.py --l2 0.00001

# python run.py --l2 0.00001 --word_dropout 0.3
