#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /dropbox/20-21/575k/env

# put your command for running word2vec.py here

python /dropbox/20-21/575k/hw2/analysis.py --save_vectors vectors.tsv --save_plot vectors.png
