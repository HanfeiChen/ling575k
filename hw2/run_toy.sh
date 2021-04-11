#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /dropbox/20-21/575k/env

# put your command for running word2vec.py here
python word2vec.py \
    --num_epochs 5 \
    --save_vectors toy-vectors.tsv \
    --embedding_dim 5 \
    --learning_rate 0.2 \
    --window_size 1 \
    --min_freq 1 \
    --num_negatives 5 \
    --training_data /dropbox/20-21/575k/data/toy-reviews.txt
