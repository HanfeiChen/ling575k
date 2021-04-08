#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /dropbox/20-21/575k/env

# put your command for running word2vec.py here
python word2vec.py \
    --num_epochs 15 \
    --save_vectors vectors.tsv \
    --embedding_dim 15 \
    --learning_rate 0.2 \
    --min_freq 5 \
    --num_negatives 15 \
    --training_data /dropbox/20-21/575k/data/sst/train-reviews.txt

python analysis.py --training_data /dropbox/20-21/575k/data/sst/train-reviews.txt --save_vectors vectors.tsv --save_plot vectors.png
