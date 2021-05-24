#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /dropbox/20-21/575k/env

python run.py \
    --train_source /dropbox/20-21/575k/data/europarl-v7-es-en/train.en.txt \
    --train_target /dropbox/20-21/575k/data/europarl-v7-es-en/train.es.txt \
    --output_file test.en.txt.es \
    --num_epochs 8 \
    --embedding_dim 16 \
    --hidden_dim 64 \
    --num_layers 2 \
    --temp 4.0

python chrF++.py -nw 0 \
    -R /dropbox/20-21/575k/data/europarl-v7-es-en/test.es.txt \
    -H test.en.txt.es > test.en.txt.es.score
