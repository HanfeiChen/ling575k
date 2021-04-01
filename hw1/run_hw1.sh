#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
# if you install anaconda in a different directory, try the following command
# source path_to_anaconda3/anaconda3/etc/profile.d/conda.sh

conda activate /dropbox/20-21/575k/env/

# include your commands here
python main.py --text_file /dropbox/20-21/575k/data/sst/train-reviews.txt --output_file train_vocab_base.txt

python main.py --text_file /dropbox/20-21/575k/data/sst/train-reviews.txt --output_file train_vocab_freq5.txt --min_freq 5
