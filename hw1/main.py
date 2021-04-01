import argparse
from vocabulary import Vocabulary


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", type=str, default="/dropbox/20-21/575k/data/toy-reviews.txt")
    parser.add_argument("--output_file", type=str, default="toy-vocab.txt")
    parser.add_argument("--min_freq", type=int, default=1)
    args = parser.parse_args()

    # DONE: your code here! (~2-3 lines)
    # produce a Vocabulary object from the text_file supplied in the args
    # write that Vocabulary object to the output_file specified in the args
    voc = Vocabulary.from_text_files([args.text_file], min_freq=args.min_freq)
    voc.save_to_file(args.output_file)
