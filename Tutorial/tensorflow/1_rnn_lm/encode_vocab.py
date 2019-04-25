import codecs
import sys

RAW_DATA = 'data/ptb.train.txt'
VOCAB = 'ptb.vocab'
OUTPUT_DATA = 'ptb.train'

with codecs.open(VOCAB, 'r', 'utf-8') as f_vocab:
    vocab = [w.strip() for w in f_vocab.readlines()]

word_to_id = {v : idx for idx, v in enumerate(vocab)}

def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id['<unk>']

with codecs.open(RAW_DATA, 'r', 'utf-8') as fin, codecs.open(OUTPUT_DATA, 'w', 'utf-8') as fout:
    for line in fin:
        words = line.strip().split() + ['<eos>']
        out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
        fout.write(out_line)

