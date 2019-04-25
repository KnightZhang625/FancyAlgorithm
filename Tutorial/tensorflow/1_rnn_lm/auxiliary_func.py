import codecs           # better read function
import collections
from operator import itemgetter     # get the data from the list, tuple, etc.

from pathlib import Path
cur_path = Path(__file__).absolute().parent

import sys
sys.path.insert(0, str(cur_path))

RAW_DATA = 'data/ptb.train.txt'
VOCAB_OUTPUT = 'ptb.vocab'

counter = collections.Counter()
with codecs.open(RAW_DATA, 'r', 'utf-8') as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1

sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
# sorted_word_to_cnt_2 = sorted(counter.items(), key=lambda x : x[1], reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]
sorted_words = ['<unk>', '<sos>', '<sos>'] + sorted_words

if len(sorted_words) > 10000:
    sorted_words = sorted_words[:10000]

with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word + '\n')