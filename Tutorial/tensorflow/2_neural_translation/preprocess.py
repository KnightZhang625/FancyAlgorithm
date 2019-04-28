import re
import pickle
import jieba
import codecs
from collections import Counter

class PreProcess(object):
    def __init__(self, *, path, lang_type):
        self.path = path
        assert lang_type in ['en', 'zh'], TypeError
        self.lang_type = lang_type
    
    def process(self, *, save_path):
        with codecs.open(self.path, 'r', 'utf-8') as file:
            sentence_raw = (sentence.strip() for sentence in file.readlines())
        
        vocab_count = self.extract_vocab(sentence_raw)
        vocab_sorted = sorted(vocab_count.items(), key=lambda x : x[1], reverse=True)
        
        word_idx, idx_word = self.build_vocab(vocab_sorted)

        try:
            with codecs.open(save_path + '/word_idx_%s.pickle'%(self.lang_type), 'wb') as file_1, \
                 codecs.open(save_path + '/idx_word_%s.pickle'%(self.lang_type), 'wb') as file_2:
                pickle.dump(word_idx, file_1)
                pickle.dump(idx_word, file_2)
        except Exception as e:
            print(e)
        else:
            print('NOTICE : file has been saved successfully')

    def extract_vocab(self, sentence_raw):
        vocab_count = Counter()
        if self.lang_type == 'en':
            for sentence in sentence_raw:
                for word in sentence.split(' '):
                    vocab_count[word] += 1
        elif self.lang_type == 'zh':
            for sentence in sentence_raw:
                word_cut = jieba.cut(sentence)
                for word in word_cut:
                    vocab_count[word] += 1
        return vocab_count

    def build_vocab(self, vocab_sorted):
        word_idx = {}
        idx_word = {}
        vocab_sorted.extend([('<sos>', 0), ('<eos>', 0), ('<unk>', 0)])
        for idx, word_and_count in enumerate(vocab_sorted):
            word = word_and_count[0]
            word_idx[word] = idx
            idx_word[idx] = word
        return word_idx, idx_word

class Transfer(PreProcess):
    def __init__(self, *, path, lang_type):
        super(Transfer, self).__init__(path=path, lang_type=lang_type)
        with codecs.open('word_idx_%s.pickle'%(self.lang_type), 'rb') as file:
            self.word_idx = pickle.load(file)
         
    def transfer(self, *, save_path):
        try:
            with codecs.open(self.path, 'r', 'utf-8') as file, \
                codecs.open(save_path + '/train_%s'%(self.lang_type), 'w', 'utf-8') as file_write:
                for sentence in file.readlines():
                    if self.lang_type == 'en':
                        cache = ' '.join([str(self.word_idx[word]) for word in sentence.strip().split(' ')])
                        cache = str(self.word_idx['<sos>']) + ' ' + cache + ' ' + str(self.word_idx['<eos>'])
                        file_write.writelines(cache + '\n')
                    elif self.lang_type == 'zh':
                        cache = ' '.join([str(self.word_idx[word]) for word in jieba.cut(sentence.strip())])
                        cache = str(self.word_idx['<sos>']) + ' ' + cache + ' ' + str(self.word_idx['<eos>'])
                        file_write.write(cache + '\n')
        except Exception as KeyError:
            print(KeyError)
        else:
            print('NOTICE : file has been transferred successfully')

if __name__ == '__main__':
    from pathlib import Path
    cur_path = str(Path(__file__).absolute().parent)

    # preprocess = PreProcess(path='train.txt.zh', lang_type='zh')
    # preprocess.process(save_path=cur_path)

    transfer = Transfer(path='train.txt.zh', lang_type='zh')
    transfer.transfer(save_path=cur_path)