# coding:utf-8
# converting the vocab dictionary to the vocab_idx, idx_vocob format

import codecs
import pickle
import argparse

from log import log_info as _info
from log import log_error as _error

def make_dict(path, save_dir):
  vocab_idx = {}
  idx_vocab = {}
  with codecs.open(path, 'r', 'utf-8') as file:
    for idx, vocab in enumerate(file):
      vocab = vocab.strip()
      if len(vocab) > 0:
        vocab_idx[vocab] = idx
        idx_vocab[idx] = vocab
  
  with codecs.open(save_dir + 'vocab_idx.bin', 'wb') as vocab_idx_save,\
       codecs.open(save_dir + 'idx_vocab.bin', 'wb') as idx_vocab_save:
      pickle.dump(vocab_idx, vocab_idx_save)
      pickle.dump(idx_vocab, idx_vocab_save)

  _info('Vocab length: {}, the dictionary has been saved to: {}'.format(len(vocab_idx), save_dir)) 


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-v', '--vocab_path', help='the path for vocab dictionary')
  parser.add_argument('-s', '--save_path', help='the path to save the vocab_idx, idx_vocab')
  args = parser.parse_args()

  make_dict(args.vocab_path, args.save_path)