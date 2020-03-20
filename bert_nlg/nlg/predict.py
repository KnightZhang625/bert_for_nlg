# coding:utf-8

import re
import codecs
import functools
import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent.parent

import config as cg
from data_utils import make_mask
from data_utils import str_to_idx, vocab_idx, idx_vocab

from log import log_info as _info
from log import log_error as _error

def restore_model(pb_path):
  """Restore the latest model from the given path."""
  subdirs = [x for x in Path(pb_path).iterdir()
             if x.is_dir() and 'temp' not in str(x)]
  latest_model = str(sorted(subdirs)[-1])
  predict_fn = predictor.from_saved_model(latest_model)

  return predict_fn

"""Decorator for the predict function."""
def predict_decorator(save_path=None):
  def predict(func):
    functools.wraps(func)
    def predict_inner(model, *args):
      for input_x, input_mask, sentence in func(model, args[0]):
        features = {'input_x': input_x, 'input_mask': input_mask}
        predictions = model(features)
        # single predict
        if save_path is None: 
          sample_id = predictions['sample_id'][0]
          ppls = predictions['ppls'][0]
          reply_sentence = ''.join([idx_vocab[idx] for idx in sample_id])
          # delete the <eos> tag
          reply_sentence_without_endtag = re.sub(r'</s>.*', '', reply_sentence)
          return reply_sentence, ppls
        # batch predict
        else:
          with codecs.open(save_path, 'a', 'utf-8') as file:
            sample_ids = predictions['sample_id']
            ppls = predictions['ppls']
            for sent, ids, ppl in zip(sentence, sample_ids, ppls):
              reply_sentence = ''.join([idx_vocab[idx] for idx in ids])
              # delete the <eos> tag
              reply_sentence_without_endtag = re.sub(r'</s>.*', '', reply_sentence)
              to_write = sent + '=' + reply_sentence_without_endtag + '\t' + str(ppl) + '\n'
              file.write(to_write)
              file.flush()
    
    return predict_inner
  return predict

@predict_decorator(save_path=None)
def predict_single(model, sentence):
  """predict single data.
  
  Args:
    model: The restore model.
    sentence: String. The test sentence.

  Returns:
    reply_sentence: String.
    ppls: Float.
  """
  # [1, length]
  input_x = [str_to_idx(sentence)]
  input_mask = make_mask(input_x)
  
  yield input_x, input_mask, sentence

padding_func = lambda string, max_length : string + [vocab_idx['<padding>'] for _ in range(max_length - len(string))]
def padding(array):
  """padding the given array to the same length."""
  max_length = max([len(sent) for sent in array])
  padding_func_with_args = functools.partial(padding_func, max_length=max_length)
  
  return list(map(padding_func_with_args, array))

@predict_decorator(save_path=MAIN_PATH / 'data/infer_data.txt')
def predict_batch(model, path):
  """predict the given test data on batch mode,
    then save in the `save_path` defined in decorator.
  
  Args:
    model: The restore model.
    path: The test data path.
  """
  input_x = []
  sentence_cache = []
  batch_size = 2
  with codecs.open(path, 'r', 'utf-8') as file:
    for line in file:
      if len(line) > 0:
        input_x.append(str_to_idx(line.strip()))
        sentence_cache.append(line.strip())
        if len(input_x) == batch_size:
          input_x_padded = padding(input_x)
          input_mask = make_mask(input_x_padded)
          yield input_x_padded, input_mask, sentence_cache
          input_x = []
          sentence_cache = []
  
  if len(input_x) > 0:
    input_x_padded = padding(input_x)
    input_mask = make_mask(input_x_padded)
    yield input_x_padded, input_mask, sentence_cache

if __name__ == '__main__':
  # restore the model
  model = restore_model(cg.pb_model_path)

  # predict single sentence
  sentence = '我也以为是女的'
  reply, ppl = predict_single(model, sentence)
  print(reply, ppl)

  # predict batch
  predict_batch(model, MAIN_PATH / 'data/test_data.txt')
