# coding:utf-8

import tensorflow as tf
from tensorflow.contrib import predictor
from pathlib import Path

from data_utils import make_mask
from data_utils import str_to_idx, idx_vocab

def restore_model(pb_path):
  subdirs = [x for x in Path(pb_path).iterdir()
             if x.is_dir() and 'temp' not in str(x)]
  latest_model = str(sorted(subdirs)[-1])
  predict_fn = predictor.from_saved_model(latest_model)

  return predict_fn

def predict_single(model, sentence):
  # [1, length]
  input_x = [str_to_idx(sentence)]
  input_mask = make_mask(input_x)
  
  features = {'input_x': input_x, 'input_mask': input_mask}
  perdictions = model(features)

  sample_id = perdictions['sample_id'][0]
  ppls = perdictions['ppls'][0]
  
  reply_sentence = ''.join([idx_vocab[idx] for idx in sample_id])

  return reply_sentence, ppls
  

if __name__ == '__main__':
  model = restore_model('pb_models/')

  sentence = '这是什么意思'
  reply, ppl = predict_single(model, sentence)
  print(reply, ppl)