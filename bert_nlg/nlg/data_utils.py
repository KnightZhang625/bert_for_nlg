# coding:utf-8

import codecs
import random
import numpy as np
import tensorflow as tf
from functools import partial
tf.enable_eager_execution()

import config as cg
batch_size = cg.batch_size

def train_generator(path):
  """this could achieve random by dataset.shuffle(10).batch(2).repeat(3)."""
  with codecs.open(path, 'r', 'utf-8') as file:
    for line in file:
      if len(line) != 0:
        line = [i for i in line][:-1]
        line = np.array(list(map(int, line)))
        
        features = {'input': line}
        yield features

str_to_int = lambda string : list(map(int, string))
def train_generator_test(path):
  """this could achieve padding among each batch."""
  with codecs.open(path, 'r', 'utf-8') as file:
    data = file.read().split('\n')[:-1]
  data = list(map(str_to_int, data))
  
  random.shuffle(data)
  lines = []
  for idx, line in enumerate(data):    
    if len(lines) < batch_size:
      lines.append(line)
      if len(lines) == batch_size:
        features = {'input': lines}
        yield features
        lines = []
      if idx > (len(data) // batch_size) * batch_size - 1 and len(data) % batch_size != 0:
        lines.extend(random.sample(data, 2 - len(lines)))
        features = {'input': lines}
        yield features
        lines = []

def train_input_fn():
  output_types = {'input': tf.int32}
  output_shapes = {'input': [None, None]}

  generator = partial(train_generator_test, path=cg.DATA_PATH)
  dataset = tf.data.Dataset.from_generator(
    generator,
    output_types=output_types,
    output_shapes=output_shapes)

  dataset = dataset.repeat(3)

  return dataset

if __name__ == '__main__':
  for data in train_input_fn():
    print(data)
    input()
  
  # for data in train_generator(cg.DATA_PATH):
  #   print(data)
  
  # for data in train_generator_test(cg.DATA_PATH):
  #   print(data)
  #   input()