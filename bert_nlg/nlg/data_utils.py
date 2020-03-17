# coding:utf-8

import codecs
import numpy as np
import tensorflow as tf
from functools import partial
tf.enable_eager_execution()

import config as cg

def train_generator(path):
  with codecs.open(path, 'r', 'utf-8') as file:
    for line in file:
      if len(line) != 0:
        line = [i for i in line][:-1]
        line = np.array(list(map(int, line)))
        
        features = {'input': line}
        yield features

def train_input_fn():
  output_types = {'input': tf.int32}
  output_shapes = {'input': [None]}

  generator = partial(train_generator, path=cg.DATA_PATH)
  dataset = tf.data.Dataset.from_generator(
    generator,
    output_types=output_types,
    output_shapes=output_shapes)

  dataset = dataset.shuffle(10).batch(2).repeat(3)

  return dataset

if __name__ == '__main__':
  for data in train_input_fn():
    print(data)
    input()
  
  # for data in train_generator(cg.DATA_PATH):
  #   print(data)