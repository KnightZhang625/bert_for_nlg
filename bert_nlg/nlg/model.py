# coding:utf-8

import copy
import tensorflow as tf

class BertEncoder(object):
  """Use Bert to encode the source input."""
  def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=False,
               scope=None):
    """Constructor for BertEncoder.
    
    Args:
      token_type_ids: maybe for positional embedding for encoder.
    
    TODO
    """
    # create a copy of config, prevent from changing the original configuration.
    config = copy.deepcopy(config)
    
