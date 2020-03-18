# coding:utf-8

import sys
import copy
import tensorflow as tf

from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(MAIN_PATH))

import function_toolkit as ft
import model_LEGO as lego
from config import BertEncoderConfig as config

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
    if not is_training:
      config.hidden_drouput_prob = 0.0
      config.attention_probs_dropout_prob = 0.0
    
    input_shape = ft.get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
    
    self.build_graph(config, input_ids, input_shape, token_type_ids, use_one_hot_embeddings)

  def build_graph(self,
                  config,
                  input_ids,
                  input_mask,
                  token_type_ids,
                  use_one_hot_embeddings,
                  scope=None):
    """Build forward graph.
      Do not change the scope name if wanna to use the pretrained model.
    """
    with tf.variable_scope(scope, default_name='bert'):
      with tf.variable_scope('embeddings'):
        # embedding
        embedding_output, _ = lego.embedding_lookup(
          input_ids=input_ids,
          vocab_size=config.vocab_size,
          embedding_size=config.embedding_size,
          initializer_range=config.initializer_range,
          word_embedding_name='word_embeddings',
          use_one_hot_embeddings=use_one_hot_embeddings)
    
        # positional embedding
        positional_embedding_output = lego.embedding_postprocessor(
          input_tensor=embedding_output,
          use_positional_embeddings=True,
          positional_embedding_name='positional_embeddings',
          initializer_range=config.initializer_range,
          max_positional_embeddings=config.max_positional_embeddings,
          dropout_prob=config.hidden_dropout_prob)

      with tf.variable_scope('encoder'):
        # [batch_size, seq_length] -> [batch_size, seq_length, seq_length]
        # attention_mask = ft.create_attention_mask_from_input_mask(
        #   input_ids, input_mask)
        attention_mask = tf.cast(input_mask, tf.float32)

        all_encoder_layers = lego.transformer_model(input_tensor=positional_embedding_output,
                                                    attention_mask=attention_mask,
                                                    hidden_size=config.hidden_size,
                                                    num_hidden_layers=config.num_hidden_layers,
                                                    num_attention_heads=config.num_attention_heads,
                                                    intermediate_size=config.intermediate_size,
                                                    intermediate_act_fn=ft.get_activation(config.hidden_act),
                                                    hidden_dropout_prob=config.hidden_dropout_prob,
                                                    attention_dropout_prob=config.attention_dropout_prob,
                                                    initializer_range=config.initializer_range,
                                                    do_return_all_layers=True)

      self.sequence_output = all_encoder_layers[-1]

  def get_sequence_output(self):
    return self.sequence_output