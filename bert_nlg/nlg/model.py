# coding:utf-8

import sys
import copy
import tensorflow as tf

from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(MAIN_PATH))

import function_toolkit as ft
import model_LEGO as lego

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
    
    self.build_graph(config, input_ids, input_mask, token_type_ids, use_one_hot_embeddings)

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
        embedding_output, self.embedding_table = lego.embedding_lookup(
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
  
  def get_embedding_table(self):
    return self.embedding_table

class Decoder(object):
  def __init__(self,
               config,
               is_training,
               encoder_state,
               embedding_table,
               decoder_intput_data=None,
               seq_length_decoder_input_data=None,
               scope=None):
    config = copy.deepcopy(config)
    self.is_training = is_training
    self.embedding_table = embedding_table

    input_shape = ft.get_shape_list(encoder_state, expected_rank=2)
    self.batch_size = input_shape[0]
    
    self.tgt_vocab_size = config.tgt_vocab_size
    self.unit_type = config.unit_type
    self.num_units = config.num_units
    self.forget_bias = config.forget_bias

    if not is_training:
      self.dropout = 0.0
    else:
      self.dropout = config.dropout

    initializer_range = config.initializer_range
    
    self.tgt_sos_id = tf.constant(config.sos_id, dtype=tf.int32)
    self.tgt_eos_id = tf.constant(config.eos_id, dtype=tf.int32)
    self.max_len_infer = config.max_len_infer

    self.build_graph(encoder_state, initializer_range, seq_length_decoder_input_data, decoder_intput_data, scope)
  
  def build_graph(self,
                  encoder_state,
                  initializer_range,
                  seq_length_decoder_input_data=None,
                  decoder_input_data=None,
                  scope=None):
    with tf.variable_scope(scope, default_name='decoder'):
      output_layer = tf.layers.Dense(self.tgt_vocab_size, 
                                     name='decoder_output', 
                                     kernel_initializer=ft.create_initialzer(initializer_range=initializer_range))
      maximum_iterations = 0
      if self.max_len_infer != None:
        maximum_iterations = self.max_len_infer
      else:
        decoding_length_factor = 5.0
        max_encoder_length = tf.reduce_max(seq_length_decoder_input_data)
        maximum_iterations = tf.to_int32(tf.round(tf.to_float(max_encoder_length) * decoding_length_factor))
    
      with tf.variable_scope('rnn_decoder') as scope:
        cells, decoder_initial_state = None, None
        cells = lego.create_cell_list_for_RNN(unit_type=self.unit_type,
                                              num_units=self.num_units,
                                              dropout=self.dropout,
                                              forget_bias=self.forget_bias)
        decoder_initial_state = encoder_state
        logits, sample_id, final_context_state = None, None, None

        if self.is_training:
          target_input = decoder_input_data
          decoder_emb_inp = tf.nn.embedding_lookup(self.embedding_table, target_input)

          helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, seq_length_decoder_input_data)
          my_decoder = tf.contrib.seq2seq.BasicDecoder(cells, helper, decoder_initial_state)

          outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, swap_memory=True, scope=scope)
          self.logits = output_layer(outputs.rnn_output)
          self.sample_id = tf.argmax(self.logits, axis=-1)
        else:
          start_tokens = tf.fill([self.batch_size], self.tgt_sos_id)
          end_token = self.tgt_eos_id
          
          helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_table, start_tokens, end_token)
          my_decoder = tf.contrib.BasicDecoder(cells, helper, decoder_initial_state, output_layer=output_layer)

          outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder,
                                                                              maximum_iterations=maximum_iterations,
                                                                              swap_memory=True,
                                                                              scope=scope)
          self.logits = outputs.rnn_output
          self.sample_id = outputs.sample_id
    
    self.ppl_seq, self.ppl = ft.get_ppl(self.logits)
        
  def get_decoder_output(self):
    return self.logits, self.sample_id, self.ppl_seq, self.ppl    