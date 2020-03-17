# coding:utf-8

import math
import tensorflow as tf

import function_toolkit as ft

def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size,
                     initializer_range,
                     word_embedding_name='word_embeddings',
                     use_one_hot_embeddings=False):
  """Looks up words embeddings for id tensor.

  Args:
    input_ids: int32 Tensor of shape [batch_size, seq_length] containing word ids.
    vocab_size: int. Size of the embedding vocabulary.
    embedding_size: int. Width of the word embeddings.
    initializer_range: float. Embedding initialation range.
    word_embedding_name: string. Name of the embedding table.
    use_one_hot_embeddings: bool. If True. use one-hot method for word embedding.
      If False, use 'tf.gather()'.
  
  Returns:
    float Tensor of shape [batch_size, seq_length, embedding_size].
  """
  
  embedding_table = tf.get_variable(
    name=word_embedding_name,
    shape=[vocab_size, embedding_size],
    initializer=ft.create_initialzer(initializer_range=initializer_range))
  
  if use_one_hot_embeddings:
    input_shape = ft.get_shape_list(input_ids, expected_rank=2)
    input_ids_squeeze = tf.reshape(input_ids, [-1])
    one_hot_input_ids = tf.one_hot(input_ids_squeeze, depth=vocab_size)
    output = tf.matmul(one_hot_input_ids, embedding_table)
    output = tf.reshape(output, [input_shape[0], input_shape[1], -1])
  else:
    output = tf.nn.embedding_lookup(embedding_table, input_ids)
  
  return output, embedding_table

def embedding_postprocessor(input_tensor,
                            use_positional_embeddings=True,
                            positional_embedding_name='positional_embeddings',
                            initializer_range=0.02,
                            max_positional_embeddings=512,
                            dropout_prob=0.1):
  """Perform positional embeddings on a word embedding tensor.

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, embedding_size].
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.
    
  Returns:
    float tensor with same sahpe as 'input_tensor'.
  """
  input_shape = ft.get_shape_list(input_tensor, expected_rank=3)
  seq_length = input_shape[1]
  width = input_shape[2]

  if use_positional_embeddings:
    assert_op = tf.assert_less_equal(seq_length, max_positional_embeddings)
    with tf.control_dependencies([assert_op]):
      full_positional_embeddings = tf.get_variable(
        name=positional_embedding_name,
        shape=[max_positional_embeddings, width],
        initializer=ft.create_initialzer(initializer_range=initializer_range))
    
    positional_embeddings = tf.slice(full_positional_embeddings, [0, 0], [seq_length, -1])  # [seq_length, width]
    positional_embeddings = tf.expand_dims(positional_embeddings, [0])  # [1, seq_length, width]

    output = input_tensor + positional_embeddings
  
  output = ft.layer_norm_and_dropout(output, dropout_prob)
  return output

def transformer_model(input_tensor,
                      attention_mask,
                      hidden_size,
                      num_hidden_layers,
                      num_attention_heads,
                      intermediate_size,
                      intermediate_act_fn,
                      hidden_dropout_prob,
                      attention_dropout_prob,
                      initializer_range,
                      do_return_all_layers=False):
  """Multi-headed, multi-layer Transformer from 'Attention is All you need'.
  
  This is almost an exact implementation of the original Transformer encoder.

  Args:
  input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
  attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
    seq_length], with 1 for positions that can be attended to and 0 in
    positions that should not be.
  hidden_size: int. Hidden size of the Transformer.
  num_hidden_layers: int. Number of layers (blocks) in the Transformer.
  num_attention_heads: int. Number of attention heads in the Transformer.
  intermediate_size: int. The size of the "intermediate" (a.k.a., feed
    forward) layer.
  intermediate_act_fn: function. The non-linear activation function to apply
    to the output of the intermediate/feed-forward layer.
  hidden_dropout_prob: float. Dropout probability for the hidden layers.
  attention_probs_dropout_prob: float. Dropout probability of the attention
    probabilities.
  initializer_range: float. Range of the initializer (stddev of truncated
    normal).
  do_return_all_layers: Whether to also return all layers or just the final
    layer.

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.
  """
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
      'The hidden size ({}) is not a multiple of the number of attention head\
         ({}).'.format(hidden_size, num_attention_heads))
  
  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = ft.get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2]

  if input_width != hidden_size:
    raise ValueError('The width of the input tensor ({}) != hidden size ({}).'.format(input_width, hidden_size))

  prev_output = input_tensor
  all_layers_outputs = []
  for layer_idx in range(num_hidden_layers):
    with tf.variable_scope('layer_{}'.format(layer_idx)):
      layer_input = prev_output

      with tf.variable_scope('attention'):
        with tf.variable_scope('self'):
          # [b, s, n * a]
          attention_head = attention_layer(input_tensor=layer_input,
                                           attention_mask=attention_mask,
                                           num_attention_heads=num_attention_heads,
                                           size_per_head=attention_head_size,
                                           query_act=None,
                                           key_act=None,
                                           value_act=None,
                                           attention_dropout_prob=attention_dropout_prob,
                                           initializer_range=initializer_range,
                                           batch_size=batch_size,
                                           seq_length=seq_length)
        
        with tf.variable_scope('output'):
          # [b, s, h]
          attention_output = tf.layers.dense(
            attention_head,
            hidden_size,
            kernel_initializer=ft.create_initialzer(initializer_range=initializer_range))
          attention_output = ft.dropout(attention_output, hidden_dropout_prob)
          attention_output = ft.layer_norm(attention_output + layer_input)

      with tf.variable_scope('intermediate'):
        # [b, s, i]
        intermediate_output = tf.layers.dense(
          attention_head,
          intermediate_size,
          activation=intermediate_act_fn,
          kernel_initializer=ft.create_initialzer(initializer_range=initializer_range))
      
      with tf.variable_scope('output'):
        # [b, s, h]
        layer_output = tf.layers.dense(
          intermediate_output,
          hidden_size,
          kernel_initializer=ft.create_initialzer(initializer_range=initializer_range))
        layer_output = ft.dropout(layer_output, hidden_dropout_prob)
        layer_output = ft.layer_norm(layer_output + attention_output)
        prev_output = layer_output
        all_layers_outputs.append(prev_output)
  
  if do_return_all_layers:
    return all_layers_outputs
  else:
    return all_layers_outputs[-1]

def attention_layer(input_tensor,
                    attention_mask,
                    num_attention_heads,
                    size_per_head,
                    query_act,
                    key_act,
                    value_act,
                    attention_dropout_prob,
                    initializer_range,
                    batch_size,
                    seq_length):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.

  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.

  In practice, the multi-headed attention are done with transposes and
  reshapes rather than actual separate tensors.

  Args:
    input_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
      * from_seq_length, num_attention_heads * size_per_head]. If False, the
      output will be of shape [batch_size, from_seq_length, num_attention_heads
      * size_per_head].
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `input_tensor`.

  Returns:
    float Tensor of shape [batch_size, from_seq_length,
      num_attention_heads * size_per_head].

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """
  
  def transpose_for_scores(input_tensor, batch_size, seq_length, num_attention_heads, width):
    output_tensor = tf.reshape(
      input_tensor, [batch_size, seq_length, num_attention_heads, width])
    
    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  # [b, s, n * a]
  query_layer = tf.layers.dense(
    input_tensor,
    num_attention_heads * size_per_head,
    activation=query_act,
    name='query',
    kernel_initializer=ft.create_initialzer(initializer_range=initializer_range))

  key_layer = tf.layers.dense(
    input_tensor,
    num_attention_heads * size_per_head,
    activation=key_act,
    name='key',
    kernel_initializer=ft.create_initialzer(initializer_range=initializer_range))
  
  value_layer = tf.layers.dense(
    input_tensor,
    num_attention_heads * size_per_head,
    activation=value_act,
    name='value',
    kernel_initializer=ft.create_initialzer(initializer_range=initializer_range))
  
  # [b, n, s, a]
  query_layer = transpose_for_scores(query_layer, batch_size, seq_length, 
                                     num_attention_heads, size_per_head)
  
  key_layer = transpose_for_scores(key_layer, batch_size, seq_length, 
                                   num_attention_heads, size_per_head)
  
  # [b, n, s, s]
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
  attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))

  if attention_mask is not None:
    # [b, 1, s, s]
    attention_mask = tf.expand_dims(attention_mask, axis=[1])
    adder = (1.0 - attention_mask) * -10000.0
    attention_scores += adder

  attention_prob = tf.nn.softmax(attention_scores)
  attention_prob = ft.dropout(attention_prob, attention_dropout_prob)

  # [b, n, s, a]
  value_layer = transpose_for_scores(value_layer, batch_size, seq_length,
                                     num_attention_heads, size_per_head)
  
  # [b, n, s, a]
  context_layer = tf.matmul(attention_prob, value_layer)
  # [b, s, n, a]
  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
  # [b, s, n * a]
  context_layer = tf.reshape(context_layer, [batch_size, seq_length, -1])

  return context_layer