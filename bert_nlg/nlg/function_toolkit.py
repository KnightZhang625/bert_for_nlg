# coding:utf-8

import six
import numpy as np
import tensorflow as tf

def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns the shape of the tensor as a list format, prefer static dimensions.
  
  Args:
    tensor: A tf.Tensor to find the shape.
    expected_rank: (optional) int. If this argument is specified,
      an exception will be thrown if the actual rank not match
      the expected rank.
  
  Returns:
    A list of dimensions of the shape of the tensor, All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned as
    tf.Tensor scalars.
  """

  if name is None:
    name = tensor.name
  
  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)
  
  shape = tensor.shape.as_list()

  non_static_indexes = []
  for idx, dim in enumerate(shape):
    if dim is None:
      non_static_indexes.append(idx)
  
  # no dynamic dimension exits
  if not non_static_indexes:
    return shape
  
  dynamic_shape = tf.shape(tensor)
  for idx in non_static_indexes:
    shape[idx] = dynamic_shape[idx]
  
  return shape

def assert_rank(tensor, expected_rank, name):
  if name is None:
    name = tensor.name
  
  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    # could expect multiple rank choices
    for x in expected_rank:
      expected_rank_dict[x] = True
  
  actual_rank = tensor.shape.ndims  # this is the rank number, not the shape.
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        'For the tensor `{}` in scope `{}`, the actual rank \
        `{}` (shape = {}) is not equal to the expected rank `{}`'.format(
          name, scope_name, actual_rank, tensor.shape, expected_rank))

def create_initialzer(init_type='trunc', initializer_range=0.02):
  if init_type is 'trunc':
    return tf.truncated_normal_initializer(stddev=initializer_range)
  else:
    raise NotImplementedError('Initialize Type: `{}` not implemented.'.format(init_type))

def layer_norm(input_tensor, name=None):
  return tf.contrib.layers.layer_norm(inputs=input_tensor, scope=name)

def dropout(input_tensor, dropout_prob):
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor
  
  output = tf.nn.dropout(input_tensor, keep_prob=(1-dropout_prob))

  return output

def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
  output_tensor = layer_norm(input_tensor, name)
  output_tensor = dropout(output_tensor, dropout_prob)
  return output_tensor

def create_attention_mask_from_input_mask(from_tensor, to_mask):
  """Create 3D attention mask from a 2D tensor mask."""
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  from_seq_length = from_shape[1]

  to_mask = tf.expand_dims(to_mask, [1])  # [batch_size, 1, to_seq_length]
  to_mask = tf.tile(to_mask, [1, from_seq_length, 1]) # [batch_size, from_seq_length, to_seq_length]
  to_mask = tf.cast(to_mask, tf.float32)

  return to_mask

def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return tf.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)

def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf

def get_ppl(logits):
  max_time = tf.cast(get_shape_list(logits, expected_rank=3)[1], tf.float32)
  # [b, s, v]
  probs = tf.nn.softmax(logits, axis=-1)
  # [b, s]
  probs_max = tf.reduce_max(probs, axis=-1)
  log_prob =  -tf.log(probs_max)
  # [b]
  log_prob_seq = tf.reduce_sum(log_prob, axis=-1) / max_time
  log_prob_mean = tf.reduce_mean(log_prob_seq)

  return log_prob_seq, log_prob_mean