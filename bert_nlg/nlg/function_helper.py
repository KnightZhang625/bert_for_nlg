# coding:utf-8

import six
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