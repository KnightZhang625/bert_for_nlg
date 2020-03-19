# coding:utf-8

import sys
import logging
import argparse
import tensorflow as tf
from pathlib import Path

import config as cg
import function_toolkit as ft
from model import BertEncoder, Decoder
from data_utils import train_input_fn, server_input_fn

from log import log_info as _info
from log import log_error as _error

MAIN_PATH = Path(__file__).absolute().parent

# log record
class Setup(object):
    """Setup logging"""
    def __init__(self, log_name='tensorflow', path=str(MAIN_PATH / 'log')):
        Path('log').mkdir(exist_ok=True)
        tf.compat.v1.logging.set_verbosity(logging.INFO)
        handlers = [logging.FileHandler(str(MAIN_PATH / 'log/main.log')),
                    logging.StreamHandler(sys.stdout)]
        logging.getLogger('tensorflow').handlers = handlers
setup = Setup()

def model_fn_builder():
  """returns `model_fn` closure for the Estimator."""
  
  def model_fn(features, labels, mode, params):
    # features name and shape
    _info('*** Features ****')
    for name in sorted(features.keys()):
      tf.logging.info(' name = {}, shape = {}'.format(name, features[name].shape))

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # get data
    input_x = features['input_x']
    input_mask = features['input_mask']
    if is_training:
      input_y = features['input_y']
      seq_length = features['seq_length']
    else:
      input_y = None
      seq_length = None

    # build encoder
    model = BertEncoder(
      config=cg.BertEncoderConfig,
      is_training=is_training,
      input_ids=input_x,
      input_mask=input_mask)
    embedding_table = model.get_embedding_table()
    encoder_output = tf.reduce_sum(model.get_sequence_output(), axis=1)

    # build decoder
    decoder_model = Decoder(
      config=cg.DecoderConfig,
      is_training=is_training,
      encoder_state=encoder_output,
      embedding_table=embedding_table,
      decoder_intput_data=input_y,
      seq_length_decoder_input_data=seq_length)
    logits, sample_id, ppl_seq, ppl = decoder_model.get_decoder_output()

    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {'sample_id': sample_id, 'ppls': ppl_seq}
      output_spec = tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
      if mode == tf.estimator.ModeKeys.TRAIN:
        max_time = ft.get_shape_list(labels, expected_rank=2)[1]
        target_weights = tf.sequence_mask(seq_length, max_time, dtype=logits.dtype)
        batch_size = tf.cast(ft.get_shape_list(labels, expected_rank=2)[0], tf.float32)

        loss = tf.reduce_sum(
          tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits) * target_weights) / batch_size

        learning_rate = tf.train.polynomial_decay(cg.learning_rate,
                                          tf.train.get_or_create_global_step(),
                                          cg.train_steps / 100,
                                          end_learning_rate=1e-4,
                                          power=1.0,
                                          cycle=False)

        lr = tf.maximum(tf.constant(cg.lr_limit), learning_rate)
        optimizer = tf.train.AdamOptimizer(lr, name='optimizer')
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss, tvars, colocate_gradients_with_ops=cg.colocate_gradients_with_ops)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, tvars), global_step=tf.train.get_global_step())


        # this is excellent, because it could display the result each step, i.e., each step equals to batch_size.
        # the output_spec, display the result every save checkpoints step.
        logging_hook = tf.train.LoggingTensorHook({'loss' : loss, 'ppl': ppl, 'lr': lr}, every_n_iter=cg.print_info_interval)

        output_spec = tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])
      elif mode == tf.estimator.ModeKeys.EVAL:
        # TODO
        raise NotImplementedError
    
    return output_spec
  
  return model_fn

def main():
  # create directory to save the model
  Path(cg.save_model_path).mkdir(exist_ok=True)

  model_fn = model_fn_builder()

  gpu_config = tf.ConfigProto()
  gpu_config.gpu_options.allow_growth = True

  run_config = tf.contrib.tpu.RunConfig(
    session_config=gpu_config,
    keep_checkpoint_max=cg.keep_checkpoint_max,
    save_checkpoints_steps=cg.save_checkpoints_steps,
    model_dir=cg.save_model_path)

  estimator = tf.estimator.Estimator(model_fn, config=run_config)
  estimator.train(train_input_fn)

def package_model(ckpt_path, pb_path):
  model_fn = model_fn_builder()
  estimator = tf.estimator.Estimator(model_fn, ckpt_path)
  estimator.export_saved_model(pb_path, server_input_fn)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=_info('python train.py [train | package]', head='USAGE:'))
  parser.add_argument('mode')
  
  args = parser.parse_args()
  mode = args.mode
  if mode == 'train':
    main()
  elif mode == 'pacakge':
    pakage_model('models/', 'pb_models/')