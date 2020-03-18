# coding:utf-8

import sys
import logging
import tensorflow as tf
from pathlib import Path

import config as cg
from model import BertEncoder
from data_utils import train_input_fn

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

    input_x = features['input_x']
    input_mask = features['input_mask']

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    model = BertEncoder(
      config=cg.BertEncoderConfig,
      is_training=is_training,
      input_ids=input_x,
      input_mask=input_mask)
    encoder_output = model.get_sequence_output() 
    _info(encoder_output)

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

if __name__ == '__main__':
  main()