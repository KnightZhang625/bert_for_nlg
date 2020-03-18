# coding:utf-8

from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent.parent

# data
DATA_PATH = MAIN_PATH / 'data/test_data.txt'
VOCAB_IDX_PATH = MAIN_PATH / 'data/vocab_idx.bin'
IDX_VOCAB_PATH = MAIN_PATH / 'data/idx_vocab.bin'

# model path
save_model_path = 'models/'
keep_checkpoint_max = 1
save_checkpoints_steps = 3

# global
batch_size = 3
train_steps = 3
print_info_interval = 2

# Bert
class BertEncoderConfig(object):
  hidden_dropout_prob = 0.1
  attention_dropout_prob = 0.1

  vocab_size = 7819
  embedding_size = 128
  max_positional_embeddings = 30
  hidden_size = 128
  num_hidden_layers = 4
  num_attention_heads = 4
  intermediate_size = 256

  initializer_range = 0.02
  hidden_act = 'gelu'