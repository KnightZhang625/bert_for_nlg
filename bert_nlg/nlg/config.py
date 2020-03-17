# coding:utf-8

from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent.parent

DATA_PATH = MAIN_PATH / 'data/test_data.txt'

class BertEncoderConfig(object):
  hidden_dropout_prob = 0.1
  attention_probs_dropout_prob = 0.1

  vocab_size = -1
  embedding_size = -1
  max_positional_embeddings = -1
  hidden_size = -1
  num_hidden_layers = -1
  num_attention_heads = -1
  intermediate_size = -1

  initializer_range = 0.02
  hidden_act = 'gelu'

if __name__ == '__main__':
  print(BertEncoderConfig.hidden_dropout_prob)