# coding:utf-8
# This file is used for extracting loss from the log file.

import re
import sys
import codecs
import argparse
from pathlib import Path

MAIN_PATH = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(MAIN_PATH))

from log import log_info as _info
from log import log_error as _error

PATTERN = r'^(loss = )\d{1,5}\.\d{1,10}' 
PATTERN_2 = r'(ppl = )\d{1,5}\.\d{1,10}'

def extract(log_path, save_path, save_path_2):
    with codecs.open(log_path, 'r', 'utf-8') as file, \
         codecs.open(save_path, 'w', 'utf-8') as file_2, \
         codecs.open(save_path_2, 'w', 'utf-8') as file_3:
        for line in file:
            if re.search(PATTERN, line):
                match = re.search(PATTERN, line).group()
                loss = match.split(' ')[2]
                file_2.write('sup_avg:' + loss + '\n')
                file_2.flush()
            if re.search(PATTERN_2, line):
              match = re.search(PATTERN_2, line).group()
              ppl = match.split(' ')[2]
              file_3.write('sup_avg:' + ppl + '\n')
              file_3.flush()
    _info('The loss record have been save to {}.'.format(save_path))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-p')
  
  args = parser.parse_args()
  path = args.p

  log_path = path
  extract(log_path, 'loss_record', 'ppl_record')