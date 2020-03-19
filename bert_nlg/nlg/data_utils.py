# coding:utf-8

import copy
import codecs
import pickle
import random
import functools
import numpy as np
import tensorflow as tf
from pathlib import Path
# tf.enable_eager_execution()

from log import log_info as _info
from log import log_error as _error
from log import print_process as _process

# load global configuration
import config as cg
batch_size = cg.batch_size
train_steps = cg.train_steps
sos_id = cg.DecoderConfig.sos_id
eos_id = cg.DecoderConfig.eos_id

# # abandon generator
# def train_generator(path):
#   """this could achieve random by dataset.shuffle(10).batch(2).repeat(3)."""
#   with codecs.open(path, 'r', 'utf-8') as file:
#     for line in file:
#       if len(line) != 0:
#         line = [i for i in line][:-1]
#         line = np.array(list(map(int, line)))
				
#         features = {'input': line}
#         yield features

# load the vocab dictionary
def load_vocab():
	global vocab_idx
	global idx_vocab
	with codecs.open(cg.VOCAB_IDX_PATH, 'rb') as vocab_idx_read,\
			 codecs.open(cg.IDX_VOCAB_PATH, 'rb') as idx_vocab_read:
			vocab_idx = pickle.load(vocab_idx_read)
			idx_vocab = pickle.load(idx_vocab_read)

load_vocab()

# covert vocab to idx function
str_to_idx = lambda line : [vocab_idx[v] if v in vocab_idx.keys() else vocab_idx['<unk>'] for v in line]

"""check whether the coverted data exists."""
def check_data(need_exit):
	def de_func(func):
		@functools.wraps(func)
		def de_func_inner():
			data_path = Path(__file__).absolute().parent / 'processed_data'
			files = list(data_path.rglob('*.bin'))
			if need_exit:
				if len(files) != 2:
					_error('No data exists.')
					raise FileNotFoundError
			else:
				if len(files) > 0:
					_error('The data exists.')
					raise FileExistsError
			if func.__name__ == 'process_data':
				func()
			else:
				for data in func():
					yield data
		return de_func_inner
	return de_func

def make_process_data():
	"""due to `yield` in decorator, use another function to execute `process_data()`."""
	for _ in process_data():
		pass

@check_data(need_exit=False)
def process_data():
	"""covert the string data to idx, and save."""
	with codecs.open(cg.DATA_PATH, 'r', 'utf-8') as file:
		data = file.read().split('\n')

	if len(data[-1]) == 0:
		_error('The last line which is empty has been removed.')
		data = data[:-1]

	questions = []
	answers = []
	for line in data:
		line_split = line.split('=')
		que, ans = line_split[0], line_split[1]
		questions.append(que)
		answers.append(ans)
	assert len(questions) == len(answers),\
		_error('The number of quesiton: {} not equal to the number of answer: {}'.format(len(questions), len(answers)))

	que_idx = [str_to_idx(que) for que in questions]
	ans_idx = [str_to_idx(ans) for ans in answers]

	Path('processed_data/').mkdir(exist_ok=True)
	with codecs.open('processed_data/questions.bin', 'wb') as file:
		pickle.dump(que_idx, file)
	with codecs.open('processed_data/answers.bin', 'wb') as file:
		pickle.dump(ans_idx, file)
	
	_info('Coverted questions and answers have been saved into `processed_data` directory.')

def make_mask(que_batch, reverse=False):
	mask = []
	for que in que_batch:
		mask_per_que = []
		for idx, v in enumerate(que):
			if v != vocab_idx['<padding>']:
				prev = idx + 1
				rear = len(que) - prev
				if not reverse:
					mask_per_vocab = [1 for _ in range(prev)] + [0 for _ in range(rear)]
				else:
					mask_per_vocab = [0 for _ in range(prev)] + [1 for _ in range(rear)]
			else:
				mask_per_vocab = [0 for _ in range(len(que))]
			mask_per_que.append(mask_per_vocab)
		mask.append(mask_per_que)
	return mask

def padding_data(que_batch, input_ans_batch, output_ans_batch):
	"""padding each data in the batch with same length."""
	max_que_length = max(list(map(len, que_batch)))
	max_ans_length = max(list(map(len, input_ans_batch)))

	padding = lambda line, max_length : line + [vocab_idx['<padding>'] for _ in range(max_length - len(line))]

	padding_que = functools.partial(padding, max_length=max_que_length)
	padding_ans = functools.partial(padding, max_length=max_ans_length)

	que_batch = list(map(padding_que, que_batch))
	input_ans_batch = list(map(padding_ans, input_ans_batch))
	output_ans_batch = list(map(padding_ans, output_ans_batch))

	mask = make_mask(que_batch)
	
	return que_batch, input_ans_batch, output_ans_batch, mask

@check_data(need_exit=True)
def train_generator():
	"""this could achieve padding among each batch."""
	# load the data
	with codecs.open('processed_data/questions.bin', 'rb') as file:
		questions = pickle.load(file)
	with codecs.open('processed_data/answers.bin', 'rb') as file:
		answers = pickle.load(file)
	assert len(questions) == len(answers),\
		 _error('The number of quesiton: {} not equal to the number of answer: {}'.format(len(questions), len(answers)))
	
	# random shuffle the data
	questions_answers = list(zip(questions, answers))
	random.shuffle(questions_answers)
	questions, answers = zip(*questions_answers)
	questions = list(questions)
	answers = list(answers)

	que_batch = []
	input_ans_batch = []
	output_ans_batch = []
	seq_length_decoder_input_data = []
	batch_num = len(questions) // batch_size
	for idx, que in enumerate(questions):
		if len(que_batch) < batch_size:
			# que
			que_batch.append(que)

			# ans
			inp_ans = copy.deepcopy(answers[idx])
			out_ans = copy.deepcopy(answers[idx])
			inp_ans.insert(0, sos_id)
			out_ans.append(eos_id)
			input_ans_batch.append(inp_ans)
			output_ans_batch.append(out_ans)
			seq_length_decoder_input_data.append(len(inp_ans))

			# check whether a batch is full
			if len(que_batch) == batch_size:
				que_batch_padded, inp_ans_batch_padded, out_ans_batch_padded, mask = padding_data(que_batch, input_ans_batch, output_ans_batch)
				features = {'input_x': que_batch_padded, 'input_mask': mask, 'input_y': inp_ans_batch_padded, 'seq_length': seq_length_decoder_input_data}
				yield(features, out_ans_batch_padded)
				que_batch = []
				input_ans_batch = []
				output_ans_batch = []
				seq_length_decoder_input_data = []

			if idx > (batch_num * batch_size - 1) and len(questions) % batch_size != 0:
					que_batch = copy.deepcopy(questions[idx:])
					input_ans_batch = copy.deepcopy(answers[idx:])
					output_ans_batch = copy.deepcopy(answers[idx:])
					for _ in range(batch_num - len(que_batch)):
						aug_que = random.choice(questions)
						aug_que_idx = questions.index(aug_que)						
						que_batch.append(aug_que)

						inp_ans = copy.deepcopy(answers[aug_que_idx])
						out_ans = copy.deepcopy(answers[aug_que_idx])
						inp_ans.insert(0, sos_id)
						out_ans.append(eos_id)
						input_ans_batch.append(inp_ans)
						output_ans_batch.append(out_ans)
					
					for idx, _ in enumerate(input_ans_batch):
						input_ans_batch[idx].insert(0, sos_id)
					for idx, _ in enumerate(output_ans_batch):
						output_ans_batch[idx].append(eos_id)
					seq_length_decoder_input_data = [len(ans) for ans in input_ans_batch]

					assert len(que_batch) == len(input_ans_batch) == len(output_ans_batch) == batch_size

					que_batch_padded, inp_ans_batch_padded, out_ans_batch_padded, mask = padding_data(que_batch, input_ans_batch, output_ans_batch)
					features = {'input_x': que_batch_padded, 'input_mask': mask, 'input_y': inp_ans_batch_padded, 'seq_length': seq_length_decoder_input_data}
					yield(features, out_ans_batch_padded)
					break
		
def train_input_fn():
	output_types = {'input_x': tf.int32, 'input_mask': tf.int32, 'input_y': tf.int32, 'seq_length': tf.int32}
	output_shapes = {'input_x': [None, None], 'input_mask': [None, None, None], 'input_y': [None, None], 'seq_length': [None]}

	dataset = tf.data.Dataset.from_generator(
		train_generator,
		output_types=(output_types, tf.int32),
		output_shapes=(output_shapes, [None, None]))

	dataset = dataset.repeat(train_steps)

	return dataset

def server_input_fn():
	input_x = tf.placeholder(tf.int32, shape=[None, None], name='input_x')
	input_mask = tf.placeholder(tf.int32, shape=[None, None, None], name='input_mask')

	receive_tensors = {'input_x': input_x, 'input_mask': input_mask}
	features = {'input_x': input_x, 'input_mask': input_mask}

	return tf.estimator.export.ServingInputReceiver(features, receive_tensors)

if __name__ == '__main__':
	# for data in train_input_fn():
	#   print(data)
	#   input()
	
	# for data in train_generator():
	# 	print(data)
	# 	input()
	
	make_process_data()