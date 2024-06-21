import torch
import torch.utils
import torch.utils.data
import numpy as np


class Zh2EnDataSet(torch.utils.data.Dataset):
	def __init__(self, src_filename, trg_filename, src_vocab, trg_vocab):
		super().__init__()
		self.src_filename=src_filename
		self.trg_filename=trg_filename
		self.src_vocab=src_vocab
		self.trg_vocab=trg_vocab
		self.src_lines, self.trg_lines=self.__read_data()
		self.max_src_len=max(len(line) for line in self.src_lines)
		self.max_trg_len=max(len(line) for line in self.trg_lines)


	def __len__(self):
		return len(self.src_lines)


	def __getitem__(self, index):
		src_data=self.src_lines[index]
		trg_data=self.trg_lines[index]

		# word encoding
		s_w2i=self.src_vocab.word2id
		t_w2i=self.trg_vocab.word2id
		src_data=[s_w2i[word] if word in s_w2i.keys() else s_w2i['<unk>'] for word in src_data]
		trg_data=[t_w2i[word] if word in t_w2i.keys() else t_w2i['<unk>'] for word in trg_data]
		src=torch.LongTensor(self.max_src_len).fill_(s_w2i['<pad>'])
		trg=torch.LongTensor(self.max_trg_len).fill_(t_w2i['<pad>'])
		src[:len(src_data)]=torch.LongTensor(src_data)
		trg[:len(trg_data)]=torch.LongTensor(trg_data)
		return src, trg
	

	def __read_data(self):
		with open(self.src_filename, 'r', encoding='utf-8') as f:
			src_lines=np.array(f.readlines())
		with open(self.trg_filename, 'r', encoding='utf-8') as f:
			trg_lines=np.array(f.readlines())
		assert len(src_lines)==len(trg_lines)
		return self.__preprocessing_data(src_lines, trg_lines)


	def __preprocessing_data(self, src_lines, trg_lines):
		src_lines=[['<sos>']+line.strip().split('\t')+['<eos>'] for line in src_lines]
		trg_lines=[['<sos>']+line.strip().split('\t')+['<eos>'] for line in trg_lines]
		return src_lines, trg_lines






