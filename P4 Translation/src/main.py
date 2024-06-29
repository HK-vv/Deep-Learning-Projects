from torch.utils.data import dataset, dataloader
from dataprocess import Vocab, Zh2EnDataSet
from setting import *
import numpy as np
import torch
from training import BertTraining

# get vocab of zh and en
vocab_zh=Vocab(FileName.vocab_zh)
vocab_en=Vocab(FileName.vocab_en)

# make dataset
trainset=Zh2EnDataSet(FileName.train_zh, FileName.train_en, vocab_zh, vocab_en)
testset=Zh2EnDataSet(FileName.test_zh, FileName.test_en, vocab_zh, vocab_en)


# creat model
config=ModelConfig()
max_src_len, max_trg_len=trainset.max_len()
bert=config.get_model(vocab_zh.size(), vocab_en.size(), max_src_len, max_trg_len)
optimizer=torch.optim.Adam(bert.parameters(), lr=LR)
loss_func=torch.nn.CrossEntropyLoss()
print("model init")

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"we are using {device}")
bert_training=BertTraining(bert, optimizer, loss_func, trainset, testset, device=device, load=True)
bert_training.train()
# for _ in range(5):
# 	bert_training.demo()

