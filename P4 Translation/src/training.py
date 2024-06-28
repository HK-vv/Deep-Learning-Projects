import random
import torch.utils
from setting import *
from torch.utils.data.dataloader import DataLoader
import torch
import time
from torchtext.data.metrics import bleu_score

class BertTraining():
	def __init__(self, bert, optimizer, criterion, trainset, testset, device, load):
		self.bert=bert.to(device)
		self.optimizer=optimizer
		self.criterion=criterion
		self.trainset=trainset
		self.testset=testset
		self.trainloader=DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
		self.validloader=DataLoader(testset, batch_size=BATCH_SIZE)
		self.testloader=DataLoader(testset, batch_size=1)
		self.device=device
		self.train_loss=[]
		self.valid_loss=[]
		self.valid_bleu=[]
		self.test_bleu=[]
		self.ep=0
		if load:
			self._load()

	def _load(self):
		cp=torch.load(FileName.checkpoint)
		self.bert.load_state_dict(cp['model_state'])
		self.optimizer.load_state_dict(cp['optimizer_state'])
		self.ep=cp['current_epoch']
		self.train_loss=cp['train_loss']
		self.valid_loss=cp['valid_loss']
		self.valid_bleu=cp['valid_bleu']
		self.test_bleu=cp['test_bleu']

	def train(self):	
		while self.ep<EPOCH:
			self.ep+=1
			start_time=time.time()
			epoch_train_loss=self._train_epoch()
			print(f"trained, using {time.time()-start_time:.0f}s")
			epoch_valid_res=self._valid_epoch()
			print(f"validated, using {time.time()-start_time:.0f}s")
			t=self.ep%TEST_FREQ==0
			if t:
				epoch_test_bleu=self._test_epoch()
				print(f"tested, using {time.time()-start_time:.0f}s")
				self.test_bleu.append(epoch_test_bleu)
			self.train_loss.append(epoch_train_loss)
			self.valid_loss.append(epoch_valid_res[0])
			self.valid_bleu.append(epoch_valid_res[1])
			elapse_time=time.time()-start_time
			if self.ep%SAVE_FREQ==0:
				checkpoint={
					'model_state': self.bert.state_dict(),
					'optimizer_state': self.optimizer.state_dict(),
					'current_epoch': self.ep,
					'train_loss': self.train_loss,
					'valid_loss': self.valid_loss,
					'valid_bleu': self.valid_bleu,
					'test_bleu': self.test_bleu
				}
				torch.save(checkpoint, FileName.checkpoint)
			print(f"Epoch {self.ep:} train_loss={epoch_train_loss:.2f}, valid_loss={epoch_valid_res[0]:.2f}, valid_bleu={epoch_valid_res[1]:.2f}"
		 		+(f", test_bleu={epoch_test_bleu:.2f}" if t else "")+f". using {elapse_time/60:.2f} minutes")


	def _train_epoch(self):
		self.bert.train()
		total_loss=0.0
		for step, (src, trg) in enumerate(self.trainloader):
			# print(f"step {step}")
			src, trg=src.to(self.device), trg.to(self.device)
			src_pad_mask, trg_pad_mask, trg_att_mask=self._make_masks(src, trg[:,:-1])
			self.optimizer.zero_grad()
			out=self.bert(src, trg[:,:-1], src_pad_mask, trg_pad_mask, trg_att_mask)
			# out.shape=(batch_size, trg_len-1, trg_vocab_size)

			trg_vocab_size=len(self.trainset.trg_vocab)

			# flatten the batch dim
			out=out.contiguous().view(-1, trg_vocab_size)
			trg=trg[:,1:].contiguous().view(-1)

			loss=self.criterion(out, trg)
			loss.backward()
			torch.nn.utils.clip_grad.clip_grad_norm_(self.bert.parameters(), 1)
			self.optimizer.step()
			total_loss+=loss.item()

		mean_loss=total_loss/len(self.trainloader)
		return mean_loss
		

	def _valid_epoch(self):
		self.bert.eval()
		val_loss=0
		_bleu_score=0
		with torch.no_grad():
			for step, (src, trg) in enumerate(self.validloader):
				src, trg= src.to(self.device), trg.to(self.device)
				src_pad_mask, trg_pad_mask, trg_att_mask=self._make_masks(src, trg[:,:-1])
				out=self.bert(src, trg[:,:-1], src_pad_mask, trg_pad_mask, trg_att_mask)
				out=torch.log_softmax(out, dim=-1)
				_bleu_score+=self._compute_bleu_score(out.argmax(dim=-1), trg, self.testset.trg_vocab)
				trg_vocab_size=len(self.testset.trg_vocab)
				out=out.contiguous().view(-1, trg_vocab_size)
				trg=trg[:,1:].contiguous().view(-1)
				val_loss+=self.criterion(out, trg).item()
				# TODO: insert torchtext.data.metrics.bleu_score
		b_num=len(self.validloader)
		return val_loss/b_num, _bleu_score/b_num
	
	def _test_epoch(self):
		# TODO: do predict task and compute mean bleu
		_bleu_score=0
		for step, (src, trg) in enumerate(self.testloader):
			trg_out=self._translate(src)
			# compare trg_out and trg.
			trg_out=trg_out.cpu()
			_bleu_score+=self._compute_bleu_score(trg_out, trg, self.testset.trg_vocab)
		return _bleu_score/len(self.testloader)
				
	def _translate(self, src: torch.Tensor):
		if len(src.shape)==1:
			src=src.view(1, -1)
		src=src.to(self.device)
		src_pad_mask=self._make_masks(src)
		trg_out=torch.zeros(1, self.bert.max_trg_len, dtype=int).fill_(PAD).to(self.device)
		trg_out[0, 0]=self.testset.trg_vocab.word2id['<s>']
		encoder_out=self.bert.encoder(src, src_pad_mask)
		for i in range(self.bert.max_trg_len-1):
			_, trg_pad_mask, trg_att_mask=self._make_masks(src, trg_out)
			decoder_out=self.bert.decoder(trg_out, encoder_out, trg_pad_mask, trg_att_mask)
			trg_out[0, i+1]=decoder_out.argmax(dim=-1)[0, i]
			if trg_out[0, i+1].cpu().item()==self.testset.trg_vocab.word2id['</s>']:
				break
		return trg_out

	def _compute_bleu_score(self, out, trg, vocab):
		shifted_trg=trg[:,1:]
		b_size=out.shape[0]
		out_2word=[]
		trg_2word=[]
		for i in range(b_size):
			out_2word.append(self._ids2words(out[i], vocab))
			trg_2word.append([self._ids2words(shifted_trg[i], vocab)])
		return bleu_score(out_2word, trg_2word)

	def _ids2words(self, iter, vocab):
		r=[]
		for id in iter:
			word=vocab.id2word[id.cpu().item()]
			r.append(word)
			if word=='</s>':
				break
		return r

	def _ids2line(self, iter, vocab, sep=' '):
		return sep.join(self._ids2words(iter, vocab))

	def _make_masks(self, src, trg=None):
		# src_len=src.shape[-1]
		src_pad_mask=(src==PAD).to(self.device)
		if trg is None:
			return src_pad_mask
		trg_len=trg.shape[-1]
		trg_pad_mask=(trg==PAD).to(self.device)
		trg_att_mask=torch.triu(torch.ones(trg_len, trg_len), diagonal=1).to(dtype=bool).to(self.device)
		return src_pad_mask, trg_pad_mask, trg_att_mask
	
	def demo(self):
		random_index=random.choice(range(len(self.testset)))
		src, trg=self.testset[random_index]
		trg_out=self._translate(src)
		print(f"source text:\n{self._ids2line(src.view(-1), self.testset.src_vocab, sep='')}")
		print(f"model output:\n{self._ids2line(trg_out.view(-1), self.testset.trg_vocab)}")
		print(f"reference translation:\n{self._ids2line(trg.view(-1), self.testset.trg_vocab)}")
		