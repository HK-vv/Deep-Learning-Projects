from model import Bert

SAVE_FREQ=2
TEST_FREQ=5

EPOCH=200
BATCH_SIZE=100
LR=0.001
PAD=0

class FileName():
	data_prefix='../data/'
	vocab_zh=data_prefix+'vocab.zh'
	vocab_en=data_prefix+'vocab.en'
	train_zh=data_prefix+'train.zh'
	train_en=data_prefix+'train.en'
	test_zh=data_prefix+'test.zh'
	test_en=data_prefix+'test.en'
	fig_prefix='../doc/pic/'
	checkpoint='../checkpoint/bert.pth'
	fig_prefix='../doc/pic/'
	train_loss_fig=fig_prefix+'train_loss.png'
	valid_loss_fig=fig_prefix+'valid_loss.png'
	valid_bleu_fig=fig_prefix+'valid_bleu.png'
	test_bleu_fig=fig_prefix+'test_bleu_fig'

		
class ModelConfig():
	src_vocab_size=None
	trg_vocab_size=None 
	h_dim=256
	enc_n_heads=8
	dec_n_heads=8
	enc_ff_dim=256
	dec_ff_dim=256
	enc_n_layers=4
	dec_n_layers=4
	dropout=0.1
	
	def get_model(self, src_vocab_size, trg_vocab_size, max_src_len, max_trg_len):
		bert=Bert(src_vocab_size=src_vocab_size,
			trg_vocab_size=trg_vocab_size,
			h_dim=self.h_dim,
			enc_n_heads=self.enc_n_heads,
			dec_n_heads=self.dec_n_heads,
			enc_ff_dim=self.enc_ff_dim,
			dec_ff_dim=self.dec_ff_dim,
			enc_n_layers=self.enc_n_layers,
			dec_n_layers=self.dec_n_layers,
			dropout=self.dropout,
			max_src_len=max_src_len, 
			max_trg_len=max_trg_len)
		return bert