import torch
from model import Poet
import setting

INIT_S=40

def predict(text, gen_len=100):
	cp=torch.load('../checkpoint/poet.pth')
	poet=setting.poet
	poet.load_state_dict(cp['model_state'])
	poet.eval()
	# convert text to tensor
	text=list(map(lambda t: setting.word2ix[t], text))
	h=poet.init_state()
	x=torch.tensor(text)

	for i in range(gen_len):
		y, h=poet(x, h)
		y_pred=torch.argmax(y[-1]).detach()
		if y_pred.numpy()==setting.word2ix['<EOP>']:
			break
		x=torch.cat([x, y_pred.view(1)])
	
	x=x[INIT_S+1:]
	predict_words=[setting.ix2word[_] for _ in x.detach().numpy()]
	predict_text=''.join(predict_words)
	return predict_text

input_text='秦时明月汉时关'
print("input text: "+input_text)
ptext=predict(['</s>']*INIT_S+['<START>']+list(input_text), gen_len=100)
print("output text: "+ptext)

