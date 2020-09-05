import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from lstm import LSTMCellb,LSTMCello,LSTMCelli,LSTMCellf,LSTMCell

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,vocab_size,batch_size):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, input_dim)
	# Change the line below to change lstm cell structure from lstm.py(can use LSTMCell, LSTMCellb, LSTMCello, LSTMCelli and LSTMCellf)
        self.lstm_cell = LSTMCellf(input_dim, hidden_dim,batch_size)
        self.fc = nn.Linear(hidden_dim, output_dim)
     
    def forward(self, x):
        x=self.word_embeddings(x)
        batch_size = self.batch_size
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(batch_size,self.hidden_dim).cuda())
            
        else:
            h0 = Variable(torch.zeros(batch_size,self.hidden_dim))
        if torch.cuda.is_available():
            c0 = Variable(torch.zeros(batch_size,self.hidden_dim).cuda())
            
        else:
            c0 = Variable(torch.zeros(batch_size,self.hidden_dim))
            
            
        outs = torch.zeros(batch_size,x.size(1),self.hidden_dim).cuda()
        
        hn=h0
        cn=c0
        
        for seq in range(x.size(1)):
            cn,hn = self.lstm_cell(x[:,seq,:], (hn,cn)) 
            outs[:,seq,:]=hn
            
        out = self.fc(outs)
        out = F.log_softmax(out,dim=2)
        return out
    
