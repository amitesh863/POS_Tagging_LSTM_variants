# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 17:52:10 2019

@author: Amitesh863
"""

import math
import torch
from torch.nn import Parameter
import torch.nn as nn


#Standard LSTM Cell
class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size,batch_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh =Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias = Parameter(torch.Tensor(batch_size,4 * hidden_size))
        self.reset_parameters()



    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
   
    def forward(self, x, hidden):
       hx, cx = hidden
       dim_h = self.hidden_size

       wxi = self.weight_ih[:dim_h]
       wxj = self.weight_ih[dim_h:2 * dim_h]
       wxf = self.weight_ih[2 * dim_h:3 * dim_h]
       wxo = self.weight_ih[3 * dim_h:]

     
       whi =self.weight_hh[:dim_h]
       whj = self.weight_hh[dim_h:2 * dim_h]
       whf=self.weight_hh[2 * dim_h:3 * dim_h]
       who=self.weight_hh[3 * dim_h:]

       bi = self.bias[:,:dim_h]
       bj = self.bias[:,dim_h:2 * dim_h]
       bf = self.bias[:,2 * dim_h:3 * dim_h]
       bo= self.bias[:,3 * dim_h:]
       

       it = torch.tanh(x@wxi.t()+hx@whi.t()+bi)
       jt = torch.sigmoid(x@wxj.t()+hx@whj.t()+bj)
       ft = torch.sigmoid(x@wxf.t()+hx@whf.t()+bf)
       ot = torch.tanh(x@wxo.t()+hx@who.t()+bo)
 
       ct = cx* ft + it *jt
       ht = torch.tanh(ct)*ot
       return ct,ht

# LSTM Cell without the forget gate
class LSTMCellf(nn.Module):

    def __init__(self, input_size, hidden_size,batch_size, bias=True):
        super(LSTMCellf, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh =Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.bias = Parameter(torch.Tensor(batch_size,3 * hidden_size))
        self.reset_parameters()



    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
   
    def forward(self, x, hidden):
       hx, cx = hidden
       dim_h = self.hidden_size

       wxi = self.weight_ih[:dim_h]
       wxj = self.weight_ih[dim_h:2 * dim_h]
       wxo = self.weight_ih[2 * dim_h:]
    

     
       whi =self.weight_hh[:dim_h]
       whj = self.weight_hh[dim_h:2 * dim_h]
       
       who=self.weight_hh[2 * dim_h:]

       bi = self.bias[:,:dim_h]
       bj = self.bias[:,dim_h:2 * dim_h]
       
       bo= self.bias[:,2 * dim_h:]
       

       it = torch.tanh(x@wxi.t()+hx@whi.t()+bi)
       jt = torch.sigmoid(x@wxj.t()+hx@whj.t()+bj)
       ot = torch.tanh(x@wxo.t()+hx@who.t()+bo)

       

       
       ct =it *jt
       ht = torch.tanh(ct)*ot
       return ct,ht

# LSTM Cell without the output gate
class LSTMCello(nn.Module):

    def __init__(self, input_size, hidden_size,batch_size, bias=True):
        super(LSTMCello, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh =Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.bias = Parameter(torch.Tensor(batch_size,3 * hidden_size))
        self.reset_parameters()



    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
   
    def forward(self, x, hidden):
       hx, cx = hidden
       dim_h = self.hidden_size

       wxi = self.weight_ih[:dim_h]
       wxj = self.weight_ih[dim_h:2 * dim_h]
       wxf = self.weight_ih[2 * dim_h:]

     
       whi =self.weight_hh[:dim_h]
       whj = self.weight_hh[dim_h:2 * dim_h]
       whf=self.weight_hh[2 * dim_h:]

       bi = self.bias[:,:dim_h]
       bj = self.bias[:,dim_h:2 * dim_h]
       bf= self.bias[:,2 * dim_h:]
       

       it = torch.tanh(x@wxi.t()+hx@whi.t()+bi)
       jt = torch.sigmoid(x@wxj.t()+hx@whj.t()+bj)
       ft = torch.sigmoid(x@wxf.t()+hx@whf.t()+bf)

       

       
       ct = cx* ft + it *jt
       ht = torch.tanh(ct)
       return ct,ht

# LSTM Cell w/o the input gate
class LSTMCelli(nn.Module):

    def __init__(self, input_size, hidden_size,batch_size, bias=True):
        super(LSTMCelli, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(2 * hidden_size, input_size))
        self.weight_hh =Parameter(torch.Tensor(2 * hidden_size, hidden_size))
        self.bias = Parameter(torch.Tensor(batch_size,2 * hidden_size))
        self.reset_parameters()



    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
   
    def forward(self, x, hidden):
       hx, cx = hidden
       dim_h = self.hidden_size

       wxf = self.weight_ih[:dim_h]
       wxo = self.weight_ih[dim_h:]

     
       whf =self.weight_hh[:dim_h]
       who=self.weight_hh[ dim_h:]

       bf = self.bias[:,:dim_h]
       bo= self.bias[:, dim_h:]
       

       ft = torch.sigmoid(x@wxf.t()+hx@whf.t()+bf)
       ot = torch.tanh(x@wxo.t()+hx@who.t()+bo)

       

       
       ct = cx* ft
       ht = torch.tanh(ct)*ot
       return ct,ht

# LSTM cell with forget gate bias set to 1
class LSTMCellb(nn.Module):

    def __init__(self, input_size, hidden_size,batch_size, bias=True):
        super(LSTMCellb, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh =Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias = Parameter(torch.Tensor(batch_size,4 * hidden_size))
        self.reset_parameters()




    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
   
    def forward(self, x, hidden):
       hx, cx = hidden
       dim_h = self.hidden_size

       wxi = self.weight_ih[:dim_h]
       wxj = self.weight_ih[dim_h:2 * dim_h]
       wxf = self.weight_ih[2 * dim_h:3 * dim_h]
       wxo = self.weight_ih[3 * dim_h:]

     
       whi =self.weight_hh[:dim_h]
       whj = self.weight_hh[dim_h:2 * dim_h]
       whf=self.weight_hh[2 * dim_h:3 * dim_h]
       who=self.weight_hh[3 * dim_h:]

       bi = self.bias[:,:dim_h]
       bj = self.bias[:,dim_h:2 * dim_h]
       bf = torch.ones((self.batch_size,dim_h))
       bf=bf.cuda()

       bo= self.bias[:,3 * dim_h:]
       

       it = torch.tanh(x@wxi.t()+hx@whi.t()+bi)
       jt = torch.sigmoid(x@wxj.t()+hx@whj.t()+bj)
       ft = torch.sigmoid(x@wxf.t()+hx@whf.t()+bf)
       ot = torch.tanh(x@wxo.t()+hx@who.t()+bo)
 
       ct = cx* ft + it *jt
       ht = torch.tanh(ct)*ot
       return ct,ht
