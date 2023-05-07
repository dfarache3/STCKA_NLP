# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math

class STCK_Atten_DeepLinear(nn.Module):
    def __init__(self, text_vocab_size, concept_vocab_size, embedding_dim, txt_embedding_weights,\
                        cpt_embedding_weights, hidden_size, output_size, gama=0.5, num_layer=1, finetuning=True):
        super(STCK_Atten_DeepLinear, self).__init__()

        self.gama = gama
        print(gama)
        da = hidden_size
        print(hidden_size)
        db = int(da/2)
        print(db)
     
        self.txt_word_embed = nn.Embedding(text_vocab_size, embedding_dim)
        print(self.txt_word_embed)

        if isinstance(txt_embedding_weights, torch.Tensor):
            self.txt_word_embed.weight = nn.Parameter(txt_embedding_weights, requires_grad=finetuning)
        
        print(self.txt_word_embed)
        self.cpt_word_embed = nn.Embedding(concept_vocab_size, embedding_dim)
        
        print(self.cpt_word_embed)

        if isinstance(cpt_embedding_weights, torch.Tensor):
            self.cpt_word_embed.weight = nn.Parameter(cpt_embedding_weights, requires_grad=finetuning)
        
        print(self.cpt_word_embed)

        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layer, batch_first=True, bidirectional=True)
        print(self.lstm)
        
        self.W1 = nn.Linear(2*hidden_size+embedding_dim, da*5)
        print(da*5)
        self.W1_2 = nn.Linear(da*5, da*4)
        self.W1_3 = nn.Linear(da*4, da*3)
        self.W1_4 = nn.Linear(da*3, da*2)
        self.W1_5 = nn.Linear(da*2, da)
        self.w1 = nn.Linear(da, 1, bias=False)

        self.W2 = nn.Linear(embedding_dim, db*5)
        self.W2_2 = nn.Linear(db*5, db*4)
        self.W2_3 = nn.Linear(db*4, db*3)
        self.W2_4 = nn.Linear(db*3, db*2)
        self.W2_5 = nn.Linear(db*2, db)
        self.w2 = nn.Linear(db, 1, bias=False)

        self.output = nn.Linear(2*hidden_size+embedding_dim, output_size)
        print(self.output)
            

    def self_attention(self, txt_wordid):
        # H: batch_size, seq_len, 2*hidden_size
        input_txt = self.txt_word_embed(txt_wordid) # input_: batch_size, seq_len, emb_dim
        H, (hn, cn) = self.lstm(input_txt) # output: batch_size, seq_len, 2*hidden_size
        # H: batch_size, seq_len, 2*hidden_size
        hidden_size = H.size()[-1]
        Q = K = V = H
        # batch_size, seq_len, seq_len
        atten_weight = F.softmax(torch.bmm(Q, K.permute(0, 2, 1))/math.sqrt(hidden_size), -1)
        A = torch.bmm(atten_weight, V) # batch_size, seq_len, 2*hidden_size
        A = A.permute(0, 2, 1)
        # q: short text representation
        q = F.max_pool1d(A, A.size()[2]).squeeze(-1) # batch_size, 2*hidden_size
        return q

    def cst_attention(self, c, q):
        # c: batch_size, concept_seq_len, embedding_dim
        # q: batch_size, 2*hidden_size
        # print(q.size())
        # print(c.size())
        q = q.unsqueeze(1)
        q = q.expand(q.size(0), c.size(1), q.size(2))
        c_q = torch.cat((c, q), -1) # batch_size, concept_seq_len, embedding_dim+2*hidden_size
        firstLayer = F.tanh(self.W1(c_q))
        secLayer = F.tanh(self.W1_2(firstLayer))
        thirdLayer = F.tanh(self.W1_3(secLayer))
        fourthLayer = F.tanh(self.W1_4(thirdLayer))
        fifthLayer = F.tanh(self.W1_5(fourthLayer))

        c_q = self.w1(fifthLayer) # batch_size, concept_seq_len, 1
        alpha = F.softmax(c_q.squeeze(-1), -1) # batch_size, concept_seq_len

        return alpha

    def ccs_attention(self, c):
        # c: batch_size, concept_seq_len, embedding_dim
        firstLayer = F.tanh(self.W2(c))
        secLayer = F.tanh(self.W2_2(firstLayer))
        thirdLayer = F.tanh(self.W2_3(secLayer))
        fourthLayer = F.tanh(self.W2_4(thirdLayer))
        fifthLayer = F.tanh(self.W2_5(fourthLayer))
        c = self.w2(fifthLayer) # batch_size, concept_seq_len, 1
        beta = F.softmax(c.squeeze(-1), -1) # batch_size, concept_seq_len
        return beta

    def forward(self, txt_wordid, cpt_wordid):
        # txt_wordid: batch_size, text_seq_len
        # cpt_wordid: batch_size, concept_seq_len

        q = self.self_attention(txt_wordid) # text representation

        input_cpt = self.cpt_word_embed(cpt_wordid) # input_: batch_size, concept_seq_len, emb_dim
        alpha = self.cst_attention(input_cpt, q) # batch_size, concept_seq_len
        beta = self.ccs_attention(input_cpt) # batch_size, concept_seq_len

        cpt_atten = F.softmax(self.gama*alpha+(1-self.gama)*beta, -1) # batch_size, concept_seq_len
        p = torch.bmm(cpt_atten.unsqueeze(1), input_cpt).squeeze(1) # batch_size, emb_dim

        hidden_rep = torch.cat((q, p), -1) # batch_size, 2*hidden_size+emb_dim

        logit = self.output(hidden_rep)

        return logit
