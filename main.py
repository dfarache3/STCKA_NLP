# -*- coding: utf-8 -*-

import time
import torch
import torch.nn.functional as F
import numpy as np
import logging
import argparse
from model import STCK_Atten
from model import STCKA_relu
from model import STCKA_sigmoid
from model import STCKA_leakyrelu
from model import STCKA_DeepLinear
from model import STCKA_Convo
from model import STCKA_DeepConvo

from utils import dataset, metrics, config
import copy
from tqdm import tqdm

start_time = time.time()

#logger: log message structure that store logRecord Files
#set structure for logger files
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

#create class for logger
logger = logging.getLogger(__name__)

#
def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def train_model(model, train_iter, dev_iter, epoch, lr, loss_func):
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    all_loss = 0.0
    model.train()
    ind = 0.0
    for idx, batch in enumerate(train_iter):
        txt_text = batch.text[0]
        cpt_text = batch.concept[0]
        # batch_size = text.size()[0]
        target = batch.label
        
        if torch.cuda.is_available():
            txt_text = txt_text.cuda()
            cpt_text = cpt_text.cuda()
            target = target.cuda()
            
        optim.zero_grad()
        # pred: batch_size, output_size
        logit = model(txt_text, cpt_text)
        
        loss = loss_func(logit, target)

        loss.backward()
        # clip_gradient(model, 1e-1)
        optim.step()

        if idx % 10 == 0:
            logger.info('Epoch:%d, Idx:%d, Training Loss:%.4f', epoch, idx, loss.item())
            # dev_iter_ = copy.deepcopy(dev_iter)
            # p, r, f1, eval_loss = eval_model(model, dev_iter, id_label)
        all_loss += loss.item()
        ind += 1

    eval_loss, acc, p, r, f1 = 0.0, 0.0, 0.0, 0.0, 0.0
    eval_loss, acc, p, r, f1 = eval_model(model, dev_iter, loss_func)
    # return all_loss/ind
    return all_loss/ind, eval_loss, acc, p, r, f1

def eval_model(model, val_iter, loss_func):
    eval_loss = 0.0
    ind = 0.0
    score = 0.0
    pred_label = None
    target_label = None
    # flag = True
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_iter)):
            txt_text = batch.text[0]
            cpt_text = batch.concept[0]
            # batch_size = text.size()[0]
            target = batch.label
            
            if torch.cuda.is_available():
                txt_text = txt_text.cuda()
                cpt_text = cpt_text.cuda()
                target = target.cuda()
            logit = model(txt_text, cpt_text)
            
            loss = loss_func(logit, target)
            eval_loss += loss.item()
            if ind > 0:
                pred_label = torch.cat((pred_label, logit), 0)
                target_label = torch.cat((target_label, target))
            else:
                pred_label = logit
                target_label = target

            ind += 1

    acc, p, r, f1 = metrics.assess(pred_label, target_label)
    return eval_loss/ind, acc, p, r, f1


def main():
    #set config file
    args = config.config()
    
    #print(args)
    #(cpt_embedding_path = dataset/glove.6B.300d.txt, dev_batch_size = 64, dev_data_path = None, early_stopping = 15, embedding_dim = 300, epoch = 1, fine_tunning = True, 
    # hidden_size = 64, load_model = None, lr = 2e-4, output_size = 8, test_batch_size = 64, test_data_path = None, test_data_path = None, train_batch_size = 128,
    # train_data_path = 'dataset/snippets.tsv)

    #get path to dataset if not given
    if not args.train_data_path:
        logger.info("please input train dataset path")
        exit()
    # if not (args.dev_data_path or args.test_data_path):
    #     logger.info("please input dev or test dataset path")
    #     exit()
    
    #batch data
    all_ = dataset.load_dataset(args.train_data_path, args.dev_data_path, args.test_data_path, \
                      args.txt_embedding_path, args.cpt_embedding_path, args.train_batch_size, \
                                                          args.dev_batch_size, args.test_batch_size)
    

    #transfer dataset processing into sections (input embedding)
    txt_TEXT, cpt_TEXT, txt_vocab_size, cpt_vocab_size, txt_word_embeddings, cpt_word_embeddings, \
            train_iter, dev_iter, test_iter, label_size = all_

    #print(txt_TEXT)
    #print(txt_word_embeddings)
    #print(train_iter)
    
    #initialize model
    selection = input("Choose model (tanh, relu, sigmoid, leakyRelu, deepLinear, conv, convDeep) to run: ")
    actFunc = selection

    #actFunc = 'deepLinear'
    if(actFunc=='tanh'):
        model = STCK_Atten(txt_vocab_size, cpt_vocab_size, args.embedding_dim, txt_word_embeddings,\
                             cpt_word_embeddings, args.hidden_size, label_size)
    elif(actFunc=='relu'):
        model = STCKA_relu.STCK_Atten_relu(txt_vocab_size, cpt_vocab_size, args.embedding_dim, txt_word_embeddings,\
                             cpt_word_embeddings, args.hidden_size, label_size)
    elif(actFunc=='sigmoid'):
        model = STCKA_sigmoid.STCK_Atten_sigmoid(txt_vocab_size, cpt_vocab_size, args.embedding_dim, txt_word_embeddings,\
                             cpt_word_embeddings, args.hidden_size, label_size)
    elif(actFunc=='leakyRelu'):
        model = STCKA_leakyrelu.STCK_Atten_leakyrelu(txt_vocab_size, cpt_vocab_size, args.embedding_dim, txt_word_embeddings,\
                             cpt_word_embeddings, args.hidden_size, label_size)
    elif(actFunc=='deepLinear'):
        model = STCKA_DeepLinear.STCK_Atten_DeepLinear(txt_vocab_size, cpt_vocab_size, args.embedding_dim, txt_word_embeddings,\
                             cpt_word_embeddings, args.hidden_size, label_size)
    elif(actFunc=='conv'):
        model = STCKA_Convo.STCK_Atten_Convo(txt_vocab_size, cpt_vocab_size, args.embedding_dim, txt_word_embeddings,\
                             cpt_word_embeddings, args.hidden_size, label_size)
    elif(actFunc=='convDeep'):
        model = STCKA_DeepConvo.STCK_Atten_DeepConvo(txt_vocab_size, cpt_vocab_size, args.embedding_dim, txt_word_embeddings,\
                         cpt_word_embeddings, args.hidden_size, label_size)

    if torch.cuda.is_available():
         model = model.cuda()
    
    #split dataset
    train_data, test_data = dataset.train_test_split(train_iter, 0.8)
    train_data, dev_data = dataset.train_dev_split(train_data, 0.8)
    loss_func = torch.nn.CrossEntropyLoss()

    #load model if given
    if args.load_model:
         model.load_state_dict(torch.load(args.load_model))
         test_loss, acc, p, r, f1 = eval_model(model, test_data, loss_func)
         logger.info('Test Loss:%.4f, Test Acc:%.4f, Test P:%.4f, Test R:%.4f, Test F1:%.4f', test_loss, acc, p, r, f1)
         print('loaded model')
         return
    
    #train/eval model
    best_score = 0.0
    test_loss, test_acc, test_p, test_r, test_f1 = 0, 0, 0, 0, 0

    for epoch in range(args.epoch):
         train_loss, eval_loss, acc, p, r, f1 = train_model(model, train_data, dev_data, epoch, args.lr, loss_func) #train
        
         logger.info('Epoch:%d, Training Loss:%.4f', epoch, train_loss)
         logger.info('Epoch:%d, Eval Loss:%.4f, Eval Acc:%.4f, Eval P:%.4f, Eval R:%.4f, Eval F1:%.4f', epoch, eval_loss, acc, p, r, f1)
        
         if f1 > best_score:
             best_score = f1
             torch.save(model.state_dict(), 'results/%d_%s_%s.pt' % (epoch, 'Model', str(best_score))) #save current best model
             test_loss, test_acc, test_p, test_r, test_f1 = eval_model(model, test_data, loss_func) #eval
         logger.info('Test Loss:%.4f, Test Acc:%.4f, Test P:%.4f, Test R:%.4f, Test F1:%.4f, Time:%.4f', test_loss, test_acc, test_p, test_r, test_f1, (time.time() - start_time))

if __name__ == "__main__":
    main()