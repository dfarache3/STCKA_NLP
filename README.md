# Deep Short Text Classification with Knowledge Powered Attention
The code is an improvement and testing of the STCKA model given by the original authors of the paper. In this the model is improved and tested upon by changing the activation function and the neural network architecture in the C-ST and C-CS attention mechanisms. 

#### For the purpose of reproducing this paper, we implemented this code.

## Requirements
* Python==3.7.4
* pytorch==1.3.1
* torchtext==0.3.1
* numpy
* tqdm

## Input data format
Snippets and TagMyNews Dataset can be available in dataset folder. The data format is as follows('\t' means TAB):

```
origin text \t concepts
...
```
## File Assingment
Copied files:
- model/init.py
- utils/init.py
- utils/config.py
- utils/dataset.py
- utils/metrics.py

Modified files:
- main.py: changed to take input of selection of model
- model/STCKA.py: adjusted for understanding purposes and errors that were run into

Added/Created files
- STCKA_relu.py: Model with activation function changed to relu
- STCKA_sigmoid.py: Model with activation function changed to sigmoid
- STCKA_leakyRelu.py: Model with activation function changed to leakyRelu
- STCKA_DeepLinear.py: Model with activation function changed to a linear deeper network
- STCKA_Convo.py: Model with activation function changed to convolution single layer architecture 
- STCKA_DeepConvo.py: Model with activation function changed to convolution deeper network


## How to run
Train & Dev & Test:
Original dataset is randomly split into 80% for training and 20% for test. 20% of randomly selected training instances are used to form development set.

```console
python main.py --epoch 20 --lr 2e-4 --train_data_path dataset/tagmynews.tsv --txt_embedding_path dataset/glove.6B.300d.txt --cpt_embedding_path dataset/glove.6B.300d.txt  --embedding_dim 300 --train_batch_size 128 --hidden_size 64
```

--epoch: Number of epochs
--lr: Learning rate
--train_data_path: Training dataset file path
--txt_embedding_path: The file path to access the pretrained word vectors for input short text
--cpt_embedding_path: The file path to access the pretrained word vectors for concept set
--embedding_dim: Embedding dimension
--train_batch_size: Batch_size
--hidden_size: Hidden dimension

More detailed configurations can be found in `config.py`, which is in utils folder.

The program will print:
Choose model (tanh, relu, sigmoid, leakyRelu, deepLinear, conv, convDeep) to run:
For the input you will type any of these inputs
tanh, relu, sigmoid, leakyRelu, deepLinear, conv, convDeep

For refrence tanh is both for the activation function model an single linear
## Cite
```
Author paper: https://aaai.org/ojs/index.php/AAAI/article/view/4585/4463 
Chen, J., Hu, Y., Liu, J., Xiao, Y., and Jiang, H. Deep
short text classification with knowledge powered atten-
tion. In Proceedings of the AAAI Conference on Artificial
Intelligence, volume 33, pp. 6252–6259, 2019.

Author GitHub Repository: https://github.com/AIRobotZhang/STCKA 
Chen J, Hu Y, L. J. e. a. Deep short text classification with
knowledge powered attention. 2019

Standford Vector File GloVe: https://www.kaggle.com/thanakomsn/glove6b300dtxt
```
