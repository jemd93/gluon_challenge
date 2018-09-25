# import necessary dependencies
from __future__ import print_function
import math
import os
import time
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn
import pandas as pd
import mxnet.ndarray as F
import logging
import sys
import argparse

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64
# ssh -A ubuntu@10.30.29.216
py_new = True #  Checks for python version
if sys.version[:1] == '2':
    py_new = False

# Check context available and load the necessary context (cpu/gpu)
try:
    a = mx.nd.ones((2,3), mx.gpu(0))
    context = mx.gpu(0)
except:
    context = mx.cpu(0)

try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen


# Class to create model objects.
class GluonRNNModel(gluon.Block):

    def __init__(self, mode, vocab_size, num_embed, num_hidden,
                 num_layers, dropout=0.5, **kwargs):
        super(GluonRNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(vocab_size, num_embed,
                                        weight_initializer=mx.init.Uniform(0.1))

            if mode == 'lstm':
                #  we create a LSTM layer with certain number of hidden LSTM cell and layers
                #  in our example num_hidden is 1000 and num of layers is 2
                #  The input to the LSTM will only be passed during the forward pass (see forward function below)
                self.rnn = rnn.LSTM(num_hidden, num_layers, dropout=dropout,
                                    input_size=num_embed)
            elif mode == 'gru':
                #  we create a GRU layer with certain number of hidden GRU cell and layers
                #  in our example num_hidden is 1000 and num of layers is 2
                #  The input to the GRU will only be passed during the forward pass (see forward function below)
                self.rnn = rnn.GRU(num_hidden, num_layers, dropout=dropout,
                                   input_size=num_embed)
            else:
                #  we create a vanilla RNN layer with certain number of hidden vanilla RNN cell and layers
                #  in our example num_hidden is 1000 and num of layers is 2
                #  The input to the vanilla will only be passed during the forward pass (see forward function below)
                self.rnn = rnn.RNN(num_hidden, num_layers, activation='relu', dropout=dropout,
                                   input_size=num_embed)
            self.decoder = nn.Dense(vocab_size, in_units=num_hidden)
            self.num_hidden = num_hidden

#  define the forward pass of the neural network
    def forward(self, inputs, hidden):
        emb = self.encoder(inputs)
        #  emb, hidden are the inputs to the hidden
        output, hidden = self.rnn(emb, hidden)
        #  the ouput from the hidden layer to passed to drop out layer
        output = self.drop(output)
        #  print('output forward',output.shape)
        #  Then the output is flattened to a shape for the dense layer
        decoded = self.decoder(output.reshape((-1, self.num_hidden)))
        return decoded, hidden

    # Initial state of RNN layer
    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


def generate_random_text(model, input_string, seq_length, batch_size, sentence_length):
    count = 0
    new_string = ''
    cp_input_string = input_string
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)
    while count < sentence_length:
        idx = [char_indices[c] for c in input_string]
        if(len(input_string) != seq_length):
            raise ValueError('there was a error in the input ')
        sample_input = mx.nd.array(np.array([idx[0:seq_length]]).T, ctx=context)
        output, hidden = model(sample_input, hidden)
        index = mx.nd.argmax(output, axis=1)
        index = index.asnumpy()
        count = count + 1
        new_string = new_string + indices_char[index[-1]]
        input_string = input_string[1:] + indices_char[index[-1]]
    print(cp_input_string + new_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reads data and trains a model to generate random utterances"
                                                 "for a given intent",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_data', type=str,
                        help='Name of the file to read the data from (output from the last program')
    parser.add_argument('model_name', type=str,
                        help='Name of the model to load')
    parser.add_argument('input_string', type=str,
                        help='String to test the model with')
    parser.add_argument('--mode', type=str, default='lstm',
                        help='Mode for the RNN to train on')
    parser.add_argument('--embed-size', type=int, default=50,
                        help='Size of embeddings')
    parser.add_argument('--hidden-layers', type=int, default=2,
                        help='Number of hidden layers')
    parser.add_argument('--hidden-units', type=int, default=1000,
                        help='Number of hidden units in hidden layer')
    parser.add_argument('--seq-length', type=int, default=20,
                        help='Sequence length (size of window of learning)')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate')


    args = parser.parse_args()

    suffix = '.pickle'
    df_intent = pd.read_pickle(os.path.join('./data', args.input_data + suffix))
    utterances = df_intent['utterance']

    text = '\n'.join(utterances.values)
    print("--------------------------------------")
    print("This is how the training data looks : ")
    print("--------------------------------------")
    print(text[0:300])
    print("--------------------------------------")

    # total of characters in dataset
    chars = sorted(list(set(text)))
    vocab_size = len(chars)+1

    # zeros for padding
    chars.insert(0, "\0")

    ''.join(chars[1:-6])

    # maps character to unique index e.g. {a:1,b:2....}
    char_indices = dict((c, i) for i, c in enumerate(chars))
    # maps indices to character (1:a,2:b ....)
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # mapping the dataset into index
    idx = [char_indices[c] for c in text]

    # testing the mapping
    ''.join(indices_char[i] for i in idx[:70])

    # define the lstm
    mode = 'lstm'
    # number of characters in vocab_size
    vocab_size = len(chars) + 1
    embedsize = 50
    hidden_units = 1000
    number_layers = 2
    dropout = 0.4

    # GluonRNNModel
    model = GluonRNNModel(mode, vocab_size, embedsize, hidden_units, number_layers, dropout)

    model.load_params(os.path.join('./MODELS', args.model_name), context)

    test_input = args.input_string
    seq_length = len(test_input)

    print("User input : " + test_input)
    print("Intent : " + df_intent['intent'].unique()[0])
    print("--------------------------------------")
    print("Model output with 1000 characters and seq_length of " + str(seq_length) + " : ")
    print("--------------------------------------")
    generate_random_text(model, test_input, seq_length, 1, 1000)
    print("--------------------------------------")