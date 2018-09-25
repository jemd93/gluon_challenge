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


# prepares rnn batches
# The batch will be of shape is (num_example * batch_size) because of RNN uses sequences of input     x
# for example if we use (a1,a2,a3) as one input sequence , (b1,b2,b3) as another input sequence and (c1,c2,c3)
# if we have batch of 3, then at timestep '1'  we only have (a1,b1.c1) as input, at timestep '2' we have (a2,b2,c2) as input...
# hence the batchsize is of order
# In feedforward we use (batch_size, num_example)
def rnn_batch(data, batch_size):
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data


# get the batch
def get_batch(source, i, seq):
    seq_len = min(seq, source.shape[0] - 1 - i)
    data = source[i: i + seq_len]
    target = source[i + 1: i + 1 + seq_len]
    return data, target.reshape((-1,))


# detach the hidden state, so we dont accidentally compute gradients
def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden


def trainGluonRNN(epochs, train_data, seq, clip, seq_length, batch_size):
    for epoch in range(epochs):
        total_L = 0.0
        hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)
        for ibatch, i in enumerate(range(0, train_data.shape[0] - 1, seq_length)):
            data, target = get_batch(train_data, i, seq)
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = loss(output, target) # this is total loss associated with seq_length
                L.backward()

            grads = [i.grad(context) for i in model.collect_params().values()]
            # Here gradient is for the whole batch.
            # So we multiply max_norm by batch_size and seq_length to balance it.
            gluon.utils.clip_global_norm(grads, clip * seq_length * batch_size)

            trainer.step(batch_size)
            total_L += mx.nd.sum(L).asscalar()

            if ibatch % log_interval == 0 and ibatch > 0:
                cur_L = total_L / seq_length / batch_size / log_interval
                print('[Epoch %d Batch %d] loss %.2f' % (epoch + 1, ibatch, cur_L))
                total_L = 0.0
            print("Epoch : {} ".format(epoch))
        model.save_params(os.path.join('./MODELS',args.model_name))


if __name__== "__main__":
    parser = argparse.ArgumentParser(description="Reads data and trains a model to generate random utterances"
                                                 "for a given intent",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_data', type=str,
                        help='Name of the file to read the data from')
    parser.add_argument('output_data', type=str,
                        help='Name of the file to save the data to')
    parser.add_argument('model_name', type=str,
                        help='Name of the model to save')
    parser.add_argument('intent', type=str,
                        help='Intent to train the model on')
    parser.add_argument('--mode', type=str, default='lstm',
                        help='Mode for the RNN to train on')
    parser.add_argument('--embed-size', type=int, default=50,
                        help='Size of embeddings')
    parser.add_argument('--hidden-layers', type=int, default=2,
                        help='Number of hidden layers')
    parser.add_argument('--hidden-units', type=int, default=1000,
                        help='Number of hidden units in hidden layer')
    parser.add_argument('--clip', type=float, default=0.2,
                        help='Clipping')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train the model for')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Size of each training batch')
    parser.add_argument('--seq-length', type=int, default=20,
                        help='Sequence length (size of window of learning)')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use')

    args = parser.parse_args()
    input_suffix = '.pickle'
    output_suffix = '.pickle'
    # Select data for the specified intent
    df = pd.read_pickle(os.path.join('./data', args.input_data + input_suffix))
    df_intent = df.loc[df['intent'] == args.intent]
    df_intent = df_intent.sample(n=200)
    utterances = df_intent['utterance']

    df_intent.to_pickle(os.path.join('./data', args.output_data + output_suffix))

    # Transform the dataframe into a "document" with 1 utterance per line
    text = '\n'.join(utterances.values)

    print("--------------------------------------")
    print("This is how the training data looks : ")
    print("--------------------------------------")
    print(text[0:300])
    print("--------------------------------------")
    print("Length of the input text : {}".format(len(text)))
    print("Number of utterances used : {}".format(len(df_intent.index)))

    # total of characters in dataset
    chars = sorted(list(set(text)))
    vocab_size = len(chars)+1
    print("Total # of characters in the text : {}".format(vocab_size))

    # zeros for padding
    chars.insert(0, "\0")

    ''.join(chars[1:-6])

    # maps character to unique index e.g. {a:1,b:2....}
    char_indices = dict((c, i) for i, c in enumerate(chars))
    # maps indices to character (1:a,2:b ....)
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # mapping the dataset into index
    idx = [char_indices[c] for c in text]
    print("--------------------------------------")
    print("This is how the data set looks after index mapping : ")
    print("--------------------------------------")
    print(idx[0:300])
    print("--------------------------------------")
    print("And this is how the mapping dictionary looks : ")
    print("--------------------------------------")
    print(char_indices)
    print("--------------------------------------")

    # testing the mapping
    ''.join(indices_char[i] for i in idx[:70])

    # number of characters in vocab_size
    vocab_size = len(chars) + 1
    log_interval = 64

    # GluonRNNModel
    model = GluonRNNModel(args.mode, vocab_size, args.embed_size, args.hidden_units,
                          args.hidden_layers, args.dropout)
    # Initialize weights randomly
    model.collect_params().initialize(mx.init.Xavier(), ctx=context)
    # Pick trainer/optimizer : Adam trainer
    trainer = gluon.Trainer(model.collect_params(), args.optimizer)
    # Pick loss function : softmax cros entropy loss
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    idx_nd = mx.nd.array(idx)
    # convert the idex of characters to be ingested by the RNN
    train_data_rnn_gluon = rnn_batch(idx_nd, args.batch_size).as_in_context(context)

    print("And this is the input for the RNN : ")
    print("--------------------------------------")
    print(train_data_rnn_gluon)
    print("--------------------------------------")
    print('The train data shape is : {}'.format(train_data_rnn_gluon.shape))

    # The train data shape
    trainGluonRNN(args.epochs, train_data_rnn_gluon, seq=args.seq_length, clip=args.clip,
                  seq_length=args.seq_length, batch_size=args.batch_size)
