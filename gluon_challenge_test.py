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

def readAndStoreFile(url):
    import os
    try:
        # For Python 3.0 and later
        from urllib.request import urlopen
    except ImportError:
        # Fall back to Python 2's urllib2
        from urllib2 import urlopen
    if not os.path.isdir("data"):
        response = urlopen(url)
        #  There might be problem decoding the response
        #  Please checkt he data folder and the file before trying the execrise
        #  in windows operating system  'data = response.read()' is enough
        data = response.read().decode('UTF-8')
        os.mkdir("data")
        with open("data/nietzsche.txt", "w+") as f:
            f.write(data)
            f.close()

url = "https://s3.amazonaws.com/text-datasets/nietzsche.txt"
readAndStoreFile(url)

# loading https://s3.amazonaws.com/text-datasets/nietzsche.txt
# You can load anyother text you want
# (https://cs.stanford.edu/people/karpathy/char-rnn/)
if py_new:
    with open("data/nietzsche.txt", errors='ignore') as f:
        text = f.read()
else:
    with open("data/nietzsche.txt") as f:
        text = f.read()

df = pd.read_pickle('./data_english.pickle')
df_moneymovement = df.loc[df['intent'] == 'i.fai.moneymovement']
utterances = df_moneymovement['utterance']

text = '\n'.join(utterances.values)

print(len(text))

# total of characters in dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)+1
print('total chars:', vocab_size)

# zeros for padding
chars.insert(0, "\0")

''.join(chars[1:-6])

# maps character to unique index e.g. {a:1,b:2....}
char_indices = dict((c, i) for i, c in enumerate(chars))
# maps indices to character (1:a,2:b ....)
indices_char = dict((i, c) for i, c in enumerate(chars))

# mapping the dataset into index
idx = [char_indices[c] for c in text]
print(len(idx))

# testing the mapping
''.join(indices_char[i] for i in idx[:70])

# input for neural network(our basic rnn has 3 inputs, n samples)
cs = 3
c1_dat = [idx[i] for i in range(0, len(idx)-1-cs, cs)]
c2_dat = [idx[i+1] for i in range(0, len(idx)-1-cs, cs)]
c3_dat = [idx[i+2] for i in range(0, len(idx)-1-cs, cs)]
# the output of rnn network (single vector)
c4_dat = [idx[i+3] for i in range(0, len(idx)-1-cs, cs)]

# stacking the inputs to form (3 input features )
x1 = np.stack(c1_dat[:-2])
x2 = np.stack(c2_dat[:-2])
x3 = np.stack(c3_dat[:-2])

# the output (1 X N data points)
y = np.stack(c4_dat[:-2])

col_concat = np.array([x1, x2, x3])
t_col_concat = col_concat.T
print(t_col_concat.shape)

# our sample inputs for the model
x1_nd = mx.nd.array(x1)
x2_nd = mx.nd.array(x2)
x3_nd = mx.nd.array(x3)
sample_input = mx.nd.array([[x1[0], x2[0], x3[0]], [x1[1], x2[1], x3[1]]])

simple_train_data = mx.nd.array(t_col_concat)
simple_label_data = mx.nd.array(y)

# Set the batchsize as 32, so input is of form 32 X 3
# output is 32 X 1
batch_size = 32


def get_batch(source, label_data, i, batch_size=32):
    bb_size = min(batch_size, source.shape[0] - 1 - i)
    data = source[i: i + bb_size]
    target = label_data[i: i + bb_size]
    # print(target.shape)
    return data, target.reshape((-1, ))


test_bat, test_target = get_batch(simple_train_data,
                                  simple_label_data, 5, batch_size)
print(test_bat.shape)
print(test_target.shape)

# simple UnRollredRNN_Model
from mxnet.gluon import Block, nn
from mxnet import ndarray as F


class UnRolledRNN_Model(Block):
    def __init__(self, vocab_size, num_embed, num_hidden, **kwargs):
        super(UnRolledRNN_Model, self).__init__(**kwargs)
        self.num_embed = num_embed
        self.vocab_size = vocab_size

        # use name_scope to give child Blocks appropriate names.
        # It also allows sharing Parameters between Blocks recursively.
        with self.name_scope():
            self.encoder = nn.Embedding(self.vocab_size, self.num_embed)
            self.dense1 = nn.Dense(num_hidden, activation='relu', flatten=True)
            self.dense2 = nn.Dense(num_hidden, activation='relu', flatten=True)
            self.dense3 = nn.Dense(vocab_size, flatten=True)

    def forward(self, inputs):
        emd = self.encoder(inputs)
        # print( emd.shape )
        # since the input is shape (batch_size, input(3 characters) )
        # we need to extract 0th, 1st, 2nd character from each batch
        character1 = emd[:, 0, :]
        character2 = emd[:, 1, :]
        character3 = emd[:, 2, :]
        # green arrow in diagram for character 1
        c1_hidden = self.dense1(character1)
        # green arrow in diagram for character 2
        c2_hidden = self.dense1(character2)
        # green arrow in diagram for character 3
        c3_hidden = self.dense1(character3)
        # yellow arrow in diagram
        c1_hidden_2 = self.dense2(c1_hidden)
        addition_result = F.add(c2_hidden, c1_hidden_2)  # Total c1 + c2
        addition_hidden = self.dense2(addition_result)  # the yellow arrow
        addition_result_2 = F.add(addition_hidden, c3_hidden)  # Total c1 + c2
        final_output = self.dense3(addition_result_2)
        return final_output


vocab_size = len(chars) + 1  # the vocabsize
num_embed = 30
num_hidden = 256
# model creation
simple_model = UnRolledRNN_Model(vocab_size, num_embed, num_hidden)
# model initilisation
simple_model.collect_params().initialize(mx.init.Xavier(), ctx=context)
trainer = gluon.Trainer(simple_model.collect_params(), 'adam')
loss = gluon.loss.SoftmaxCrossEntropyLoss()

# check point file
try:
    os.makedirs('checkpoints')
except:
    print("directory already exists")
filename_unrolled_rnn = "checkpoints/rnn_gluon_abc.params"


# the actual training
def UnRolledRNNtrain(train_data, label_data, batch_size=32, epochs=10):
    epochs = epochs
    smoothing_constant = .01
    for e in range(epochs):
        for ibatch, i in enumerate(range(0, train_data.shape[0] - 1, batch_size)):
            data, target = get_batch(train_data, label_data, i, batch_size)
            data = data.as_in_context(context)
            target = target.as_in_context(context)
            with autograd.record():
                output = simple_model(data)
                L = loss(output, target)
            L.backward()
            trainer.step(data.shape[0])

            ##########################
            #  Keep a moving average of the losses
            ##########################
            if ibatch == 128:
                curr_loss = mx.nd.mean(L).asscalar()
                moving_loss = 0
                moving_loss = (curr_loss if ((i == 0) and (e == 0))
                               else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)
                print("Epoch %s. Loss: %s, moving_loss %s" % (e, curr_loss, moving_loss))

# simple_model.save_params(filename_unrolled_rnn)
#
# epochs = 10
# UnRolledRNNtrain(simple_train_data, simple_label_data, batch_size, epochs)
# # loading the model back
# simple_model.load_params(filename_unrolled_rnn, ctx=context)

# evaluating the model
def evaluate(input_string):
    idx = [char_indices[c] for c in input_string]
    sample_input = mx.nd.array([[idx[0], idx[1], idx[2]]], ctx=context)
    output = simple_model(sample_input)
    index = mx.nd.argmax(output, axis=1)
    return index.asnumpy()[0]

# predictions
# begin_char = 'mone'
# answer = evaluate(begin_char)
# print('the predicted answer is ', indices_char[answer])

model = mx.gluon.rnn.SequentialRNNCell()
with model.name_scope():
    model.add(mx.gluon.rnn.LSTMCell(20))
    model.add(mx.gluon.rnn.LSTMCell(20))
states = model.begin_state(batch_size=32)
inputs = mx.nd.random.uniform(shape=(5, 32, 10))

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

# define the lstm
mode = 'lstm'
# number of characters in vocab_size
vocab_size = len(chars) + 1
embedsize = 50
hididen_units = 1000
number_layers = 2
clip = 0.2
epochs = 200  # use 200 epochs for good result
batch_size = 32
seq_length = 100  # sequence length
dropout = 0.4
log_interval = 64
# checkpoints/gluonlstm_2 (prepared for seq_lenght 100, 200 epochs)
rnn_save = 'checkpoints/gluonlstm_200.params'


# GluonRNNModel
model = GluonRNNModel(mode, vocab_size, embedsize, hididen_units,
                      number_layers, dropout)
# initalise the weights of models to random weights
model.collect_params().initialize(mx.init.Xavier(), ctx=context)
# Adam trainer
trainer = gluon.Trainer(model.collect_params(), 'adam')
# softmax cros entropy loss
loss = gluon.loss.SoftmaxCrossEntropyLoss()

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

idx_nd = mx.nd.array(idx)
# convert the idex of characters
train_data_rnn_gluon = rnn_batch(idx_nd, batch_size).as_in_context(context)

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

def trainGluonRNN(epochs, train_data, seq=seq_length):
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
        print(epoch)
        model.save_params(rnn_save)

print('the train data shape is', train_data_rnn_gluon.shape)

# The train data shape
trainGluonRNN(10, train_data_rnn_gluon, seq=seq_length)

model.load_params(rnn_save, context)


# evaluates a seqtoseq model over input string
def evaluate_seq2seq(model, input_string, seq_length, batch_size):
    idx = [char_indices[c] for c in input_string]
    if(len(input_string) != seq_length):
        raise ValueError("input string should be equal to sequence length")
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)
    sample_input = mx.nd.array(np.array([idx[0:seq_length]]).T, ctx=context)
    output, hidden = model(sample_input, hidden)
    index = mx.nd.argmax(output, axis=1)
    index = index.asnumpy()
    return [indices_char[char] for char in index]


# maps the input sequence to output sequence
def mapInput(input_str, output_str):
    for i, _ in enumerate(input_str):
        partial_input = input_str[:i+1]
        partial_output = output_str[i:i+1]
        print(partial_input + "->" + partial_output[0])

test_input = 'I would like to transf'
print(len(test_input))
result = evaluate_seq2seq(model, test_input, seq_length, 1)
mapInput(test_input, result)

# a nietzsche like text generator
import sys


def generate_random_text(model, input_string, seq_length, batch_size, sentence_length):
    count = 0
    new_string = ''
    cp_input_string = input_string
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)
    while count < sentence_length:
        idx = [char_indices[c] for c in input_string]
        if(len(input_string) != seq_length):
            print(len(input_string))
            raise ValueError('there was a error in the input ')
        sample_input = mx.nd.array(np.array([idx[0:seq_length]]).T, ctx=context)
        output, hidden = model(sample_input, hidden)
        index = mx.nd.argmax(output, axis=1)
        index = index.asnumpy()
        count = count + 1
        new_string = new_string + indices_char[index[-1]]
        input_string = input_string[1:] + indices_char[index[-1]]
    print(cp_input_string + new_string)

generate_random_text(model,
                        "probably the time is at hand when it will be once and again understood WHAT has actually sufficed an",
                        seq_length, 1, 200)