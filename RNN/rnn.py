"""
Implementation of vanilla recurrent neural network. Inspired by karpathy.
"""

import numpy as np


# data I/O
data = open('./shaks.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print (f'data has {data_size} characters, and {vocab_size} unique.')

char_to_ix = {ch:i for i, ch in enumerate(chars)}
ix_to_char = {i:ch for i, ch in enumerate(chars)}

# hyperparameters
hidden_size = 100
seq_length = 25  # number of steps to unroll RNN for
learning_rate = 1e-1

# model parameters
W_xh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
W_hh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
W_hy = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output

bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1))  # otuput bias

# loss function

def LossFun(inputs, targets, hprev):
    """ 
    inputs, targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    
    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1)) # encode 1-of-k representation
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(W_xh, xs[t]) + np.dot(W_hh, hs[t-1]) + bh) # hidden state
        ys[t] = np.dot(W_hy, hs[t]) + by  # unnormalized log probabilities for next chars
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probabilties for next chars
        loss += -np.log(ps[t][targets[t], 0]) # softmax (cross-entropy loss)
        
    # backward pass

    dW_xh, dW_hh, dW_hy = np.zeros_like(W_xh), np.zeros_like(W_hh), np.zeros_like(W_hy)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dh_next = np.zeros_like(hs[0])
    
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1 # backprop into y
        dW_hy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(W_hy.T, dy) + dh_next  # backprop into h
        dh_raw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
        dbh += dh_raw
        dW_xh += np.dot(dh_raw, xs[t].T)
        dW_hh += np.dot(dh_raw, hs[t-1].T)
        dh_next = np.dot(W_hh.T, dh_raw)
    for dparam in [dW_xh, dW_hh, dW_hy, dbh, dby]:
        np.clip(dparam, -5, -5, out=dparam) # clip to mitigate exploding gradients
    return loss, dW_xh, dW_hh, dW_hy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for the first time step
    """
    
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        h = np.tanh(np.dot(W_xh, x) + np.dot(W_hh, h) + bh)
        y = np.dot(W_hy, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes

n,p = 0,0
mWxh, mWhh, mWhy = np.zeros_like(W_xh), np.zeros_like(W_hh), np.zeros_like(W_hy)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # memory variables for adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

while True:
    # prepare inputs ( from left to right in steps seq_length long)
    if p+seq_length+1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size, 1)) # reset RNN memory
        p = 0
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
    
    # sample from the model now and then
    if n%100 == 0:
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print(f'{txt, }\r\n')
        
    # forward seq_length characters through the next and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = LossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 100 == 0: print(f"iter: {n}, loss: {smooth_loss}") # print progress
    
    
    # perform parameter update with adagrad
    for param, dparam, mem in zip([W_xh, W_hh, W_hy, bh, by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param = learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
        
    p += seq_length
    n += 1