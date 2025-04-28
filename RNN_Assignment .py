#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random

# Vocabulary
vocab = ['i', 'like', 'deep', 'learning']
vocab_size = len(vocab)

# Word to index
word_to_index = {word: idx for idx, word in enumerate(vocab)}
index_to_word = {idx: word for word, idx in word_to_index.items()}

# One-hot encoding
def one_hot(word):
    vec = np.zeros((vocab_size, 1))
    vec[word_to_index[word]] = 1
    return vec

# Initialize weights
hidden_size = 5  # Number of hidden units
Wx = np.random.uniform(-1, 1, (hidden_size, vocab_size))
Wh = np.random.uniform(-1, 1, (hidden_size, hidden_size))
Wy = np.random.uniform(-1, 1, (vocab_size, hidden_size))

# Inputs
inputs = ['i', 'like', 'deep']  # Predict 'learning'

# Initialize hidden state
h_prev = np.zeros((hidden_size, 1))

# Forward pass through time
for word in inputs:
    x = one_hot(word)
    a = np.dot(Wx, x) + np.dot(Wh, h_prev)
    h = np.tanh(a)
    h_prev = h  # Update hidden state

# Output layer
y = np.dot(Wy, h_prev)
y_pred = np.exp(y) / np.sum(np.exp(y))  # softmax

# Prediction
predicted_index = np.argmax(y_pred)
predicted_word = index_to_word[predicted_index]

print("Predicted next word:", predicted_word)


# In[ ]:




