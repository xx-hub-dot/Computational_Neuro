#!/usr/bin/env python
# coding: utf-8

# In[92]:


import random

def init_weights():
    return [random.uniform(-0.5, 0.5) for _ in range(8)]

def init_biases():
    return [0.5, 0.7]

def tanh(x):
    return (2 / (1 + 2.71828**(-2 * x))) - 1

def forward_pass(inputs, weights, biases):
    neth1 = weights[0] * inputs[0] + weights[1] * inputs[1] + biases[0]
    out_h1 = tanh(neth1)
    neth2 = weights[2] * inputs[0] + weights[3] * inputs[1] + biases[0]
    out_h2 = tanh(neth2)
    
    neto1 = weights[4] * out_h1 + weights[5] * out_h2 + biases[1]
    out_o1 = tanh(neto1)
    neto2 = weights[6] * out_h1 + weights[7] * out_h2 + biases[1]
    out_o2 = tanh(neto2)
    
    return out_h1, out_h2, out_o1, out_o2

inputs = [0.05, 0.1]
targets = [0.1, 0.99]

weights = init_weights()
biases = init_biases()

out_h1, out_h2, out_o1, out_o2 = forward_pass(inputs, weights, biases)

print(f"(h1): {out_h1}")
print(f"(h2): {out_h2}")
print(f"(o1): {out_o1}")
print(f"(o2): {out_o2}")


# In[ ]:





# In[ ]:




