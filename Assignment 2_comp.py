#!/usr/bin/env python
# coding: utf-8

# In[16]:


def sigmoid(x):
    return 1 / (1 + 2.71828**(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self):
        self.learning_rate = 0.5
        
    def forward_pass(self, inputs, weights_hidden, weights_output):
        # Hidden layer calculations
        net_h1 = inputs[0] * weights_hidden[0] + inputs[1] * weights_hidden[1]
        net_h2 = inputs[0] * weights_hidden[2] + inputs[1] * weights_hidden[3]
        out_h1 = sigmoid(net_h1)
        out_h2 = sigmoid(net_h2)
        
        # Output layer calculations
        net_o1 = out_h1 * weights_output[0] + out_h2 * weights_output[1]
        net_o2 = out_h1 * weights_output[2] + out_h2 * weights_output[3]
        out_o1 = sigmoid(net_o1)
        out_o2 = sigmoid(net_o2)
        
        return (out_h1, out_h2), (out_o1, out_o2), (net_h1, net_h2), (net_o1, net_o2)

    def update_weights(self, inputs, weights_hidden, weights_output, targets):
        # Forward pass
        hidden_outputs, final_outputs, net_hidden, net_output = self.forward_pass(inputs, weights_hidden, weights_output)
        out_h1, out_h2 = hidden_outputs
        out_o1, out_o2 = final_outputs
        
        # Output layer deltas
        delta_o1 = (out_o1 - targets[0]) * out_o1 * (1 - out_o1)
        delta_o2 = (out_o2 - targets[1]) * out_o2 * (1 - out_o2)
        
        # Hidden layer deltas
        delta_h1 = out_h1 * (1 - out_h1) * (weights_output[0] * delta_o1 + weights_output[2] * delta_o2)
        delta_h2 = out_h2 * (1 - out_h2) * (weights_output[1] * delta_o1 + weights_output[3] * delta_o2)
        
        # Update output weights
        w5 = weights_output[0] - self.learning_rate * delta_o1 * out_h1
        w6 = weights_output[1] - self.learning_rate * delta_o1 * out_h2
        w7 = weights_output[2] - self.learning_rate * delta_o2 * out_h1
        w8 = weights_output[3] - self.learning_rate * delta_o2 * out_h2
        
        # Update hidden weights
        w1 = weights_hidden[0] - self.learning_rate * delta_h1 * inputs[0]
        w2 = weights_hidden[1] - self.learning_rate * delta_h1 * inputs[1]
        w3 = weights_hidden[2] - self.learning_rate * delta_h2 * inputs[0]
        w4 = weights_hidden[3] - self.learning_rate * delta_h2 * inputs[1]
        
        return [w1, w2, w3, w4], [w5, w6, w7, w8]

def train():
    nn = NeuralNetwork()
    inputs = [0.05, 0.1]
    weights_hidden = [0.15, 0.2, 0.25, 0.3]
    weights_output = [0.4, 0.45, 0.5, 0.55]
    targets = [0.01, 0.99]
    
    weights_hidden_new, weights_output_new = nn.update_weights(inputs, weights_hidden, weights_output, targets)
    return weights_hidden_new, weights_output_new

# Run training
final_hidden_weights, final_output_weights = train()
print("Hidden Weights (w1, w2, w3, w4):", [round(w, 4) for w in final_hidden_weights])
print("Output Weights (w5, w6, w7, w8):", [round(w, 4) for w in final_output_weights])


# In[ ]:




