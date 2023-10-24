#ICNLPBrain.py

import os
import re
import json
import math
import random
import numpy as np
import csv
import requests
from ICMLCore import ICMLCoreclass

class ICNLPBrain:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.word_to_index = {}
        self.index_to_word = {}
        self.embeddings = None  
        self.icmlcore_instance = ICMLCoreclass(input_size=10, hidden_sizes=[5], output_size=1)
        print("ICNLPBrain initialized with vocab size:", self.vocab_size)
        print("ML model initialized.")
        
    #Function to clean text
    def preprocess_text(self, text):
       
       #Lowercase the text
        text = text.lower()
        
        #Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text

    #Function for data tokenization
    def tokenize(self, text):
        tokens = text.split()
        return tokens

    #Function for vocabulary building
    def build_vocab(self, tokenized_texts):
        vocab = {}
        for tokens in tokenized_texts:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
        return vocab

    #Function to convert tokenized text to integer
    def text_to_integers(self, tokenized_text, vocab):
        return [vocab[token] for token in tokenized_text]

    #Function to convert integers back into text
    def integers_to_text(self, integers, vocab):
        reverse_vocab = {v: k for k, v in vocab.items()}
        return [reverse_vocab[i] for i in integers]
    
    #Function for forward propagation
    def forward_propagation(self, input_data):
        try:
            target, context_indices = input_data
            softmax_output = self.skipgram_forward_prop(target, context_indices)
            
            print(f"Shape of softmax_output: {softmax_output.shape}")

            if softmax_output is not None:
                #ML function call
                ml_output = self.icmlcore_instance.forward_propagation(softmax_output)
                print(f"Forward propagation successful. ML Output: {ml_output}")
                return softmax_output, ml_output
            else:
                print("Forward propagation failed.")
                return None
        except Exception as e:
            print(f"Exception in forward_propagation: {e}")
            return None
    
    #Function to initialize weights
    def initialize_weights(self, embed_size):
        #Initialize the weights for the neural network
        self.W1 = np.random.randn(self.vocab_size, embed_size) * np.sqrt(1. / self.vocab_size)
        self.W2 = np.random.randn(embed_size, self.vocab_size) * np.sqrt(1. / self.vocab_size)
        self.embeddings = np.random.randn(self.vocab_size, embed_size) * np.sqrt(1. / self.vocab_size)
        self.context_embeddings = np.random.randn(self.vocab_size, embed_size) * np.sqrt(1. / self.vocab_size)
        print("Weights initialized.")
    
    #Function for skip-gram model
    def skipgram_forward_prop(self, target, context_indices):
        pass
    
    #Function for backward propagation in skip-gram
    def skipgram_backward_prop(self, target, context, softmax_output):
        try:    
            #Initialize gradients to 0
            dW1 = np.zeros_like(self.W1)
            dW2 = np.zeros_like(self.W2)
                    
            #Calculate the loss gradient w.r.t softmaxoutput
            grad_softmax = softmax_output
            grad_softmax[context] -= 1
        
            #Calculate gradients for W2 (context embeddings)
            dW2 = np.outer(self.embeddings[target], grad_softmax)
        
            #Calculate gradients for W1 (target embeddings)
            dW1[target, :] = np.dot(self.W2, grad_softmax.T)
        
            print(f"Gradient dW1: {dW1}")
            print(f"Gradient dW2: {dW2}")
        
            return dW1, dW2
        except Exception as e:
            print(f"Backward propagation failed: {e}")
            return None, None
    
    #Function for for updating weights
    def update_weights(self, dW1, dW2, learning_rate):
        if dW1 is not None and dW2 is not None:
            self.W1 -= learning_rate * dW1
            self.W2 -= learning_rate * dW2
            print("Weights Updated")
        else:
            print("Gradients are None. Weights not updated.")
                
    #Function for training the skip-gram
    def train_skipgram(self, tokenized_texts, epochs, learning_rate):
        self.initialize_weights(embed_size=300)
        
        #Training loop for skip-gram model
        print("Training Skip-gram model...")
        
        for epoch in range(epochs):
            total_loss = 0
            for tokenized_text in tokenized_texts:
                

def main():
    print("Welcome to ICNLPBrain")
    nlp_brain = ICNLPBrain(vocab_size = 5000)
    
    #Initialize weights
    nlp_brain.initialize_weights(embed_size=100)
    
    #Placeholder text, replace with actual data
    texts = ["Hello, world!", "ICNLPBrain is awesome.", "Natural language processing is cool!"]
    
    #Preprocess and tokenize
    processed_texts = [nlp_brain.preprocess_text(text) for text in texts]
    tokenized_texts = [nlp_brain.tokenize(text) for text in processed_texts]
    
    #Build Vocabulary
    vocab = nlp_brain.build_vocab(tokenized_texts)    
    
    #Convert text to integer
    encoded_texts = [nlp_brain.text_to_integers(tokenized_text, vocab) for tokenized_text in tokenized_texts]    
    
    print("Vocabulary:", vocab)
    print("Encoded Texts:", encoded_texts)
    
    #Test forward and backward propagation
    target_word_index = 2
    context_word_indices = [3, 4]
      
    #Forward Propagation
    forward_output, ml_output = nlp_brain.forward_propagation((target_word_index, context_word_indices))
    print("Forward propagation output: {forward_output}, ML Output: {ml_output}")
    
    #Backward Propagation
    context_word_index = 3
    dW1, dW2 = nlp_brain.skipgram_backward_prop(target_word_index, context_word_index, forward_output)
    print (f"Backward propagation gradients: dW1: {dW1}, dW2: {dW2}")
    
    #Update Weights
    learning_rate = 0.01
    nlp_brain.update_weights(dW1, dW2, learning_rate)
    print("Weights updated.")
        
if __name__ == "__main__":
    main()
