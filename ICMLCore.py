#ICMLCore.py

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class ICMLCoreclass:
    def __init__(self, input_size, hidden_sizes, output_size):
        #Initialize weights and biases
        self.weights = []
        self.biases = []
        
        #Input layer for first hidden layer
        self.weights.append(np.random.randn(5000, hidden_sizes[0]))
        self.biases.append(np.zeros((1, hidden_sizes[0])))
        
        #Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.weights.append(np.random.randn(hidden_sizes[i], hidden_sizes[i+1]))
            self.biases.append(np.zeros((1, hidden_sizes[i+1])))
            
        #Last hidden layer to output layer
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size))
        self.biases.append(np.zeros((1, output_size)))
          
    def forward_propagation(self, input_data, nlp_output=None):
        self.activations = []
        self.activations.append(input_data)
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            a = sigmoid(z)
            self.activations.append(a)
            
        if nlp_output is not None:
            print(f"Received NLP Output: {nlp_output}")
            
        return self.activations[-1]
    
    def cost_function(self, y_true, y_pred):
        """
        Compute the Binary Cross-Entropy Loss
        """
        epsilon = 1e-15 #to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -1 * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.mean(loss)
   
    def backpropagation(self):  
        #Backpropagation code here

        pass
    
    def train(self):
        #Training loop code here

        pass
    
    def predict(self):
        #Prediction code here

        pass
        
if __name__ == "__main__":
    #Initialize the ICMLCoreclass class with 3 layers: input, one hidden, and output
    model = ICMLCoreclass(input_size=10, hidden_sizes=[5], output_size=1)
    
    #Test initialization
    print("Weights:", model.weights)
    print("Biases:", model.biases)
    
    #Test forward propagation with random input
    input_data = np.random.randn(1, 10)
    output = model.forward_propagation(input_data)
    print("Output after propagation:", output)
    
    y_true = np.array([[1]])
    y_pred = model.forward_propagation(input_data)
    loss = model.cost_function(y_true, y_pred)
    print(f"Loss: {loss}")
