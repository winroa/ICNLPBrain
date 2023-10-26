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
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]))
        self.biases.append(np.zeros((1, hidden_sizes[0])))
        
        #Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.weights.append(np.random.randn(hidden_sizes[i], hidden_sizes[i+1]))
            self.biases.append(np.zeros((1, hidden_sizes[i+1])))
            
        #Last hidden layer to output layer
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size))
        self.biases.append(np.zeros((1, output_size)))
          
    def forward_propagation(self, input_data, nlp_output=None):
        print(f"Received input_data in forward_propagation: {input_data}")
        
        input_data = np.reshape(input_data, (1, -1))

        self.activations = []
        self.activations.append(input_data)
        
        #Pre-processing layer: Expanding from 10 to 5000 dimensions
        pre_process_weights = np.random.randn(10, 5000)
        pre_process_biases = np.zeros((1, 5000))
        pre_processed_input = np.dot(input_data, pre_process_weights) + pre_process_biases
        
        self.activations.append(pre_processed_input)
        
        self.weights[0] = np.random.randn(5000,5)
        
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
   
    def backpropagation(self, y_train, y_pred):  
        #Initialize a list to store the gradients for each layer
        gradients = []
        
        #Calculate the output layer gradient
        output_gradient = self.activations[-1] * (1 - self.activations[-1])
        print (f"Initial output_gradient shape: {output_gradient.shape}")
        
        #Loop backward through the layers to calculate gradients
        for i in range(len(self.weights) - 1, -1, -1):
            print (f"Layer {i} output_gradient shape: {output_gradient.shape}")
            layer_gradient = output_gradient.dot(self.weights[i].T)
            print (f"Layer {i} layer_gradient shape: {layer_gradient.shape}")
            gradients.append(layer_gradient)
            
            #Update the output gradient for the next iteration
            print(f"self.activations[{i}] shape: {self.activations[i].shape}")
            output_gradient = layer_gradient * (self.activations[i] * (1 - self.activations[i]))
            print(f"New output_gradient shape: {output_gradient.shape}")
    
        gradients.reverse()
            
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * self.activations[i].T.dot(gradients[i])
            self.biases[i] += self.learning_rate * np.sum(gradients[i], axis=0, keepdims=True)
    
    def train(self, X_train, y_train, epochs):
        for epoch in range(epochs):
            y_pred = self.forward_propagation(input_data)
            loss = self.cost_function(y_train, y_pred)
            self.backpropagation(y_train, y_pred)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss}")
    
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
    
    #Generate ground truth data
    y_true = np.array([[1]])
    
    #Calculate Loss
    y_pred = model.forward_propagation(input_data)
    loss = model.cost_function(y_true, y_pred)
    print(f"Loss: {loss}")
    
    #Training
    epochs = 10
    model.train(input_data, y_true, epochs)
