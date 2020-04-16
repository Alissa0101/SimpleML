import numpy as np
import math
class model():
    def __init__(self):
        #super(model, self).__init__()
        self.layers = []
        #print(input_size, output_size)
        
    def add_layer(self, layer):
        if(layer.name == None):
            layer.name = "Layer_" + str(len(self.layers)+1)
        self.layers.append(layer)

    def compile(self):
        for layerIndex in range(0, len(self.layers)-1):
            layer = self.layers[layerIndex]
            nextLayer = self.layers[layerIndex+1]
            weightShape = np.concatenate((np.asarray(layer.shape), np.asarray(nextLayer.shape)))
            layer.weights = np.full(weightShape, 0.5)
            layer.bias = np.full(nextLayer.shape, -1)

        self.layers[-1].bias = np.full(self.layers[-1].shape, -1)

    def display(self):
        print("\n\n")
        for layerIndex in range(0, len(self.layers)):
            layer = self.layers[layerIndex]
            weightsAmount = layer.weights.size
            print(layer.name + "  -=-  " + str(layer.shape) + "  -=-  " + str(weightsAmount))
        print("\n\n")

    def train(self, input=None, target=None, epochs=None, loss_function=None):
        if(loss_function == None): raise ValueError("Invalid loss_function: " + str(loss_function))
        for epoch in range(1, epochs+1):
            for inputs, targets in zip(input, target):
                prediction = self.step_forward(inputs)
                lossValue = loss_function(self, targets, prediction)
                self.step_backward(targets, prediction)
                print(targets, prediction)

    def step_forward(self, inputs):
        currentValues = inputs

        for layerIndex in range(0, len(self.layers)-1):
            layer = self.layers[layerIndex]
            layer.neurons = currentValues
            currentValues = layer.activation(self, np.dot(layer.neurons, layer.weights) + layer.bias)

        lastLayer = self.layers[-1]
        print(lastLayer.use_bias)
        if(lastLayer.use_bias == False): return lastLayer.activation(self, currentValues)
        if(lastLayer.use_bias == True): return lastLayer.activation(self, currentValues+lastLayer.bias)

    def step_backward(self, actual, prediction):
        weightChanges = []
        previousLayerNeurons = actual
        for layerIndex in range(0, len(self.layers)):
            index = (len(self.layers)-1) - layerIndex
            layer = self.layers[index]
            print(layer.name)
            neurons = layer.neurons
            if(index == len(self.layers)-1):
                print("at output")
            print(previousLayerNeurons, neurons)
            np.dot(previousLayerNeurons, neurons)
            previousLayerNeurons = neurons

class _layer():
    def __init__(self, shape=0, name=None, activation=None, use_bias=False):
        #if(not isinstance(shape, )): raise ValueError("Invalid value for shape: " + str(shape))
        self.shape = np.array(shape)
        self.name = name
        self.weights = np.array([])
        self.bias = np.array([])
        self.neurons = np.full(shape, 1)
        if(activation == None): activation = activations.none
        self.activation = activation
        self.use_bias = use_bias

    

class layers:
    class fully_connected(_layer):
        def __init__(self, **args):
            _layer.__init__(self, **args)

class activations:
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    def relu(self, x):
       return np.maximum(0,x)
    def softmax(self, x):
        expo = np.exp(x)
        expo_sum = np.sum(np.exp(x))
        return expo/expo_sum
    def none(self, x):
        return x
    def sigmoid_dev(self, x):
        return x * (1 - x)

class loss:
    def mean_squared_error(self, actual, predicted):
        mse = (np.square(actual - predicted)).mean(axis=None)
        return mse