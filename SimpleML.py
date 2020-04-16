import numpy as np
import math
class model():
    def __init__(self, input_size=None, output_size=None, input_activation=None, output_activation=None):
        #super(model, self).__init__()
        if(input_size==None or output_size==None): raise ValueError("Either input_size or output_size are not defined")
        self.layers = []
        self.input_size = input_size
        self.output_size = output_size
        self.add_layer(layers.fully_connected(shape=input_size, name="Input", activation=input_activation))
        self.output_activation = np.vectorize(output_activation)
        #print(input_size, output_size)
        
    def add_layer(self, layer):
        if(layer.name == None):
            layer.name = "Layer_" + str(len(self.layers)+1)
        self.layers.append(layer)

    def compile(self):
        self.add_layer(layers.fully_connected(shape=self.output_size, name="Output"))
        for layerIndex in range(0, len(self.layers)-1):
            layer = self.layers[layerIndex]
            nextLayer = self.layers[layerIndex+1]
            weightShape = np.concatenate((np.asarray(layer.shape), np.asarray(nextLayer.shape)))
            layer.weights = np.full(weightShape, 0.5)

    def display(self):
        print("\n\n")
        for layerIndex in range(0, len(self.layers)):
            layer = self.layers[layerIndex]
            weightsAmount = layer.weights.size
            print(layer.name + "  -=-  " + str(layer.shape) + "  -=-  " + str(weightsAmount))
        print("\n\n")

    def train(self, input=None, target=None, epochs=None):
        for epoch in range(1, epochs+1):
            for inputs in input:
                result = self.step(inputs)
                print(result)

    def step(self, inputs):
        currentValues = inputs

        for layerIndex in range(0, len(self.layers)-1):
            layer = self.layers[layerIndex]
            layer.neurons = layer.activation(self, currentValues)
            currentValues = np.dot(layer.neurons, layer.weights)

        
        return self.output_activation(self, currentValues)


class _layer():
    def __init__(self, shape=0, name=None, activation=None):
        #if(not isinstance(shape, )): raise ValueError("Invalid value for shape: " + str(shape))
        self.shape = np.array(shape)
        self.name = name
        self.weights = np.array([])
        self.neurons = np.full(shape, 1)
        if(activation == None): activation = np.vectorize(activations.none)
        self.activation = np.vectorize(activation)

    

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