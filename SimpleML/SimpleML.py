import numpy as np
import math
import sys
class model():
    def __init__(self):
        np.random.seed(1)
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
            #layer.weights = np.full(weightShape, 0.5)
            layer.weights = np.full(weightShape, 1,dtype='float32')#np.random.random_sample(weightShape)
            layer.bias = np.full(nextLayer.shape, 1,dtype='float32')#np.random.random_sample(nextLayer.shape)

        self.layers[-1].bias = np.full(self.layers[-1].shape, 1,dtype='float32')

    def display(self):
        print("\n\n")
        for layerIndex in range(0, len(self.layers)):
            layer = self.layers[layerIndex]
            weightsAmount = layer.weights.size
            print(layer.name + "  -=-  " + str(layer.shape) + "  -=-  " + str(weightsAmount))
        print("\n\n")

    def train(self, input=None, target=None, epochs=None, loss_function=None, learn_rate=0.00001):
        if(loss_function == None): raise ValueError("Invalid loss_function: " + str(loss_function))
        lossHistory = []
        accuracyHistory = []
        for epoch in range(1, epochs+1):
            weight_changes = []
            bias_changes = []
            lossValue = 0
            accuracyValue = 0
            stepCount = 0
            for inputs, targets in zip(input, target):
                prediction = self.predict(inputs)
                lossValue += loss_function(self, targets, prediction)
                accuracyValue +=  1 - np.sum(targets-prediction)
                #print(targets, prediction)
                wc, bc = self.step_backward(targets, prediction, learn_rate)
                weight_changes.append(wc)
                bias_changes.append(bc)
                #self._applyChanges(wc, bc)
                progress = stepCount/len(input)
                #print(progress)
                self.update_progress(progress, extra=(lossValue/stepCount))
                stepCount += 1
            lossHistory.append(lossValue/len(input))
            accuracyHistory.append(accuracyValue/len(input))
            print(" - ", epoch,"/",epochs, "Loss:", lossValue/len(input), "Accracy:", accuracyValue/len(input))
            self._applyChanges(np.average(weight_changes, axis=0), np.average(bias_changes, axis=0))
        return lossHistory, accuracyHistory

    def predict(self, inputs):
        currentValues = inputs

        for layerIndex in range(0, len(self.layers)-1):
            layer = self.layers[layerIndex]
            layer.neurons = currentValues
            
            if(layer.use_bias == True): currentValues = layer.activation(self, np.dot(layer.neurons, layer.weights) + layer.bias)
            if(layer.use_bias == False): currentValues = layer.activation(self, np.dot(layer.neurons, layer.weights))


        lastLayer = self.layers[-1]
        result = None
        if(lastLayer.use_bias == False): result = lastLayer.activation(self, currentValues)
        if(lastLayer.use_bias == True): result = lastLayer.activation(self, currentValues+lastLayer.bias)
        lastLayer.neurons = result
        return result


    def step_backward(self, actual, prediction, learn_rate):
        nextLayersTarget = actual
        weight_changes = []
        bias_changes = []
        for layerIndex in range(0, len(self.layers)-1):
            index = (len(self.layers)-1) - layerIndex
            layer = self.layers[index]
            nextLayer = self.layers[index-1]
            #print(layer.name, " --> ", nextLayer.name)
            accuracy = layer.activation(self,nextLayersTarget - layer.neurons, deriv=True)
            #print(accuracy.shape, nextLayer.weights.shape)
            weightAccuracy = nextLayer.weights * accuracy
            #print(weightAccuracy)
            #print(nextLayer.weights)
            #nextLayer.weights += weightAccuracy*learn_rate
            weight_changes.append(weightAccuracy*learn_rate)
            #print(nextLayer.weights)
            nextLayersTarget = np.average(weightAccuracy, axis=1)
            #print(nextLayersTarget)
            biasAccuracy = nextLayer.bias * accuracy
            #print(biasAccuracy)
            #print(nextLayer.bias)
            #nextLayer.bias += biasAccuracy*learn_rate
            bias_changes.append(biasAccuracy*learn_rate)
            #print(nextLayer.bias)

            #print("\n")
        return weight_changes, bias_changes
            
    def _applyChanges(self, weight_changes, bias_changes):
        a = 1
        #print(weight_changes, bias_changes)
        for layerIndex in range(0, len(self.layers)-1):
            index = (len(self.layers)-1) - layerIndex
            layer = self.layers[index]
            nextLayer = self.layers[index-1]
            nextLayer.weights += weight_changes[layerIndex]
            nextLayer.bias += bias_changes[layerIndex]

    def update_progress(self, progress, extra=0):
        barLength = 10 # Modify this to change the length of the progress bar
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(barLength*progress))
        text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status) + " - " + str(extra)
        sys.stdout.write(text)
        sys.stdout.flush()

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
    def sigmoid(self, x, deriv=False):
        if(deriv == True): return x * (1 - x)
        return 1/(1+np.exp(-x))
    def relu(self, x, deriv=False):
        if(deriv == True):
            x[x<=0] = 0
            x[x>0] = 1
            return x
        return np.maximum(0,x)
    def softmax(self, x, deriv=False):
        expo = np.exp(x)
        expo_sum = np.sum(np.exp(x))
        return expo/expo_sum
    def none(self, x, deriv=False):
        return x

class loss:
    def mean_squared_error(self, actual, predicted):
        mse = (np.square(actual - predicted)).mean(axis=None)
        return mse