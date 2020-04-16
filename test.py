import numpy as np
import SimpleML as sml

inputData = np.array([[2, 2]])
outputData = np.array([[1,1]])

model = sml.model(input_size=inputData.shape[1:], output_size=outputData.shape[1:], output_activation=sml.activations.sigmoid)

hiddenLayer1 = sml.layers.fully_connected(shape=[4])
hiddenLayer2 = sml.layers.fully_connected(shape=[4])

model.add_layer(hiddenLayer1)
model.add_layer(hiddenLayer2)

model.compile()

model.display()

model.train(input=inputData, target=outputData, epochs=1)