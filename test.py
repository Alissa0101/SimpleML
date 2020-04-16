import numpy as np
import SimpleML as sml

inputData = np.array([[2]])
outputData = np.array([[1]])

model = sml.model()

model.add_layer(sml.layers.fully_connected(shape=inputData.shape[1:], name="Input"))
model.add_layer(sml.layers.fully_connected(shape=[2]))
model.add_layer(sml.layers.fully_connected(shape=[2]))
model.add_layer(sml.layers.fully_connected(shape=outputData.shape[1:], name="Output"))

model.compile()

model.display()

model.train(input=inputData, target=outputData, epochs=1, loss_function=sml.loss.mean_squared_error)