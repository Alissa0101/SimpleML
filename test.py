import numpy as np
import SimpleML as sml

inputData = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
outputData = np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0]])

useBias = False

model = sml.model()

model.add_layer(sml.layers.fully_connected(shape=inputData.shape[1:], name="Input", activation=sml.activations.relu, use_bias=useBias))
model.add_layer(sml.layers.fully_connected(shape=[24], activation=sml.activations.relu, use_bias=useBias))
#model.add_layer(sml.layers.fully_connected(shape=[5], activation=sml.activations.softmax, use_bias=useBias))
model.add_layer(sml.layers.fully_connected(shape=outputData.shape[1:], name="Output", activation=sml.activations.sigmoid, use_bias=True))

model.compile()

model.display()

model.train(input=inputData, target=outputData, epochs=100000, loss_function=sml.loss.mean_squared_error, learn_rate=0.0001)

for inp, out in zip(inputData, outputData):
    print(out, model.predict(inp))
