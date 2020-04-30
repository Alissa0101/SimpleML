import numpy as np
import SimpleML.SimpleML as sml
import matplotlib.pyplot as plt

inputData = np.array([[1, 1], [0, 0], [1, 0], [0, 1]])
outputData = np.array([[1], [0], [1], [1]])

useBias = False

model = sml.model()

model.add_layer(sml.layers.fully_connected(shape=inputData.shape[1:], name="Input", activation=sml.activations.none, use_bias=useBias))
model.add_layer(sml.layers.fully_connected(shape=[24], activation=sml.activations.none, use_bias=useBias))
model.add_layer(sml.layers.fully_connected(shape=[24], activation=sml.activations.none, use_bias=useBias))
model.add_layer(sml.layers.fully_connected(shape=outputData.shape[1:], name="Output", activation=sml.activations.none, use_bias=False))

model.compile()

model.display()

for inp, out in zip(inputData, outputData):
    print(out, model.predict(inp))

history, acc = model.train(input=inputData, target=outputData, epochs=1000, loss_function=sml.loss.mean_squared_error, learn_rate=0.001)


for layer in model.layers:
    print(layer.name)
    print(layer.weights, "\n")


plt.plot(history)
plt.show()
plt.plot(acc)
plt.show()

for inp, out in zip(inputData, outputData):
    print(out, model.predict(inp))
