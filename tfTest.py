import numpy as np
import SimpleML as sml
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

inputData = np.array([[1, 1], [0, 0]])
outputData = np.array([[1], [0]])

#data_x = tf.data.Dataset.from_tensor_slices((inputData))
#data_y = tf.data.Dataset.from_tensor_slices((outputData))

model = keras.Sequential()  

model.add(keras.layers.Dense(2, input_shape=[2]))
model.add(keras.layers.Dense(24))
model.add(keras.layers.Dense(24))
model.add(keras.layers.Dense(1, use_bias=True))

model.compile(keras.optimizers.sgd(), loss=keras.losses.mean_squared_error)
print(model.summary())


history = model.fit(inputData, outputData, epochs=1000)

plt.plot(history.history['loss'])
plt.show()

for inp, out in zip(inputData, outputData):
    inp = np.array([inp])
    print(inp, out, model.predict(inp))