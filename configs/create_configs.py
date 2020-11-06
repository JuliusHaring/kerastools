from tensorflow import keras

model1 = keras.models.Sequential()
model1.add(keras.layers.Input(shape=(0)))
model1.add(keras.layers.Dense(2, activation='softmax'))
with open('./configs/m1.json', 'w') as f:
    f.write(model1.to_json())

model2 = keras.models.Sequential()
model2.add(keras.layers.Input(shape=(None, 0)))
model2.add(keras.layers.LSTM(20))
model2.add(keras.layers.Dense(2, activation='softmax'))
with open('./configs/m2.json', 'w') as f:
    f.write(model2.to_json())
model2.save_weights('./configs/m2.h5')

ae = keras.models.Sequential()
ae.add(keras.layers.Dense(10, activation='sigmoid'))
ae.add(keras.layers.Dense(5, activation='sigmoid'))
ae.add(keras.layers.Dense(10, activation='softmax'))
with open('./configs/ae.json', 'w') as f:
    f.write(ae.to_json())


