import matplotlib.pyplot as plt
import numpy as np
from numpy import load
from sklearn.model_selection import train_test_split
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models
from numpy import save

pos = load('feauter_pos_new.npy')
pos = pos/pos.max()
input_size = pos.shape[1]
code_size = 30
X_train, X_Test = train_test_split(pos, test_size=0.1, random_state=42)
X_valid, X_test = train_test_split(X_Test, test_size=0.1, random_state=42)


def build_autoencoder(input_size, code_size):

    # The encoder
    encoder = models.Sequential()
    encoder.add(layers.InputLayer((input_size,)))
    encoder.add(layers.Dense(code_size))

    # The decoder
    decoder = models.Sequential()
    decoder.add(layers.InputLayer((code_size,)))
    decoder.add(layers.Dense(np.prod(input_size)))

    return encoder, decoder


encoder, decoder = build_autoencoder(input_size, code_size)

inp = layers.Input(shape=(input_size,))
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = models.Model(inp,reconstruction)
autoencoder.compile(optimizer='adam', loss='mse',  metrics=['accuracy'])
print(autoencoder.summary())

history = autoencoder.fit(x=X_train, y=X_train, epochs=40,validation_data=[X_valid, X_valid])

loss, acc = autoencoder.evaluate(X_test,X_test)
print(loss)
print(acc)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


def visualize(X,encoder,decoder):
    code = encoder.predict(X)
    reco = decoder.predict(code)


    print("Original")
    print(X)

    print("Code")
    print(code)

    print("Reconstructed")
    print(reco)


for i in range(5):
    test = X_test[i]
    test = np.reshape(test,(1,-1))
    visualize(test,encoder,decoder)

encode_topics = encoder.predict(pos)
save('encode_pos_new.npy', pos)