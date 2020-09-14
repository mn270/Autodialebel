from numpy import load
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from tensorflow.keras.models import Sequential, Model
import numpy as np
import matplotlib.pyplot as plt


topics = load('topics.npy')
input_size = topics.shape[1]
code_size = 30
X_train, X_Test = train_test_split(topics, test_size=0.1, random_state=42)
X_valid, X_test = train_test_split(X_Test, test_size=0.1, random_state=42)


def build_autoencoder(input_size, code_size):
    # The encoder
    encoder = Sequential()
    encoder.add(InputLayer(input_size))
    encoder.add(Dense(code_size))

    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    decoder.add(Dense(np.prod(input_size))) # np.prod(img_shape) is the same as 32*32*3, it's more generic than saying 3072

    return encoder, decoder


encoder, decoder = build_autoencoder(input_size, code_size)

inp = Input(input_size)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inp,reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')
print(autoencoder.summary())

history = autoencoder.fit(x=X_train, y=X_train, epochs=20,validation_data=[X_valid, X_valid])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


def visualize(X,encoder,decoder):
    """Draws original, encoded and decoded images"""
    # img[None] will have shape of (1, 32, 32, 3) which is the same as the model input
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