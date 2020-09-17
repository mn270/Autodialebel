import matplotlib.pyplot as plt
import numpy as np
from numpy import load
from sklearn.model_selection import train_test_split
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models
from numpy import save

embb_words = load('feauter_word_new.npy')
timestamp = embb_words.shape[1]
input_size = embb_words.shape[2]
code_size = 30
X_train, X_Test = train_test_split(embb_words, test_size=0.1, random_state=42)
X_valid, X_test = train_test_split(X_Test, test_size=0.1, random_state=42)


def build_autoencoder(timestamp, input_size, code_size):
    # The encoder
    encoder = models.Sequential()
    encoder.add(layers.LSTM(100, return_sequences=True, input_shape=(timestamp, input_size)))
    encoder.add(layers.LSTM(code_size))

    # The decoder
    decoder = models.Sequential()
    decoder.add(layers.RepeatVector(timestamp, input_shape=[code_size]))
    decoder.add(layers.LSTM(100, return_sequences=True))
    decoder.add(layers.TimeDistributed(layers.Dense(input_size, activation="sigmoid")))
    decoder.add(layers.Dense(np.prod((input_size))))

    return encoder, decoder


encoder, decoder = build_autoencoder(timestamp,input_size, code_size)

inp = layers.Input(shape=(timestamp, input_size,))
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = models.Model(inp,reconstruction)
autoencoder.compile(optimizer='adam', loss='mse',  metrics=['accuracy'])
print(autoencoder.summary())

history = autoencoder.fit(x=X_train, y=X_train, epochs=20,validation_data=[X_valid, X_valid],)
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

encode_topics = encoder.predict(embb_words)
save('encode_words_new.npy', encode_topics)