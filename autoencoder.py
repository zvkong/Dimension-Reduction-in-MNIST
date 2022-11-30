import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

train_data = pd.read_csv("./mnist/train.csv")
test_data = pd.read_csv("./mnist/test.csv")

X_train = train_data.iloc[:, 1:].values.astype('float32') / 255.
y_train = train_data.label.values.astype('float32') / 255.
X_test = test_data.values

X_train_img = X_train.reshape(-1, 28, 28)
X_test_img = X_test.reshape(-1, 28, 28)

# model
encoding_dim = 32
input_img = tf.keras.Input(shape=(784,))
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_img)
decoded = tf.keras.layers.Dense(784, activation='sigmoid')(encoded)
autoencoder = tf.keras.Model(input_img, decoded)
# encoder
encoder = tf.keras.Model(input_img, encoded)
# decoder
encoded_input = tf.keras.Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = tf.keras.Model(encoded_input, decoder_layer(encoded_input))
# compile
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# train
autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_split=0.2)

# predict
encoded_imgs = encoder.predict(X_train)
decoded_imgs = decoder.predict(encoded_imgs)
encoded_imgs_df = pd.DataFrame(encoded_imgs)
encoded_imgs_df.to_csv("./output/mnist_autoencoder.csv", index=False)

# plot decoded
idx = 14
plt.subplot(1, 2, 1)
plt.imshow(X_train[idx].reshape([28, 28]))
plt.subplot(1, 2, 2)
plt.imshow(decoded_imgs[idx].reshape([28, 28]))
plt.show()
