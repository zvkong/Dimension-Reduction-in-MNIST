import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

train_data = pd.read_csv("./mnist/train.csv")
test_data = pd.read_csv("./mnist/test.csv")

X_train = train_data.iloc[:, 1:].values.astype('float32') / 255.
X_test = test_data.values.astype('float32') / 255.

X_train_img = X_train.reshape(-1, 28, 28)
X_test_img = X_test.reshape(-1, 28, 28)
X_img = np.concatenate((X_train_img, X_test_img), axis=0)

# model
input_img = tf.keras.layers.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same')(x)
encoded = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder = tf.keras.models.Model(input_img, decoded)
autoencoder.summary()
# compile
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# train
autoencoder.fit(X_img, X_img,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_split=0.2)

# predict
encoder = tf.keras.Model(input_img, encoded)
encoded_input = tf.keras.Input(shape=(encoded.shape[1:]))
decoder_layer = autoencoder.layers[-1]
decoder = tf.keras.Model(encoded, decoded)
encoded_imgs = encoder.predict(X_img)
decoded_imgs = decoder.predict(encoded_imgs)

# plot original digits decoded digits
idxs = np.random.randint(0, X_img.shape[0], 5)
i = 0
plt.figure(figsize=[12, 4])
for idx in idxs:
    plt.subplot(2, 5, 1 + i)
    plt.imshow(X_img[idx])
    plt.subplot(2, 5, 6 + i)
    plt.imshow(decoded_imgs[idx])
    i += 1
    pass
plt.tight_layout()
plt.savefig("./img/autoencoder_decoded.png")

# plot features
i = 0
plt.figure(figsize=[12, 10])
for idx in idxs:
    for j in range(4):
        plt.subplot(4, 5, j*5 + i + 1)
        plt.imshow(encoded_imgs[idx, :, :, j])
        plt.title("{}th digit, {}th feature".format(i + 1, j + 1))
        pass
    i += 1
    pass
plt.tight_layout()
plt.savefig("./img/autoencoder_feature.png")

# save low dimentional representations
encoded_imgs_df = pd.DataFrame(encoded_imgs.reshape(-1, np.prod(encoded.shape[1:])))
encoded_imgs_df.to_csv("./output/mnist_autoencoder.csv", index=False)
