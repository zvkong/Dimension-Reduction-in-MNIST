import matplotlib.pyplot as plt
import numpy as np

train_data = pd.read_csv("./mnist/train.csv")
X_train = train_data.iloc[:, 1:].values.astype('float32')
y_train = train_data.iloc[:, 0].values.astype('float32')
X_train_img = X_train.reshape(-1, 28, 28)

digit = 6
X_train_digit = X_train_img[y_train == 6]
plt.figure(figsize=[12, 4])
for i in range(4):
    idx = np.random.randint(0, 4137)
    plt.subplot(1, 4, i+1)
    plt.imshow(X_train_digit[idx])
    i += 1
    pass
plt.tight_layout()
plt.savefig("./img/digits_sample.png")
