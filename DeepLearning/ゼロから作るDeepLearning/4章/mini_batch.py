#%%
from  keras.datasets import mnist
import numpy as np 
(x_train, t_train), (x_test, t_test) = mnist.load_data()
print(x_train.shape)
t_train_one_hot = np.identity(10)[t_train]
print(t_train_one_hot.shape)
# %%
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train_one_hot[batch_mask]
# %%
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size