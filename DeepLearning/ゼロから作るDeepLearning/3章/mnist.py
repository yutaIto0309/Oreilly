#%%
from keras.datasets import mnist

(x_train, t_train), (x_test, t_test) = mnist.load_data()
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
# %%
# データの確認
import numpy as np 
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

img_show(x_train[0])
# %%
import pickle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)

    return exp_a / sum_exp_a

def init_network():
    with open("C:\オライリー\DeepLearning\ゼロから作るDeepLearning\ゼロから作るDeepLearningサンプル\ch03\sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y

x_test = np.reshape(x_test, (10000, 784))

network = init_network()
print(network.shape)
accuracy_cnt = 0
for i in range(len(x_test)):
    y = predict(network, x_test[i])
    p = np.argmax(y)

    if p == t_test[i]:
        accuracy_cnt += 1
print('Accuracy:' + str(float(accuracy_cnt) / len(t_test)))
# %%
# バッチ処理
batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t_test[i:i+batch_size])
print('Accuracy:' + str(float(accuracy_cnt) / len(t_test)))
# %%
