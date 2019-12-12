# %%
import import_path
import utils
import numpy as np
from oracle import Oracle
from without_teacher import Without_teacher
from load_data import read_csv, split_data, read_W
from omniscient_teacher import Omniscient
from surrogate_teacher import Surrogate
# %%
df = read_csv('output/weebil_vespula.csv', header=0)
train_X, test_X, train_y, test_y = split_data(df)
my_oracle = Oracle(eta=0.01, lambd=0.01)
min_w = np.random.normal(loc=0, scale=0.01, size=train_X.shape[1])
# train_X = train_X[0:10]
# train_y = train_y[0:10]
eta, lambd, alpha = 0.01, 0.01, 0.01
training_epochs = 10
oracle = Oracle(eta, lambd)
min_w = oracle.estimate_min_w(train_X, train_y, training_epochs)

print(utils.predict(test_X, test_y, min_w))
print(oracle.min_w)

W_init = oracle.make_W_init(J=10)

path = "test"
utils.write_np2csv(W_init, path)


train_X_, train_y_ = utils.return_copy_dateset(train_X, train_y)
# %%
# Omniscient
ot = Omniscient(min_w=min_w, alpha=alpha)
w_t = np.random.normal(loc=0, scale=eta, size=train_X_.shape[1])
print(train_X_.shape, train_y_.shape)
X_t, y_t, index = ot.return_textbook(train_X_, train_y_, w_t, min_w)
print(train_X_.shape, train_y_.shape)
# %%
# Surrogate
st = Surrogate(min_w=min_w, alpha=alpha)
w_t = np.random.normal(loc=0, scale=eta, size=train_X.shape[1])
print(train_X_.shape, train_y_.shape)
X_t, y_t, index = st.return_textbook(train_X_, train_y_, w_t, min_w)
print(train_X_.shape, train_y_.shape)


# %%
w_init = np.random.normal(size=train_X.shape[1])
W = read_W('output/test.csv', header=None)
wot = Without_teacher(w_init, W, eta, lambd, alpha)
print(utils.predict(test_X, test_y, wot.w_star))
print(utils.predict_by_W(test_X, test_y, wot.W))
wot.learn(train_X_, training_epochs=2)
print(utils.predict(test_X, test_y, wot.w_star))
print(utils.predict_by_W(test_X, test_y, wot.W))

# %%
w_init = np.random.normal(size=train_X.shape[1])
W = read_W('output/test.csv', header=None)
wot = Without_teacher(w_init, W, eta, lambd, alpha)
# wot = Without_teacher(min_w, W, eta, lambd, alpha)
for i in range(10):
    print('w_star: {}'.format(utils.predict(test_X, test_y, wot.w_star)))
    print('W: {}'.format(utils.predict_by_W(test_X, test_y, wot.W)))
    X_pool, y_pool, index_set = wot.return_text_book_omni(
        train_X_, train_y_, W)

    wot.learn(X_pool, training_epochs=10, loops=10)

# %%
w_init = np.random.normal(size=train_X_.shape[1])
W = read_W('output/test.csv', header=None)
wot = Without_teacher(w_init, W, eta, lambd, alpha)
# wot = Without_teacher(min_w, W, eta, lambd, alpha)
for i in range(10):
    print(utils.predict(test_X, test_y, wot.w_star))
    print(utils.predict_by_W(test_X, test_y, wot.W_star))
    X_pool, y_pool, index_set = wot.return_text_book_omni(
        train_X_, train_y_, wot.W_star)
    wot.learn(X_pool, 10, 10)
    print(utils.rmse_W(wot.W,wot.W_star))


# %%
