# %%
import import_path
import utils
import numpy as np
import pandas as pd
from oracle import Oracle
from without_teacher import Without_teacher
from load_data import read_csv, split_data, read_W
from omniscient_teacher import Omniscient
from surrogate_teacher import Surrogate
from sklearn.metrics import roc_auc_score
# %%


df = read_csv('output/weebil_vespula.csv', header=0)
train_X, test_X, train_y, test_y = split_data(df, False)
my_oracle = Oracle(eta=0.01, lambd=0.01)
min_w = np.random.normal(loc=0, scale=0.01, size=train_X.shape[1])
# train_X = train_X[0:10]
# train_y = train_y[0:10]
eta, lambd, alpha = 0.01, 0.01, 0.01
training_epochs = 10
oracle = Oracle(eta, lambd)
min_w = oracle.estimate_min_w(
    pd.concat([train_X, test_X]), pd.concat([train_y, test_y]))

print(utils.predict(test_X, test_y, min_w))
print(oracle.min_w)

W_init = oracle.make_W_init(J=10)

path = "test_test"
utils.write_np2csv(W_init, path)


train_X_, train_y_ = utils.return_copy_dateset(train_X, train_y)
# %%
w_init = np.random.normal(loc=0, scale=eta, size=train_X_.shape[1])
W = read_W('output/{}.csv'.format(path), header=None)

wot = Without_teacher(w_init, W, eta, lambd, alpha)
print(wot.w_star, utils.predict(test_X, test_y, wot.w_star))
print(utils.predict_by_W(test_X, test_y, wot.W_star),
      utils.rmse_W(W, wot.W_star, axis=1))
w_star, W_ = wot.learn(train_X_, training_epochs=10, loops=10)
print(w_star, utils.predict(test_X, test_y, w_star))
print(utils.predict_by_W(test_X, test_y, W_), utils.rmse_W(W, W_, axis=1))
# %%
