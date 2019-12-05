# %%
import import_path
import utils
import numpy as np
from oracle import Oracle
from load_data import read_csv, split_data

# %%
df = read_csv('output/weebil_vespula.csv')
train_X, test_X, train_y, test_y = split_data(df)

my_oracle = Oracle(eta=0.01, lambd=0.01)
min_w = np.random.normal(loc=0, scale=0.01, size=train_X.shape[1])
# train_X = train_X[0:10]
# train_y = train_y[0:10]
eta, lambd = 0.01, 0.01
training_epochs = 100
oracle = Oracle(eta, lambd)
min_w = oracle.estimate_min_w(train_X, train_y, training_epochs)

print(utils.predict(test_X, test_y, min_w))
print(oracle.min_w)
# %%
