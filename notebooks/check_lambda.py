# %%
import pandas as pd
from models.oracle import Oracle
from utils.utils import predict, predict_by_W, rmse_W, write_np2csv, rmse_w, make_random_mask
from utils.load_data import read_W, read_csv, split_data
from sklearn.metrics import roc_auc_score


# %%
df_insect = read_csv('../output/weebil_vespula_pm1.csv', header=0)
df_wine = read_csv('../output/wine-quality-pm1.csv',header=0)
# %%

eta = 1
J = 10
lambds = [0, 1, 2, 5, 10]
#%%
train_X, test_X, train_y, test_y = split_data(df_insect, False)
print('{}insect{}'.format('-'*10, '-'*10))
for lambd in lambds:
    oracle_insect = Oracle(eta=eta, lambd=lambd)
    min_w = oracle_insect.estimate_min_w(
        pd.concat([train_X, test_X]), pd.concat([train_y, test_y]))
    print('[min_w] {}: {}'.format(min_w, predict(test_X, test_y, min_w)))
    W_init = oracle_insect.make_W_init(J=J)

    print("lambd = {} : {}".format(lambd, predict_by_W(test_X, test_y, W_init)))


# %%
train_X, test_X, train_y, test_y = split_data(df_wine, True)
print('{}wine{}'.format('-'*10, '-'*10))
for lambd in lambds:
    oracle_wine = Oracle(eta=eta, lambd=lambd)
    min_w_wine = oracle_wine.estimate_min_w(
        pd.concat([train_X, test_X]), pd.concat([train_y, test_y]))
    print('[min_w] {}: {}'.format(min_w_wine, predict(test_X, test_y, min_w_wine)))
    W_init = oracle_wine.make_W_init(J=J)

    print("lambd = {} : {}".format(lambd, predict_by_W(test_X, test_y, W_init)))


# %%
