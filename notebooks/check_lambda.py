# %%
import pandas as pd
from models.oracle import Oracle
from utils.utils import predict, predict_by_W, rmse_W, write_np2csv, rmse_w, make_random_mask, predict_wj
from utils.load_data import read_W, read_csv, split_data
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %%
df_insect = read_csv('../output/weebil_vespula_pm1.csv', header=0)
df_wine = read_csv('../output/wine-quality-pm1.csv', header=0)
# %%

eta = 1
J = 100
lambds = [1, 2, 3, 4, 5, 10]
# %%


def save_graph(fig, save_path):
    fig.savefig('./{}.png'.format(save_path), bbox_inches='tight')
    print('saved')


def draw_boxplot(df, save_path=None, is_save=False):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Initial AUC', fontsize=18)
    ax.set_ylabel('lambda', fontsize=18)
    sns.boxplot(data=df,
                orient='h',
                palette='Pastel1',
                width=0.5,
                sym=''
                )
    if save_path is not None and is_save == True:
        save_graph(fig, save_path)


# %%
train_X, test_X, train_y, test_y = split_data(df_insect, False)
print('{}insect{}'.format('-'*10, '-'*10))
df_wj = pd.DataFrame([])
for lambd in lambds:
    oracle_insect = Oracle(eta=eta, lambd=lambd)
    min_w = oracle_insect.estimate_min_w(
        pd.concat([train_X, test_X]), pd.concat([train_y, test_y]))
    # print('[min_w] {}: {}'.format(min_w, predict(test_X, test_y, min_w)))
    W_init = oracle_insect.make_W_init(J=J)
    tmp = pd.Series(predict_wj(test_X, test_y, W_init))
    df_wj = pd.concat([df_wj, tmp], axis=1)
    # print("lambd = {} : {}".format(lambd, predict_by_W(test_X, test_y, W_init)))

df_wj.columns = ['0', '1', '2', '3', '4', '5', '10']
draw_boxplot(df_wj)
# %%
print('{}wine{}'.format('-'*10, '-'*10))
df_wj = pd.DataFrame([])
for lambd in lambds:
    oracle_wine = Oracle(eta=eta, lambd=lambd)
    min_w = oracle_wine.estimate_min_w(
        pd.concat([train_X, test_X]), pd.concat([train_y, test_y]))
    # print('[min_w] {}: {}'.format(min_w, predict(test_X, test_y, min_w)))
    W_init = oracle_wine.make_W_init(J=J)
    tmp = pd.Series(predict_wj(test_X, test_y, W_init))
    df_wj = pd.concat([df_wj, tmp], axis=1)
    # print("lambd = {} : {}".format(lambd, predict_by_W(test_X, test_y, W_init)))

df_wj.columns = ['0', '1', '2', '3', '4', '5', '10']
draw_boxplot(df_wj)
# %%
