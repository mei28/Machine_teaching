# %%
import import_path
import numpy as np
import pandas as pd
from models.oracle import Oracle
from models.surrogate_teacher import Surrogate
from models.omniscient_teacher import Omniscient
from models.random_teacher import Random
from models.without_teacher import Without_teacher
from utils import predict, predict_by_W, rmse_W, write_np2csv, rmse_w, make_random_mask, predict_wj
from load_data import read_W, read_csv, split_data
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import logging
import datetime
# %%
df = read_csv('output/wine-quality-pm1.csv', header=0)
train_X, test_X, train_y, test_y = split_data(df, True)
eta, lambd, alpha = 1, 2, 0.01
training_epochs, loops = 10, 10
J = 10
# 提示する教材合計数
textbook = 500
# 推定に使う教材数
test_textbook_list = [100]
# 推定間に提示する教材数
between_textbook_list = [1]
# 組
k = 1
lambds = [1, 2, 3, 4, 5]

for lambd in lambds:
    oracle = Oracle(eta=eta, lambd=lambd)
    min_w = oracle.estimate_min_w(
        pd.concat([train_X, test_X]), pd.concat([train_y, test_y]))
    print('{}: {}'.format(min_w, predict(test_X, test_y, min_w)))
    W_init = oracle.make_W_init(J=J)
    W = W_init.copy()
    train_X_ = train_X.copy()
    train_y_ = train_y.copy()

    now = datetime.datetime.now()
    now_str = now.strftime('%Y%m%d%H%M%S')
    result_path = 'result/wine_{}_{}_{}'.format(now_str, k, lambd)
    logging.basicConfig(
        filename='./logs/log_wine_{0:%Y%m%d%H%M%S}_{1}_{2}.log'.format(now, k, lambd), level=logging.DEBUG
    )

    logging.debug(
        './logs/log_{0:%Y%m%d%H%M%S}_{1}_{2}.log'.format(now, k, lambd))
    logging.debug('lambd')
    logging.debug(lambd)
    logging.debug('min_w')
    logging.debug(min_w)
    logging.debug(predict(test_X, test_y, min_w))
    logging.debug('eta,lambd')
    logging.debug([eta, lambd])
    # %%

    # %%
    # Omniscient
    logging.debug('Omniscient')
    train_X_ = train_X.copy()
    train_y_ = train_y.copy()
    W = W_init.copy()
    omt = Omniscient(min_w, W, N=train_X_.shape[0], alpha=alpha)

    a = np.zeros(1)
    b = np.zeros(J)
    for i in range(textbook):
        a = np.vstack((a, predict_by_W(test_X, test_y, omt.W)))
        b = np.vstack((b, predict_wj(test_X, test_y, omt.W)))
        print("{}: {}".format(i, predict_by_W(test_X, test_y, omt.W)))
        omt.show_textbook(X=train_X_, y=train_y_, N=1, option='min_w')
        logging.debug(predict_by_W(test_X, test_y, omt.W))
    a = a[1:]
    b = b[1:]
    write_np2csv(
        a, '{}_{}.csv'.format(result_path, 'omniscient'))
    write_np2csv(
        b, '{}_{}_wj.csv'.format(result_path, 'omniscient')
    )
    print('{}: finished.'.format(k))
    # %%
    # Random

    train_X_ = train_X.copy()
    train_y_ = train_y.copy()
    W = W_init.copy()

    rat = Random(min_w, W, N=train_X_.shape[0], alpha=alpha)
    logging.debug('Random')
    a = np.zeros(1)
    b = np.zeros(J)
    for i in range(textbook):
        a = np.vstack((a, predict_by_W(test_X, test_y, rat.W)))
        b = np.vstack((b, predict_wj(test_X, test_y, rat.W)))
        print("{}: {}".format(i, predict_by_W(test_X, test_y, rat.W)))
        rat.show_textbook(train_X_, y=train_y_, N=1, option='min_w')
        logging.debug(predict_by_W(test_X, test_y, rat.W))
    a = a[1:]
    b = b[1:]
    write_np2csv(a, '{}_{}.csv'.format(result_path, 'random'))
    write_np2csv(b, '{}_{}_wj.csv'.format(result_path, 'random'))
    print('{}: finished.'.format(k))

    # %%
    for t_num in test_textbook_list:
        for b_num in between_textbook_list:
            # wot
            logging.debug('without teacher')
            w_init = np.random.normal(loc=0, scale=lambd, size=min_w.shape)
            W = W_init.copy()
            train_X_ = train_X.copy()
            train_y_ = train_y.copy()

            wot = Without_teacher(
                w_init, W, N=train_X_.shape[0], eta=eta, lambd=lambd, alpha=alpha)

            a = np.zeros(7)
            b = np.zeros(J)
            for i in range(textbook):
                tmp = np.append([], predict(test_X, test_y, min_w))
                tmp = np.append(tmp, predict(test_X, test_y, wot.w_star))
                tmp = np.append(tmp, predict_by_W(test_X, test_y, wot.W))
                tmp = np.append(tmp, predict_by_W(test_X, test_y, wot.W_star))
                tmp = np.append(tmp, rmse_w(wot.w_star, min_w))
                tmp = np.append(tmp, rmse_W(wot.W, wot.W_star))
                tmp = np.append(tmp, roc_auc_score(
                    test_y, wot.predict_y(test_X, wot.w_star)))
                a = np.vstack((a, tmp))
                b = np.vstack((b, predict_wj(test_X, test_y, wot.W)))
                if i % b_num == 0:
                    masked_textbook, _ = make_random_mask(train_X_, t_num)
                    wot.learn(masked_textbook, 10, 10)
                print('{}: {}'.format(i, predict_by_W(test_X, test_y, wot.W)))
                wot.show_textbook(train_X_, y=None, N=1, option='w_star')
                print(predict(test_X, test_y, min_w), roc_auc_score(
                    test_y, wot.predict_y(test_X, wot.w_star)))
                logging.debug([predict_by_W(test_X, test_y, wot.W),
                               predict_by_W(test_X, test_y, wot.W_star)])

            a = a[1:]
            b = b[1:]
            write_np2csv(
                a, '{}_{}_{}_{}.csv'.format(result_path, "wot", t_num, b_num))
            write_np2csv(
                b, '{}_{}_{}_{}_wj.csv'.format(
                    result_path, 'wot', t_num, b_num)
            )
            print('t{}b{}: finished.'.format(t_num, b_num))

    # %%
