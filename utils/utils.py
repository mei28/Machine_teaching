
import pandas as pd
import numpy as np
import theano
import theano.tensor as T
from sklearn.metrics import roc_auc_score
import scipy.stats as stats


def predict_wj(X, y, W):
    J, D = W.shape
    val = 0
    auc_list = np.zeros(J)
    for j in range(J):
        w_j = W[j, :]
        tmp = predict(X, y, w_j)
        auc_list[j] = tmp
    return auc_list


def make_grad_loss_matrix(X, y, w):
    """return grad loss matrix

    Parameters
    ----------
    X : pandas matrix
        feature matrix
    y : pandas vector
        predict
    w : numpy
        model parameter
    Returns
    -------
    numpy matrix
        grad loss matrix
    """
    grad_loss_matrix = np.zeros((X.shape[0], w.shape[0]))
    grad_loss = grad_loss_function()
    for i in range(X.shape[0]):
        grad_loss_ = grad_loss(X.iloc[i], y.iloc[i], w)
        a = np.array(grad_loss_).flatten()
        grad_loss_matrix[i] = a

    return grad_loss_matrix


def grad_loss_function():
    """
    return grad loss function

    Returns
    -------
    grad loss function
    inputs = [X,y,w]
    outputs = [grad_loss]
    """
    X = T.vector(name='X')
    y = T.scalar(name='y')
    w = T.vector(name='w')

    logit = y * T.dot(X, w)
    loss = T.log(1+T.exp(-logit)).mean()
    grad_loss = T.grad(loss, w)

    grad_loss_ = theano.function(
        inputs=[X, y, w],
        outputs=[grad_loss],
        on_unused_input='ignore'
    )

    return grad_loss_


def loss_function():
    """
    return loss function

    Returns
    -------
    return loss function
    inputs = [X,y,w]
    outputs = [loss]
    """
    X = T.matrix(name='X')
    y = T.vector(name='y')
    w = T.vector(name='w')

    logit = y * T.dot(X, w)
    loss = T.log(1+T.exp(-logit))
    loss_ = theano.function(
        inputs=[X, y, w],
        outputs=[loss],
        on_unused_input='ignore'
    )
    return loss_


def return_argmin_index(X):
    """
    return one argmin index

    Parameters
    ----------
    X : list
        loss list

    Returns
    -------
    index
        return one argmin index with random
    """
    X = np.array(X).flatten()
    index_list = np.where(X == np.min(X))[0]
    # if len(index_list) > 1:
    #     print('random choiced')
    index = np.random.choice(index_list)
    return index


def predict(X, y, w):
    logit = np.dot(X, w)
    pred_y = 1/(1+np.exp(-logit))
    # pred_y = [1 if i > 0.5 else 0 for i in pred_y]
    return(roc_auc_score(y, pred_y))
    print(roc_auc_score(y, pred_y))


def predict_by_W(X, y, W):
    J, D = W.shape
    val = 0
    auc_list = np.zeros(J)
    for j in range(J):
        w_j = W[j, :]
        tmp = predict(X, y, w_j)
        auc_list[j] = tmp
        # print('j: {}'.format(tmp))
    return auc_list.mean()


def write_np2csv(X, path):
    np.savetxt(path, X, delimiter=',')


def return_copy_dateset(X, y):
    return X.copy(deep=True), y.copy(deep=True)
    # return date.copy(deep=True)


def rmse_W(W, W_star, axis=None):
    ans = np.sqrt(((W - W_star)**2).mean(axis))
    return ans


def rmse_w(min_w, w_star):
    ans = np.sqrt((min_w - w_star) ** 2).mean()
    return ans


def return_mode(y):
    return stats.mode(y)[0]


def return_answer_matrix(W, X, J):
    N = X.shape[0]
    W = W.copy()

    Y = np.zeros(shape=(N, J))

    for n in range(N):
        X_t = X.iloc[n, :]
        for j in range(J):
            w_j = W[j, :]
            logit = np.dot(X_t, w_j)
            p_1 = 1 / (1 + np.exp(-logit))
            Y[n, j] = np.random.choice([-1, 1], p=[1 - p_1, p_1])
    return Y


def make_random_mask(X, n):
    N, D = X.shape
    mask = np.full(N, False, dtype=bool)
    mask_index = np.random.choice(range(N), size=n, replace=False)
    new_X = X.copy().iloc[mask_index]
    return new_X, n


def change_label(y, prob=1.0):
    p_1 = prob
    for index, label in enumerate(y):
        flag = np.random.choice([-1, 1], p=[1 - p_1, p_1])
        y.iat[index] = y.iat[index]*flag

    return y
