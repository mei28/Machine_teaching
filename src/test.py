# %%
import sys
sys.path.append('src')
import theano.tensor as T
import theano
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from make_Y_w import estimate_w_star

# %%


def predict(X, y, w):
    logit = np.dot(X, w)
    pred_y = T.nnet.sigmoid(logit).eval()
    # pred_y = [1 if i > 0.5 else 0 for i in pred_y]
    return(roc_auc_score(y, pred_y))
    print(roc_auc_score(y, pred_y))


# %%
df = pd.read_csv('output/winequality-red.csv', sep=';')
X = df.drop(df.columns[-1], axis=1)
ms = MinMaxScaler()
X = ms.fit_transform(X)
X = pd.DataFrame(X)
y = [1 if i > 4 else 0 for i in df[df.columns[-1]]]

train_X, test_X, train_y, test_y = train_test_split(
    X, y, shuffle=True
)

# Wの生成
J = 3
eta = 0.01
lambd = 0.01

min_w, w_init = estimate_w_star(X, eta, lambd, J=J, epochs=10)

print('-' * 20)
print(w_init)
print('w_init: {}'.format(predict(test_X, test_y, w_init)))
print(min_w)
print('w_*_train: {}'.format(predict(train_X, train_y, min_w)))
print('w_*_test: {}'.format(predict(test_X, test_y, min_w)))


# %%

# %%
