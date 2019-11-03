# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import theano.tensor as T
import theano
from sklearn.metrics import roc_auc_score

# %%


class Optimizer(object):
    def __init__(self, params=None):
        if params is None:
            return NotImplementedError()
        self.params = params

    def updates(self, loss=None):
        if loss is None:
            return NotImplementedError()

        self.updates = OrderedDict()
        self.gparams = [T.grad(loss, param) for param in self.params]


# %%
class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, params=None):
        super(SGD, self).__init__(params=params)
        self.learning_rate = 0.01

    def updates(self, loss=None):
        super(SGD, self).updates(loss=loss)

        for param, gparam in zip(self.params, self.gparams):
            self.updates[param] = param - self.learning_rate * gparam

        return self.updates
# %%


def model(X, y, lambd, w_init):

    X = T.matrix(name="X")
    y = T.vector(name="y")
    w = theano.shared(w_init, name="w")

    logit = T.nnet.sigmoid(T.dot(X, w))
    xent = T.nnet.binary_crossentropy(logit, y)
    loss = xent.mean() + lambd * (w ** 2).sum()/2

    params = [w]
    updates = SGD(params=params).updates(loss)

    print('start: compile model')

    train = theano.function(
        inputs=[X, y],
        outputs=[loss, w],
        updates=updates,
        on_unused_input='ignore'
    )

    print('complete: compile model')

    return train
# %%


def predict(X, y, w):
    logit = np.dot(X, w)
    pred_y = T.nnet.sigmoid(logit).eval()
    # pred_y = [1 if i > 0.5 else 0 for i in pred_y]
    print(roc_auc_score(y, pred_y))
# %%


def main():
    # ワーカ数
    J = 10
    df = pd.read_csv('output/weebil_vespula.csv')

    X = df.drop('Spe', axis=1)
    y = df['Spe']

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, shuffle=True, random_state=28
    )

    np.random.seed(seed=28)
    lambd = 0.01
    eta = 0.01
    training_epochs = 100
    w_init = np.random.normal(loc=0.0, scale=lambd, size=train_X.shape[1])

    train = model(train_X, train_y, lambd, w_init)

    min_w = 9999
    for t in range(training_epochs):
        loss, w = train(train_X, train_y)
        print('{}: loss: {}'.format(t, loss))
        min_w = w

    predict(test_X, test_y, min_w)

    pred_random = [1 if np.random.rand(
    ) > 0.5 else 0 for i in range(len(test_y))]
    print(roc_auc_score(test_y, pred_random))
    pass


main()
# %%

if __name__ == "__main__":
    main()
