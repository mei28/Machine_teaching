# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import theano.tensor as T
import theano
from sklearn.metrics import roc_auc_score


theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"
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


def make_grad_loss_matrix(X, y, w):
    grad_loss_matrix = np.zeros((X.shape[0], w.shape[0]))
    grad_loss = grad_loss_function()
    for i in range(X.shape[0]):
        grad_loss_ = grad_loss(X.iloc[i], y.iloc[i], w)
        a = np.array(grad_loss_).flatten()
        grad_loss_matrix[i] = a

    return grad_loss_matrix


# %%


def grad_loss_function():
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

# %%


def loss_function():
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

# %%


def omniscient_teacher(w_t, w_, X, y, eta=0.01):
    grad_loss_ = make_grad_loss_matrix(X, y, w_t)
    choicer = omniscient_teacher_function(eta)
    index = choicer(grad_loss_, w_t, w_)
    index = np.array(index)
    index = index[0]
    X_t = X.iloc[index]
    y_t = y.iloc[index]
    # print(index)
    return X_t, y_t
# %%


def omniscient_teacher_function(eta=0.01):
    grad_loss = T.matrix(name='grad_loss')
    w_ = T.vector(name='w_')
    w_t = T.vector(name='w_t')

    first = (eta ** 2) * (grad_loss ** 2).sum(axis=1)
    second = -2 * eta * (T.dot(grad_loss, w_t - w_))
    loss = first + second
    argmin = T.argmin(loss)
    function = theano.function(
        inputs=[grad_loss, w_t, w_],
        outputs=[argmin],
        on_unused_input='ignore'
    )
    return function

# %%


def surrogate_teacher(w_t, w_, X, y, eta=0.01):
    grad_loss_ = make_grad_loss_matrix(X, y, w_t)
    loss = loss_function()
    loss_t = np.array(loss(X, y, w_t)).flatten()
    loss__ = np.array(loss(X, y, w_)).flatten()

    choicer = surrogate_teacher_function(eta)
    index = choicer(grad_loss_, loss_t, loss__)
    index = np.array(index).flatten()
    index = index[0]

    X_t = X.iloc[index]
    y_t = y.iloc[index]
    print(index)
    return X_t, y_t
# %%


def surrogate_teacher_function(eta=0.01):
    grad_loss_ = T.matrix(name='grad_loss')
    loss_t = T.vector(name='loss_t')
    loss__ = T.vector(name='loss__')
    first = (eta**2)*(grad_loss_ ** 2).sum()
    second = -2*eta*(loss_t-loss__)
    argmin_ = T.argmin(first + second)
    t = theano.function(
        inputs=[grad_loss_, loss_t, loss__],
        outputs=[argmin_],
    )
    return t

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


def student_model(lambd, w_init):
    X = T.vector(name='X')
    y = T.scalar(name='y')
    w = theano.shared(w_init, name='w')

    logit = T.nnet.sigmoid(T.dot(X, w))
    xent = T.nnet.binary_crossentropy(logit, y)
    loss = xent + lambd * (w ** 2).sum() / 2

    print('start: compile model')
    params = [w]
    updates = SGD(params=params).updates(loss)

    student = theano.function(
        inputs=[X, y],
        outputs=[loss, w],
        updates=updates,
        on_unused_input='ignore'
    )

    print('complete: compile model')

    return student
# %%


def predict(X, y, w):
    logit = np.dot(X, w)
    pred_y = T.nnet.sigmoid(logit).eval()
    # pred_y = [1 if i > 0.5 else 0 for i in pred_y]
    return(roc_auc_score(y, pred_y))
    print(roc_auc_score(y, pred_y))
# %%


def main():
    # ワーカ数
    J = 10
    df = pd.read_csv('output/weebil_vespula.csv')

    X = df.drop('Spe', axis=1)
    y = df['Spe']

    # train_X, test_X, train_y, test_y = train_test_split(
    #     X, y, shuffle=True, random_state=88
    # )
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, shuffle=True
    )

    # np.random.seed(seed=28)
    lambd = 0.01
    eta = 0.01
    training_epochs = 1000
    w_init = np.random.normal(loc=0.0, scale=lambd, size=train_X.shape[1])

    train = model(train_X, train_y, lambd, w_init)

    min_w = 9999
    for t in range(training_epochs):
        loss, w = train(train_X, train_y)

        # print('{}: loss: {}'.format(t, loss))
        min_w = w

    predict(test_X, test_y, min_w)

    print('-' * 20)

    training_epochs = 5
    surrogate_student = student_model(lambd, w_init)
    w_t = w_init
    surrogate_w = 0
    for t in range(training_epochs):
        X_t, y_t = surrogate_teacher(w_t, min_w, train_X, train_y, eta=0.01)
        loss, w = surrogate_student(X_t, y_t)
        # print('{}: loss: {}'.format(t, loss))
        surrogate_w = w

    random_w = 0
    random_select = student_model(lambd, w_init)
    for t in range(training_epochs):
        index = np.random.randint(0, train_X.shape[0])

        _, random_w = random_select(train_X.iloc[index], train_y.iloc[index])
    omni_student = student_model(lambd, w_init)
    w_t = w_init
    omni_w = 99999
    for t in range(training_epochs):
        X_t, y_t = omniscient_teacher(w_t, min_w, train_X, train_y, eta=0.01)
        loss, w = omni_student(X_t, y_t)
        omni_w = w

    print('-' * 20)
    print('w_init: {}'.format(predict(test_X, test_y, w_init)))
    print('min_w: {}'.format(predict(test_X, test_y, min_w)))
    print('random_w: {}'.format(predict(test_X, test_y, random_w)))
    print('surrogate_w: {}'.format(predict(test_X, test_y, surrogate_w)))
    print('omni_w: {}'.format(predict(test_X, test_y, omni_w)))


main()
# %%

if __name__ == "__main__":
    main()

# %%
# %%

# %%
