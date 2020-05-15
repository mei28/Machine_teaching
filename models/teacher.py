import numpy as np
import pandas as pd
import theano
import theano.tensor as T


class Teacher():
    def __init__(self, min_w, W, N, alpha=0.01):
        """
        teacher base class

        Parameters
        ----------
        min_w : numpy
            true model parameter
        W : numpy
            worker's parameter
        N : int
            the number of text book pool
        alpha : float, optional
            learning rate, by default 0.01
        """
        super().__init__()
        self.min_w = min_w.copy()
        self.W = W.copy()
        self.alpha = alpha
        J, D = W.shape
        self.mask = np.full((J, N), True, dtype=bool)

    def make_grad_loss_matrix(self, X, y, w):
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
        grad_loss = self.grad_loss_function()
        for i in range(X.shape[0]):
            grad_loss_ = grad_loss(X.iloc[i], y.iloc[i], w)
            a = np.array(grad_loss_).flatten()
            grad_loss_matrix[i] = a

        return grad_loss_matrix

    def return_textbook(self):
        pass

    def grad_loss_function(self):
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

    def loss_function(self):
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

    def return_argmin_index(self, X):
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
        # if len(index_list) < 1:
        #     print('Not exist')
        index = np.random.choice(index_list)
        return index

    def update_w_j(self, X_t, y_t, w_j):
        """
        update w_j

        Parameters
        ----------
        X_t : pandas
            text book
        y_t : pandas
            true label
        w_j : numpy
            worker's parameter

        Returns
        -------
            return updated w_j
        """
        grad_loss = self.grad_loss_function()
        grad_loss_ = grad_loss(X_t, y_t, w_j)[0]

        w_j = w_j - self.alpha * grad_loss_
        return w_j

    def show_textbook(self, X, y=None, N=1, option='None'):
        J, D = self.W.shape
        if y is not None:
            # print('use y')
            y = y
        else:
            if option == 'mix':
                # print('use mix')
                y = self.decision_Y_by_mix(X, self.W)
            elif option == 'majority':
                # print('use majority')
                y = self.decision_Y_by_majority(X, self.W)
            elif option == 'prob':
                # print('use prob')
                y = self.decision_Y_by_prob(X, self.W)
            elif option == 'min_w':
                # print('use w_star')
                y = self.predict_y(X, self.min_w)
            else:
                print('default: min_w')
                y = self.predict_y(X, self.min_w)

        for j in range(J):
            w_j = self.W[j, :]
            for n in range(N):
                mask = self.mask[j]
                X_j, y_j = X[mask], y[mask]

                X_t, y_t = self.return_textbook(
                    X_j, y_j, w_j, self.min_w)
                index = np.where(X == X_t)[0][0]
                self.mask[j, index] = False
                w_j_new = self.update_w_j(X_t, y_t, w_j)
                self.W[j, :] = w_j_new

    def decision_Y_by_majority(self, X, W):
        """
        return label from worker decision

        Parameters
        ----------
        X : pandas
            text book pool

        Returns
        -------
        y pandas
            decision by majority
        """
        N = X.shape[0]
        J = self.J

        y = np.zeros(shape=(N))
        Y = return_answer_matrix(W, X, J=J)

        for n in range(N):
            Y_n = Y[n, :]
            y[n] = return_mode(Y_n)
        y = pd.Series(y)
        return y

    def decision_Y_by_prob(self, X, W):
        N = X.shape[0]
        J = self.J

        y = np.zeros(shape=N)
        Y = return_answer_matrix(W, X, J)

        for i, tmp in enumerate(Y):
            y[i] = np.random.choice(tmp)
        y = pd.Series(y)
        return y

    def decision_Y_by_mix(self, X, W):
        N = X.shape[0]
        J = self.J

        y = np.zeros(shape=N)
        Y = return_answer_matrix(W, X, J)

        threshhold = 0.2
        for i, tmp in enumerate(Y):
            sum_num = tmp.sum()
            if J * threshhold < sum_num and sum_num < (1 - threshhold) * J:
                y[i] = np.random.choice(tmp)
            else:
                y[i] = return_mode(tmp)
        y = pd.Series(y)
        return y

    def predict_y(self, X, w):
        """
        return predicted y
        Parameters
        ----------
        X : pandas
        w : numpy
            model parameter

        Returns
        -------
        return predicted y pandas
        y = {-1,1}
        for logistic loss
        """
        N, D = X.shape
        y = np.zeros(N)
        for n in range(N):
            logit = np.dot(X.iloc[n], w)
            p_1 = 1 / (1 + np.exp(-logit))
            y[n] = 1 if p_1 > 0.5 else -1
        y = pd.Series(y)
        return y
