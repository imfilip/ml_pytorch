"""
Some basic models implementation.
"""

from typing import List
import numpy.typing as npt
import numpy as np

class Perceptron:
    """Perceptron classifier class.

    Attributes:
        w_: List of model weights.
        b_: Bias weight.
        X_: Training set.
        y_: Target values.
        errors_: Errors history from fitting a model.

    """
    w_: List[float] = None
    X_: npt.ArrayLike = None
    y_: npt.ArrayLike = None
    b_: float = None
    errors_: List[float] = None

    def __init__(self, eta: float = 0.01, n_iter: int = 100, random_state: int = 1) -> None:
        """ Class initializer.
        Args:
            eta (float): Learning rate. Defaults to 0.01.
            n_iter (int): Number of iteration of fitting algorithm. Defaults to 100.
            random_state (int): Random seed to random numbers generator. Defaults to 1. 
        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike):
        """Fit training data.
        Args:
            X (ArrayLike): Training set.
            y (ArrayLike): Target value vector.
        Returns:
            Class itself.
        """
        self.X_ = X
        self.y_ = y

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = self.X_.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(self.X_, self.y_):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input - linear predictor.
        Args:
            X (ArrayLike): Training set.
        Returns:
            Linear predictor.
        """
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Predict class label.
        Args:
            X (ArrayLike): Training set.
        Returns:
            Prediction - class label: 0 or 1.
        """
        return np.where(self.net_input(X) >= 0.0, 1, 0)

class AdalineGD:
    """ADAptive LInear NEuron classifier class.

    Attributes:
        w_: List of model weights.
        b_: Bias weight.
        X_: Training set.
        y_: Target values.
        losses_: Errors history from fitting a model.

    """
    w_: List[float] = None
    X_: npt.ArrayLike = None
    y_: npt.ArrayLike = None
    b_: float = None
    losses_: List[float] = None

    def __init__(self, eta: float = 0.01, n_iter: int = 100, random_state: int = 1) -> None:
        """ Class initializer.
        Args:
            eta (float): Learning rate. Defaults to 0.01.
            n_iter (int): Number of iteration of fitting algorithm. Defaults to 100.
            random_state (int): Random seed to random numbers generator. Defaults to 1. 
        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike):
        """Fit training data.
        Args:
            X (ArrayLike, shape = [n_examples, n_features]): Training set.
            y (ArrayLike, shape = [n_examples, 1]): Target value vector.
        Returns:
            Class itself.
        """
        self.X_ = X
        self.y_ = y

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = self.X_.shape[1])
        self.b_ = np.float_(0.)
        self.losses_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            losses = self.cost_function(self.X_, self.y_)
            
            self.w_ += self.eta * 2.0 * np.matmul(X.T, y - output) / X.shape[0]
            self.b_ += self.eta * 2.0 * (y - output).mean()
            self.losses_.append(losses)
        return self

    def cost_function(self, X, y):
        """Calculate cost function.
        Args:
            X (ArrayLike): Training set.
            y (ArrayLike): Target values.
        Returns:
            (float). Cost function.
        """
        return ((y - (np.matmul(X, self.w_) + self.b_))**2).mean()

    def net_input(self, X):
        """Calculate net input - linear predictor.
        Args:
            X (ArrayLike): Training set.
        Returns:
            Linear predictor.
        """
        return np.matmul(X, self.w_) + self.b_

    def activation(self, X):
        """Compute linear activation
        Args:
            X (ArrayLike): Training set.
        """
        return X

    def predict(self, X):
        """Predict class label.
        Args:
            X (ArrayLike): Training set.
        Returns:
            Prediction - class label: 0 or 1.
        """
        return np.where(self.activation(self.net_input(X))>= 0.5, 1, 0)

if __name__ == "__main__":

    test = Perceptron(n_iter = 1000)
    print(test.eta)
    print(test.n_iter)
    print(test.random_state)

    X = np.ones((10, 2))
    y = X[:, 0]

    test.fit(X, y)
    print(test.X_)
    print(test.y_)
    print(test.w_)
