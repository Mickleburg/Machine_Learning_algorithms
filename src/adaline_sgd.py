import numpy as np


class AdalineSGD:
    """Адаптивный линейный нейрон (стохастический градиентный спуск).

    Параметры
    ---------
    eta : float, default=0.01
        Скорость обучения (0 < eta <= 1).
    n_iter : int, default=10
        Количество эпох обучения.
    shuffle : bool, default=True
        Перемешивать обучающие данные перед каждой эпохой.
    random_state : int, default=None
        Начальное значение генератора случайных чисел.
    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.losses_ = []
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = [self._update_weights(xi, target) for xi, target in zip(X, y)]
            self.losses_.append(np.mean(losses))
        return self

    def partial_fit(self, X, y):
        """Дообучение без повторной инициализации весов (онлайн-обучение)."""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float64(0.)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_ += self.eta * 2.0 * xi * error
        self.b_ += self.eta * 2.0 * error
        return error ** 2

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
