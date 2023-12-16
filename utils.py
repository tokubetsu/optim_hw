import sympy as sp
from abc import abstractmethod
import numpy as np
from visualize import OptimVisualizer
from itertools import combinations


class BaseOptimizer:
    def __init__(self,
                 xk: np.ndarray,
                 max_iter: int = 1000,
                 zero: int = 10 ** -9):

        self._xk = xk

        self._max_iter = max_iter
        self._zero = zero

        self._xs = []
        self._fs = []

        self._k = 0

    @property
    def x_history(self) -> list[np.ndarray]:
        return self._xs

    @property
    def f_history(self) -> list[np.ndarray]:
        return self._fs

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def descend(self):
        pass


class Optimizer(BaseOptimizer):
    def __init__(self,
                 x1: sp.Symbol,
                 x2: sp.Symbol,
                 f: sp.Add,
                 xk: np.ndarray,
                 max_iter: int = 1000,
                 zero: int = 10 ** -9,
                 init_f: bool = True):

        super().__init__(xk=xk, max_iter=max_iter, zero=zero)

        if init_f:
            self._fx = self._get_fx(x1, x2, f)
            self._dx = self._get_dx(x1, x2, f)

    @staticmethod
    def _get_fx(x1, x2, f):
        return sp.lambdify((x1, x2), f)

    @staticmethod
    def _get_dx(x1, x2, f):
        return [
            sp.lambdify((x1, x2), sp.diff(f, x1)),
            sp.lambdify((x1, x2), sp.diff(f, x2))
        ]

    def count_func(self, args: np.ndarray) -> np.ndarray:
        if len(args.shape) == 1:
            return self._fx(*args)
        else:
            return np.apply_along_axis(lambda x: self._fx(x[0], x[1]), 0, args)

    def _count_diff_x(self):
        res = []
        for func in self._dx:
            res.append(func(*self._xk))
        return np.array(res)

    def _stop_criterion(self):
        df = self._count_diff_x()
        norm = np.linalg.norm(df, ord=2)
        if self._k > 0:
            diff = self.count_func(self.x_history[-2]) - self.count_func(self._xk)
        else:
            diff = norm
        return (norm <= self._zero) or (diff <= self._zero)

    @abstractmethod
    def _update_xk(self):
        pass

    def step(self):
        self._xs.append(self._xk)
        self._fs.append(self.count_func(self._xk))

        if self._stop_criterion():
            return True

        self._update_xk()
        self._k += 1
        return False

    def descend(self):
        done = False
        n_iter = 0
        while (not done) & (n_iter <= self._max_iter):
            n_iter += 1
            done = self.step()


class GradientDescent(Optimizer):
    def __init__(self,
                 x1: sp.Symbol,
                 x2: sp.Symbol,
                 f: sp.Add,
                 xk: np.ndarray,
                 alpha,
                 max_iter: int = 1000,
                 zero: int = 10 ** -9):

        super().__init__(x1, x2, f, xk, max_iter=max_iter, zero=zero)
        self._alpha = alpha

    @property
    def alpha(self):
        return self._alpha

    def _update_xk(self):
        dx = self._count_diff_x()
        self._xk = self._xk - self.alpha * dx


class GradientArmijoDescent(GradientDescent):
    def __init__(self, x1, x2, f, xk: np.ndarray, zero: int = 10 ** -9, max_iter: int = 1000):
        super().__init__(x1,x2, f, xk, 1, max_iter=max_iter, zero=zero)

        self._cur_alpha = 1
        self._theta = 0.5
        self._epsilon = 0.5

    def __check_condition(self):
        dx = self._count_diff_x()
        left_x = self._xk + self._cur_alpha * (-dx)
        left = self.count_func(left_x)

        right_part = np.dot(dx, (-dx))
        right = self.count_func(self._xk) + self._epsilon * self._cur_alpha * right_part
        return left <= right

    def __find_alpha(self):
        self._cur_alpha = self._alpha
        got = self.__check_condition()
        while not got:
            self._cur_alpha = self._cur_alpha * self._theta
            got = self.__check_condition()

    @property
    def alpha(self):
        self.__find_alpha()
        return self._cur_alpha


class NewtonMethod(Optimizer):
    def __init__(self, x1, x2, f, xk: np.ndarray, zero: int = 10 ** -9, max_iter: int = 1000):
        super().__init__(x1, x2, f, xk, max_iter=max_iter, zero=zero)

        self.__dxx = self.__get_dxx(x1, x2, f)

    @staticmethod
    def __get_dxx(x1, x2, f):
        diff = (sp.diff(f, x1), sp.diff(f, x2))
        dxx = []
        for x in (x1, x2):
            dxx.append([])
            for df in diff:
                dxx[-1].append(sp.lambdify((x1, x2), sp.diff(df, x)))
        return dxx

    def __count_diff_xx(self):
        res = []
        for row in self.__dxx:
            res.append([])
            for func in row:
                res[-1].append(func(*self._xk))
        return np.array(res)

    def _update_xk(self):
        dxx = self.__count_diff_xx()
        dx = self._count_diff_x()
        stp = np.dot(np.linalg.inv(dxx), dx)
        self._xk = self._xk - stp


class ConjugateGradientDescent(Optimizer):
    def __init__(self, a: np.ndarray, b: np.ndarray, xk: np.ndarray, zero: int = 10 ** -9, max_iter: int = 1000):
        super().__init__(None, None, None, xk, zero=zero, max_iter=max_iter, init_f=False)

        self.__a = a
        self.__b = b

        self.__dk = 0
        self.__beta = 0
        self.__alpha = 0

    def count_func(self, args: np.ndarray) -> np.ndarray:
        inner = np.dot(np.dot(args.T, self.__a), args)
        if len(args.shape) > 1:
            inner = np.diag(inner)
        return inner + self.__b

    def _count_diff_x(self):
        diff = np.dot((self.__a.T + self.__a), self._xk)
        return diff

    def __update_dk(self):
        self.__dk = - self._count_diff_x() + self.__beta * self.__dk

    def __update_beta(self):
        mid = np.dot(self.__a, self.__dk)
        num = np.dot(mid, self._count_diff_x())
        den = np.dot(mid, self.__dk)
        self.__beta = num / den

    def __update_alpha(self):
        num = np.dot(2 * np.dot(self.__a, self._xk) + self.__b, self.__dk)
        den = 2 * np.dot(np.dot(self.__a, self.__dk), self.__dk)
        self.__alpha = - num / den

    def _update_xk(self):
        if self._k >= 1:
            self.__update_beta()

        self.__update_dk()
        self.__update_alpha()

        self._xk = self._xk + np.dot(self.__alpha, self.__dk)


class QuadraticLossDescent(Optimizer):
    def __init__(self, x1, x2, f, xk: np.ndarray, f_cond, k, ck, alpha, zero: int = 10 ** -9, max_iter: int = 1000):
        super().__init__(x1, x2, f, xk, zero=zero, max_iter=max_iter)

        self.__alpha = alpha

        self.__f = f
        self.__x1 = x1
        self.__x2 = x2
        self.__f_cond = f_cond

        self.f_cond_x = sp.lambdify((x1, x2), f_cond)
        self.__ck = sp.lambdify(k, ck)

        self.__fk = self.count_func(self._xk)

    def __get_phi(self):
        cur_ck = self.__ck(self._k)
        phi = self.__f + cur_ck / 2 * (self.__f_cond ** 2)
        return phi

    def _stop_criterion(self):
        diff_f = self.__fk - self.count_func(self._xk)
        diff_c = self.f_cond_x(*self._xk)

        if (diff_f <= self._zero) and (diff_c <= self._zero):
            return True

    def _update_xk(self):
        phi = self.__get_phi()

        inner_gd = GradientDescent(self.__x1, self.__x2, phi, self._xk, alpha=self.__alpha)
        inner_gd.descend()

        xs = inner_gd.x_history
        self._xs.extend(xs[:-1])
        self._fs.extend(self.count_func(np.array(xs[:-1]).T))

        self._xk = xs[-1]
        self.__fk = self.count_func(self._xk)


class SpaceSet:
    def __init__(self, conditions):
        self.conditions = conditions
        self.__get_special_points()

    def __get_special_points(self):
        points = []

        for condition in combinations(self.conditions, r=2):
            a = np.vstack(condition)[:, :-2]
            b = np.vstack(condition)[:, -2:-1].reshape(-1, )
            if np.linalg.det(a) != 0:
                points.append(np.linalg.solve(a, b))

        points = np.vstack(points)
        self.points = points[np.lexsort(points.T)]

    def __contains__(self, item):
        f = np.sign(np.dot(self.conditions[:, :-2], item).reshape(-1, 1) - self.conditions[:, -2:-1])
        not_lines = np.where(f != 0)[0]
        inner = np.equal(f[not_lines], self.conditions[not_lines, -1:]).all()
        return inner

    def projection(self, item):
        if item in self:
            return item

        else:
            y_zeros = self.conditions[np.where(self.conditions[:, 1] == 0)]
            y_zeros_p = np.hstack([(y_zeros[:, 2] / y_zeros[:, 0]).reshape(-1, 1),
                                   np.tile(item[1], y_zeros.shape[0]).reshape(-1, 1)])

            x_zeros = self.conditions[np.where(self.conditions[:, 1] != 0)]
            x_zeros_p = np.hstack([np.tile(item[0], x_zeros.shape[0]).reshape(-1, 1),
                                   (x_zeros[:, 2] / x_zeros[:, 1]).reshape(-1, 1)])

            variants = np.vstack([self.points, y_zeros_p, x_zeros_p])
            variants = variants[np.apply_along_axis(lambda x: x in self, 1, variants)]
            p_dist = np.linalg.norm(variants - item, ord=2, axis=1)
            return variants[np.argmin(p_dist)]


class ConditionalGradientDescent(GradientArmijoDescent):
    def __init__(self, x1, x2, f, xk: np.ndarray, conditions, zero: int = 10 ** -9, max_iter: int = 1000, crit=0.1):
        super().__init__(x1, x2, f, xk, max_iter=max_iter, zero=zero)
        self._set = SpaceSet(conditions=conditions)
        self.__crit = 0.1

    def _stop_criterion(self):
        if self._k > 0:
            prev = self.x_history[-2]
            return np.linalg.norm(prev - self._xk, ord=2) < self.__crit
        else:
            return False

    def _update_xk(self):
        dx = self._count_diff_x()
        new_x = self._xk - self.alpha * dx
        self._xk = self._set.projection(new_x)
