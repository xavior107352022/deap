import operator

import numpy as np


def add(x, y):
    return operator.add(x, y)


def sub(x, y):
    return operator.sub(x, y)


def mul(x, y):
    return operator.mul(x, y)


def div(x, y):
    result = (x / y).replace([np.inf, -np.inf], np.nan)
    result[(x == 0) & (y == 0)] = 0
    return result


def mulint(x, y):
    return y * x


# TS
# def pct_chg_test(t, t1):
#     return (t - t1) / abs(t1)
# tp1_list = [3, 1, -3, -1, -3, -1, 3, 1]
# t_list = [1, 3, 1, 3, -1, -3, -1, -3]
# tf1_list = [3, 1, -3, -1, -3, -1, 3, 1]
# [pct_chg_test(t, t1) for t, t1 in zip(t_list, t1_list)]
# A = pd.DataFrame([t1_list, t_list, tf1_list])
# pct_chg(A, 1)
# pct_chg(A, -1)
def delay(x, t):
    return x.shift(t)


def delta(x, t):
    if t >= 0:
        return x - delay(x, t)
    else:
        return delay(x, t) - x


def pct_chg(x, t):
    if t >= 0:
        return delta(x, t) / delay(x, t).abs()
    else:
        return delta(x, t) / x.abs()


def ts_sum(x, t):
    return x.rolling(t).sum()


def ts_mean(x, t):
    return x.rolling(t).mean()


def ts_std(x, t):
    return x.rolling(t).std()


def ts_min(x, t):
    return x.rolling(t).min()


def ts_max(x, t):
    return x.rolling(t).max()


def ts_corr(x, y, t):
    x_rolling = x.rolling(t, t)
    y_rolling = y.rolling(t, t)
    X = (x - x_rolling.mean()) / x_rolling.std()
    Y = (y - y_rolling.mean()) / x_rolling.std()
    return (X * Y).rolling(t, t).sum() / (t - 1)
