import numpy as np
import pandas as pd


def ic_t_stat(
    individual,
    toolbox_compile,
    target,
    market_constraint,
    time_constraint,
    scope,
    require_cols,
):
    func = toolbox_compile(expr=individual)
    scope["func"] = func
    yp = eval("func({})".format(", ".join(require_cols)), scope)
    yp = yp[market_constraint]
    yp = yp[yp.index >= time_constraint].rank(axis=1)
    target = target[market_constraint]
    target = target[target.index >= time_constraint].rank(axis=1)
    ic_t = target.corrwith(yp, axis=1)
    result = abs(ic_t.mean()) * 1
    if (yp == 0).sum().sum() >= 0.2 * yp.count().sum():  # 為0的數據占有值的數據20%以上
        return (-10,)
    elif (yp.std(axis=1) == 0).sum() > 0:  # 一期標準差為0
        return (-10,)
    elif np.isnan(result):
        return (-10,)
    else:
        return (result,)


def rank_return(
    individual,
    toolbox_compile,
    target,
    market_constraint,
    time_constraint,
    scope,
    require_cols,
):
    func = toolbox_compile(expr=individual)
    scope["func"] = func
    yp = eval("func({})".format(", ".join(require_cols)), scope)
    yp = yp[market_constraint]
    yp = yp[yp.index >= time_constraint]
    target_test = target[market_constraint]
    target_test = target_test[target_test.index >= time_constraint]
    mean_return_pos = (
        ((target_test[yp.rank(axis=1, ascending=False) <= 50] + 1) * (1 - 0.001425))
        * (1 - 0.004425)
    ).mean(axis=1)
    mean_return_pos_nacount = pd.isna(mean_return_pos).sum()
    mean_return_pos = mean_return_pos.prod()
    mean_return_neg = (
        ((target_test[yp.rank(axis=1, ascending=True) <= 50] + 1) * (1 - 0.001425))
        * (1 - 0.004425)
    ).mean(axis=1)
    mean_return_neg_nacount = pd.isna(mean_return_neg).sum()
    mean_return_neg = mean_return_neg.prod()
    return_all_nan_count = (target_test.count(axis=1) == 0).sum()
    if (
        mean_return_pos_nacount > return_all_nan_count
        or mean_return_neg_nacount > return_all_nan_count
    ):
        return (-10,)
    else:
        return (max(mean_return_pos, mean_return_neg),)
