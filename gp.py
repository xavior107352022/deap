# %%
import numpy as np
import pandas as pd

from custom_fitness import ic_t_stat
from custom_operators import (
    add,
    delay,
    delta,
    div,
    mul,
    pct_chg,
    sub,
    ts_corr,
    ts_max,
    ts_mean,
    ts_min,
    ts_std,
)
from deap import algorithms, base, creator, gp, tools
from utils import process_cols

# %%
# 整理資料
sub_list = ["\(元\)", "\(千股\)", "\(千元\)", "\(千股\)", "\%"]
df_price = pd.read_csv("data/price.txt", sep="\t", encoding="cp950")
df_price["證券代碼"] = (
    df_price["證券代碼"].astype("str").str.replace(" ", "").replace("-", "")
)
df_price["市場別"] = df_price["市場別"].astype("str").str.replace(" ", "")
df_price["開盤價(元)"] = (
    df_price["開盤價(元)"]
    .astype("str")
    .str.replace(" ", "")
    .replace("-", np.nan)
    .astype("float")
)
df_price.columns = [process_cols(x, sub_list) for x in df_price.columns]

df_chip = pd.read_csv("data/chip.txt", sep="\t", encoding="cp950")
df_chip = df_chip.rename({"證券名稱": "證券代碼"}, axis=1)
df_chip["證券代碼"] = (
    df_chip["證券代碼"].astype("str").str.replace(" ", "").replace("-", "")
)
for i in range(3, 10):
    df_chip.iloc[:, i] = (
        df_chip.iloc[:, i]
        .astype("str")
        .str.replace(" ", "")
        .replace("-", np.nan)
        .astype("float")
    )
df_chip.columns = [process_cols(x, sub_list) for x in df_chip.columns]

base_cols = ["年月日", "證券代碼"]
same_cols = list((set(df_price.columns) & set(df_chip.columns)) - set(base_cols))
df_price_chip = pd.merge(
    left=df_price, right=df_chip.drop(same_cols, axis=1), on=base_cols, how="left"
)
del df_chip, df_price

df_price_chip = df_price_chip.fillna(0)
ratios_cols = []
for x in ["外資買賣超", "投信買賣超", "自營買賣超"]:
    df_price_chip[x + "比率"] = div(df_price_chip[x], df_price_chip["成交量"])
    ratios_cols.append(x + "比率")
df_pivot = df_price_chip.pivot(index=base_cols[0], columns=base_cols[1])
require_cols = [
    "開盤價",
    "最高價",
    "最低價",
    "收盤價",
    "成交量",
    "成交值",
    "流通在外股數",
    "外資買賣超",
    "投信買賣超",
    "自營買賣超",
    "外資總投資股率",
    "投信持股率",
    "自營持股率",
] + ratios_cols

for x in ["市場別"] + require_cols:
    locals()[x] = df_pivot[x]

target = delay(df_pivot["收盤價"] / df_pivot["開盤價"].replace(0, np.nan) - 1, -1)
market_constraint = df_pivot["市場別"] == "TSE"
time_constraint = 20050101
del df_pivot, df_price_chip
# %%
# gp 程式開始
df_type = pd.core.frame.DataFrame
pset = gp.PrimitiveSetTyped(
    "MAIN", [df_type] * len(require_cols), df_type, require_cols
)
# normal operator
pset.addPrimitive(add, [df_type, df_type], df_type)
pset.addPrimitive(sub, [df_type, df_type], df_type)
pset.addPrimitive(mul, [df_type, df_type], df_type)
pset.addPrimitive(div, [df_type, df_type], df_type)
# time series operator
pset.addPrimitive(delay, [df_type, int], df_type)
pset.addPrimitive(delta, [df_type, int], df_type)
pset.addPrimitive(pct_chg, [df_type, int], df_type)
pset.addPrimitive(ts_mean, [df_type, int], df_type)
pset.addPrimitive(ts_std, [df_type, int], df_type)
pset.addPrimitive(ts_max, [df_type, int], df_type)
pset.addPrimitive(ts_min, [df_type, int], df_type)
pset.addPrimitive(ts_corr, [df_type, df_type, int], df_type)
# terminals
pset.addTerminal(3, int)
pset.addTerminal(5, int)
pset.addTerminal(10, int)
pset.addTerminal(20, int)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

scope = {k: v for k, v in locals().items() if k in require_cols}
toolbox = base.Toolbox()
int_terminal_types = [int]
toolbox.register(
    "expr",
    gp.genHalfAndHalf,
    pset=pset,
    min_=0,
    max_=2,
    terminal_types=int_terminal_types,
)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register(
    "evaluate",
    ic_t_stat,
    toolbox_compile=toolbox.compile,
    target=target,
    market_constraint=market_constraint,
    time_constraint=time_constraint,
    scope=scope,
    require_cols=require_cols,
)
toolbox.register("select", tools.selTournament, tournsize=10)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register(
    "expr_mut",
    gp.genFull,
    min_=0,
    max_=1,
    terminal_types=int_terminal_types,
)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.nanmean)
stats.register("std", np.nanstd)
stats.register("min", np.nanmin)
stats.register("max", np.nanmax)
# %%
# gp 主程式
pop = toolbox.population(n=20)
hof = tools.HallOfFame(5)
population, logbook = algorithms.eaSimple(
    pop, toolbox, 0.3, 0.3, 3, stats, halloffame=hof
)
df_result = pd.DataFrame()
df_result["cal"] = [x.__str__() for x in hof]
df_result["fitness"] = [x.fitness.values[0] for x in hof]
df_result.to_excel("df_result.xlsx", index=False)
# %%
