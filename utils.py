import gc

import numpy as np
import pandas as pd
import pyodbc
import regex as re
from scipy import stats


def pd_read_mssql_data(sql_query, server="XAVIOR\SQLEXPRESS", database="cmoney"):
    con = pyodbc.connect(
        "DRIVER={SQL Server};SERVER=" + server + ";DATABASE=" + database
    )
    df = pd.read_sql_query(sql_query, con)
    con.close()
    return df


def pd_descriptive_stat(df, save_path):
    df_describe = df.describe().T
    df_describe["nonna%"] = df_describe["count"] / len(df)
    df_describe = df_describe.sort_values("nonna%")
    if save_path:
        df_describe.to_csv(save_path, encoding="cp950")
    else:
        pass


def cal_factor_return_df(df_pivot, cal_factor_dict):
    basefactor = cal_factor_dict["basefactor"]
    cal = cal_factor_dict["cal"]
    params = cal_factor_dict["params"]
    if cal == "pctchg":
        period = params["period"]
        df_pivot_shift = df_pivot.shift(period)
        df_pivot_result = ((df_pivot - df_pivot_shift) / abs(df_pivot_shift)).replace(
            [np.inf, -np.inf], np.NaN
        )
        df_pivot_result[(df_pivot == 0) & (df_pivot_result == 0)] = 0
    elif cal == "chg":
        period = params["period"]
        df_pivot_shift = df_pivot.shift(period)
        df_pivot_result = df_pivot - df_pivot_shift
    return df_pivot_result


def pivot_melt_cal_factor(df, index, columns, cal_factor_dict):
    basefactor = cal_factor_dict["basefactor"]
    factorname = cal_factor_dict["factorname"]
    df_pivot = df.pivot(index=index, columns=columns, values=basefactor)
    df_pivot = cal_factor_return_df(df_pivot, cal_factor_dict)
    df_melt = df_pivot.reset_index().melt(
        id_vars=df_pivot.index.name,
        value_name=factorname,
        var_name=df_pivot.columns.name,
    )
    return pd.merge(left=df, right=df_melt, on=[index, columns], how="left")


def make_day_m(year_month):
    year = year_month[:4]
    month = year_month[4:]
    if year <= "2012":
        if month in ["03"]:
            return [year_month + "11", year + "04" + "01"]  # 3/31
        elif month in ["08"]:
            return [year_month + "11", year + "09" + "01"]  # 8/31
        elif month in ["10"]:
            return [year_month + "11", year + "11" + "01"]  # 10/31
        elif month in ["04"]:
            return [year_month + "11", year + "05" + "01"]  # 4/30
        else:
            return year_month + "11"
    else:
        if month in ["08", "11"]:
            return [year_month + "11", year_month + "15"]
        elif month in ["03"]:
            return [year_month + "11", year + "04" + "01"]
        elif month in ["05"]:
            return [year_month + "11", year + "05" + "16"]
        else:
            return year_month + "11"


def make_day_q(year_month):
    year = year_month[:4]
    month = year_month[4:]
    if year <= "2012":
        if month in ["03"]:
            return [year + "04" + "01"]  # 3/31
        elif month in ["08"]:
            return [year + "09" + "01"]  # 8/31
        elif month in ["10"]:
            return [year + "11" + "01"]  # 10/31
        elif month in ["04"]:
            return [year + "05" + "01"]  # 4/31
        return None

    else:
        if month in ["08", "11"]:
            return [year_month + "15"]
        elif month in ["03"]:
            return [year + "04" + "01"]
        elif month in ["05"]:
            return [year + "05" + "16"]
        return None


def create_rebalance_date(date_type="q"):
    df = pd_read_mssql_data(
        "SELECT DISTINCT [日期] FROM price_adj_d;", database="cmoney_price"
    )
    if date_type == "q":
        make_day = make_day_q
    else:
        make_day = make_day_m
    # 創建年月
    rebalance_year = [str(x) for x in np.arange(1980, 9999)]
    rebalance_month = [
        str(z) if len(str(z)) == 2 else "0" + str(z) for z in np.arange(1, 13)
    ]
    rebalance_year_month = []
    for i in range(len(rebalance_year)):
        for j in range(len(rebalance_month)):
            rebalance_year_month.append(rebalance_year[i] + rebalance_month[j])
    # 依據規則加入日期
    rebalance_year_month_day = []
    for i in range(len(rebalance_year_month)):
        if type(make_day(rebalance_year_month[i])) == list:
            rebalance_year_month_day += list(make_day(rebalance_year_month[i]))
        else:
            if make_day(rebalance_year_month[i]):
                rebalance_year_month_day += [make_day(rebalance_year_month[i])]

    df_date = pd.DataFrame()
    df_date["rebalance_day"] = rebalance_year_month_day
    target = np.array(df_date["rebalance_day"])
    A = np.sort(df["日期"].unique())
    idx = A.searchsorted(target)
    idx = np.clip(idx, 0, len(A) - 1)
    Result = A[idx]
    Result = np.where(Result < target, np.NaN, Result)
    df_date["rebalance_day_trade"] = Result
    df_date = df_date.dropna()
    df_date = df_date.drop_duplicates(subset=["rebalance_day_trade"], keep="last")
    df_date["rebalance_day_trade_next"] = df_date["rebalance_day_trade"].shift(-1)
    df_date["rebalance_day_trade_next"] = df_date["rebalance_day_trade_next"].fillna(
        A[-1]
    )
    return df_date[["rebalance_day_trade", "rebalance_day_trade_next"]]


def make_key_q(date):
    date = str(date)
    year = date[:4]
    month = date[4:]
    if year >= "2013":
        if month <= "0331":
            return str(int(year) - 1) + "03"

        elif (month > "0331") & (month <= "0515"):
            return str(int(year) - 1) + "04"

        if (month > "0515") & (month <= "0814"):
            return year + "01"

        elif (month > "0814") & (month <= "1114"):
            return year + "02"

        elif (month > "1114") & (month <= "1231"):
            return year + "03"
    else:
        if month <= "0331":  # 3/31
            return str(int(year) - 1) + "03"

        elif (month > "0331") & (month <= "0430"):  # 4/30
            return str(int(year) - 1) + "04"

        if (month > "0430") & (month <= "0831"):  # 8/31
            return year + "01"

        elif (month > "0831") & (month <= "1031"):  # 10/31
            return year + "02"

        elif (month > "1031") & (month <= "1231"):
            return year + "03"


def make_key_m(date):
    date = str(date)
    year = date[:4]
    month = date[4:6]
    day = date[6:]
    if month == "01":
        if day > "10":
            return str(int(year) - 1) + "12"
        else:
            return str(int(year) - 1) + "11"
    elif month == "02":
        if day > "10":
            return str(int(year)) + "01"
        else:
            return str(int(year) - 1) + "12"
    else:
        if day > "10":
            if len(str(int(month) - 1)) == 1:
                return year + "0" + str(int(month) - 1)
            else:
                return year + str(int(month) - 1)
        else:
            if len(str(int(month) - 2)) == 1:
                return year + "0" + str(int(month) - 2)
            else:
                return year + str(int(month) - 2)


def read_info():
    df_firm_info_listed = pd_read_mssql_data("SELECT * FROM firm_info_listed;")
    df_firm_info_delisted = pd_read_mssql_data("SELECT * FROM firm_info_delisted;")
    df_firm_info_listed = df_firm_info_listed[
        ["年度", "股票代號", "股票名稱", "產業代號", "產業名稱"]
    ]
    df_firm_info_delisted = df_firm_info_delisted[
        ["年度", "股票代號", "股票名稱", "產業代號", "產業名稱"]
    ]
    df_firm_info = pd.concat([df_firm_info_listed, df_firm_info_delisted])
    len(df_firm_info["股票代號"].unique()) == len(df_firm_info)  # 必須為正
    del df_firm_info_listed, df_firm_info_delisted
    return df_firm_info


def read_return(date_type="q"):
    df_date = create_rebalance_date(date_type)
    df_firm_info = read_info()
    df = pd_read_mssql_data(
        "SELECT [日期],[股票代號],[收盤價] FROM price_adj_d;", database="cmoney_price"
    )
    df_return = (
        df.pivot(columns="股票代號", index="日期", values="收盤價").shift(-1)
        / df.pivot(columns="股票代號", index="日期", values="收盤價")
        - 1
    )
    df_return = df_return.replace([np.inf, -np.inf], np.nan)
    df_return["日期"] = df_return.index
    df_return = df_return.melt(id_vars="日期", value_name="return")
    df["return"] = pd.merge(
        left=df, right=df_return, on=["日期", "股票代號"], how="left"
    )["return"]
    del df_return
    gc.collect()
    # 對齊基本資料表以及股價資料表所有的公司
    cross_set = set(df_firm_info["股票代號"]) & set(df["股票代號"])
    df_firm_info = df_firm_info[df_firm_info["股票代號"].isin(cross_set)]
    df = df[df["股票代號"].isin(cross_set)]
    print("合併報酬股價資料長度 : {}".format(len(df)))

    # 創建每個日期對應的轉倉日欄位
    df = df.reset_index(drop=True)
    # 製作
    target = np.array(df["日期"])
    A = np.array(df_date["rebalance_day_trade"])
    # A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A) - 1)
    left = A[idx - 1]
    right = A[idx]
    idx += target >= right

    df["date_rebalance"] = pd.Series(np.where(target < A.min(), np.NaN, A[idx - 1]))
    return df[["日期", "股票代號", "date_rebalance", "return"]]


def return_last_valid_value(x):
    if pd.isna(x).sum() == len(x):
        return np.NaN
    else:
        return x[x.last_valid_index()]


def performance_stats(Return, Turnover_list, Return_Benchmark):
    # 權益或是累積報酬
    Return = pd.Series(Return)
    Equity = (Return + 1).cumprod()
    Geo_a_Return_annual = (Equity[-1]) ** (250 / len(Equity)) - 1
    Arith_a_Return_annual = (Return.mean() + 1) ** (250) - 1
    # 投組勝率
    Hitrate = sum(Return > 0) / len(Return)
    Return_p_values = stats.ttest_1samp(Return, 0).pvalue
    STD_Return_annual = Return.std() * np.sqrt(250)

    Sharpe_annual = (Geo_a_Return_annual) / STD_Return_annual

    # 計算MDD
    D = Equity.cummax() - Equity
    MDD = D.max() * 1
    d = D / (D + Equity)
    mdd = d.max()

    # 投組勝率
    winrate = sum((Return.values - Return_Benchmark.values.flatten()) > 0) / len(Return)
    # 贏過大盤pvalues
    win_Return_p_values = stats.ttest_1samp(
        Return.values - Return_Benchmark.values.flatten(), 0
    ).pvalue
    # 周轉率
    turnover = np.mean(Turnover_list)
    result = pd.Series(
        [
            Equity[-1],
            Geo_a_Return_annual,
            Arith_a_Return_annual,
            STD_Return_annual,
            Sharpe_annual,
            mdd,
            Return_p_values,
            Hitrate,
            win_Return_p_values,
            winrate,
            turnover,
        ]
    )
    result.index = [
        "Total Return",
        "Geometric Mean Return",
        "Arithmetric Mean Return",
        "Standard Deviation",
        "Sharpe Ratios",
        "mdd",
        "P-Value",
        "Hit Rate",
        "Win-P-Value",
        "Win Rate",
        "turnover",
    ]
    return result.round(2)


def process_cols(col_name, sub_list):
    return re.sub("|".join(sub_list), "", col_name)
