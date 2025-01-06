# %%
import sqlite3

import pandas as pd

# %%
con = sqlite3.connect("./data/data.db")
cursor = con.cursor()
cursor.execute("SELECT * FROM daily_data_new_v2 where [商品代號]='TX     '")
rows = cursor.fetchall()
# %%
df_price = pd.DataFrame(
    rows,
    columns=[
        "成交日期",
        "商品代號",
        "到期月份(週別)",
        "成交時間",
        "成交價格",
        "成交數量",
        "近月價格",
        "遠月價格",
        "開盤集合競價",
    ],
)

# %%
